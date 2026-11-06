"""ECHONEXT (AMENDMENT_ECHONEXT.md): frozen EchoNext-Mini (Hughes et al., NEJM AI 2026; Poterucha et al., Nature 2025)
applied to the header-free 10-s reconstruction of the printed ECGs, then added to the ULT ensemble.
Upstream files are staged (not committed) from a local clone of PierreElias/IntroECG, folder "7-EchoNext Minimodel":
    git clone --depth 1 https://github.com/PierreElias/IntroECG.git <dir>
    ECHONEXT_MINI_DIR="<dir>/7-EchoNext Minimodel" python -m research_experiments.phase_6b_threshold_sampling_lstm.echonext_mini --stage
    python -m research_experiments.phase_6b_threshold_sampling_lstm.echonext_mini --features   (inference + transfer check)
    python -m research_experiments.phase_6b_threshold_sampling_lstm.echonext_mini --members --combine
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.signal import medfilt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.p5common import blockstd, corr_ci, corr_t, holm, make_X
from research_experiments.phase_6.p6common import PAT_GROUP, inner_oof
from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
from research_experiments.phase_6b_threshold_sampling_lstm import enh_build as EB
from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import P2, outer_ecgfmlp, sfx

warnings.filterwarnings("ignore")
UPSTREAM_COMMIT = "15233e9392e9dc10c136a5624abf10199207bce9"
PD = B.PRIV / "echonext_mini"
FILES = {"weights.pt": "models/echonext_multilabel_minimodel/weights.pt",
         "tabular_transformer.joblib": "models/echonext_multilabel_minimodel/tabular_transformer.joblib",
         "waveform_normalization_params.json": "models/echonext_multilabel_minimodel/waveform_normalization_params.json",
         "resnet1d_tabular.py": "cradlenet/models/resnet1d_tabular.py"}
OUTPUTS = ["lvef_lte_45", "lvwt_gte_13", "aortic_stenosis", "aortic_regurgitation", "mitral_regurgitation",
           "tricuspid_regurgitation", "pulmonary_regurgitation", "rv_systolic_dysfunction", "pericardial_effusion",
           "pasp_gte_45", "tr_max_gte_32", "shd"]
MUSE_UV = 4.88
Y = B.Y; nD = B.nD; EF = E.EF


def stage():
    up = Path(os.environ["ECHONEXT_MINI_DIR"])
    PD.mkdir(parents=True, exist_ok=True)
    sha = {}
    for name, rel in FILES.items():
        shutil.copyfile(up / rel, B.guard(PD / name)); sha[name] = hashlib.sha256((PD / name).read_bytes()).hexdigest()
    (B.guard(PD / "provenance.json")).write_text(json.dumps(dict(upstream="PierreElias/IntroECG/7-EchoNext Minimodel", commit=UPSTREAM_COMMIT, sha256=sha), indent=1))
    print(json.dumps(sha, indent=1))


def model():
    spec = importlib.util.spec_from_file_location("echonext_resnet1d_tabular", PD / "resnet1d_tabular.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
    m.load_state_dict(torch.load(PD / "weights.pt", map_location="cpu", weights_only=True)["model"])
    return m.eval()


def en_filter(x):
    """EchoNext baseline wander removal (median 0.2 s then 0.6 s), applied at 500 Hz."""
    x = np.asarray(x, float); b = medfilt(medfilt(x, 101), 301)
    return x - b


def waveforms(kind):
    L, R = EB.dev_signals()
    orig = EB.bandpass
    if kind == "en":
        EB.bandpass = en_filter
    try:
        out = [EB.build_one(l, r) for l, r in zip(L, R)]
    finally:
        EB.bandpass = orig
    S = np.stack([o[0] for o in out])[:, :, ::2] * (1000.0 / MUSE_UV)                    # N x 12 x 2500, MUSE units
    hr = np.array([o[2]["hr"] for o in out])
    p = json.loads((PD / "waveform_normalization_params.json").read_text())
    for i in range(12):
        S[:, i] = (np.clip(S[:, i], p["lowerbound"][i], p["upperbound"][i]) - p["mean"][i]) / p["std"][i]
    return np.transpose(S, (0, 2, 1))[:, None].astype(np.float32), hr                      # N x 1 x 2500 x 12


def tabular(hr, use_hr=True):
    pipe = joblib.load(PD / "tabular_transformer.joblib")
    sc, im = pipe.steps[0][1], pipe.steps[1][1]
    T = np.tile(np.r_[0.5, im.statistics_], (nD, 1))                                      # sex unknown; others median
    if use_hr:
        T[:, 2] = (hr - sc.mean_[1]) / sc.scale_[1]                                       # image-measured heart rate
    return T.astype(np.float32)


@torch.no_grad()
def infer(m, W, T):
    return np.concatenate([torch.sigmoid(m((torch.from_numpy(W[i:i + 64]), torch.from_numpy(T[i:i + 64])))).numpy() for i in range(0, nD, 64)])


def features():
    torch.set_num_threads(4)
    m = model()
    W_en, hr = waveforms("en"); W_bp, _ = waveforms("bp")
    P = dict(primary=infer(m, W_en, tabular(hr)), bandpass=infer(m, W_bp, tabular(hr)), no_hr=infer(m, W_en, tabular(hr, False)))
    np.savez(B.guard(PD / "probs.npz"), ecg_id=np.asarray(B.IDS, dtype="U32"), **P)
    mu = np.mean([_mu_outer(r) for r in (0, 1, 2)], 0)
    out = dict(n=nD, image_heart_rate_median=float(np.median(hr)), variants={})
    for k, p in P.items():
        lv = p[:, 0]
        out["variants"][k] = dict(auroc_ef_le_45=float(roc_auc_score(EF <= 45, lv)),
                                  auroc_normal_vs_mild=float(roc_auc_score(Y[np.isin(Y, (0, 1))] == 1, lv[np.isin(Y, (0, 1))])),
                                  auroc_mild_vs_moderate=float(roc_auc_score(Y[np.isin(Y, (1, 2))] == 2, lv[np.isin(Y, (1, 2))])),
                                  auroc_normal_vs_reduced=float(roc_auc_score(Y > 0, lv)),
                                  spearman_with_ef=float(spearmanr(lv, EF).correlation),
                                  mean_prob_by_class={c: float(lv[Y == i].mean()) for i, c in enumerate(B.CLASSES)})
    out["in_house_ef_ridge_reference"] = dict(auroc_ef_le_45=float(roc_auc_score(EF <= 45, -mu)),
                                              auroc_normal_vs_mild=float(roc_auc_score(Y[np.isin(Y, (0, 1))] == 1, -mu[np.isin(Y, (0, 1))])),
                                              auroc_mild_vs_moderate=float(roc_auc_score(Y[np.isin(Y, (1, 2))] == 2, -mu[np.isin(Y, (1, 2))])),
                                              spearman_with_ef=float(spearmanr(mu, EF).correlation))
    (B.guard(B.RES) / "echonext_transfer.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _mu_outer(r):
    mu = np.zeros(nD)
    for k in range(5):
        z = Bd.mu_job(r, k); mu[z["TE"]] = z["outer"]
    return mu


# ------------------------------------------------------------------ members M13 / M14 and ULT2
def en_logits():
    z = np.load(PD / "probs.npz"); assert (z["ecg_id"] == np.asarray(B.IDS, dtype="U32")).all()
    p = np.clip(z["primary"], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def member(m, r, k):
    f = B.guard(D2 / f"{m}_r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    TR, TE, inner = Bd.splits(r, k)
    fm = np.zeros(nD, bool); fm[TR] = True
    X = U.std(en_logits(), fm) if m == "M13" else blockstd([make_X(["ecgf", "feat:" + E.FAMS], fm)[0], en_logits()], fm)
    best = None
    for C in E.CS:
        P = np.zeros((len(TR), 4))
        for ia, ib in inner:
            P[ib] = U.fit_pred("lr", C, X[TR[ia]], Y[TR[ia]], X[TR[ib]])
        ll = U.logloss(Y[TR], P)
        if best is None or ll < best[1]:
            best = (C, ll, P)
    out = dict(TR=TR, TE=TE, param=best[0], inner=best[2], outer=U.fit_pred("lr", best[0], X[TR], Y[TR], X[TE]))
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


D2 = B.PRIV / "echonext_members"


def fold(r, k, Pv, with_en):
    TR, TE, _ = Bd.splits(r, k)
    inn = [inner_oof(b, r, k, H.SPEC[b]) for b in ("ecgfmlp", "ecgflindist")]
    Ins, Outs = {"M1": np.mean([x[1] for x in inn], 0)}, {"M1": Pv[TE]}
    for m in U.OLD:
        j = E.member_job(m, r, k); Ins[m], Outs[m] = j["inner"], j["outer"]
    for m in U.NEW:
        j = U.member(m, r, k); Ins[m], Outs[m] = j["inner"], j["outer"]
    h = np.load(B.PRIV / "hier_ext" / f"r{r}_k{k}.npz")
    vS1i, vqi = H.vote_parts(Ins["M1"]); vS1o, vqo = H.vote_parts(Pv[TE])
    Ins["H"] = H.combine((vS1i + h["in1"]) / 2, (vqi + h["in2"]) / 2, 1, 1, 0.5)[1]
    Outs["H"] = H.combine((vS1o + h["out1"]) / 2, (vqo + h["out2"]) / 2, 1, 1, 0.5)[1]
    extra_i, extra_o = [], []
    if with_en:
        for m in ("M13", "M14"):
            j = member(m, r, k); assert np.array_equal(j["TR"], TR); Ins[m], Outs[m] = j["inner"], j["outer"]
        lg = en_logits()[:, 0]; extra_i, extra_o = [lg[TR] / 5], [lg[TE] / 5]
    mu = Bd.mu_job(r, k); yt = Y[TR]; res = {}
    w = U.caruana(Ins, yt); si = sum(w[n] * Ins[n] for n in Ins); so = sum(w[n] * Outs[n] for n in Outs)
    names = list(Ins)
    Zi = np.column_stack([np.concatenate([U.lp(Ins[n]) for n in names], 1), mu["inner"] / 50, (mu["inner"] / 50) ** 2, *extra_i])
    Zo = np.column_stack([np.concatenate([U.lp(Outs[n]) for n in names], 1), mu["outer"] / 50, (mu["outer"] / 50) ** 2, *extra_o])
    sp = list(StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, yt, PAT_GROUP()[TR]))
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        mo = np.zeros((len(TR), 4))
        for ia, ib in sp:
            mo[ib] = LogisticRegression(C=C, max_iter=5000).fit(Zi[ia], yt[ia]).predict_proba(Zi[ib])
        ll = U.logloss(yt, mo)
        if best is None or ll < best[1]:
            best = (C, ll, mo)
    sto = LogisticRegression(C=best[0], max_iter=5000).fit(Zi, yt).predict_proba(Zo)
    vi, vo = np.mean(list(Ins.values()), 0), np.mean(list(Outs.values()), 0)
    tag = "ULT2" if with_en else "ULT"
    for name, Pi, Po in ((f"{tag}-select", si, so), (f"{tag}-stack", best[2], sto), (f"{tag}-vote", vi, vo)):
        wt = U.tune_floor(Pi, yt, 0.18); res[name] = (Po, (Po * wt).argmax(1))
    if with_en:
        res["M13 EchoNext-Mini LR (alone)"] = (Outs["M13"], Outs["M13"].argmax(1))
    else:
        prior = np.bincount(yt, minlength=4) / len(TR); a = choose_alpha(Ins["M1"], yt, prior, "protected")
        res["IMG-C3"] = (Pv[TE], B.decide(Pv[TE], "prior", a, prior))
    return TE, res, dict(select_weights=w, stack_C=best[0])


def combine():
    preds, probs, cells, infos = {}, {}, {}, []
    for r in (0, 1, 2):
        Pm, _ = outer_ecgfmlp(r)
        Pv = (Pm + B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]) / 2
        pr, pb = {}, {}
        for k in range(5):
            for with_en in (False, True):
                TE, res, info = fold(r, k, Pv, with_en)
                if with_en:
                    infos.append(dict(r=r, k=k, stack_C=info["stack_C"], select_weights={a: round(b, 3) for a, b in info["select_weights"].items()}))
                for n, (P, p) in res.items():
                    pr.setdefault(n, np.zeros(nD, int))[TE] = p; pb.setdefault(n, np.zeros((nD, 4)))[TE] = P
        Pc, pc = B.load_run(f"P6A-c3-B1-v1__s0_r{r}"); pr["C3 (with header)"], pb["C3 (with header)"] = pc, Pc
        fo = B.fold_vec(r)
        for n in pr:
            preds.setdefault(n, []).append(pr[n]); probs.setdefault(n, []).append(pb[n])
            cells.setdefault(n, []).extend(float(np.mean(pr[n][fo == k] == Y[fo == k])) for k in range(5))
    out = dict(models={}, comparisons=[], fold_info=infos)
    for n in preds:
        ms = [B.metrics(Y, P, p) for P, p in zip(probs[n], preds[n])]
        s = {k: float(np.mean([m[k] for m in ms])) for k in ["accuracy", "balanced_accuracy", "macro_f1", "min_recall"] + [f"recall_{c}" for c in B.CLASSES]}
        for c, cn in enumerate(B.CLASSES):
            s[f"precision_{cn}"] = float(np.mean([np.mean(Y[p == c] == c) if (p == c).any() else 0.0 for p in preds[n]]))
        s.update(accuracy_ci95_corrected=corr_ci(cells[n]), accuracy_per_repeat=[round(m["accuracy"], 4) for m in ms], admissible=bool(s["min_recall"] >= 0.15))
        out["models"][n] = s
    pv = []
    for a, b in (("ULT2-stack", "ULT-stack"), ("ULT2-stack", "C3 (with header)"), ("ULT2-stack", "IMG-C3")):
        if b not in cells:
            continue
        dl = np.array(cells[a]) - np.array(cells[b]); md, p = corr_t(dl); pv.append(p)
        out["comparisons"].append(dict(model=a, vs=b, delta_accuracy=md, p=p, same_sign_all_repeats=bool(len({np.sign(dl[5 * r:5 * r + 5].mean()) for r in range(3)}) == 1)))
    for c, h in zip(out["comparisons"], holm(pv)):
        c["p_holm"] = float(h)
    (B.guard(B.RES) / "echonext_ult2.json").write_text(json.dumps(out, indent=1, default=float))
    for n, s in out["models"].items():
        print(f"{n:30s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} mF1 {s['macro_f1']:.3f} rec {[round(s[f'recall_{c}'], 3) for c in B.CLASSES]} adm {s['admissible']}")
    for c in out["comparisons"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("stage", "features", "members", "combine"):
        ap.add_argument(f"--{a}", action="store_true")
    a = ap.parse_args()
    if a.stage:
        stage()
    if a.features:
        features()
    if a.members:
        Parallel(n_jobs=4)(delayed(member)(m, r, k) for m in ("M14", "M13") for r in (0, 1, 2) for k in range(5))
    if a.combine:
        combine()
