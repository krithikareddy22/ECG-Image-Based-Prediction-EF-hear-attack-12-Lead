"""ENH members, combiners, statistics and error analysis (AMENDMENT_ENH.md). Header-free; development CV only.
Needs enh_build.py and enh_rocket.py first.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.enh --members     (M2-M6, nested; cached)
    python -m research_experiments.phase_6b_threshold_sampling_lstm.enh --combine     (vote / stack, stats, errors)
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.p5common import blockstd, corr_ci, corr_t, ecgf_f6, feature_family, holm, impute, make_X
from research_experiments.phase_6.p6common import PAT_GROUP, inner_oof
from research_experiments.phase_6a_normal_mild_lstm import p6acommon as P6A
from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import P2, outer_ecgfmlp, sfx

warnings.filterwarnings("ignore")
D = B.PRIV / "enh"; MEM = D / "members"
Y = B.Y; nD = B.nD
FAMS = H.FAMS
CS = 10.0 ** np.arange(-4, 1.01, 0.5)
ALPHAS_R = (1.0, 10.0, 100.0, 1e3, 1e4)
WG = np.round(np.arange(0.4, 2.51, 0.1), 2)
MEMBERS = ("M2", "M3", "M4", "M5", "M6")
LT = P6A.label_table().loc[B.IDS]
EF = LT.ef.values.astype(float)
_BUILD = None


def build():
    global _BUILD
    if _BUILD is None:
        z = np.load(D / "build.npz")
        assert (z["ecg_id"] == np.asarray(B.IDS, dtype="U32")).all()
        _BUILD = {k: z[k] for k in ("f6", "logits", "beats")}
    return _BUILD


def std_fit(X, fm):
    mu, sd = X[fm].mean(0), X[fm].std(0) + 1e-6
    return (X - mu) / sd


def features(m, r, k, fm):
    b = build()
    if m == "M2":
        return blockstd([b["f6"].astype(float), b["logits"].astype(float)], fm)
    if m == "M2ref":                                       # Diagnostic D1: M2 head on the aligned input
        return blockstd([ecgf_f6().astype(float), feature_family("logits150")[0]], fm)
    if m == "M3":
        return std_fit(b["beats"].reshape(nD, -1).astype(float), fm)
    if m == "M4":
        return std_fit(np.load(D / f"rocket_r{r}_k{k}.npy").astype(np.float32), fm)
    if m == "M5":
        return make_X(["ecgf", "feat:" + FAMS], fm)[0]
    if m == "M6":
        W = np.concatenate([impute(feature_family(f)[0], fm) for f in FAMS.split("+")], 1)
        return blockstd([ecgf_f6().astype(float), b["f6"].astype(float), W], fm)
    raise KeyError(m)


def ef_probs(mu, sd):
    c = norm.cdf((np.array([31.0, 41.0, 53.0])[None] - mu[:, None]) / sd)
    return np.c_[1 - c[:, 2], c[:, 2] - c[:, 1], c[:, 1] - c[:, 0], c[:, 0]]      # Normal, Mild, Moderate, Severe


def member_job(m, r, k):
    f = B.guard(MEM / f"{m}_r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); TE = np.flatnonzero(fv == k)
    fm = np.zeros(nD, bool); fm[TR] = True
    X = features(m, r, k, fm).astype(np.float32 if m == "M4" else np.float64)
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], PAT_GROUP()[TR]))
    if m == "M6":
        best = None
        for a in ALPHAS_R:
            mu = np.zeros(len(TR))
            for ia, ib in inner:
                mu[ib] = Ridge(alpha=a).fit(X[TR[ia]], EF[TR[ia]]).predict(X[TR[ib]])
            rmse = float(np.sqrt(np.mean((mu - EF[TR]) ** 2)))
            if best is None or rmse < best[1]:
                best = (a, rmse, mu)
        a, sd, mu = best
        inn = ef_probs(mu, sd)
        out = ef_probs(Ridge(alpha=a).fit(X[TR], EF[TR]).predict(X[TE]), sd)
        res = dict(TR=TR, TE=TE, param=a, inner=inn, outer=out, sd=sd)
    else:
        oof = {C: np.zeros((len(TR), 4)) for C in CS}
        for ia, ib in inner:
            lr = LogisticRegression(class_weight="balanced", max_iter=2000, warm_start=True)
            for C in CS:                                   # increasing C, warm-started
                lr.set_params(C=C).fit(X[TR[ia]], Y[TR[ia]]); oof[C][ib] = lr.predict_proba(X[TR[ib]])
        ll = {C: float(-np.mean(np.log(np.clip(P[np.arange(len(TR)), Y[TR]], 1e-9, 1)))) for C, P in oof.items()}
        C = min(ll, key=ll.get)
        out = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X[TR], Y[TR]).predict_proba(X[TE])
        res = dict(TR=TR, TE=TE, param=C, inner=oof[C], outer=out, edge=float(C in (CS[0], CS[-1])))
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **res)
    return res


# ------------------------------------------------------------------ decision rule (vectorised grid)
W = np.array([[1.0, a, b, c] for a in WG for b in WG for c in WG])


def grid_stats(P, y):
    pred = (P[:, None, :] * W[None]).argmax(2)                         # n x G
    acc = (pred == y[:, None]).mean(0)
    rec = np.stack([(pred[y == c] == c).mean(0) for c in range(4)])    # 4 x G
    f1 = []
    for c in range(4):
        tp = ((pred == c) & (y[:, None] == c)).sum(0); pp = (pred == c).sum(0); ap = (y == c).sum()
        f1.append(np.where(tp > 0, 2 * tp / np.maximum(pp + ap, 1), 0.0))
    return acc, rec, np.mean(f1, 0)


def tune(P, y):
    acc, rec, f1 = grid_stats(P, y)
    base = int(np.flatnonzero((W == [1, 1, 1, 1]).all(1))[0])
    ok = (rec.min(0) >= 0.15) & (f1 >= f1[base] - 0.01)
    if not ok.any():
        return W[base]
    i = np.flatnonzero(ok)[np.argmax(acc[ok])]
    return W[i]


def logit(P):
    return np.log(np.clip(P, 1e-6, 1))


def per_fold(r, k, Pv, J):
    """Members' inner/outer probabilities -> ENH-vote and ENH-stack predictions for outer fold k."""
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); TE = np.flatnonzero(fv == k)
    inn = [inner_oof(b, r, k, H.SPEC[b]) for b in ("ecgfmlp", "ecgflindist")]
    assert np.array_equal(inn[0][0], TR)
    Ins = [np.mean([x[1] for x in inn], 0)] + [J[(m, r, k)]["inner"] for m in MEMBERS]
    Outs = [Pv[TE]] + [J[(m, r, k)]["outer"] for m in MEMBERS]
    for m in MEMBERS:
        assert np.array_equal(J[(m, r, k)]["TR"], TR)
    res = {}
    vi, vo = np.mean(Ins, 0), np.mean(Outs, 0)
    w = tune(vi, Y[TR]); res["ENH-vote"] = (vo, (vo * w).argmax(1), w)
    Zi, Zo = np.concatenate([logit(p) for p in Ins], 1), np.concatenate([logit(p) for p in Outs], 1)
    meta_oof = np.zeros((len(TR), 4))
    for ia, ib in StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, Y[TR], PAT_GROUP()[TR]):
        meta_oof[ib] = LogisticRegression(C=1.0, max_iter=3000).fit(Zi[ia], Y[TR][ia]).predict_proba(Zi[ib])
    w2 = tune(meta_oof, Y[TR])
    so = LogisticRegression(C=1.0, max_iter=3000).fit(Zi, Y[TR]).predict_proba(Zo)
    res["ENH-stack"] = (so, (so * w2).argmax(1), w2)
    for j, m in enumerate(("M1",) + MEMBERS):                       # diagnostics: each member alone, argmax
        res[m] = (Outs[j], Outs[j].argmax(1), None)
    o = J[("M2ref", r, k)]["outer"]; res["M2ref"] = (o, o.argmax(1), None)   # Diagnostic D1
    # reference IMG-C3 (as imgonly.py) and HIER-ext (cached Extension-1 folds)
    prior = np.bincount(Y[TR], minlength=4) / len(TR)
    a = choose_alpha(Ins[0], Y[TR], prior, "protected")
    res["IMG-C3"] = (Pv[TE], B.decide(Pv[TE], "prior", a, prior), None)
    j = np.load(B.PRIV / "hier_ext" / f"r{r}_k{k}.npz")
    vS1i, vqi = H.vote_parts(Ins[0]); vS1o, vqo = H.vote_parts(Pv[TE])
    S1i, qi = (vS1i + j["in1"]) / 2, (vqi + j["in2"]) / 2; S1o, qo = (vS1o + j["out1"]) / 2, (vqo + j["out2"]) / 2
    H.WS = np.round(np.arange(0.3, 2.01, 0.1), 2)
    wm, ws, t = H.tune(S1i, qi, Y[TR]); p, Pp = H.combine(S1o, qo, wm, ws, t)
    res["HIER-ext"] = (Pp, p, None)
    return TE, res


NEIGH = {"Normal->Mild": (0, 1), "Mild->Normal": (1, 0), "Mild->Moderate": (1, 2), "Moderate->Mild": (2, 1),
         "Moderate->Severe": (2, 3), "Severe->Moderate": (3, 2)}


def describe(y, p):
    out = {k: float(np.mean(p[y == a] == b)) for k, (a, b) in NEIGH.items()}
    err = p != y
    out["error_rate"] = float(err.mean())
    out["share_of_errors_2plus_classes_away"] = float(np.mean(np.abs(p[err] - y[err]) >= 2)) if err.any() else 0.0
    return out


def errors_by(y, preds, groups):
    """Mean over repeats of the error rate per group (descriptive)."""
    out = {}
    for g in pd.unique(groups):
        m = groups == g
        out[str(g)] = dict(n=int(m.sum()), error_rate=float(np.mean([np.mean(p[m] != y[m]) for p in preds])))
    return out


def combine():
    jobs = {(m, r, k): member_job(m, r, k) for m in MEMBERS + ("M2ref",) for r in (0, 1, 2) for k in range(5)}
    names = ["ENH-stack", "ENH-vote", "IMG-C3", "HIER-ext", "C3 (with header)", "M1", *MEMBERS, "M2ref"]
    preds = {n: [] for n in names}; probs = {n: [] for n in names}; cell = {n: [] for n in names}; wts = []
    for r in (0, 1, 2):
        Pm, _ = outer_ecgfmlp(r)
        Pv = (Pm + B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]) / 2
        pr = {n: np.zeros(nD, int) for n in names}; pb = {n: np.zeros((nD, 4)) for n in names}
        for k in range(5):
            TE, res = per_fold(r, k, Pv, jobs)
            for n, (P, p, w) in res.items():
                pb[n][TE], pr[n][TE] = P, p
            wts.append(dict(r=r, k=k, vote=res["ENH-vote"][2].tolist(), stack=res["ENH-stack"][2].tolist()))
        Pc, pc = B.load_run(f"P6A-c3-B1-v1__s0_r{r}"); pb["C3 (with header)"], pr["C3 (with header)"] = Pc, pc
        fo = B.fold_vec(r)
        for n in names:
            preds[n].append(pr[n]); probs[n].append(pb[n])
            cell[n] += [float(np.mean(pr[n][fo == k] == Y[fo == k])) for k in range(5)]
    out = dict(n_ecgs=nD, class_counts=[int((Y == c).sum()) for c in range(4)], models={}, comparisons=[], decision_weights=wts,
               member_params={m: [float(jobs[(m, r, k)]["param"]) for r in (0, 1, 2) for k in range(5)] for m in MEMBERS},
               member_param_at_grid_edge={m: float(np.mean([jobs[(m, r, k)].get("edge", 0.0) for r in (0, 1, 2) for k in range(5)])) for m in MEMBERS})
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "macro_auroc", "min_recall"] + [f"recall_{c}" for c in B.CLASSES]
    for n in names:
        ms = [B.metrics(Y, P, p) for P, p in zip(probs[n], preds[n])]
        s = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        for c, cn in enumerate(B.CLASSES):
            s[f"precision_{cn}"] = float(np.mean([np.mean(Y[p == c] == c) if (p == c).any() else 0.0 for p in preds[n]]))
        s["accuracy_ci95_corrected"] = corr_ci(cell[n]); s["accuracy_per_repeat"] = [round(m["accuracy"], 4) for m in ms]
        s["admissible"] = bool(s["min_recall"] >= 0.15)
        s["confusion_mean"] = np.mean([[[int(np.sum((Y == a) & (p == b))) for b in range(4)] for a in range(4)] for p in preds[n]], 0).round(1).tolist()
        d = [describe(Y, p) for p in preds[n]]
        s["neighbour_confusion"] = {k: float(np.mean([x[k] for x in d])) for k in d[0]}
        out["models"][n] = s
    comps = [(a, b) for a in ("ENH-stack", "ENH-vote") for b in ("IMG-C3", "C3 (with header)", "HIER-ext")]
    pv = []
    for a, b in comps:
        dlt = np.array(cell[a]) - np.array(cell[b]); md, p = corr_t(dlt); pv.append(p)
        out["comparisons"].append(dict(model=a, vs=b, delta_accuracy=md, p=p,
                                       same_sign_all_repeats=bool(len({np.sign(dlt[5 * r:5 * r + 5].mean()) for r in range(3)}) == 1)))
    for c, h in zip(out["comparisons"], holm(pv)):
        c["p_holm"] = float(h)
    # ---- error analysis ("why"), descriptive
    nb = np.where(LT.male.values, 52.0, 54.0)
    dist = np.min(np.abs(EF[:, None] - np.c_[np.full(nD, 31.0), np.full(nD, 41.0), nb]), 1)
    dbin = pd.cut(dist, [-0.01, 3, 6, 10, 1e9], labels=["<3", "3-6", "6-10", ">=10"]).astype(str)
    from research_experiments.phase_6b_threshold_sampling_lstm.hc30 import HC
    q = feature_family("sq")[0][:, feature_family("sq")[1].index("quality_score")]
    qbin = pd.cut(q, np.nanpercentile(q, [0, 33.3, 66.7, 100]), include_lowest=True, duplicates="drop").astype(str)
    mild = Y == 1
    pos = np.where(mild, (EF - 41.0) / (nb - 41.0), np.nan)
    mbin = np.where(~mild, "not Mild", np.where(pos < 1 / 3, "Mild lower third (near Moderate)", np.where(pos < 2 / 3, "Mild middle third", "Mild upper third (near Normal)")))
    ea = {}
    for n in ("ENH-stack", "C3 (with header)"):
        ea[n] = dict(by_ef_distance_to_boundary=errors_by(Y, preds[n], dbin),
                     by_echo_link=errors_by(Y, preds[n], np.where(HC, "HC30 (exact name, echo <= 30 d)", "other links")),
                     by_digitisation_quality=errors_by(Y, preds[n], qbin),
                     by_mild_band_position={g: dict(n=int((mbin == g).sum()),
                                                    predicted_share={cn: float(np.mean([np.mean(p[mbin == g] == c) for p in preds[n]])) for c, cn in enumerate(B.CLASSES)})
                                            for g in pd.unique(mbin) if g != "not Mild"})
    out["error_analysis"] = ea
    (B.guard(B.RES) / "enh_exploratory.json").write_text(json.dumps(out, indent=1))
    for n, s in out["models"].items():
        print(f"{n:18s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} mF1 {s['macro_f1']:.3f} "
              f"rec {[round(s[f'recall_{c}'], 3) for c in B.CLASSES]} adm {s['admissible']}")
    for c in out["comparisons"]:
        print(c)
    print(json.dumps(out["member_param_at_grid_edge"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--members", action="store_true"); ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.members:
        Parallel(n_jobs=4)(delayed(member_job)(m, r, k) for m in ("M4", "M2", "M3", "M5", "M6") for r in (0, 1, 2) for k in range(5))
    if a.combine:
        combine()
