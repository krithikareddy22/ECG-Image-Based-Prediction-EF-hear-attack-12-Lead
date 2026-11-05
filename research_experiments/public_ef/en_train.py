"""Train on EchoNext (printout format), evaluate on the 837 hospital development ECGs (PUBLIC_EF_PLAN.md). Laptop only.
Needs: en_prep.py done; phase_6b enh_build.py run (synth f6 of the hospital ECGs); Phase 6 caches restored.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.enh_build        (if data/private/enh/build.npz is missing)
    python -m research_experiments.public_ef.en_train
Writes per-ECG outputs only to the git-ignored data/private/; commits aggregates (results/en_train_results.json).
"""
import json
import warnings

import numpy as np
from scipy.stats import norm, pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from research_experiments.phase_5.p5common import corr_ci, corr_t, ecgf_f6, holm
from research_experiments.phase_6.p6common import PAT_GROUP, inner_oof
from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import P2, outer_ecgfmlp, sfx
from research_experiments.public_ef.en_prep import PRIV, RES

warnings.filterwarnings("ignore")
CS = 10.0 ** np.arange(-4, 1.01, 0.5)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)
Y, EF, nD = B.Y, E.EF, B.nD
NCUT = Bd.normal_cut(52.0, 2.0)                       # per-ECG Normal limit of the hospital ECGs (52 men / 54 women)


def metrics(y, p):
    rec = [float(np.mean(p[y == c] == c)) for c in range(4)]
    pre = [float(np.mean(y[p == c] == c)) if (p == c).any() else 0.0 for c in range(4)]
    return dict(accuracy=float(np.mean(p == y)), min_recall=min(rec), **{f"recall_{c}": rec[i] for i, c in enumerate(B.CLASSES)},
                **{f"precision_{c}": pre[i] for i, c in enumerate(B.CLASSES)})


def T_metrics_ef(y, p, mu, ef):
    return dict(**metrics(y, p), ef_r=float(pearsonr(mu, ef)[0]), ef_rmse=float(np.sqrt(np.mean((mu - ef) ** 2))))


def train_en():
    from research_experiments.public_ef.en_split import locked      # Amendment 3: EchoNext locked test patients
    z = np.load(PRIV / "en_cache.npz", allow_pickle=False)
    X = np.c_[z["f6_aligned"], z["f6_synth"]].astype(float); y = z["y"]; ef = z["ef"]; g = z["patient_key"]
    L = locked(g); D = np.flatnonzero(~L)
    mu_, sd_ = X[D].mean(0), X[D].std(0) + 1e-6
    Xs = (X - mu_) / sd_
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(Xs[D], y[D], g[D]))
    tr, va = D[tr], D[va]
    best_c = min(CS, key=lambda C: -np.mean(np.log(np.clip(
        LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xs[tr], y[tr]).predict_proba(Xs[va])[np.arange(len(va)), y[va]], 1e-9, 1))))
    best_a = min(ALPHAS, key=lambda a: float(np.mean((Ridge(alpha=a).fit(Xs[tr], ef[tr]).predict(Xs[va]) - ef[va]) ** 2)))
    lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000).fit(Xs[tr], y[tr])
    rg = Ridge(alpha=best_a).fit(Xs[tr], ef[tr])
    internal = dict(n_train=int(len(tr)), n_heldout=int(len(va)), C=float(best_c), alpha=float(best_a),
                    heldout=metrics(y[va], lr.predict_proba(Xs[va]).argmax(1)),
                    heldout_ef_r=float(pearsonr(rg.predict(Xs[va]), ef[va])[0]),
                    heldout_ef_rmse=float(np.sqrt(np.mean((rg.predict(Xs[va]) - ef[va]) ** 2))),
                    class_counts={c: int((y == i).sum()) for i, c in enumerate(B.CLASSES)})
    lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000).fit(Xs[D], y[D])  # refit on all non-locked rows
    rg = Ridge(alpha=best_a).fit(Xs[D], ef[D])
    if L.any():
        pl = rg.predict(Xs[L])
        internal["echonext_locked_test"] = dict(n=int(L.sum()), patients=int(len(set(g[L]))),
                                                **T_metrics_ef(y[L], lr.predict_proba(Xs[L]).argmax(1), pl, ef[L]))
    Xh = (np.c_[ecgf_f6(), E.build()["f6"]].astype(float) - mu_) / sd_
    P, mu = lr.predict_proba(Xh), rg.predict(Xh)
    np.savez(PRIV / "en_hospital_outputs.npz", ecg_id=np.asarray(B.IDS, dtype="U32"), P=P, mu=mu)
    return internal, P, mu


def band(mu, sd, idx):
    cN, cM, cS = (norm.cdf((c - mu) / sd) for c in (NCUT[idx], 41.0, 31.0))
    return np.c_[1 - cN, cN - cM, cM - cS, cS]


def fold(r, k, Pv, Pen, mu_en, plus):
    TR, TE, _ = Bd.splits(r, k)
    inn = [inner_oof(b, r, k, H.SPEC[b]) for b in ("ecgfmlp", "ecgflindist")]
    Ins, Outs = {"M1": np.mean([x[1] for x in inn], 0)}, {"M1": Pv[TE]}
    for m in ("M5", "M6"):
        j = E.member_job(m, r, k); Ins[m], Outs[m] = j["inner"], j["outer"]
    mu = Bd.mu_job(r, k)
    zi, zo = [mu["inner"] / 50], [mu["outer"] / 50]
    if plus:
        sd = float(np.std(mu_en[TR] - EF[TR]))
        Ins["EN-cls"], Outs["EN-cls"] = Pen[TR], Pen[TE]
        Ins["EN-band"], Outs["EN-band"] = band(mu_en[TR], sd, TR), band(mu_en[TE], sd, TE)
        zi.append(mu_en[TR] / 50); zo.append(mu_en[TE] / 50)
    names = list(Ins); yt = Y[TR]
    Zi = np.column_stack([np.concatenate([U.lp(Ins[n]) for n in names], 1), *zi])
    Zo = np.column_stack([np.concatenate([U.lp(Outs[n]) for n in names], 1), *zo])
    sp = list(StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, yt, PAT_GROUP()[TR]))
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        mo = np.zeros((len(TR), 4))
        for ia, ib in sp:
            mo[ib] = LogisticRegression(C=C, max_iter=5000).fit(Zi[ia], yt[ia]).predict_proba(Zi[ib])
        ll = U.logloss(yt, mo)
        if best is None or ll < best[1]:
            best = (C, ll, mo)
    so = LogisticRegression(C=best[0], max_iter=5000).fit(Zi, yt).predict_proba(Zo)
    return TE, (so * U.tune_floor(best[2], yt, 0.18)).argmax(1)


def main():
    internal, Pen, mu_en = train_en()
    out = dict(internal_echonext=internal, transfer={}, models={}, comparisons=[])
    m01, m12 = np.isin(Y, (0, 1)), np.isin(Y, (1, 2))
    out["transfer"]["auroc_ef_le_45"] = float(roc_auc_score(EF <= 45, -mu_en))
    out["transfer"]["auroc_normal_vs_mild"] = float(roc_auc_score(Y[m01] == 1, -mu_en[m01]))
    out["transfer"]["auroc_mild_vs_moderate"] = float(roc_auc_score(Y[m12] == 2, -mu_en[m12]))
    out["transfer"]["ef_r_vs_echo"] = float(pearsonr(mu_en, EF)[0])
    out["transfer"]["zero_shot_argmax"] = metrics(Y, Pen.argmax(1))
    preds, cells = {}, {}
    for r in (0, 1, 2):
        Pm, _ = outer_ecgfmlp(r)
        Pv = (Pm + B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]) / 2
        pr = {n: np.zeros(nD, int) for n in ("EN zero-shot + tuned decision", "stack-BASE", "stack-PLUS")}
        for k in range(5):
            TR, TE, _ = Bd.splits(r, k)
            pr["EN zero-shot + tuned decision"][TE] = (Pen[TE] * U.tune_floor(Pen[TR], Y[TR], 0.18)).argmax(1)
            for name, plus in (("stack-BASE", False), ("stack-PLUS", True)):
                te, p = fold(r, k, Pv, Pen, mu_en, plus); pr[name][te] = p
        pr["C3 (with header)"] = B.load_run(f"P6A-c3-B1-v1__s0_r{r}")[1]
        fo = B.fold_vec(r)
        for n, p in pr.items():
            preds.setdefault(n, []).append(p); cells.setdefault(n, []).extend(float(np.mean(p[fo == k] == Y[fo == k])) for k in range(5))
    for n, ps in preds.items():
        ms = [metrics(Y, p) for p in ps]
        s = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        s.update(accuracy_ci95_corrected=corr_ci(cells[n]), accuracy_per_repeat=[round(m["accuracy"], 4) for m in ms], admissible=bool(s["min_recall"] >= 0.15))
        out["models"][n] = s
    pv = []
    for a, b in (("stack-PLUS", "stack-BASE"), ("stack-PLUS", "C3 (with header)")):
        d = np.array(cells[a]) - np.array(cells[b]); md, p = corr_t(d); pv.append(p)
        out["comparisons"].append(dict(model=a, vs=b, delta_accuracy=md, p=p, same_sign_all_repeats=bool(len({np.sign(d[5 * r:5 * r + 5].mean()) for r in range(3)}) == 1)))
    for c, h in zip(out["comparisons"], holm(pv)):
        c["p_holm"] = float(h)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "en_train_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("internal_echonext", "transfer", "comparisons")}, indent=1, default=float))
    for n, s in out["models"].items():
        print(f"{n:32s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} rec {[round(s[f'recall_{c}'], 3) for c in B.CLASSES]} adm {s['admissible']}")


if __name__ == "__main__":
    main()
