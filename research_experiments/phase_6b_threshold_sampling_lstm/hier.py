"""HIER exploratory recipe (AMENDMENT_HIER.md): two-stage, header-free four-class model.
Stage 1 Normal / Middle / Severe; stage 2 Mild vs Moderate inside Middle; one combined pipeline.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.hier          (pre-registered grid)
    python -m research_experiments.phase_6b_threshold_sampling_lstm.hier --ext    (Extension 1 grid)
"""
import json
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.experiments.nested import BASES, MLPF
from research_experiments.phase_5.p5common import corr_ci, corr_t, holm, make_X, metrics
from research_experiments.phase_6.p6common import PAT_GROUP, inner_oof
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import P2, outer_ecgfmlp, sfx

warnings.filterwarnings("ignore")
FAMS = "qrs+st+hrv+freq+wavelet+leadstat+sq+logits150"
BLOCKS = ["ecgf", "feat:" + FAMS]
EXT = "--ext" in sys.argv                                    # grid extension, AMENDMENT_HIER.md "Extension 1"
CS = (0.1, 0.3, 1.0, 3.0, 10.0) if EXT else (0.001, 0.003, 0.01, 0.03, 0.1)
WS = np.round(np.arange(0.3 if EXT else 0.6, 2.01, 0.1), 2)
TS = np.round(np.arange(0.30, 0.701, 0.02), 2)
SPEC = {"ecgfmlp": (["ecgf"], "p2heads", MLPF, B.SEEDS), "ecgflindist": BASES["ecgflindist"]}
CACHE = B.PRIV / ("hier_ext" if EXT else "hier")
Y = B.Y
Y3 = np.where(Y == 0, 0, np.where(Y == 3, 2, 1))            # Normal / Middle / Severe


def lr(C, n):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def logloss(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))


def fold_job(r, k):
    """LR members of both stages for outer fold (r, k): chosen C, inner OOF on the training portion, outer predictions."""
    f = B.guard(CACHE / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); TE = np.flatnonzero(fv == k)
    fm = np.zeros(B.nD, bool); fm[TR] = True
    X, _ = make_X(BLOCKS, fm)
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], PAT_GROUP()[TR]))
    mid = np.isin(Y, (1, 2))
    best1 = best2 = None
    for C in CS:
        P1 = np.zeros((len(TR), 3)); q = np.zeros(len(TR))
        for ia, ib in inner:
            a, b = TR[ia], TR[ib]
            P1[ib] = lr(C, 3).fit(X[a], Y3[a]).predict_proba(X[b])
            am = a[mid[a]]
            q[ib] = lr(C, 2).fit(X[am], (Y[am] == 2).astype(int)).predict_proba(X[b])[:, 1]
        l1 = logloss(Y3[TR], P1)
        mm = mid[TR]; l2 = logloss((Y[TR][mm] == 2).astype(int), np.c_[1 - q[mm], q[mm]])
        if best1 is None or l1 < best1[1]:
            best1 = (C, l1, P1)
        if best2 is None or l2 < best2[1]:
            best2 = (C, l2, q)
    o1 = lr(best1[0], 3).fit(X[TR], Y3[TR]).predict_proba(X[TE])
    tm = TR[mid[TR]]
    o2 = lr(best2[0], 2).fit(X[tm], (Y[tm] == 2).astype(int)).predict_proba(X[TE])[:, 1]
    out = dict(TR=TR, TE=TE, C1=best1[0], C2=best2[0], in1=best1[2], in2=best2[2], out1=o1, out2=o2)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


def combine(S1, q, w_mid, w_sev, t):
    s = S1 * np.array([1.0, w_mid, w_sev])[None]
    g = s.argmax(1)
    pred = np.where(g == 0, 0, np.where(g == 2, 3, np.where(q >= t, 2, 1)))
    P4 = np.c_[S1[:, 0], S1[:, 1] * (1 - q), S1[:, 1] * q, S1[:, 2]]
    return pred, P4


def fast(y, pred):
    rc = np.array([np.mean(pred[y == c] == c) for c in range(4)])
    f1 = []
    for c in range(4):
        tp = np.sum((pred == c) & (y == c)); pp = np.sum(pred == c); ap = np.sum(y == c)
        f1.append(0 if tp == 0 else 2 * tp / (pp + ap))
    return float(np.mean(pred == y)), rc, float(np.mean(f1))


def tune(S1, q, y):
    base_pred, _ = combine(S1, q, 1.0, 1.0, 0.5)
    _, _, f1b = fast(y, base_pred)
    best = (1.0, 1.0, 0.5, -1)
    for wm in WS:
        for ws in WS:
            s = S1 * np.array([1.0, wm, ws])[None]; g = s.argmax(1)
            for t in TS:
                pred = np.where(g == 0, 0, np.where(g == 2, 3, np.where(q >= t, 2, 1)))
                acc, rc, f1 = fast(y, pred)
                if rc.min() >= 0.15 and f1 >= f1b - 0.01 and acc > best[3] + 1e-12:
                    best = (float(wm), float(ws), float(t), acc)
    return best[:3]


def vote_parts(P4):
    S1 = np.c_[P4[:, 0], P4[:, 1] + P4[:, 2], P4[:, 3]]
    q = P4[:, 2] / np.clip(P4[:, 1] + P4[:, 2], 1e-9, None)
    return S1, q


def main():
    jobs = Parallel(n_jobs=4)(delayed(fold_job)(r, k) for r in (0, 1, 2) for k in range(5))
    J = {(r, k): jobs[5 * r + k] for r in (0, 1, 2) for k in range(5)}
    res = {n: [] for n in ("HIER", "HIER-vote", "IMG-C3", "C3 (with header)")}
    cellacc = {n: [] for n in res}; extra = {"HIER": [], "HIER-vote": []}; params = []
    for r in (0, 1, 2):
        Pm, _ = outer_ecgfmlp(r)
        Pl = B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]
        Pv = (Pm + Pl) / 2
        fo = B.fold_vec(r)
        preds = {n: np.zeros(B.nD, int) for n in res}; probs = {n: np.zeros((B.nD, 4)) for n in res}
        for k in range(5):
            j = J[(r, k)]; TR, TE = j["TR"], j["TE"]
            inn = [inner_oof(b, r, k, SPEC[b]) for b in ("ecgfmlp", "ecgflindist")]
            assert np.array_equal(inn[0][0], TR)
            Vin = np.mean([x[1] for x in inn], 0)
            vS1i, vqi = vote_parts(Vin); vS1o, vqo = vote_parts(Pv[TE])
            # HIER: vote + LR in both stages
            S1i = (vS1i + j["in1"]) / 2; qi = (vqi + j["in2"]) / 2
            S1o = (vS1o + j["out1"]) / 2; qo = (vqo + j["out2"]) / 2
            wm, ws, t = tune(S1i, qi, Y[TR])
            preds["HIER"][TE], probs["HIER"][TE] = combine(S1o, qo, wm, ws, t)
            params.append(dict(r=r, k=k, C_stage1=float(j["C1"]), C_stage2=float(j["C2"]), w_mid=wm, w_sev=ws, t=t))
            # HIER-vote: two-stage decision on the IMG-vote probabilities only
            wm2, ws2, t2 = tune(vS1i, vqi, Y[TR])
            preds["HIER-vote"][TE], probs["HIER-vote"][TE] = combine(vS1o, vqo, wm2, ws2, t2)
            # IMG-C3 reference (as imgonly.py)
            prior = np.bincount(Y[TR], minlength=4) / len(TR)
            a = choose_alpha(Vin, Y[TR], prior, "protected")
            preds["IMG-C3"][TE] = B.decide(Pv[TE], "prior", a, prior); probs["IMG-C3"][TE] = Pv[TE]
        P, p = B.load_run(f"P6A-c3-B1-v1__s0_r{r}")
        preds["C3 (with header)"], probs["C3 (with header)"] = p, P
        for n in res:
            m = B.metrics(Y, probs[n], preds[n])
            pr = [float(np.mean(Y[preds[n] == c] == c)) if (preds[n] == c).any() else 0.0 for c in range(4)]
            m.update({f"precision_{c}": pr[i] for i, c in enumerate(B.CLASSES)})
            m["confusion"] = [[int(np.sum((Y == a) & (preds[n] == b))) for b in range(4)] for a in range(4)]
            res[n].append(m)
            cellacc[n] += [float(np.mean(preds[n][fo == k] == Y[fo == k])) for k in range(5)]
            if n in extra:
                g = np.where(preds[n] == 0, 0, np.where(preds[n] == 3, 2, 1))
                midm = np.isin(Y, (1, 2)) & np.isin(preds[n], (1, 2))
                extra[n].append(dict(stage1_3class_accuracy=float(np.mean(g == Y3)),
                                     stage2_accuracy_on_true_mid=float(np.mean(preds[n][np.isin(Y, (1, 2))] == Y[np.isin(Y, (1, 2))])),
                                     stage2_accuracy_when_both_mid=float(np.mean(preds[n][midm] == Y[midm]))))
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "macro_auroc", "min_recall"] + \
           [f"recall_{c}" for c in B.CLASSES] + [f"precision_{c}" for c in B.CLASSES]
    out = dict(n_ecgs=int(B.nD), class_counts=[int((Y == c).sum()) for c in range(4)], models={}, comparisons=[], params=params)
    for n, v in res.items():
        s = {k: float(np.mean([x[k] for x in v])) for k in keys if k in v[0]}
        s["accuracy_ci95_corrected"] = corr_ci(cellacc[n]); s["accuracy_per_repeat"] = [round(x["accuracy"], 4) for x in v]
        s["confusion_mean"] = np.mean([x["confusion"] for x in v], 0).round(1).tolist()
        if n in extra:
            s.update({k: float(np.mean([x[k] for x in extra[n]])) for k in extra[n][0]})
        out["models"][n] = s
    comps = [(a, b) for a in ("HIER", "HIER-vote") for b in ("IMG-C3", "C3 (with header)")]
    pv = []
    for a, b in comps:
        d = np.array(cellacc[a]) - np.array(cellacc[b]); md, p = corr_t(d); pv.append(p)
        out["comparisons"].append(dict(model=a, vs=b, delta_accuracy=md, p=p,
                                       same_sign_all_repeats=bool(len({np.sign(np.mean(d[5 * r:5 * r + 5])) for r in range(3)}) == 1)))
    for c, h in zip(out["comparisons"], holm(pv)):
        c["p_holm"] = float(h)
    (B.guard(B.RES) / ("hier_ext_exploratory.json" if EXT else "hier_exploratory.json")).write_text(json.dumps(out, indent=1))
    for n, s in out["models"].items():
        print(f"{n:18s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} mF1 {s['macro_f1']:.3f} "
              f"rec {[round(s[f'recall_{c}'], 3) for c in B.CLASSES]}", {k: round(s[k], 3) for k in s if k.startswith("stage")})
    for c in out["comparisons"]:
        print(c)


if __name__ == "__main__":
    main()
