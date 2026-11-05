"""Phase 7 E8 (PHASE7_PLAN.md Amendment 4): doctor-style hierarchical classification with refer zones, header-free.
Specialists (class-balanced logistic regression) on [ECGFounder f6 || 121 image clinician features || 150 ECGFounder
diagnostic outputs]: A abnormal vs Normal, S Severe vs Mild+Moderate (abnormal rows), M Mild vs Moderate.
Decision: Normal / refer / abnormal -> Severe / refer / not severe -> Mild / refer / Moderate. Thresholds chosen per
outer fold on nested inner OOF; selective variants maximise the answered share under accuracy and Mild / Severe recall
constraints; H-full answers everyone. Aggregates only.
    python -m research_experiments.phase_7.e8_doctor --run
    python -m research_experiments.phase_7.e8_doctor --report
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations_with_replacement, product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.p5common import blockstd, ecgf_f6, impute
from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.e1_ecgf import rep

B = C.B
CS = (0.003, 0.01, 0.03, 0.1)
QS = np.linspace(0.1, 0.9, 9)
VARIANTS = [(T, mr, sr) for T in (0.65, 0.70) for mr in (0.0, 0.05, 0.10) for sr in (0.0, 0.15)]
OUTD = C.PRIV7 / "e8"
SPEC = {"A": lambda y: (np.ones(len(y), bool), (y >= 1).astype(int)),
        "S": lambda y: (y >= 1, (y == 3).astype(int)),
        "M": lambda y: ((y == 1) | (y == 2), (y == 1).astype(int))}


def design(fm):
    from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import clin
    return blockstd([ecgf_f6().astype(float), impute(clin()[0], fm), rep("dx")], fm)


def fit(X, y, rows, C_):
    m = LogisticRegression(C=C_, class_weight="balanced", max_iter=3000)
    return m.fit(X[rows], y[rows])


def specialists(X, Y, fit_rows, val_rows, pred_rows, Cs=None):
    """-> dict name -> P(positive) on pred_rows, and the chosen C per specialist."""
    out, chosen = {}, {}
    for s, f in SPEC.items():
        ok, yy = f(Y)
        fr = fit_rows[ok[fit_rows]]
        if Cs is None:
            vr = val_rows[ok[val_rows]]
            sc = {c: roc_auc_score(yy[vr], fit(X, yy, fr, c).predict_proba(X[vr])[:, 1]) for c in CS}
            chosen[s] = max(sc, key=sc.get)
        else:
            chosen[s] = Cs[s]
        out[s] = fit(X, yy, fr, chosen[s]).predict_proba(X[pred_rows])[:, 1]
    return out, chosen


def decide(pA, pS, pM, t):
    """t = (a_lo, a_hi, s_lo, s_hi, m_lo, m_hi) -> (pred, answered)."""
    a_lo, a_hi, s_lo, s_hi, m_lo, m_hi = t
    pred = np.full(len(pA), -1)
    pred[pA <= a_lo] = 0
    ab = pA >= a_hi
    sev = ab & (pS >= s_hi); pred[sev] = 3
    ns = ab & (pS <= s_lo)
    pred[ns & (pM >= m_hi)] = 1; pred[ns & (pM <= m_lo)] = 2
    return pred, pred >= 0


def job(r):
    warnings.filterwarnings("ignore")
    Y = C.Y; fv = B.fold_vec(r)
    res = {"full": np.zeros(C.nD, int), **{f"v{i}": np.full(C.nD, -1) for i in range(len(VARIANTS))}}
    info = []
    for k in range(5):
        itr, iva, ote = B.split_indices(r, k); TR = np.flatnonzero(fv != k)
        fm = np.zeros(C.nD, bool); fm[itr] = True; X = design(fm)
        po, Cs = specialists(X, Y, itr, iva, ote)
        pin = {s: np.zeros(len(TR)) for s in SPEC}
        for ia, ib in StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], C.PAT[TR]):
            fm2 = np.zeros(C.nD, bool); fm2[TR[ia]] = True; X2 = design(fm2)
            p, _ = specialists(X2, Y, TR[ia], None, TR[ib], Cs)
            for s in SPEC:
                pin[s][ib] = p[s]
        yT = Y[TR]; nM, nS = (yT == 1).sum(), (yT == 3).sum()
        q = {s: np.quantile(pin[s], QS) for s in SPEC}
        pairs = {s: [(lo, hi) for lo, hi in combinations_with_replacement(q[s], 2)] for s in SPEC}
        combos = []                                                         # (t, coverage, acc, mild_rec, sev_rec, min_rec)
        for (al, ah), (sl, sh), (ml, mh) in product(pairs["A"], pairs["S"], pairs["M"]):
            t = (al, ah, sl, sh, ml, mh); pr, ans = decide(pin["A"], pin["S"], pin["M"], t)
            n = ans.sum()
            if n < 20:
                continue
            acc = np.mean(pr[ans] == yT[ans])
            combos.append((t, n / len(TR), acc, np.sum(ans & (pr == 1) & (yT == 1)) / nM, np.sum(ans & (pr == 3) & (yT == 3)) / nS))
        for i, (T, mr, sr) in enumerate(VARIANTS):
            ok = [c for c in combos if c[2] >= T and c[3] >= mr and c[4] >= sr]
            if ok:
                t = max(ok, key=lambda c: c[1])[0]
                pr, ans = decide(po["A"], po["S"], po["M"], t); res[f"v{i}"][ote] = np.where(ans, pr, -1)
            info.append(dict(fold=k, variant=i, feasible=bool(ok)))
        # H-full: lo = hi, floor objective (accuracy s.t. every recall >= 0.15) on the inner OOF
        best = None
        for a, s_, m in product(q["A"], q["S"], q["M"]):
            pr, _ = decide(pin["A"], pin["S"], pin["M"], (a, a, s_, s_, m, m))
            rec = [np.mean(pr[yT == c] == c) for c in range(4)]
            acc = np.mean(pr == yT)
            if min(rec) >= 0.15 and (best is None or acc > best[0]):
                best = (acc, (a, a, s_, s_, m, m))
        t = best[1] if best else (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        res["full"][ote] = decide(po["A"], po["S"], po["M"], t)[0]
        info.append(dict(fold=k, C=Cs, full_feasible=best is not None))
    d = C.guard7(OUTD); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"r{r}.npz", **res); (d / f"info_r{r}.json").write_text(json.dumps(info, default=float))
    print(f"[r{r}] full acc {np.mean(res['full'] == Y):.3f} | " + " ".join(
        f"v{i} cov {np.mean(res[f'v{i}'] >= 0):.2f}" for i in range(len(VARIANTS))), flush=True)
    return r


def report():
    y = C.Y; rows = []
    Z = [np.load(OUTD / f"r{r}.npz") for r in (0, 1, 2)]
    full = [z["full"] for z in Z]
    o = {}; ms, cl = [], []
    for r, p in enumerate(full):
        P = np.eye(4)[p]; ms.append(B.metrics(y, P, p)); cl += [c["accuracy"] for c in C.cells(P, p, r)]
    o = {k: float(np.mean([m[k] for m in ms])) for k in ["accuracy", "balanced_accuracy", "macro_f1"] + [f"{a}_{c}" for c in C.CLASSES for a in ("precision", "recall", "f1")]}
    ci = C.corr_ci(cl)
    rows.append(dict(model="H-full (all patients)", answered=1.0, accuracy=o["accuracy"], accuracy_ci95=f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                     macro_f1=o["macro_f1"], mild_precision=o["precision_Mild"], mild_recall_answered=o["recall_Mild"], mild_recall_overall=o["recall_Mild"],
                     severe_recall_answered=o["recall_Severe"], severe_recall_overall=o["recall_Severe"],
                     recall_answered_N_Mi_Mo_S="/".join(f"{o[f'recall_{c}']:.2f}" for c in C.CLASSES), predicted_mix="all", infeasible_folds=0))
    for i, (T, mr, sr) in enumerate(VARIANTS):
        reps, cells = [], []
        for r, z in enumerate(Z):
            p = z[f"v{i}"]; ans = p >= 0; fv = B.fold_vec(r)
            for k in range(5):
                s = ans & (fv == k)
                if s.any():
                    cells.append(float(np.mean(p[s] == y[s])))
            reps.append(dict(cov=ans.mean(), acc=np.mean(p[ans] == y[ans]) if ans.any() else np.nan,
                             mprec=np.mean(y[ans & (p == 1)] == 1) if (ans & (p == 1)).any() else np.nan,
                             mrec_a=np.mean(p[ans & (y == 1)] == 1) if (ans & (y == 1)).any() else np.nan,
                             mrec_o=np.sum(ans & (p == 1) & (y == 1)) / (y == 1).sum(),
                             srec_a=np.mean(p[ans & (y == 3)] == 3) if (ans & (y == 3)).any() else np.nan,
                             srec_o=np.sum(ans & (p == 3) & (y == 3)) / (y == 3).sum(),
                             rec=[np.mean(p[ans & (y == c)] == c) if (ans & (y == c)).any() else np.nan for c in range(4)],
                             mix=np.bincount(p[ans], minlength=4) if ans.any() else np.zeros(4)))
        info = [json.loads((OUTD / f"info_r{r}.json").read_text()) for r in (0, 1, 2)]
        inf = sum(1 for x in info for e in x if e.get("variant") == i and not e["feasible"])
        ci = C.corr_ci(cells) if len(cells) > 2 else [np.nan, np.nan]
        f = lambda key: float(np.nanmean([x[key] for x in reps]))                    # noqa: E731
        rows.append(dict(model=f"Hierarchy + refer: acc >= {T:.2f}, Mild >= {mr:.0%}, Severe >= {sr:.0%}", answered=f("cov"), accuracy=f("acc"),
                         accuracy_ci95=f"[{ci[0]:.3f}, {ci[1]:.3f}]", macro_f1=np.nan, mild_precision=f("mprec"), mild_recall_answered=f("mrec_a"),
                         mild_recall_overall=f("mrec_o"), severe_recall_answered=f("srec_a"), severe_recall_overall=f("srec_o"),
                         recall_answered_N_Mi_Mo_S="/".join(f"{v:.2f}" for v in np.nanmean([x["rec"] for x in reps], 0)),
                         predicted_mix="/".join(f"{v:.0f}" for v in np.mean([x["mix"] for x in reps], 0)), infeasible_folds=inf))
    D = pd.DataFrame(rows)
    D.to_csv(C.guard7(C.RES7 / "e8_doctor.csv"), index=False, float_format="%.4f")
    (C.guard7(C.RES7) / "e8_full_metrics.json").write_text(json.dumps(dict(**o, accuracy_ci95_corrected=C.corr_ci(cl)), indent=1))
    pd.set_option("display.width", 260)
    print(D.to_string(index=False, float_format="%.3f"))
    print(json.dumps({k: round(v, 3) for k, v in o.items()}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        rep("dx"); B.table()
        with Pool(3) as pool:
            for r in pool.imap_unordered(job, (0, 1, 2)):
                print("done", r, flush=True)
    if a.report:
        report()


if __name__ == "__main__":
    main()
