"""BANDS sensitivity sweep (AMENDMENT_BANDS.md): four-class EF band limits vs header-free ECG performance.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.bands --sweep      (EF grading, all definitions)
    python -m research_experiments.phase_6b_threshold_sampling_lstm.bands --confirm    (retrained vote, selected points)
"""
import argparse
import itertools
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.p5common import corr_ci
from research_experiments.phase_6.p6common import PAT_GROUP
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

warnings.filterwarnings("ignore")
EF = E.EF; MALE = E.LT.male.values.astype(bool); nD = B.nD
D = E.D / "bands"
G = np.arange(20.0, 71.0, 1.0)
ORIG = (52.0, 2.0, 41.0, 31.0)
ESC = (49.9, 0.1, 41.0, 30.0)
MIN_N, MIN_W = 60, 5.0


def normal_cut(cN, o):
    """Per-ECG Normal limit: men EF > cN; women EF >= cN + o (o = 0: same strict rule as men)."""
    return np.where(MALE, cN, cN + o - (0.05 if o > 0 else 0.0))


def labels(cN, o, cMi, cS):
    return np.where(EF > normal_cut(cN, o), 0, np.where(EF >= cMi, 1, np.where(EF >= cS, 2, 3)))


def splits(r, k):
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); TE = np.flatnonzero(fv == k)
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, B.Y[TR], PAT_GROUP()[TR]))
    return TR, TE, inner


def mu_job(r, k):
    """Definition-free EF prediction (ridge on the M6 inputs): inner OOF on the training portion + outer predictions."""
    f = B.guard(D / f"mu_r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    TR, TE, inner = splits(r, k)
    fm = np.zeros(nD, bool); fm[TR] = True
    X = E.features("M6", r, k, fm)
    best = None
    for a in E.ALPHAS_R:
        mu = np.zeros(len(TR))
        for ia, ib in inner:
            mu[ib] = Ridge(alpha=a).fit(X[TR[ia]], EF[TR[ia]]).predict(X[TR[ib]])
        rmse = float(np.sqrt(np.mean((mu - EF[TR]) ** 2)))
        if best is None or rmse < best[1]:
            best = (a, rmse, mu)
    out = dict(TR=TR, TE=TE, alpha=best[0], sd=best[1], inner=best[2], outer=Ridge(alpha=best[0]).fit(X[TR], EF[TR]).predict(X[TE]))
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


IS, IM, IN = np.meshgrid(np.arange(len(G)), np.arange(len(G)), np.arange(len(G)), indexing="ij")
VALID = (IS < IM) & (IM < IN)


def tune_cuts(mu, y):
    """Cut points on predicted EF (Severe < tS <= Moderate < tMi <= Mild < tN <= Normal): max accuracy, all recalls >= 0.15."""
    n = np.array([(y == c).sum() for c in range(4)])
    cnt = [np.searchsorted(np.sort(mu[y == c]), G, side="left") for c in range(4)]   # #(class c with mu < G[i])
    corr = np.stack([n[0] - cnt[0][IN], cnt[1][IN] - cnt[1][IM], cnt[2][IM] - cnt[2][IS], cnt[3][IS]])
    rec = corr / np.maximum(n, 1)[:, None, None, None]
    acc = corr.sum(0) / len(y)
    ok = VALID & (rec.min(0) >= 0.15)
    score = np.where(ok, acc, -1) if ok.any() else np.where(VALID, rec.mean(0), -1)
    i = np.unravel_index(np.argmax(score), score.shape)
    return G[i[0]], G[i[1]], G[i[2]]


def grade(mu, t):
    return np.where(mu >= t[2], 0, np.where(mu >= t[1], 1, np.where(mu >= t[0], 2, 3)))


def summarise(y, preds):
    rows = []
    for p in preds:
        rec = [float(np.mean(p[y == c] == c)) for c in range(4)]
        f1 = []
        for c in range(4):
            tp = np.sum((p == c) & (y == c)); f1.append(0.0 if tp == 0 else 2 * tp / ((p == c).sum() + (y == c).sum()))
        rows.append(dict(accuracy=float(np.mean(p == y)), macro_f1=float(np.mean(f1)), kappa=float(cohen_kappa_score(y, p)),
                         **{f"recall_{c}": rec[i] for i, c in enumerate(B.CLASSES)}))
    s = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    pc = np.bincount(y, minlength=4) / len(y)
    s.update(majority_accuracy=float(pc.max()), chance_accuracy=float(np.sum(pc ** 2)), accuracy_per_repeat=[round(r["accuracy"], 4) for r in rows])
    return s


def eval_def(d, MU):
    y = labels(*d)
    preds = []
    for r in (0, 1, 2):
        p = np.zeros(nD, int)
        for k in range(5):
            m = MU[(r, k)]; t = tune_cuts(m["inner"], y[m["TR"]]); p[m["TE"]] = grade(m["outer"], t)
        preds.append(p)
    n = [int((y == c).sum()) for c in range(4)]
    feas = bool(min(n) >= MIN_N and d[0] - d[2] >= MIN_W and d[2] - d[3] >= MIN_W)
    return dict(c_N=d[0], female_offset=d[1], c_Mi=d[2], c_S=d[3], n_Normal=n[0], n_Mild=n[1], n_Moderate=n[2], n_Severe=n[3],
                feasible=feas, **{k: v for k, v in summarise(y, preds).items() if k != "accuracy_per_repeat"})


def sweep():
    MU = {(r, k): mu_job(r, k) for r in (0, 1, 2) for k in range(5)}
    grid = [(float(a), float(o), float(b), float(c)) for a, o, b, c in itertools.product(range(45, 57), (0, 2), range(35, 47), range(25, 37))
            if a > b > c]
    grid += [ESC]
    grid = [d for d in grid if min(np.bincount(labels(*d), minlength=4)) >= 10]
    rows = Parallel(n_jobs=4, batch_size=16)(delayed(eval_def)(d, MU) for d in grid)
    R = pd.DataFrame(rows)
    R.round(4).to_csv(B.guard(B.RES / "bands_sweep.csv"), index=False)
    F = R[R.feasible]
    det = F[(F.recall_Mild >= 0.25) & (F.recall_Moderate >= 0.25)]
    pick = lambda df, col: df.sort_values(col, ascending=False).iloc[0].to_dict() if len(df) else None
    key = lambda d: (R.c_N == d[0]) & (R.female_offset == d[1]) & (R.c_Mi == d[2]) & (R.c_S == d[3])
    one_d = {"c_N (Normal/Mild)": R[(R.female_offset == 2) & (R.c_Mi == 41) & (R.c_S == 31)].sort_values("c_N"),
             "c_Mi (Mild/Moderate)": R[(R.c_N == 52) & (R.female_offset == 2) & (R.c_S == 31)].sort_values("c_Mi"),
             "c_S (Moderate/Severe)": R[(R.c_N == 52) & (R.female_offset == 2) & (R.c_Mi == 41)].sort_values("c_S")}
    out = dict(n_definitions=int(len(R)), n_feasible=int(len(F)), n_feasible_with_detection=int(len(det)),
               ORIG_EF=R[key(ORIG)].iloc[0].to_dict(), ESC_2021=R[key(ESC)].iloc[0].to_dict(),
               MAX_ACC_DET=pick(det, "accuracy"), MAX_KAPPA=pick(F, "kappa"), MAX_ACC_FEASIBLE_ANY=pick(F, "accuracy"),
               top10_acc_with_detection=det.sort_values("accuracy", ascending=False).head(10).round(4).to_dict("records"),
               top10_kappa=F.sort_values("kappa", ascending=False).head(10).round(4).to_dict("records"),
               one_dimensional={k: v.round(4).to_dict("records") for k, v in one_d.items()},
               ef_regressor=dict(alpha=[float(MU[(r, k)]["alpha"]) for r in (0, 1, 2) for k in range(5)],
                                 inner_rmse_mean=float(np.mean([MU[(r, k)]["sd"] for r in (0, 1, 2) for k in range(5)]))))
    (B.guard(B.RES) / "bands_exploratory.json").write_text(json.dumps(out, indent=1, default=float))
    cols = ["c_N", "female_offset", "c_Mi", "c_S", "n_Normal", "n_Mild", "n_Moderate", "n_Severe", "accuracy", "kappa", "chance_accuracy",
            "recall_Normal", "recall_Mild", "recall_Moderate", "recall_Severe"]
    for k in ("ORIG_EF", "ESC_2021", "MAX_ACC_DET", "MAX_KAPPA", "MAX_ACC_FEASIBLE_ANY"):
        print(k, {c: (round(out[k][c], 3) if isinstance(out[k][c], float) else out[k][c]) for c in cols} if out[k] else None)
    for k, v in one_d.items():
        print("==", k); print(v[cols[:4] + cols[4:8] + ["accuracy", "kappa", "recall_Mild", "recall_Moderate", "feasible"]].round(3).to_string(index=False))


# ------------------------------------------------------------------ confirmation: retrained classifiers on the relabelled classes
def lr_member(m, y, r, k, tag):
    f = B.guard(D / f"{tag}_{m}_r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    TR, TE, inner = splits(r, k)
    fm = np.zeros(nD, bool); fm[TR] = True
    X = E.features(m, r, k, fm)
    oof = {C: np.zeros((len(TR), 4)) for C in E.CS}
    for ia, ib in inner:
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, warm_start=True)
        for C in E.CS:
            lr.set_params(C=C).fit(X[TR[ia]], y[TR[ia]]); oof[C][ib] = lr.predict_proba(X[TR[ib]])
    ll = {C: float(-np.mean(np.log(np.clip(P[np.arange(len(TR)), y[TR]], 1e-9, 1)))) for C, P in oof.items()}
    C = min(ll, key=ll.get)
    out = dict(TR=TR, TE=TE, C=C, inner=oof[C], outer=LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X[TR], y[TR]).predict_proba(X[TE]))
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


def band_probs(mu, sd, d, idx):
    nc = normal_cut(d[0], d[1])[idx]
    cN, cM, cS = norm.cdf((nc - mu) / sd), norm.cdf((d[2] - mu) / sd), norm.cdf((d[3] - mu) / sd)
    return np.c_[1 - cN, cN - cM, cM - cS, cS]


def confirm():
    S = json.loads((B.RES / "bands_exploratory.json").read_text())
    pts = {}
    for name in ("ORIG_EF", "ESC_2021", "MAX_ACC_DET", "MAX_KAPPA"):
        if S[name]:
            d = (S[name]["c_N"], S[name]["female_offset"], S[name]["c_Mi"], S[name]["c_S"])
            pts.setdefault(d, []).append(name)
    MU = {(r, k): mu_job(r, k) for r in (0, 1, 2) for k in range(5)}
    out = {}
    for d, names in pts.items():
        y = labels(*d); tag = "d_" + "_".join(f"{x:g}" for x in d)
        Parallel(n_jobs=4)(delayed(lr_member)(m, y, r, k, tag) for m in ("M2", "M5") for r in (0, 1, 2) for k in range(5))
        preds, cells = [], []
        for r in (0, 1, 2):
            p = np.zeros(nD, int)
            for k in range(5):
                mm = [lr_member(m, y, r, k, tag) for m in ("M2", "M5")]; mu = MU[(r, k)]
                TR, TE = mu["TR"], mu["TE"]
                vi = np.mean([mm[0]["inner"], mm[1]["inner"], band_probs(mu["inner"], mu["sd"], d, TR)], 0)
                vo = np.mean([mm[0]["outer"], mm[1]["outer"], band_probs(mu["outer"], mu["sd"], d, TE)], 0)
                w = E.tune(vi, y[TR]); p[TE] = (vo * w).argmax(1)
            preds.append(p); fo = B.fold_vec(r); cells += [float(np.mean(p[fo == k] == y[fo == k])) for k in range(5)]
        s = summarise(y, preds); s["accuracy_ci95_corrected"] = corr_ci(cells)
        s["definition"] = dict(c_N=d[0], female_offset=d[1], c_Mi=d[2], c_S=d[3]); s["class_counts"] = [int((y == c).sum()) for c in range(4)]
        for n in names:
            out[n] = s
        print(names, d, s["class_counts"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("definition", "class_counts")})
    (B.guard(B.RES) / "bands_confirm.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep", action="store_true"); ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        sweep()
    if a.confirm:
        confirm()
