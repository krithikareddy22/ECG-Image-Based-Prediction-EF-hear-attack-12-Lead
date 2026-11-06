"""ULT (AMENDMENT_ULT.md): large diverse header-free ensemble, owner objective (max accuracy, every recall >= 0.15).
    python -m research_experiments.phase_6b_threshold_sampling_lstm.ult --members
    python -m research_experiments.phase_6b_threshold_sampling_lstm.ult --combine
"""
import argparse
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier

from research_experiments.phase_5.p5common import blockstd, corr_ci, corr_t, ecgf_f6, feature_family, holm, impute, make_X
from research_experiments.phase_6.p6common import PAT_GROUP, inner_oof
from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import P2, outer_ecgfmlp, sfx

warnings.filterwarnings("ignore")
D = B.PRIV / "ult"
Y = B.Y; nD = B.nD; EF = E.EF
ZONE = np.isin(Y, (1, 2)) | ((Y == 0) & (EF < 62))
SPEC = {"M8": ("lr", tuple(E.CS)), "M9": ("knn", (15, 30, 60)), "M10": ("hgb", (100, 300)), "M12": ("lr", tuple(E.CS))}
NEW = tuple(SPEC)
OLD = ("M2", "M4", "M5", "M6")
_MS = None


def multiscale():
    global _MS
    if _MS is None:
        z = np.load(P2.parent / "models" / "ecgf_aligned_features.npz")
        pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[B.IDS].values
        _MS = {k: z[k][pos].astype(float) for k in ("f4", "f5", "f6max")}
    return _MS


def std(X, fm):
    return (X - X[fm].mean(0)) / (X[fm].std(0) + 1e-6)


def feats(m, fm):
    if m == "M8":
        ms = multiscale(); return blockstd([ms["f4"], ms["f5"], ms["f6max"]], fm)
    if m == "M9":
        X = np.c_[std(ecgf_f6().astype(float), fm), std(E.build()["f6"].astype(float), fm)]
        return PCA(64, random_state=0).fit(X[fm]).transform(X)
    if m == "M10":
        W = np.concatenate([impute(feature_family(f)[0], fm) for f in E.FAMS.split("+")], 1)
        F6 = std(ecgf_f6().astype(float), fm)
        return np.c_[W, PCA(32, random_state=0).fit(F6[fm]).transform(F6)]
    if m == "M12":
        return make_X(["ecgf", "feat:" + E.FAMS], fm)[0]
    raise KeyError(m)


def fit_pred(kind, g, Xa, ya, Xb):
    if kind == "lr":
        mdl = LogisticRegression(C=g, class_weight="balanced", max_iter=3000)
    elif kind == "knn":
        mdl = KNeighborsClassifier(n_neighbors=g, weights="distance")
    else:
        mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=g, l2_regularization=1.0,
                                             class_weight="balanced", random_state=0)
    P = mdl.fit(Xa, ya).predict_proba(Xb)
    out = np.full((len(Xb), 4), 1e-3); out[:, mdl.classes_] = P
    return out / out.sum(1, keepdims=True)


def logloss(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))


def member(m, r, k):
    f = B.guard(D / f"{m}_r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    TR, TE, inner = Bd.splits(r, k)
    fm = np.zeros(nD, bool); fm[TR] = True
    X = feats(m, fm); kind, grid = SPEC[m]
    ok = ZONE if m == "M12" else np.ones(nD, bool)
    best = None
    for g in grid:
        P = np.zeros((len(TR), 4))
        for ia, ib in inner:
            a = TR[ia][ok[TR[ia]]]
            P[ib] = fit_pred(kind, g, X[a], Y[a], X[TR[ib]])
        ll = logloss(Y[TR], P)
        if best is None or ll < best[1]:
            best = (g, ll, P)
    a = TR[ok[TR]]
    out = dict(TR=TR, TE=TE, param=best[0], inner=best[2], outer=fit_pred(kind, best[0], X[a], Y[a], X[TE]),
               edge=float(best[0] in (grid[0], grid[-1])))
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


# ------------------------------------------------------------------ combination
def tune_floor(P, y, floor):
    acc, rec, _ = E.grid_stats(P, y)
    for fl in (floor, 0.15):
        ok = rec.min(0) >= fl
        if ok.any():
            return E.W[np.flatnonzero(ok)[np.argmax(acc[ok])]]
    return np.array([1.0, 1.0, 1.0, 1.0])


def caruana(Ins, y, steps=25):
    cnt, cur, n = Counter(), None, 0
    for _ in range(steps):
        best = None
        for nm, P in Ins.items():
            Q = P if cur is None else (cur * n + P) / (n + 1)
            ll = logloss(y, Q)
            if best is None or ll < best[1]:
                best = (nm, ll, Q)
        cnt[best[0]] += 1; cur, n = best[2], n + 1
    return {nm: cnt[nm] / steps for nm in Ins}


def lp(P):
    return np.log(np.clip(P, 1e-6, 1))


def fold(r, k, Pv):
    TR, TE, _ = Bd.splits(r, k)
    inn = [inner_oof(b, r, k, H.SPEC[b]) for b in ("ecgfmlp", "ecgflindist")]
    assert np.array_equal(inn[0][0], TR)
    Ins, Outs = {"M1": np.mean([x[1] for x in inn], 0)}, {"M1": Pv[TE]}
    for m in OLD:
        j = E.member_job(m, r, k); assert np.array_equal(j["TR"], TR); Ins[m], Outs[m] = j["inner"], j["outer"]
    for m in NEW:
        j = member(m, r, k); assert np.array_equal(j["TR"], TR); Ins[m], Outs[m] = j["inner"], j["outer"]
    h = np.load(B.PRIV / "hier_ext" / f"r{r}_k{k}.npz")
    vS1i, vqi = H.vote_parts(Ins["M1"]); vS1o, vqo = H.vote_parts(Pv[TE])
    Ins["H"] = H.combine((vS1i + h["in1"]) / 2, (vqi + h["in2"]) / 2, 1, 1, 0.5)[1]
    Outs["H"] = H.combine((vS1o + h["out1"]) / 2, (vqo + h["out2"]) / 2, 1, 1, 0.5)[1]
    mu = Bd.mu_job(r, k)
    yt = Y[TR]; res = {}; info = {}
    # select
    w = caruana(Ins, yt); info["select_weights"] = w
    si = sum(w[n] * Ins[n] for n in Ins); so = sum(w[n] * Outs[n] for n in Outs)
    # stack
    names = list(Ins)
    Zi = np.c_[np.concatenate([lp(Ins[n]) for n in names], 1), mu["inner"] / 50, (mu["inner"] / 50) ** 2]
    Zo = np.c_[np.concatenate([lp(Outs[n]) for n in names], 1), mu["outer"] / 50, (mu["outer"] / 50) ** 2]
    sp = list(StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, yt, PAT_GROUP()[TR]))
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        mo = np.zeros((len(TR), 4))
        for ia, ib in sp:
            mo[ib] = LogisticRegression(C=C, max_iter=5000).fit(Zi[ia], yt[ia]).predict_proba(Zi[ib])
        ll = logloss(yt, mo)
        if best is None or ll < best[1]:
            best = (C, ll, mo)
    info["stack_C"] = best[0]
    sto = LogisticRegression(C=best[0], max_iter=5000).fit(Zi, yt).predict_proba(Zo)
    vi, vo = np.mean(list(Ins.values()), 0), np.mean(list(Outs.values()), 0)
    for tag, floor in (("", 0.18), (" [floor 0.15]", 0.15)):
        for name, Pi, Po in (("ULT-select", si, so), ("ULT-stack", best[2], sto), ("ULT-vote", vi, vo)):
            wt = tune_floor(Pi, yt, floor); res[name + tag] = (Po, (Po * wt).argmax(1))
    return TE, res, info


def combine():
    J = {(r, k): None for r in (0, 1, 2) for k in range(5)}
    for m in E.MEMBERS + ("M2ref",):
        for r in (0, 1, 2):
            for k in range(5):
                J[(m, r, k)] = E.member_job(m, r, k)
    names = None; preds, probs, cells, infos = {}, {}, {}, []
    for r in (0, 1, 2):
        Pm, _ = outer_ecgfmlp(r)
        Pv = (Pm + B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]) / 2
        pr, pb = {}, {}
        for k in range(5):
            TE, res, info = fold(r, k, Pv); infos.append(dict(r=r, k=k, **{a: (b if not isinstance(b, dict) else {x: round(y, 3) for x, y in b.items()}) for a, b in info.items()}))
            _, ref = E.per_fold(r, k, Pv, J)
            res.update({n: (ref[n][0], ref[n][1]) for n in ("ENH-vote", "IMG-C3", "HIER-ext")})
            for n, (P, p) in res.items():
                pr.setdefault(n, np.zeros(nD, int))[TE] = p; pb.setdefault(n, np.zeros((nD, 4)))[TE] = P
        Pc, pc = B.load_run(f"P6A-c3-B1-v1__s0_r{r}"); pr["C3 (with header)"], pb["C3 (with header)"] = pc, Pc
        fo = B.fold_vec(r)
        for n in pr:
            preds.setdefault(n, []).append(pr[n]); probs.setdefault(n, []).append(pb[n])
            cells.setdefault(n, []).extend(float(np.mean(pr[n][fo == k] == Y[fo == k])) for k in range(5))
    out = dict(n_ecgs=nD, class_counts=[int((Y == c).sum()) for c in range(4)], models={}, comparisons=[], fold_info=infos,
               new_member_params={m: [float(member(m, r, k)["param"]) for r in (0, 1, 2) for k in range(5)] for m in NEW})
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "qwk", "macro_auroc", "min_recall"] + [f"recall_{c}" for c in B.CLASSES]
    for n in preds:
        ms = [B.metrics(Y, P, p) for P, p in zip(probs[n], preds[n])]
        s = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        for c, cn in enumerate(B.CLASSES):
            s[f"precision_{cn}"] = float(np.mean([np.mean(Y[p == c] == c) if (p == c).any() else 0.0 for p in preds[n]]))
        s["accuracy_ci95_corrected"] = corr_ci(cells[n]); s["accuracy_per_repeat"] = [round(m["accuracy"], 4) for m in ms]
        s["admissible"] = bool(s["min_recall"] >= 0.15)
        s["confusion_mean"] = np.mean([[[int(np.sum((Y == a) & (p == b))) for b in range(4)] for a in range(4)] for p in preds[n]], 0).round(1).tolist()
        d = [E.describe(Y, p) for p in preds[n]]; s["neighbour_confusion"] = {k: float(np.mean([x[k] for x in d])) for k in d[0]}
        out["models"][n] = s
    comps = [(a, b) for a in ("ULT-select", "ULT-stack", "ULT-vote") for b in ("C3 (with header)", "IMG-C3", "ENH-vote", "HIER-ext")]
    pv = []
    for a, b in comps:
        dl = np.array(cells[a]) - np.array(cells[b]); md, p = corr_t(dl); pv.append(p)
        out["comparisons"].append(dict(model=a, vs=b, delta_accuracy=md, p=p,
                                       same_sign_all_repeats=bool(len({np.sign(dl[5 * r:5 * r + 5].mean()) for r in range(3)}) == 1)))
    for c, h in zip(out["comparisons"], holm(pv)):
        c["p_holm"] = float(h)
    (B.guard(B.RES) / "ult_exploratory.json").write_text(json.dumps(out, indent=1, default=float))
    for n, s in out["models"].items():
        print(f"{n:26s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} mF1 {s['macro_f1']:.3f} "
              f"rec {[round(s[f'recall_{c}'], 3) for c in B.CLASSES]} adm {s['admissible']}")
    for c in out["comparisons"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    sw = pd.DataFrame([i["select_weights"] for i in infos]).mean().round(3).to_dict()
    print("mean selection weights", sw, "| stack C", Counter(i["stack_C"] for i in infos))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--members", action="store_true"); ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.members:
        Parallel(n_jobs=4)(delayed(member)(m, r, k) for m in ("M10", "M8", "M9", "M12") for r in (0, 1, 2) for k in range(5))
    if a.combine:
        combine()
