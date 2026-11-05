"""MILDFEAT models (AMENDMENT_MILDFEAT.md): image-only clinician features (mildfeat.py) for four-class LVEF, focused on
Mild. No header, age or sex. Phase 2 folds, 3 repeats, original labels, all 837 development ECGs; locked test unused.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mildmodels --run --jobs 8
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mildmodels --report
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg_ef.metrics import evaluate
from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.p5common import blockstd, cells, corr_ci, corr_t, decide, ecgf_f6, holm
from research_experiments.phase_6.p6common import INNER6
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm.mildfeat import OUT as FEAT_CSV

CFG = "IMGF"
LIN = dict(head="linear", loss="ce", weight="balanced")
MLPF = dict(LIN, head="mlp", hidden=64, dropout=0.5)
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
QUALITY = ("valid_leads", "min_beats_per_column", "n_beats")
P2 = B.ROOT / "research_experiments" / "phase_2" / "results"
Y, PAT, nD = B.Y, B.PAT, B.nD
MEMBERS = {"clinlr": (0,), "clinord": (0,), "clingbm": (0,), "mlpclin": tuple(B.SEEDS)}
INNER_BASES = {"clinlr": (0,), "ecgfmlp": tuple(B.SEEDS)}            # ecgflindist inner OOF re-used from Phase 6 (INNER6)
CMD = "python -m research_experiments.phase_6b_threshold_sampling_lstm.mildmodels --run"


# ------------------------------------------------------------------ inputs
_C = None


def clin():
    global _C
    if _C is None:
        D = pd.read_csv(FEAT_CSV)
        assert (D.ecg_id.values == B.IDS).all(), "feature rows do not match the development IDs"
        cols = [c for c in D.columns if c not in ("ecg_id", "_error") + QUALITY]
        _C = (D[cols].values.astype(float), cols)
    return _C


def impute(X, fm):
    med = np.nanmedian(X[fm], 0); med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(X), med[None], X)


def design(kind, fm):
    if kind == "ecgfmlp":
        return ecgf_f6().astype(float), False
    C = impute(clin()[0], fm)
    if kind == "mlpclin":
        return blockstd([C, ecgf_f6().astype(float)], fm), True
    return C, False


def bal_w(y):
    n = np.bincount(y, minlength=4).astype(float)
    return (len(y) / (4 * np.maximum(n, 1)))[y]


# ------------------------------------------------------------------ fitters -> (predict_proba fn, info)
def _lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_lr(Xtr, ytr, Xva, yva, seed, pre):
    best = None
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(Xtr, ytr)
        sc = s3(evaluate(yva, clf.predict_proba(Xva)))
        if best is None or sc > best[0]:
            best = (sc, clf, C)
    return best[1].predict_proba, dict(C=best[2])


def ord_proba(ms, X):
    G = np.stack([m.predict_proba(X)[:, 1] for m in ms], 1)            # P(y > k), k = 0, 1, 2
    G = np.minimum.accumulate(G, axis=1)                                 # monotone in k
    P = np.column_stack([1 - G[:, 0], G[:, 0] - G[:, 1], G[:, 1] - G[:, 2], G[:, 2]])
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def fit_ord(Xtr, ytr, Xva, yva, seed, pre):
    w = bal_w(ytr); best = None
    for C in C_GRID:
        ms = [_lr(C).fit(Xtr, (ytr > k).astype(int), logisticregression__sample_weight=w) for k in range(3)]
        sc = s3(evaluate(yva, ord_proba(ms, Xva)))
        if best is None or sc > best[0]:
            best = (sc, ms, C)
    return (lambda X, ms=best[1]: ord_proba(ms, X)), dict(C=best[2])


def fit_gbm(Xtr, ytr, Xva, yva, seed, pre):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=seed, verbose=-1, n_jobs=1)
    m.fit(Xtr, ytr, sample_weight=bal_w(ytr), eval_set=[(Xva, yva)], eval_sample_weight=[bal_w(yva)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict_proba, dict(best_iteration=int(m.best_iteration_ or 400))


def fit_mlp(Xtr, ytr, Xva, yva, seed, pre):
    f = p2heads.fit(Xtr, ytr, Xva, yva, dict(MLPF, no_std=pre), seed=seed)
    return (lambda Z, f=f: f["predict"](Z)[0]), dict(l2=f.get("l2"))


FIT = {"clinlr": fit_lr, "clinord": fit_ord, "clingbm": fit_gbm, "mlpclin": fit_mlp, "ecgfmlp": fit_mlp}


def rid(model, seed, r):
    return f"P6B-{model}-{CFG}-v1__s{seed}_r{r}"


# ------------------------------------------------------------------ outer members and nested inner OOF
def outer(task):
    model, seed, r = task
    run = rid(model, seed, r)
    if B.done(run):
        return run
    torch.set_num_threads(1); B.mark_running(run, CMD)
    P = np.zeros((nD, 4)); info = []; t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(5):
            itr, iva, ote = B.split("ORIG", r, k)
            fm = np.zeros(nD, bool); fm[itr] = True
            X, pre = design(model, fm)
            pf, inf = FIT[model](X[itr], Y[itr], X[iva], Y[iva], seed, pre)
            P[ote] = pf(X[ote]); info.append(dict(fold=k, **inf))
    B.save_run(run, P, P.argmax(1), r, CFG, dict(model=model, seed=seed, repeat=r, inputs="image-only: mildfeat CLIN"
                                                  + (" + ECGFounder f6" if model == "mlpclin" else "")),
               extra=dict(folds=info, n_features=len(clin()[1]), train_minutes=(time.time() - t0) / 60))
    return run


def fit_predict(base, fit_rows, pred_rows, seed, split_seed):
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, Y[fit_rows], PAT[fit_rows]))
    tr, va = fit_rows[a], fit_rows[b]
    fm = np.zeros(nD, bool); fm[tr] = True
    X, pre = design(base, fm)
    return FIT[base](X[tr], Y[tr], X[va], Y[va], seed, pre)[0](X[pred_rows])


def inner(task):
    """Nested inner OOF (Phase 5 protocol: 4 inner folds rs 1000+10r+k, selection split rs 2000+10k+j)."""
    base, r, k = task
    if base == "ecgflindist":
        z = np.load(INNER6 / "ecgflindist" / f"r{r}_k{k}.npz"); return z["TR"], z["P"]
    f = B.guard(B.INNER / f"{base}_{CFG}" / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    torch.set_num_threads(1)
    TR = np.flatnonzero(B.fold_vec(r) != k); P = np.zeros((len(TR), 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, (ia, ib) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], PAT[TR])):
            assert not set(PAT[TR[ia]]) & set(PAT[TR[ib]]), "inner patient leakage"
            P[ib] = np.mean([fit_predict(base, TR[ia], TR[ib], s, 2000 + 10 * k + j) for s in INNER_BASES[base]], 0)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, TR=TR, P=P)
    return TR, P


# ------------------------------------------------------------------ ECGFounder members (Phase 2 runs) and votes
def sfx(r):
    return "" if r == 0 else f"_r{r}"


def ecgf_mlp(r):
    return np.mean([B.load_oof_csv(P2 / ("ecgf_aligned_f6_mlp" if (s == 42 and r == 0) else f"ecgf_f6_mlp_seed{s}{sfx(r)}") / "oof.csv")[0]
                    for s in B.SEEDS], 0)


def ecgf_lin(r):
    return B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]


def member(model, r):
    return np.mean([B.load_run(rid(model, s, r))[0] for s in MEMBERS[model]], 0)


VOTES = {"imgvote": (("ecgfmlp", "ecgflindist"), "IMG-vote: ECGF-MLP + ECGF linear (baseline, recomputed)"),
         "imgvoteclin": (("ecgfmlp", "ecgflindist", "clinlr"), "IMG-vote+CLIN: ECGF-MLP + ECGF linear + CLIN-LR")}


def vote_P(name, r):
    parts = {"ecgfmlp": ecgf_mlp(r), "ecgflindist": ecgf_lin(r)}
    return np.mean([parts[m] if m in parts else member(m, r) for m in VOTES[name][0]], 0)


def votes_and_rules():
    for name, (mem, desc) in VOTES.items():
        for r in (0, 1, 2):
            Pv = vote_P(name, r)
            if not B.done(rid(name, 0, r)):
                B.save_run(rid(name, 0, r), Pv, Pv.argmax(1), r, CFG, dict(model=desc, repeat=r))
            c3 = name.replace("vote", "c3")
            if B.done(rid(c3, 0, r)):
                continue
            fo = B.fold_vec(r); pred = np.zeros(nD, int); alphas = []
            for k in range(5):
                inn = [inner((m, r, k)) for m in mem]
                TR = inn[0][0]; assert all((x[0] == TR).all() for x in inn)
                Pin = np.mean([x[1] for x in inn], 0); prior = np.bincount(Y[TR], minlength=4) / len(TR)
                a = choose_alpha(Pin, Y[TR], prior, "protected"); alphas.append(a)
                pred[fo == k] = decide(Pv[fo == k], "prior", a, prior)
            B.save_run(rid(c3, 0, r), Pv, pred, r, CFG, dict(model=desc.split(":")[0].replace("vote", "C3") + " + nested prior rule (protected)",
                                                             repeat=r), extra=dict(alpha_per_fold=alphas))
            print(f"[{c3} r{r}] acc {np.mean(pred == Y):.3f} alpha {alphas}", flush=True)


# ------------------------------------------------------------------ evaluation
MODELS = {  # name -> (loader(r) -> (P, pred), label, family)
    "ECGF-MLP": (lambda r: (ecgf_mlp(r), ecgf_mlp(r).argmax(1)), "baseline"),
    "IMG-vote": (lambda r: B.load_run(rid("imgvote", 0, r)), "baseline"),
    "IMG-C3": (lambda r: B.load_run(rid("imgc3", 0, r)), "baseline"),
    "CLIN-LR": (lambda r: B.load_run(rid("clinlr", 0, r)), "new"),
    "CLIN-ORD": (lambda r: B.load_run(rid("clinord", 0, r)), "new"),
    "CLIN-GBM": (lambda r: B.load_run(rid("clingbm", 0, r)), "new"),
    "IMG-MLP+CLIN": (lambda r: (member("mlpclin", r), member("mlpclin", r).argmax(1)), "new"),
    "IMG-vote+CLIN": (lambda r: B.load_run(rid("imgvoteclin", 0, r)), "new"),
    "IMG-C3+CLIN": (lambda r: B.load_run(rid("imgc3clin", 0, r)), "new"),
}
PAIRS = (("IMG-vote+CLIN", "IMG-vote"), ("IMG-C3+CLIN", "IMG-C3"), ("IMG-MLP+CLIN", "ECGF-MLP"), ("CLIN-ORD", "CLIN-LR"))


def mild_aucs(P):
    out = {}
    for name, (a, b) in dict(normal_vs_mild=(0, 1), mild_vs_moderate=(1, 2)).items():
        m = np.isin(Y, [a, b]); s = P[m, b] / (P[m, a] + P[m, b] + 1e-12)
        out[f"auroc_{name}"] = float(roc_auc_score(Y[m] == b, s))
    out["auroc_reduced_vs_normal"] = float(roc_auc_score(Y > 0, 1 - P[:, 0]))
    return out


def summarise(name):
    load = MODELS[name][0]; ms, cl = [], []
    for r in (0, 1, 2):
        P, pred = load(r)
        m = B.metrics(Y, P, pred); m.update(mild_aucs(P)); ms.append(m)
        cl += [c["accuracy"] for c in cells(P, pred, r)]
    o = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
    o.update(model=name, family=MODELS[name][1], accuracy_ci95_corrected=corr_ci(cl),
             accuracy_per_repeat=[round(x["accuracy"], 4) for x in ms], admissible=bool(o["min_recall"] >= 0.15))
    return o


def paired(a, b, metric):
    d, dr = [], []
    for r in (0, 1, 2):
        Pa, pa = MODELS[a][0](r); Pb, pb = MODELS[b][0](r)
        d += [x[metric] - z[metric] for x, z in zip(cells(Pa, pa, r), cells(Pb, pb, r))]
        dr.append(B.metrics(Y, Pa, pa)[metric] - B.metrics(Y, Pb, pb)[metric])
    _, p = corr_t(d)
    return dict(candidate=a, comparator=b, metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr],
                ci95_corrected_delta=corr_ci(d), p_corrected_t=p,
                same_sign_all_repeats=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)))


def doctors_table(n_boot=1000):
    X, cols = clin(); rng = np.random.default_rng(0)
    pats = pd.unique(PAT); groups = {p: np.flatnonzero(PAT == p) for p in pats}
    boots = [np.concatenate([groups[p] for p in rng.choice(pats, len(pats))]) for _ in range(n_boot)]
    rows = []
    for j, c in enumerate(cols):
        x = X[:, j]; row = dict(feature=c, missing=float(np.isnan(x).mean()))
        for i, k in enumerate(B.CLASSES):
            row[f"median_{k}"] = float(np.nanmedian(x[Y == i])) if np.isfinite(x[Y == i]).any() else np.nan
        for name, (a, b) in dict(normal_vs_mild=(0, 1), mild_vs_moderate=(1, 2)).items():
            m = np.isin(Y, [a, b]) & np.isfinite(x)
            if (Y[m] == b).sum() < 10 or len(np.unique(x[m])) < 2:
                continue
            row[f"auroc_{name}"] = float(roc_auc_score(Y[m] == b, x[m]))
            row[f"p_{name}"] = float(mannwhitneyu(x[m & (Y == a)], x[m & (Y == b)]).pvalue)
            bs = []
            for idx in boots:
                mi = idx[np.isin(Y[idx], [a, b]) & np.isfinite(x[idx])]
                if len(np.unique(Y[mi])) == 2:
                    bs.append(roc_auc_score(Y[mi] == b, x[mi]))
            row[f"auroc_{name}_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        rows.append(row)
    T = pd.DataFrame(rows)
    for name in ("normal_vs_mild", "mild_vs_moderate"):
        p = T[f"p_{name}"].values; ok = np.isfinite(p); q = np.full(len(p), np.nan)
        o = np.argsort(p[ok]); n = ok.sum(); adj = p[ok][o] * n / np.arange(1, n + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]; tmp = np.empty(n); tmp[o] = np.minimum(adj, 1); q[ok] = tmp
        T[f"q_bh_{name}"] = q
    return T


def report():
    out = dict(n_features=len(clin()[1]), models=[summarise(n) for n in MODELS], comparisons=[])
    for a, b in PAIRS:
        for metric in ("accuracy", "macro_f1", "f1_Mild", "recall_Mild"):
            out["comparisons"].append(paired(a, b, metric))
    adj = holm([x["p_corrected_t"] for x in out["comparisons"]])
    adm = {m["model"]: m["admissible"] for m in out["models"]}
    for x, a in zip(out["comparisons"], adj):
        x["p_holm"] = float(a)
        x["genuine_improvement"] = bool(x["delta"] > 0 and a < 0.05 and x["same_sign_all_repeats"] and adm[x["candidate"]])
    for n in ("clinlr", "clinord", "clingbm", "mlpclin"):
        a = [json.loads(B.agg_path(rid(n, s, r)).read_text()).get("train_minutes", np.nan) for s in MEMBERS[n] for r in (0, 1, 2)]
        out.setdefault("train_minutes_per_run", {})[n] = float(np.mean(a))
    T = doctors_table()
    T.to_csv(B.guard(B.RES / "mildfeat_doctors_table.csv"), index=False, float_format="%.4g")
    out["doctors_table_top"] = dict(
        normal_vs_mild=T.assign(d=(T.auroc_normal_vs_mild - 0.5).abs()).sort_values("d", ascending=False).head(15)
        [["feature", "auroc_normal_vs_mild", "auroc_normal_vs_mild_ci95", "q_bh_normal_vs_mild", "median_Normal", "median_Mild"]].to_dict("records"),
        mild_vs_moderate=T.assign(d=(T.auroc_mild_vs_moderate - 0.5).abs()).sort_values("d", ascending=False).head(15)
        [["feature", "auroc_mild_vs_moderate", "auroc_mild_vs_moderate_ci95", "q_bh_mild_vs_moderate", "median_Mild", "median_Moderate"]].to_dict("records"))
    (B.guard(B.RES) / "mildfeat_models.json").write_text(json.dumps(out, indent=1, default=float))
    M = pd.DataFrame(out["models"]); M["accuracy_ci95_corrected"] = M.accuracy_ci95_corrected.map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    M.drop(columns=["accuracy_per_repeat"]).to_csv(B.guard(B.RES / "mildfeat_models.csv"), index=False, float_format="%.4f")
    pd.DataFrame(out["comparisons"]).to_csv(B.guard(B.RES / "mildfeat_comparisons.csv"), index=False, float_format="%.4f")
    plot(out, T)
    return out


def plot(out, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = out["models"]; names = [m["model"] for m in M]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, keys, title in zip(axes, (("accuracy", "macro_f1"), ("recall_Mild", "precision_Mild", "f1_Mild"),
                                      ("auroc_normal_vs_mild", "auroc_mild_vs_moderate", "auroc_reduced_vs_normal")),
                               ("Four-class accuracy / macro-F1", "Mild: recall / precision / F1", "Mild-related AUROC")):
        w = 0.8 / len(keys)
        for j, k in enumerate(keys):
            v = [m[k] for m in M]
            ax.bar(np.arange(len(M)) + j * w, v, w, label=k)
        ax.set_xticks(np.arange(len(M)) + 0.4 - w / 2); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=6); ax.set_ylim(0, 1 if "AUROC" not in title else 1)
    fig.suptitle("Image-only models with clinician ECG features (development CV, mean of 3 repeats; no header)", fontsize=10)
    fig.tight_layout(); B.guard(B.PLOTS).mkdir(parents=True, exist_ok=True)
    fig.savefig(B.PLOTS / "mildfeat_models.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, name in zip(axes, ("normal_vs_mild", "mild_vs_moderate")):
        S = T.dropna(subset=[f"auroc_{name}"]).assign(d=lambda d: (d[f"auroc_{name}"] - 0.5).abs()).sort_values("d").tail(20)
        lo = S[f"auroc_{name}_ci95"].map(lambda v: v[0]).values; hi = S[f"auroc_{name}_ci95"].map(lambda v: v[1]).values
        y = np.arange(len(S)); a = S[f"auroc_{name}"].values
        ax.errorbar(a, y, xerr=[a - lo, hi - a], fmt="o", ms=3, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels(S.feature, fontsize=7); ax.axvline(0.5, c="k", lw=0.8, ls=":")
        ax.set_title(f"Univariate AUROC {name.replace('_', ' ')} (top 20; patient-bootstrap 95 % CI)", fontsize=9)
        ax.set_xlabel("AUROC (> 0.5: higher in the more severe class)")
    fig.tight_layout(); fig.savefig(B.PLOTS / "mildfeat_doctors_table.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    if a.run:
        clin()
        tasks = [(m, s, r) for m, seeds in MEMBERS.items() for s in seeds for r in (0, 1, 2)]
        itasks = [(b, r, k) for b in INNER_BASES for r in (0, 1, 2) for k in range(5)]
        with Pool(a.jobs) as pool:
            for x in pool.imap_unordered(outer, tasks):
                print("done", x, flush=True)
            for _ in pool.imap_unordered(inner, itasks):
                pass
        print("inner OOF done", flush=True)
        votes_and_rules()
    if a.report:
        out = report()
        for m in out["models"]:
            print(f"{m['model']:14s} acc {m['accuracy']:.3f} {np.round(m['accuracy_ci95_corrected'], 3).tolist()} mF1 {m['macro_f1']:.3f} "
                  f"Mild R {m['recall_Mild']:.3f} P {m['precision_Mild']:.3f} F1 {m['f1_Mild']:.3f} | AUROC N-vs-Mild "
                  f"{m['auroc_normal_vs_mild']:.3f} Mild-vs-Mod {m['auroc_mild_vs_moderate']:.3f} reduced {m['auroc_reduced_vs_normal']:.3f} "
                  f"| min rec {m['min_recall']:.3f}")


if __name__ == "__main__":
    main()
