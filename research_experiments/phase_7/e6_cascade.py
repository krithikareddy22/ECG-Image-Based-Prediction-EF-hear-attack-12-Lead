"""Phase 7 E6 (PHASE7_PLAN.md Amendment 2): HEADER-FREE two-stage cascades. Inputs: ECGFounder f6 of the digitised
waveform and the 121 image-only clinician features (mildfeat.py); no printed machine measurements, age, sex or clin3.
Stage 1 answers confident ECGs (header-free EF grading, optionally agreeing with IMG-C3; thresholds set inside the
training folds); stage 2 classifies every referred ECG with an existing image-only model. All 837 ECGs; aggregates only.
    python -m research_experiments.phase_7.e6_cascade --run
    python -m research_experiments.phase_7.e6_cascade --report
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.p5common import blockstd, decide, ecgf_f6, impute
from research_experiments.phase_6.p6common import INNER6
from research_experiments.phase_6b_threshold_sampling_lstm import mildfix as F1
from research_experiments.phase_7 import p7common as C

B = C.B
MODELS_H = ("H1", "H2")
WEIGHTS = {"W1": 1.0, "W.3": 0.3, "W0": 0.0}
VARIANTS = [(m, w) for m in MODELS_H + ("H3",) for w in WEIGHTS]
OUTD = C.PRIV7 / "e6"
_IMG = None


def img():
    global _IMG
    if _IMG is None:
        from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import clin
        _IMG = clin()[0]
    return _IMG


def fitpred(model, wv, fit_rows, pred_rows, clean, split_seed):
    ef = B.table().ef.values; E = ecgf_f6().astype(float)
    if wv == 0.0:
        fit_rows = fit_rows[clean[fit_rows]]; w = np.ones(len(fit_rows))
    else:
        w = np.where(clean[fit_rows], 1.0, wv)
    fm = np.zeros(C.nD, bool); fm[fit_rows] = True
    if model == "H1":
        X = blockstd([E, impute(img(), fm)], fm)
        return RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[fit_rows], ef[fit_rows], sample_weight=w).predict(X[pred_rows])
    import lightgbm as lgb
    mu, sd = E[fm].mean(0), E[fm].std(0) + 1e-6
    X = np.column_stack([img(), PCA(32, random_state=0).fit(((E - mu) / sd)[fm]).transform((E - mu) / sd)])
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, C.Y[fit_rows], C.PAT[fit_rows]))
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, random_state=0, n_jobs=1, verbose=-1)
    m.fit(X[fit_rows[a]], ef[fit_rows[a]], sample_weight=w[a], eval_set=[(X[fit_rows[b]], ef[fit_rows[b]])], eval_sample_weight=[w[b]],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(X[pred_rows])


def imgc3_alphas(r):
    return json.loads((B.EXP / f"P6B-imgc3-IMGF-v1__s0_r{r}.json").read_text())["alpha_per_fold"]


def job(r):
    warnings.filterwarnings("ignore")
    Y = C.Y; ef = B.table().ef.values; bnd = B.table().sex_boundary.values; fv = B.fold_vec(r)
    clean = C.verified()[0]; img()
    _, c3img = B.load_run(f"P6B-imgc3-IMGF-v1__s0_r{r}"); al = imgc3_alphas(r)
    G = np.zeros(C.nD, int); K = {"G2T-img": np.zeros(C.nD, bool), "AGT-img": np.zeros(C.nD, bool)}
    SEL = np.zeros(C.nD); sel, params = [], []
    for k in range(5):
        ote = np.flatnonzero(fv == k)
        z1 = np.load(INNER6 / "ecgflindist" / f"r{r}_k{k}.npz"); z2 = np.load(B.INNER / "ecgfmlp_IMGF" / f"r{r}_k{k}.npz")
        TR = z1["TR"]; assert (z2["TR"] == TR).all()
        Pin = (z1["P"] + z2["P"]) / 2; yTR = Y[TR]                                  # IMG-vote inner OOF
        split = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, yTR, C.PAT[TR]))
        efo, efi = {}, {}
        for m in MODELS_H:
            for w, wv in WEIGHTS.items():
                efo[(m, w)] = fitpred(m, wv, TR, ote, clean, 3000 + 10 * r + k)
                e = np.zeros(len(TR))
                for j, (ia, ib) in enumerate(split):
                    e[ib] = fitpred(m, wv, TR[ia], TR[ib], clean, 4000 + 100 * r + 10 * k + j)
                efi[(m, w)] = e
        for w in WEIGHTS:
            efo[("H3", w)] = (efo[("H1", w)] + efo[("H2", w)]) / 2; efi[("H3", w)] = (efi[("H1", w)] + efi[("H2", w)]) / 2
        cl = clean[TR]
        rmse = {v: float(np.sqrt(np.mean((efi[v][cl] - ef[TR][cl]) ** 2))) for v in VARIANTS}
        vs = min(rmse, key=rmse.get); sel.append(f"{vs[0]}_{vs[1]}")
        e_out, e_in = efo[vs], efi[vs]; SEL[ote] = e_out
        bTR = bnd[TR]
        grid = [(d0, t1, t2) for d0 in np.arange(-15, 6.5, 0.5) for t1 in np.arange(30, 55.5, 0.5) for t2 in np.arange(20, t1, 0.5)]
        p = F1.floor_best(((q, F1.ef_grade(e_in, bTR, *q)) for q in grid), yTR) or (0.0, 41.0, 31.0)
        g_out, g_in = F1.ef_grade(e_out, bnd[ote], *p), F1.ef_grade(e_in, bTR, *p)
        d_out = np.abs(e_out[:, None] - np.column_stack([bnd[ote] + p[0], np.full(len(ote), p[1]), np.full(len(ote), p[2])])).min(1)
        d_in = np.abs(e_in[:, None] - np.column_stack([bTR + p[0], np.full(len(TR), p[1]), np.full(len(TR), p[2])])).min(1)
        prior = np.bincount(yTR, minlength=4) / len(TR)
        c3_in = decide(Pin, "prior", al[k], prior)
        ag_in, ag_out = c3_in == g_in, c3img[ote] == g_out
        t_g2 = F1.threshold(d_in, g_in == yTR); t_ag = F1.threshold(d_in[ag_in], (g_in == yTR)[ag_in])
        G[ote] = g_out; K["G2T-img"][ote] = d_out >= t_g2; K["AGT-img"][ote] = ag_out & (d_out >= t_ag)
        params.append(dict(fold=k, selected=sel[-1], cuts=[float(x) for x in p], t_g2=float(t_g2), t_ag=float(t_ag)))
        d = C.guard7(OUTD); d.mkdir(parents=True, exist_ok=True)                 # inner-fold grading, for E7 (nested)
        np.savez(d / f"inner_r{r}_k{k}.npz", TR=TR, g_in=g_in, d_in=d_in, g_out=g_out, d_out=d_out, ote=ote)
    d = C.guard7(OUTD); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"r{r}.npz", G=G, SEL=SEL, **{k.replace("-", "_"): v for k, v in K.items()})
    (d / f"params_r{r}.json").write_text(json.dumps(params))
    print(f"[r{r}] SEL {sel} G2 acc {np.mean(G == Y):.3f} | " + " ".join(f"{k} cov {v.mean():.2f} acc {np.mean(G[v] == Y[v]):.3f}" for k, v in K.items()), flush=True)
    return r


S2 = {"IMG-vote": lambda r: B.load_run(f"P6B-imgvote-IMGF-v1__s0_r{r}")[1],
      "ECGF-linear": lambda r: C.load(f"P6A-ecgflindist-B1-v1__s42_r{r}")[0].argmax(1),
      "CLIN-ORD": lambda r: B.load_run(f"P6B-clinord-IMGF-v1__s0_r{r}")[1],
      "IMG-vote+CLIN": lambda r: B.load_run(f"P6B-imgvoteclin-IMGF-v1__s0_r{r}")[1],
      "IMG-C3": lambda r: B.load_run(f"P6B-imgc3-IMGF-v1__s0_r{r}")[1]}


def stage1(s1, r):
    z = np.load(OUTD / f"r{r}.npz"); return z["G"].astype(int), z[s1.replace("-", "_")].astype(bool)


def metrics_of(preds):
    ms, cl, cm = [], [], []
    for r, pred in enumerate(preds):
        P = np.eye(4)[pred]; ms.append(B.metrics(C.Y, P, pred)); cl += [c["accuracy"] for c in C.cells(P, pred, r)]; cm.append(B.confusion(C.Y, pred))
    o = {k: float(np.mean([m[k] for m in ms])) for k in ["accuracy", "balanced_accuracy", "macro_f1"] + [f"{a}_{c}" for c in C.CLASSES for a in ("precision", "recall", "f1")]}
    o["min_recall"] = min(o[f"recall_{c}"] for c in C.CLASSES); o["admissible"] = bool(o["min_recall"] >= 0.15)
    ci = C.corr_ci(cl); o["accuracy_ci95_corrected"] = f"[{ci[0]:.3f}, {ci[1]:.3f}]"; o["confusion_mean"] = json.dumps(np.mean(cm, 0).round(1).tolist())
    return o


def paired(pa, pb, metric):
    d, dr = [], []
    for r, (a, b) in enumerate(zip(pa, pb)):
        Pa, Pb = np.eye(4)[a], np.eye(4)[b]
        d += [x[metric] - y[metric] for x, y in zip(C.cells(Pa, a, r), C.cells(Pb, b, r))]
        dr.append(B.metrics(C.Y, Pa, a)[metric] - B.metrics(C.Y, Pb, b)[metric])
    _, p = C.corr_t(d); ci = C.corr_ci(d)
    return dict(metric=metric, delta=float(np.mean(dr)), ci95=f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", p_corrected_t=p,
                same_sign=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)))


def report():
    rows, tests, absn = [], [], []
    img_c3 = [S2["IMG-C3"](r) for r in (0, 1, 2)]; c3 = [C.load(f"P6A-c3-B1-v1__s0_r{r}")[1] for r in (0, 1, 2)]
    for name, preds in (("IMG-C3 (header-free reference)", img_c3), ("C3 (uses header; context)", c3),
                        ("IMG-vote (header-free)", [S2["IMG-vote"](r) for r in (0, 1, 2)])):
        rows.append(dict(model=name, stage1="", stage2="", **metrics_of(preds), stage1_share=np.nan, stage1_accuracy=np.nan, stage2_accuracy=np.nan))
    for s1 in ("G2T-img", "AGT-img"):
        cov, acc, rec, mix = [], [], [], []
        for r in (0, 1, 2):
            g, keep = stage1(s1, r); cov.append(keep.mean()); acc.append(np.mean(g[keep] == C.Y[keep]))
            rec.append([np.mean(g[keep & (C.Y == i)] == i) if (keep & (C.Y == i)).any() else np.nan for i in range(4)])
            mix.append(np.bincount(g[keep], minlength=4))
        absn.append(dict(stage1=s1, coverage=float(np.mean(cov)), accuracy_answered=float(np.mean(acc)),
                         recall_answered=dict(zip(C.CLASSES, np.round(np.nanmean(rec, 0), 3).tolist())),
                         predicted_mix=dict(zip(C.CLASSES, np.mean(mix, 0).round(1).tolist()))))
        for s2, f in S2.items():
            preds, parts = [], []
            for r in (0, 1, 2):
                g, keep = stage1(s1, r); p2 = f(r); pred = np.where(keep, g, p2).astype(int); preds.append(pred)
                parts.append((keep.mean(), np.mean(pred[keep] == C.Y[keep]), np.mean(pred[~keep] == C.Y[~keep])))
            pt = np.mean(parts, 0); tag = f"{s1} -> {s2}"
            rows.append(dict(model=tag, stage1=s1, stage2=s2, **metrics_of(preds), stage1_share=pt[0], stage1_accuracy=pt[1], stage2_accuracy=pt[2]))
            for metric in ("accuracy", "macro_f1", "f1_Mild"):
                tests.append(dict(cascade=tag, comparator="IMG-C3", **paired(preds, img_c3, metric)))
    from research_experiments.phase_5.p5common import holm
    adj = holm([t["p_corrected_t"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = float(a); t["genuine"] = bool(t["delta"] > 0 and a < 0.05 and t["same_sign"])
    ef = B.table().ef.values; clean = C.verified()[0]
    sd = [(float(np.std(np.load(OUTD / f"r{r}.npz")["SEL"] - ef)), float(np.std((np.load(OUTD / f"r{r}.npz")["SEL"] - ef)[clean]))) for r in (0, 1, 2)]
    D = pd.DataFrame(rows); T = pd.DataFrame(tests)
    cas = D[D.stage1 != ""]; adm = cas[cas.admissible]
    out = dict(abstention_stage1=absn, header_free_ef_error_sd=dict(all=float(np.mean([x[0] for x in sd])), verified=float(np.mean([x[1] for x in sd]))),
               best_admissible_cascade=adm.sort_values("accuracy", ascending=False).iloc[0][["model", "accuracy", "accuracy_ci95_corrected", "macro_f1",
                                                                                            "precision_Mild", "recall_Mild", "f1_Mild"]].to_dict() if len(adm) else None,
               best_mild_f1_cascade=cas.sort_values("f1_Mild", ascending=False).iloc[0][["model", "accuracy", "macro_f1", "precision_Mild", "recall_Mild",
                                                                                         "f1_Mild", "admissible"]].to_dict(),
               genuine_improvements=[f"{t['cascade']}:{t['metric']}" for t in tests if t["genuine"]],
               selection={f"r{r}": json.loads((OUTD / f"params_r{r}.json").read_text()) for r in (0, 1, 2)})
    D.to_csv(C.guard7(C.RES7 / "e6_cascades.csv"), index=False, float_format="%.4f")
    T.to_csv(C.guard7(C.RES7 / "e6_cascade_tests.csv"), index=False, float_format="%.4f")
    (C.guard7(C.RES7) / "e6_cascade_summary.json").write_text(json.dumps(out, indent=1, default=float))
    plot(D)
    pd.set_option("display.width", 250)
    print(D[["model", "accuracy", "accuracy_ci95_corrected", "balanced_accuracy", "macro_f1", "precision_Mild", "recall_Mild", "f1_Mild", "min_recall",
             "stage1_share", "stage1_accuracy", "stage2_accuracy"]].to_string(index=False, float_format="%.3f"))
    print(T[T.p_corrected_t < 0.1].to_string(index=False, float_format="%.3f"))
    print(json.dumps({k: v for k, v in out.items() if k != "selection"}, indent=1, default=float))


def plot(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(len(D))
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    ax[0].bar(xs, D.accuracy, color=["#9aa5b1" if s == "" else ("#4c78a8" if a else "#c9d3dd") for s, a in zip(D.stage1, D.admissible)])
    for i, v in enumerate(D.accuracy):
        ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax[0].axhline(0.70, ls=":", c="r", lw=0.8); ax[0].set_ylim(0, 0.8)
    ax[0].set_title("Header-free cascades: four-class accuracy, all 837 ECGs (grey = references; pale = not admissible)", fontsize=8)
    for j, (k, c) in enumerate((("precision_Mild", "#4c78a8"), ("recall_Mild", "#f58518"), ("f1_Mild", "#54a24b"))):
        ax[1].bar(xs + (j - 1) * 0.27, D[k], 0.26, color=c, label=k.replace("_", " "))
    ax[1].legend(fontsize=7); ax[1].set_ylim(0, 0.6); ax[1].set_title("Mild precision / recall / F1", fontsize=9)
    for a in ax:
        a.set_xticks(xs); a.set_xticklabels(D.model, rotation=60, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(C.guard7(C.PLOTS7 / "phase7_cascades_headerfree.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        C.verified(); B.table()
        with Pool(3) as pool:
            for r in pool.imap_unordered(job, (0, 1, 2)):
                print("done", r, flush=True)
    if a.report:
        report()


if __name__ == "__main__":
    main()
