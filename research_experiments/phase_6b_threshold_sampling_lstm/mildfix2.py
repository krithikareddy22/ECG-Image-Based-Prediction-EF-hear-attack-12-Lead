"""MILDFIX2 (AMENDMENT_MILDFIX2.md): better, noise-aware EF regressors (nested selection) feeding EF grading, EF stacking
and agreement abstention. ORIG-trained vote members; reference C3 (0.562). Development CV only.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mildfix2 --run
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mildfix2 --report
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.p5common import (ALPHAS, blockstd, cells, corr_ci, corr_t, decide, ecgf_f6, feature_family,
                                                   header_numeric, holm, impute)
from research_experiments.phase_6.p6common import RUNS6
from research_experiments.phase_6b_threshold_sampling_lstm import mildfix as F1
from research_experiments.phase_6b_threshold_sampling_lstm import mmclean as M
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

MODELS_R = ("R1", "R2", "R3")
WEIGHTS = {"W1": 1.0, "W.3": 0.3, "W0": 0.0}
VARIANTS = [(m, w) for m in MODELS_R + ("R4",) for w in WEIGHTS]
BASES = ("mlpfusion", "ecgflindist", "hdrclinlr")
PRIVD = B.PRIV / "mildfix2"
_IMG = None


def img():
    """The 121 image-only clinician features (mildfeat.py; raw, NaN allowed)."""
    global _IMG
    if _IMG is None:
        from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import clin
        _IMG = clin()[0]
    return _IMG


def _rows_w(fit_rows, wv, clean):
    if wv == 0.0:
        fit_rows = fit_rows[clean[fit_rows]]
        return fit_rows, np.ones(len(fit_rows))
    return fit_rows, np.where(clean[fit_rows], 1.0, wv)


def fitpred(model, wv, fit_rows, pred_rows, clean, split_seed):
    ef = B.table().ef.values
    fit_rows, w = _rows_w(fit_rows, wv, clean)
    fm = np.zeros(B.nD, bool); fm[fit_rows] = True
    if model in ("R1", "R2"):
        mats = [header_numeric(fm), ecgf_f6().astype(float), impute(feature_family("clin3")[0], fm)]
        if model == "R2":
            mats.append(impute(img(), fm))
        X = blockstd(mats, fm)
        return RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[fit_rows], ef[fit_rows], sample_weight=w).predict(X[pred_rows])
    import lightgbm as lgb
    E = ecgf_f6().astype(float); mu, sd = E[fm].mean(0), E[fm].std(0) + 1e-6
    pca = PCA(32, random_state=0).fit(((E - mu) / sd)[fm])
    X = np.column_stack([header_numeric(fm), feature_family("clin3")[0], img(), pca.transform((E - mu) / sd)])
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, B.Y[fit_rows], B.PAT[fit_rows]))
    tr, va = fit_rows[a], fit_rows[b]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, random_state=0, n_jobs=1, verbose=-1)
    m.fit(X[tr], ef[tr], sample_weight=w[a], eval_set=[(X[va], ef[va])], eval_sample_weight=[w[b]],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(X[pred_rows])


def job(r):
    warnings.filterwarnings("ignore")
    Y, T = B.Y, B.table(); ef = T.ef.values; bnd = T.sex_boundary.values; fv = B.fold_vec(r)
    clean = ~M.flags().NF.values; img()
    Pv, _ = M.preds("vote", "ORIG", r); _, pc3 = M.preds("c3", "ORIG", r); al = M._alphas("ORIG", r)
    EFO = {f"{m}_{w}": np.zeros(B.nD) for m, w in VARIANTS}; SEL = np.zeros(B.nD); choice = []
    out = {k: dict(P=np.zeros((B.nD, 4)), pred=np.zeros(B.nD, int), keep=np.ones(B.nD, bool), params=[]) for k in ("G2", "G2s")}
    for k in ("AG", "AGT", "G2T"):
        out[k] = dict(P=np.zeros((B.nD, 4)), pred=np.zeros(B.nD, int), keep=np.zeros(B.nD, bool), params=[])
    for k in range(5):
        ote = np.flatnonzero(fv == k)
        inn = [M._inner(b, "ORIG", r, k) for b in BASES]
        TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0); yTR = Y[TR]
        split = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, yTR, B.PAT[TR]))
        efo, efi = {}, {}
        for m in MODELS_R:
            for w, wv in WEIGHTS.items():
                efo[(m, w)] = fitpred(m, wv, TR, ote, clean, 3000 + 10 * r + k)
                e = np.zeros(len(TR))
                for j, (ia, ib) in enumerate(split):
                    e[ib] = fitpred(m, wv, TR[ia], TR[ib], clean, 4000 + 100 * r + 10 * k + j)
                efi[(m, w)] = e
        for w in WEIGHTS:
            efo[("R4", w)] = (efo[("R2", w)] + efo[("R3", w)]) / 2; efi[("R4", w)] = (efi[("R2", w)] + efi[("R3", w)]) / 2
        cl = clean[TR]
        rmse = {v: float(np.sqrt(np.mean((efi[v][cl] - ef[TR][cl]) ** 2))) for v in VARIANTS}
        vs = min(rmse, key=rmse.get); choice.append(dict(fold=k, selected=f"{vs[0]}_{vs[1]}", inner_clean_rmse=rmse[vs],
                                                          r1_w1_inner_clean_rmse=rmse[("R1", "W1")]))
        for v in VARIANTS:
            EFO[f"{v[0]}_{v[1]}"][ote] = efo[v]
        e_out, e_in = efo[vs], efi[vs]; SEL[ote] = e_out
        # G2: EF grading with tuned cut points (floor objective)
        bTR = bnd[TR]
        grid = [(d0, t1, t2) for d0 in np.arange(-15, 6.5, 0.5) for t1 in np.arange(30, 55.5, 0.5) for t2 in np.arange(20, t1, 0.5)]
        p = F1.floor_best(((q, F1.ef_grade(e_in, bTR, *q)) for q in grid), yTR) or (0.0, 41.0, 31.0)
        g_out, g_in = F1.ef_grade(e_out, bnd[ote], *p), F1.ef_grade(e_in, bTR, *p)
        d_out = np.abs(e_out[:, None] - np.column_stack([bnd[ote] + p[0], np.full(len(ote), p[1]), np.full(len(ote), p[2])])).min(1)
        d_in = np.abs(e_in[:, None] - np.column_stack([bTR + p[0], np.full(len(TR), p[1]), np.full(len(TR), p[2])])).min(1)
        out["G2"]["P"][ote] = np.eye(4)[g_out]; out["G2"]["pred"][ote] = g_out; out["G2"]["params"].append([float(x) for x in p])
        # G2s: stacking + nested prior rule
        prior = np.bincount(yTR, minlength=4) / len(TR)
        Zin = np.column_stack([np.log(Pin.clip(1e-6)), e_in / 10]); Zout = np.column_stack([np.log(Pv[ote].clip(1e-6)), e_out / 10])
        Ps_out = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Zin, yTR).predict_proba(Zout)
        Ps_in = np.zeros((len(TR), 4))
        for ia, ib in split:
            Ps_in[ib] = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Zin[ia], yTR[ia]).predict_proba(Zin[ib])
        a = F1.floor_best(((a, decide(Ps_in, "prior", a, prior)) for a in ALPHAS), yTR)
        a = 0.0 if a is None else float(a)
        out["G2s"]["P"][ote] = Ps_out; out["G2s"]["pred"][ote] = decide(Ps_out, "prior", a, prior); out["G2s"]["params"].append(a)
        # abstention: agreement (AG), agreement + nested threshold (AGT), EF-grading threshold (G2T)
        c3_in = decide(Pin, "prior", al[k], prior)
        ag_out, ag_in = pc3[ote] == g_out, c3_in == g_in
        t_ag = F1.threshold(d_in[ag_in], (g_in == yTR)[ag_in])
        t_g2 = F1.threshold(d_in, g_in == yTR)
        for key, keep in (("AG", ag_out), ("AGT", ag_out & (d_out >= t_ag)), ("G2T", d_out >= t_g2)):
            out[key]["P"][ote] = np.eye(4)[g_out]; out[key]["pred"][ote] = g_out; out[key]["keep"][ote] = keep
        out["AGT"]["params"].append(float(t_ag)); out["G2T"]["params"].append(float(t_g2))
    d = B.guard(PRIVD); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"ef_r{r}.npz", SEL=SEL, **{k.replace(".", "p"): v for k, v in EFO.items()})
    for key, o in out.items():
        np.savez(d / f"{key}_r{r}.npz", P=o["P"], pred=o["pred"], keep=o["keep"])
    (d / f"params_r{r}.json").write_text(json.dumps(dict(selection=choice, **{k: o["params"] for k, o in out.items()}), default=float))
    print(f"[r{r}] SEL {[c['selected'] for c in choice]} G2 {np.mean(out['G2']['pred'] == Y):.3f} G2s {np.mean(out['G2s']['pred'] == Y):.3f} "
          f"AG cov {out['AG']['keep'].mean():.2f} acc {np.mean(out['AG']['pred'][out['AG']['keep']] == Y[out['AG']['keep']]):.3f}", flush=True)
    return r


# ------------------------------------------------------------------ report
def load(key, r):
    z = np.load(PRIVD / f"{key}_r{r}.npz"); return z["P"], z["pred"], z["keep"]


def ef_stats(pred, mask):
    ef = B.table().ef.values; res = pred[mask] - ef[mask]
    return dict(sd=float(res.std()), mae=float(np.abs(res).mean()), r=float(np.corrcoef(pred[mask], ef[mask])[0, 1]))


def a1(r):
    return pd.read_csv(RUNS6 / f"P6A1-ft-s6-joint-v1__s42_r{r}" / "oof.csv").set_index("ecg_id").loc[B.IDS, "efhat"].values


def summary(key, mask):
    ms, cl = [], []
    for r in (0, 1, 2):
        P, pred, _ = load(key, r)
        ms.append(B.metrics(B.Y[mask], P[mask], pred[mask])); cl += [c["accuracy"] for c in cells(P, pred, r, mask=mask)]
    o = {k: float(np.mean([x[k] for x in ms])) for k in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Normal",
                                                          "recall_Mild", "recall_Moderate", "recall_Severe")}
    o["min_recall"] = min(o[f"recall_{c}"] for c in B.CLASSES); o["admissible"] = bool(o["min_recall"] >= F1.FLOOR)
    o["accuracy_ci95_corrected"] = corr_ci(cl)
    return o


def selective(key, loader=load):
    reps, cl = [], []
    for r in (0, 1, 2):
        _, pred, keep = loader(key, r)[:3] if loader is load else loader(r)
        fv = B.fold_vec(r); y = B.Y
        for k in range(5):
            s = keep & (fv == k)
            if s.sum() > 0:
                cl.append(float(np.mean(pred[s] == y[s])))
        reps.append(dict(coverage=float(keep.mean()), accuracy=float(np.mean(pred[keep] == y[keep])) if keep.any() else np.nan,
                         cov={c: float(keep[y == i].mean()) for i, c in enumerate(B.CLASSES)},
                         rec={c: (float(np.mean(pred[keep & (y == i)] == i)) if (keep & (y == i)).any() else np.nan) for i, c in enumerate(B.CLASSES)},
                         mix={c: int((pred[keep] == i).sum()) for i, c in enumerate(B.CLASSES)}))
    return dict(option=key, coverage=float(np.mean([x["coverage"] for x in reps])), accuracy=float(np.nanmean([x["accuracy"] for x in reps])),
                accuracy_ci95_corrected=corr_ci(cl) if len(cl) > 2 else None, n_fold_cells=len(cl),
                correct_per_100=float(100 * np.nanmean([x["coverage"] * x["accuracy"] for x in reps])),
                coverage_by_class={c: float(np.mean([x["cov"][c] for x in reps])) for c in B.CLASSES},
                recall_answered={c: float(np.nanmean([x["rec"][c] for x in reps])) for c in B.CLASSES},
                predicted_mix={c: float(np.mean([x["mix"][c] for x in reps])) for c in B.CLASSES})


def paired(key, metric):
    d, dr = [], []
    for r in (0, 1, 2):
        Pa, pa, _ = load(key, r); Pb, pb, _, _ = F1.load("C3", "ORIG", r)
        d += [x[metric] - z[metric] for x, z in zip(cells(Pa, pa, r), cells(Pb, pb, r))]
        dr.append(B.metrics(B.Y, Pa, pa)[metric] - B.metrics(B.Y, Pb, pb)[metric])
    _, p = corr_t(d)
    return dict(model=key, metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr], ci95_corrected_delta=corr_ci(d),
                p_corrected_t=p, same_sign_all_repeats=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)))


def report():
    warnings.filterwarnings("ignore")
    Fl = M.flags(); clean = ~Fl.NF.values; allm = np.ones(B.nD, bool)
    out = dict(ef_error=[], results=[], comparisons=[], selective=[], selection={})
    names = [f"{m}_{w}" for m, w in VARIANTS] + ["SEL", "A1", "A1+SEL"]
    for nm in names:
        rows = []
        for r in (0, 1, 2):
            z = np.load(PRIVD / f"ef_r{r}.npz")
            pr = (a1(r) if nm == "A1" else (a1(r) + z["SEL"]) / 2 if nm == "A1+SEL" else z[nm.replace(".", "p")])
            rows.append((ef_stats(pr, allm), ef_stats(pr, clean)))
        out["ef_error"].append(dict(predictor=nm, **{f"all_{k}": float(np.mean([x[0][k] for x in rows])) for k in ("sd", "mae", "r")},
                                    **{f"clean_{k}": float(np.mean([x[1][k] for x in rows])) for k in ("sd", "mae", "r")}))
    for r in (0, 1, 2):
        out["selection"][f"r{r}"] = json.loads((PRIVD / f"params_r{r}.json").read_text())["selection"]
    subsets = {"all_837": allm, "NF_clear": clean & ~Fl.near_boundary.values, "HC": M.hc_mask()}
    for key in ("G2", "G2s"):
        for s, mask in subsets.items():
            out["results"].append(dict(model=key, evaluated_on=s, n_ecgs=int(mask.sum()), **summary(key, mask)))
        for metric in ("accuracy", "macro_f1", "recall_Mild"):
            out["comparisons"].append(paired(key, metric))
    adj = holm([x["p_corrected_t"] for x in out["comparisons"]])
    for x, a in zip(out["comparisons"], adj):
        x["p_holm"] = float(a)
    for key in ("AG", "AGT", "G2T"):
        out["selective"].append(selective(key))
    for name, (m, src) in {"round1_F3b_ORIG": ("F3b", "ORIG"), "round1_C3_NFS": ("C3", "NFS"), "round1_F2b_NFS": ("F2b", "NFS"),
                           "round1_C3_ORIG": ("C3", "ORIG")}.items():
        s = selective(name, loader=lambda r, m=m, src=src: (lambda z: (None, z[1], z[3]))(F1.load(m, src, r)))
        out["selective"].append(s)
    main = [x for x in out["results"] if x["evaluated_on"] == "all_837" and x["admissible"]]
    ok = [x for x in out["selective"] if x["accuracy"] >= 0.70]
    out["final"] = dict(four_class=max(main, key=lambda x: x["accuracy"]) if main else None,
                        abstention=(max(ok, key=lambda x: x["coverage"]) if ok else
                                    max([x for x in out["selective"] if x["coverage"] >= 0.2], key=lambda x: x["accuracy"])),
                        best_ef=min(out["ef_error"], key=lambda x: x["all_sd"]))
    (B.guard(B.RES) / "mildfix2_results.json").write_text(json.dumps(out, indent=1, default=float))
    pd.DataFrame(out["ef_error"]).to_csv(B.guard(B.RES / "mildfix2_ef_error.csv"), index=False, float_format="%.3f")
    pd.DataFrame(out["selective"]).to_csv(B.guard(B.RES / "mildfix2_selective.csv"), index=False, float_format="%.4f")
    rows = pd.DataFrame(out["results"]); rows["accuracy_ci95_corrected"] = rows.accuracy_ci95_corrected.map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    rows.to_csv(B.guard(B.RES / "mildfix2_results.csv"), index=False, float_format="%.4f")
    pd.DataFrame(out["comparisons"]).to_csv(B.guard(B.RES / "mildfix2_comparisons.csv"), index=False, float_format="%.4f")
    plot(out)
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    E = out["ef_error"]; xs = np.arange(len(E))
    axes[0].bar(xs - 0.2, [x["all_sd"] for x in E], 0.4, label="all 837 ECGs", color="#9aa5b1")
    axes[0].bar(xs + 0.2, [x["clean_sd"] for x in E], 0.4, label="clean echo links", color="#4c78a8")
    axes[0].axhline(9, ls=":", c="r", lw=0.8); axes[0].text(len(E) - 0.5, 9.2, "≈ 9: needed for 70 %", fontsize=7, ha="right", color="r")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([x["predictor"] for x in E], rotation=60, fontsize=7)
    axes[0].set_ylabel("EF error SD (points; out-of-fold, mean of 3 repeats)"); axes[0].legend(fontsize=7); axes[0].set_ylim(0, 16)
    axes[0].set_title("EF error of each regressor (R1 ridge, R2 + image features, R3 LightGBM, R4 mean; W = noise weighting)", fontsize=8)
    S = out["selective"]; xs = np.arange(len(S))
    axes[1].bar(xs, [x["accuracy"] for x in S], color="#4c78a8")
    for i, x in enumerate(S):
        axes[1].text(i, x["accuracy"] + 0.01, f"{x['coverage'] * 100:.0f}% answered", ha="center", fontsize=7, rotation=90)
    axes[1].axhline(0.7, ls=":", c="k", lw=0.8); axes[1].set_ylim(0, 1)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([x["option"] for x in S], rotation=30, fontsize=7)
    axes[1].set_title("Abstention options: accuracy on answered ECGs", fontsize=9)
    fig.tight_layout(); fig.savefig(B.guard(B.PLOTS / "mildfix2_summary.png"), dpi=150); plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        B.table(); M.flags()
        with Pool(3) as pool:
            for r in pool.imap_unordered(job, (0, 1, 2)):
                print("done", r, flush=True)
    if a.report:
        out = report()
        for x in out["ef_error"]:
            print(f"{x['predictor']:8s} all SD {x['all_sd']:.2f} MAE {x['all_mae']:.2f} r {x['all_r']:.3f} | clean SD {x['clean_sd']:.2f} r {x['clean_r']:.3f}")
        for x in out["results"]:
            print(x["model"], x["evaluated_on"], round(x["accuracy"], 3), np.round(x["accuracy_ci95_corrected"], 3).tolist(),
                  "Mild", round(x["recall_Mild"], 3), "min", round(x["min_recall"], 3))
        for x in out["selective"]:
            print(x["option"], "cov", round(x["coverage"], 3), "acc", round(x["accuracy"], 3), x["accuracy_ci95_corrected"] and np.round(x["accuracy_ci95_corrected"], 3).tolist(),
                  "per100", round(x["correct_per_100"], 1))
        print(json.dumps(out["final"], indent=1, default=float)[:1500])


if __name__ == "__main__":
    main()
