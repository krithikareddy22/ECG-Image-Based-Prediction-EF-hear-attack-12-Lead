"""Phase 4G analyses (no model selection; aggregates only; small cells (< 5 ECGs) suppressed in committed outputs).

  1. High-confidence cohort G (357 ECGs / 345 patients in D): composition, why folder-EF agreement is high,
     performance of each key model on G vs non-G with a class-mix adjustment (decomposes "G is easier").
  2. Mild study: Mild AUROC (one-vs-rest, vs Normal, vs Moderate), errors by distance of the nearest-echo EF to the
     clinical boundaries and by ECG-to-echo interval.
  3. Regression plots (binned / hexbin; no per-ECG values in committed outputs).

    python -m research_experiments.phase_4.analysis4
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_experiments.phase_2.cv_features import _H  # noqa: E402  (header features; no identifiers used)
from research_experiments.phase_4.benchmark4 import combined, model_groups  # noqa: E402
from research_experiments.phase_4.p4common import (COH, COHORTS, D, EF, MALE, PLOT4, PRIV, RES4, REPEATS, Y, YD,  # noqa: E402
                                                     metrics, mild_aucs, nD)

KEY = {"bench_vote": "Vote ensemble (Phase 3 benchmark)", "bench_mlpfusion": "MLP fusion (Phase 2)",
       "reg_ridge_hdr+ecgf_A__tuned": "Ridge EF regression, cohort A (tuned cut-points)",
       "reg_ridge_hdr+ecgf_A__clin": "Ridge EF regression, cohort A (clinical thresholds)",
       "reg_mlp_hdr+ecgf_A__tuned": "MLP EF regression, cohort A (tuned)",
       "mt_sel": "Multitask, lambda selected", "mt_lam1_A": "Multitask, lambda 1",
       "lq_trainG": "MLP fusion trained on G only", "lq_relabel": "MLP fusion trained on EF-derived labels",
       "fus_vote+ridgeA": "Vote + ridge-EF member", "fus_vote+mtsel": "Vote + multitask member"}
G = COHORTS["G"]
CL = ["Normal", "Mild", "Moderate", "Severe"]
SUP = 5


def sup(n, v):
    return v if n >= SUP else None


def characterise():
    age = _H.set_index("ecg_id").loc[D.ecg_id].age.values
    out = {}
    for name, m in (("G", G), ("non_G", ~G), ("all_D", np.ones(nD, bool))):
        s = COH[m]
        out[name] = dict(
            ecgs=int(m.sum()), patients=int(s.patient.nunique()),
            folder_class_counts={c: int((s.label == c).sum()) for c in CL},
            folder_class_share={c: round(float((s.label == c).mean()), 3) for c in CL},
            year_share={str(y): round(float((s.year == y).mean()), 3) for y in sorted(COH.year.unique())},
            male_share=round(float(MALE[m].mean()), 3), age_mean=round(float(np.nanmean(age[m])), 1),
            ef_mean=round(float(EF[m].mean()), 1), ef_sd=round(float(EF[m].std()), 1),
            interval_days_median=float(s.abs_days.median()), share_same_day=round(float((s.abs_days == 0).mean()), 3),
            share_single_link=round(float((s.n_links == 1).mean()), 3),
            folder_vs_nearest_echo_agreement=round(float((Y[m] == YD[m]).mean()), 3))
    one = (COH.n_links == 1).values
    out["agreement_by_number_of_links_all_D"] = {
        "single_link": dict(ecgs=int(one.sum()), agreement=round(float((Y[one] == YD[one]).mean()), 3)),
        "multiple_links": dict(ecgs=int((~one).sum()), agreement=round(float((Y[~one] == YD[~one]).mean()), 3))}
    # why the G disagreements exist (aggregate counts)
    L = pd.read_csv(PRIV / "ecg_echo_links.csv")
    L = L[L.ecg_id.isin(D.ecg_id[G])]
    L["mind"] = L.groupby("ecg_id").abs_days.transform("min")
    near = L[L.abs_days == L.mind].groupby("ecg_id").agg(lv_sheet=("lv_sheet", "first"))
    dis = D.ecg_id[G & (Y != YD)].values
    lab = COH.label
    sheet_eq = int(sum(near.loc[e, "lv_sheet"] == lab.loc[e] for e in dis if e in near.index))
    other_link = 0
    for e in dis:
        ls = L[L.ecg_id == e]
        if any(ls.derived == lab.loc[e]):
            other_link += 1
    out["G_disagreements"] = dict(n=int(len(dis)),
                                  folder_equals_sheet_class_of_nearest_echo=sup(len(dis), sheet_eq),
                                  folder_equals_class_of_another_linked_echo=sup(len(dis), other_link),
                                  note="counts only; the folder labels were produced from the same matching sheets")
    return out


def g_vs_nong(comb):
    rows = []
    piG = np.bincount(Y[G], minlength=4) / G.sum()
    for n, lab in KEY.items():
        if n not in comb:
            continue
        acc = {k: [] for k in ("G", "nonG", "nonG_classmix_to_G", "nonG_folder_eq_echo", "G_vs_echo", "balG", "balnonG")}
        for r in REPEATS:
            P, pred, _ = comb[n][r]
            ok = pred == Y
            acc["G"].append(ok[G].mean()); acc["nonG"].append(ok[~G].mean())
            rec = np.array([ok[~G & (Y == c)].mean() for c in range(4)])
            acc["nonG_classmix_to_G"].append(float(piG @ rec))
            agree = ~G & (Y == YD); acc["nonG_folder_eq_echo"].append(ok[agree].mean())
            acc["G_vs_echo"].append((pred[G] == YD[G]).mean())
            acc["balG"].append(metrics(Y[G], P[G], pred[G])["balanced_accuracy"])
            acc["balnonG"].append(metrics(Y[~G], P[~G], pred[~G])["balanced_accuracy"])
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append(dict(model=lab, tag=n, acc_G=m["G"], acc_nonG=m["nonG"], acc_nonG_reweighted_to_G_class_mix=m["nonG_classmix_to_G"],
                         acc_nonG_where_folder_eq_echo=m["nonG_folder_eq_echo"], acc_G_vs_echo_class=m["G_vs_echo"],
                         bal_acc_G=m["balG"], bal_acc_nonG=m["balnonG"],
                         gap_total=m["G"] - m["nonG"], gap_class_mix=m["nonG_classmix_to_G"] - m["nonG"],
                         gap_remaining=m["G"] - m["nonG_classmix_to_G"]))
    return pd.DataFrame(rows)


def boundary_distance():
    up = np.where(MALE == 1, 52.0, 54.0)
    return np.min(np.abs(np.stack([EF - 31, EF - 41, EF - up], 1)), 1)


def mild_study(comb):
    dist = boundary_distance()
    dbins = [(0, 2), (2, 5), (5, 10), (10, 100)]
    ibins = [(0, 0), (1, 7), (8, 30), (31, 90), (91, 10_000)]
    days = COH.abs_days.values
    out = {"auroc": {}, "by_ef_distance": {}, "by_interval": {}, "mild_confusion_row": {}}
    for n, lab in KEY.items():
        if n not in comb:
            continue
        au = [mild_aucs(comb[n][r][0]) for r in REPEATS]
        auG = [mild_aucs(comb[n][r][0], YD, G) for r in REPEATS]
        out["auroc"][lab] = {"all_D_folder": {k: float(np.mean([a[k] for a in au])) for k in au[0]},
                             "G_echo_class": {k: float(np.mean([a[k] for a in auG])) for k in auG[0]}}
        if n != "bench_vote" and "bench_vote" in comb:     # paired per-(repeat, fold) comparison vs the vote ensemble
            from research_experiments.phase_4.p4common import corr_ci, corr_t, fold_vec
            cmp_ = {}
            for k in au[0]:
                dl = []
                for r in REPEATS:
                    fo = fold_vec(r)
                    for f in range(5):
                        mk = fo == f
                        dl.append(mild_aucs(comb[n][r][0], Y, mk)[k] - mild_aucs(comb["bench_vote"][r][0], Y, mk)[k])
                d, pv = corr_t(dl); cmp_[k] = dict(diff=d, p_corrected_t=pv, ci95_corrected=corr_ci(dl))
            out["auroc"][lab]["vs_vote_all_D_folder"] = cmp_
        rows = []
        for lo, hi in dbins:
            m = (dist >= lo) & (dist < hi)
            mm = m & (Y == 1)
            acc = np.mean([(comb[n][r][1][m] == Y[m]).mean() for r in REPEATS])
            accm = np.mean([(comb[n][r][1][mm] == 1).mean() for r in REPEATS]) if mm.sum() else np.nan
            rows.append(dict(ef_distance_to_boundary=f"[{lo},{hi})", ecgs=int(m.sum()), accuracy=sup(m.sum(), float(acc)),
                             mild_ecgs=int(mm.sum()), mild_recall=sup(mm.sum(), float(accm)),
                             folder_eq_echo=sup(m.sum(), float((Y[m] == YD[m]).mean()))))
        out["by_ef_distance"][lab] = rows
        rows = []
        for lo, hi in ibins:
            m = (days >= lo) & (days <= hi)
            acc = np.mean([(comb[n][r][1][m] == Y[m]).mean() for r in REPEATS])
            rows.append(dict(abs_interval_days=f"{lo}-{hi}" if hi < 10_000 else f">{lo - 1}", ecgs=int(m.sum()),
                             accuracy=sup(m.sum(), float(acc)), folder_eq_echo=sup(m.sum(), float((Y[m] == YD[m]).mean()))))
        out["by_interval"][lab] = rows
        cm = np.mean([np.bincount(comb[n][r][1][Y == 1], minlength=4) for r in REPEATS], 0)
        out["mild_confusion_row"][lab] = {c: round(float(v), 1) for c, v in zip(CL, cm)}
    # EF of folder-Mild ECGs relative to the boundaries (descriptive)
    mild = Y == 1
    out["mild_folder_ef"] = dict(n=int(mild.sum()), within_2_of_41=int((np.abs(EF[mild] - 41) < 2).sum()),
                                 within_2_of_upper=int((np.abs(EF[mild] - np.where(MALE[mild] == 1, 52, 54)) < 2).sum()),
                                 ef_quartiles=[round(float(q), 1) for q in np.percentile(EF[mild], [25, 50, 75])])
    return out


def error_budget(comb):
    """EXPLORATORY: 4-class metrics vs folder labels if the predicted EF were the nearest-echo EF plus unbiased
    (classical) Gaussian error of SD sigma, decoded with the clinical thresholds (200 draws). Such a predictor has
    correlation r = SD_EF / sqrt(SD_EF^2 + sigma^2) with the echo EF; compare r (not RMSE) with the observed models,
    whose errors are of the shrunk / regression type (RMSE is not comparable across the two error structures)."""
    from research_experiments.phase_4.p4common import ef_class, quick
    rng = np.random.default_rng(2025); rows = []; sd = float(EF.std())
    def f1k(y, p, k):
        tp = np.sum((p == k) & (y == k)); d = np.sum(p == k) + np.sum(y == k); return 2 * tp / d if d else 0.0
    for sigma in (0, 4, 8, 10, 12, 14, 16, 18, 20, 22):
        preds = [ef_class(EF + rng.normal(0, sigma, nD) if sigma else EF, MALE) for _ in range(1 if sigma == 0 else 200)]
        ms = [quick(Y, p) for p in preds]
        r = sd / np.sqrt(sd ** 2 + sigma ** 2)
        rows.append(dict(sigma_ef_points=sigma, implied_pearson_r=float(r), implied_r2=float(r ** 2),
                         accuracy=float(np.mean([m["accuracy"] for m in ms])),
                         balanced_accuracy=float(np.mean([m["balanced_accuracy"] for m in ms])),
                         macro_f1=float(np.mean([m["macro_f1"] for m in ms])),
                         mild_f1=float(np.mean([f1k(Y, p, 1) for p in preds]))))
    obs = {}
    for n in ("reg_ridge_hdr+ecgf_A__tuned", "reg_mlp_hdr+ecgf_A__tuned", "mt_lam1_A"):
        if n in comb:
            e = np.mean([comb[n][r][2] for r in REPEATS], 0)
            obs[n] = dict(rmse=float(np.sqrt(np.mean((e - EF) ** 2))), pearson_r_all=float(np.corrcoef(e, EF)[0, 1]),
                          pearson_r_G=float(np.corrcoef(e[G], EF[G])[0, 1]), slope_pred_on_echo=float(np.polyfit(EF, e, 1)[0]))
    return dict(simulation=rows, observed_regression_error=obs, ef_sd=sd,
                note="classical-error simulation; compare implied r with observed r (RMSE not comparable)")


def boundary_regression_error(comb):
    """Regression error by distance of the nearest-echo EF to the closest clinical boundary (and class flips)."""
    dist = boundary_distance(); out = {}
    from research_experiments.phase_4.p4common import ef_class
    for n in ("reg_ridge_hdr+ecgf_A__tuned", "reg_mlp_hdr+ecgf_A__tuned", "reg_ridge_hdr+ecgf_C__tuned", "mt_lam1_A"):
        if n not in comb:
            continue
        rows = []
        for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 100)]:
            m = (dist >= lo) & (dist < hi)
            mae = np.mean([np.mean(np.abs(comb[n][r][2][m] - EF[m])) for r in REPEATS])
            flip = np.mean([np.mean(ef_class(comb[n][r][2][m], MALE[m]) != YD[m]) for r in REPEATS])
            rows.append(dict(ef_distance_to_boundary=f"[{lo},{hi})", ecgs=int(m.sum()), mae=sup(m.sum(), float(mae)),
                             clinical_threshold_class_error_vs_echo=sup(m.sum(), float(flip))))
        out[n] = rows
    return out


def mild_characteristics(comb):
    """Header characteristics (mean over ECGs) of folder-Mild ECGs predicted Mild vs predicted otherwise
    (prediction = majority over the 3 CV repeats). Aggregates only."""
    H = _H.set_index("ecg_id").loc[D.ecg_id]
    cols = ["age", "male", "Rate", "PR", "QRSD", "QT", "QTcB"]
    cols = [c for c in cols if c in H.columns]
    out = {}
    for n in ("bench_vote", "mt_sel", "reg_ridge_hdr+ecgf_A__clin"):
        if n not in comb:
            continue
        votes = np.stack([comb[n][r][1] for r in REPEATS])
        pm = np.array([np.bincount(votes[:, i], minlength=4).argmax() for i in range(nD)])
        mild = Y == 1
        grp = {"predicted_Mild": mild & (pm == 1), "predicted_Normal": mild & (pm == 0), "predicted_Moderate": mild & (pm == 2),
               "predicted_Severe": mild & (pm == 3)}
        out[KEY.get(n, n)] = {g: dict(ecgs=int(m.sum()), **({c: sup(m.sum(), round(float(H[c][m].mean()), 2)) for c in cols} |
                                                              {"echo_ef_mean": sup(m.sum(), round(float(EF[m].mean()), 1)),
                                                               "folder_eq_echo": sup(m.sum(), round(float((YD[m] == 1).mean()), 3))}))
                              for g, m in grp.items()}
    return out


def mild_extra(comb):
    """Mild-only aggregates: recall by ECG-echo interval and by position of the echo EF within the Mild band; predicted
    Mild count vs true Mild count (guards against 'predict Mild excessively'); precision / recall / F1."""
    days = COH.abs_days.values; mild = Y == 1; out = {}
    up = np.where(MALE == 1, 52.0, 54.0)
    for n, lab in KEY.items():
        if n not in comb:
            continue
        preds = [comb[n][r][1] for r in REPEATS]
        rec = lambda m: sup(int(m.sum()), float(np.mean([(p[m] == 1).mean() for p in preds])))
        o = dict(true_mild=int(mild.sum()), predicted_mild_mean=float(np.mean([(p == 1).sum() for p in preds])),
                 precision=float(np.mean([(Y[p == 1] == 1).mean() for p in preds])), recall=rec(mild))
        o["f1"] = 2 * o["precision"] * o["recall"] / (o["precision"] + o["recall"]) if o["recall"] else 0.0
        o["recall_by_interval"] = {f"{a}-{b}" if b < 10_000 else f">{a - 1}": dict(ecgs=int((mild & (days >= a) & (days <= b)).sum()),
                                   recall=rec(mild & (days >= a) & (days <= b)))
                                   for a, b in [(0, 0), (1, 14), (15, 90), (91, 10_000)]}
        o["recall_by_echo_ef_position"] = {
            "echo EF < 41 (below band)": dict(ecgs=int((mild & (EF < 41)).sum()), recall=rec(mild & (EF < 41))),
            "41 <= EF < 46": dict(ecgs=int((mild & (EF >= 41) & (EF < 46)).sum()), recall=rec(mild & (EF >= 41) & (EF < 46))),
            "46 <= EF <= upper": dict(ecgs=int((mild & (EF >= 46) & (EF <= up)).sum()), recall=rec(mild & (EF >= 46) & (EF <= up))),
            "EF > upper (above band)": dict(ecgs=int((mild & (EF > up)).sum()), recall=rec(mild & (EF > up)))}
        out[lab] = o
    return out


def plot_confusions(comb):
    from sklearn.metrics import confusion_matrix
    names = [n for n in ("bench_vote", "bench_mlpfusion", "reg_ridge_hdr+ecgf_A__tuned", "reg_ridge_hdr+ecgf_A__clin",
                         "mt_sel", "lq_relabel") if n in comb]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
    for a, n in zip(ax.ravel(), names):
        cm = np.mean([confusion_matrix(Y, comb[n][r][1], labels=range(4)) for r in REPEATS], 0)
        fr = cm / cm.sum(1, keepdims=True)
        a.imshow(fr, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{cm[i, j]:.0f}\n{fr[i, j]:.0%}", ha="center", va="center", fontsize=7, color="white" if fr[i, j] > .55 else "black")
        a.set_xticks(range(4), CL, fontsize=7); a.set_yticks(range(4), CL, fontsize=7)
        a.set_xlabel("predicted", fontsize=7); a.set_ylabel("folder label", fontsize=7); a.set_title(KEY.get(n, n), fontsize=8)
    for a in ax.ravel()[len(names):]:
        a.axis("off")
    fig.suptitle("Confusion matrices, 837 development ECGs (mean counts over 3 CV repeats; row %)", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOT4 / "confusion_matrices_key_models.png", dpi=130); plt.close(fig)


def plots(comb):
    PLOT4.mkdir(exist_ok=True)
    # 1. regression: hexbin predicted vs nearest-echo EF (repeat 0) for ridge A and MLP A, all D and G
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    for j, n in enumerate(["reg_ridge_hdr+ecgf_A__tuned", "reg_mlp_hdr+ecgf_A__tuned"]):
        if n not in comb:
            continue
        e = comb[n][0][2]
        for i, (m, t) in enumerate(((np.ones(nD, bool), "all development ECGs (837)"), (G, "high-confidence G (357)"))):
            a = ax[i, j]
            a.hexbin(EF[m], e[m], gridsize=22, extent=(5, 90, 5, 90), mincnt=SUP, cmap="Blues")   # cells < 5 ECGs suppressed
            a.plot([5, 90], [5, 90], "k:", lw=0.8)
            for x in (31, 41, 52, 54):
                a.axvline(x, c="grey", lw=0.4); a.axhline(x, c="grey", lw=0.4)
            r2 = 1 - np.sum((e[m] - EF[m]) ** 2) / np.sum((EF[m] - EF[m].mean()) ** 2)
            a.set_title(f"{'Ridge' if j == 0 else 'MLP'} (cohort A), {t}; hexes < 5 ECGs hidden\nMAE {np.mean(np.abs(e[m] - EF[m])):.1f}, R² {r2:.2f}", fontsize=8)
            a.set_xlabel("nearest-echo EF (%)"); a.set_ylabel("predicted EF (%)")
    fig.tight_layout(); fig.savefig(PLOT4 / "regression_pred_vs_echo_hexbin.png", dpi=130); plt.close(fig)
    # 2. binned residuals by true EF (regression to the mean)
    fig, a = plt.subplots(figsize=(6.5, 3.8)); bins = np.arange(5, 95, 10)
    for n, c in (("reg_ridge_hdr+ecgf_A__tuned", "#4c78a8"), ("reg_mlp_hdr+ecgf_A__tuned", "#f58518"), ("mt_lam1_A", "#54a24b")):
        if n not in comb:
            continue
        e = np.mean([comb[n][r][2] for r in REPEATS], 0); res = e - EF; mu, ns = [], []
        for lo in bins[:-1]:
            m = (EF >= lo) & (EF < lo + 10); ns.append(m.sum()); mu.append(res[m].mean() if m.sum() >= SUP else np.nan)
        a.plot(bins[:-1] + 5, mu, "o-", c=c, label=n.replace("__tuned", ""))
    a.axhline(0, c="k", lw=0.6); a.set_xlabel("nearest-echo EF bin (%)"); a.set_ylabel("mean (predicted - echo) EF")
    a.set_title("Regression to the mean: low EF over-, high EF under-predicted (mean over repeats)", fontsize=8)
    a.legend(fontsize=7); fig.tight_layout(); fig.savefig(PLOT4 / "regression_binned_residuals.png", dpi=130); plt.close(fig)


def main():
    G_ = model_groups()
    comb = {n: {r: combined(n, G_[n][r]) for r in REPEATS} for n in KEY if n in G_}
    ch = characterise()
    gv = g_vs_nong(comb)
    ms = mild_study(comb)
    ms["error_budget_exploratory"] = error_budget(comb)
    ms["regression_error_near_boundaries"] = boundary_regression_error(comb)
    ms["mild_header_characteristics"] = mild_characteristics(comb)
    ms["mild_extra"] = mild_extra(comb)
    plots(comb); plot_confusions(comb)
    (RES4 / "g_cohort_analysis.json").write_text(json.dumps(dict(characterisation=ch, g_vs_nonG=gv.to_dict("records")),
                                                            indent=1, default=float))
    gv.to_csv(RES4 / "g_vs_nonG_performance.csv", index=False, float_format="%.4f")
    (RES4 / "mild_analysis.json").write_text(json.dumps(ms, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps({k: ch[k] for k in ("G", "non_G", "agreement_by_number_of_links_all_D", "G_disagreements")}, indent=1))
    print(gv.round(3).to_string(index=False))
    print(json.dumps(ms["auroc"], indent=1)[:3000])


if __name__ == "__main__":
    main()
