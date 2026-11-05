"""Mild-EF and error-severity analysis on the TEST set for the key models (reads saved
predictions only). Outputs results/mild_analysis.csv, error_analysis.csv, subgroup_analysis.csv
and plots/mild_destinations.png.
    python -m research_experiments.analysis_mild_errors
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

from research_experiments.common import CLASSES, PLOTS, RES, ROOT, load_meta  # noqa: E402

KEY = ["B0_resnet18_image_ce", "R0_resnet18_filtered_retrain_seed42", "B_resnet18_image_cumlink_seed42",
       "B0_header_logreg_multinomial", "B_header_cumlink", "A1_ecgf_aligned_deep_multinomial",
       "AB_ecgf_aligned_deep_cumlink", "AB_ecgf_aligned_deep_coral", "AB_ecgf_aligned_deep_focal",
       "A2_ecgf_ft56_ce_3seed_ens", "A2_ecgf_ft56_cumlink_3seed_ens", "B0_default_fusion_header0.6_resnet18",
       "FU_header+ecgfA1", "FU_header+ecgfA2", "FU_ordinal_header+ecgfA2", "FU_header+image+ecgfA2"]


def main():
    meta = load_meta(); mt = meta[meta.split == "test"].reset_index(drop=True)
    h = pd.read_csv(ROOT / "data" / "header_features.csv")[["ecg_id", "male", "age"]]
    mt = mt.merge(h, on="ecg_id", how="left")
    y = mt.label_idx.values
    mild_rows, err_rows, sub_rows = [], [], []
    dest = {}
    for tag in KEY:
        f = RES / tag / "test_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("ecg_id").loc[mt.ecg_id]
        P = d[[f"p_{c}" for c in CLASSES]].values
        pred = d.pred.map({c: i for i, c in enumerate(CLASSES)}).values
        mild = y == 1
        dest[tag] = np.bincount(pred[mild], minlength=4) / mild.sum()
        tp = np.sum((pred == 1) & mild)
        mild_rows.append(dict(model=tag, mild_n=int(mild.sum()), mild_predicted_n=int((pred == 1).sum()),
                              mild_recall=tp / mild.sum(), mild_precision=tp / max(1, (pred == 1).sum()),
                              mild_auroc_ovr=roc_auc_score(mild, P[:, 1]),
                              mild_vs_normal_auroc=roc_auc_score(y[(y <= 1)] == 1, (1 - P[:, 0])[(y <= 1)]),
                              **{f"mild_to_{c}": dest[tag][k] for k, c in enumerate(CLASSES)}))
        e = np.abs(pred - y)
        err_rows.append(dict(model=tag, exact=int((e == 0).sum()), off_by_1=int((e == 1).sum()), off_by_2=int((e == 2).sum()),
                             off_by_3=int((e == 3).sum()), mean_abs_class_error=e.mean(),
                             adjacent_share_of_errors=(e == 1).sum() / max(1, (e > 0).sum()),
                             normal_called_moderate_or_severe=int(((y == 0) & (pred >= 2)).sum()),
                             severe_called_normal=int(((y == 3) & (pred == 0)).sum()),
                             under_grading=int((pred < y).sum()), over_grading=int((pred > y).sum())))
        for name, mask in (("male", mt.male.values == 1), ("female", mt.male.values == 0),
                           *[(f"year {yr}", mt.year.values == yr) for yr in sorted(mt.year.unique())]):
            if mask.sum() and len(np.unique(y[mask])) == 4:
                sub_rows.append(dict(model=tag, subgroup=name, n=int(mask.sum()), accuracy=float(np.mean(pred[mask] == y[mask])),
                                     balanced_accuracy=balanced_accuracy_score(y[mask], pred[mask]),
                                     mild_recall=float(np.mean(pred[mask & mild] == 1)) if (mask & mild).any() else np.nan))
    pd.DataFrame(mild_rows).to_csv(RES / "mild_analysis.csv", index=False)
    pd.DataFrame(err_rows).to_csv(RES / "error_analysis.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(RES / "subgroup_analysis.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(dest) + 1))
    left = np.zeros(len(dest)); cols = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    M = np.array(list(dest.values()))
    for k, c in enumerate(CLASSES):
        ax.barh(range(len(dest)), M[:, k], left=left, color=cols[k], label=f"predicted {c}"); left += M[:, k]
    ax.set_yticks(range(len(dest)), list(dest), fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("share of the 47 true-Mild test ECGs"); ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(PLOTS / "mild_destinations.png", dpi=130); plt.close(fig)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(mild_rows).round(3).to_string(index=False))
    print(pd.DataFrame(err_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
