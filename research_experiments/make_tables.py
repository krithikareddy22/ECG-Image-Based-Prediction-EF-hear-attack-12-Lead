"""Collect every run in research_experiments/results/*/metrics.json into the comparison tables
and run the PRE-SPECIFIED paired comparisons (patient-cluster paired bootstrap + exact McNemar,
Holm correction within each metric family).

    python -m research_experiments.make_tables
-> results/comparison_main.csv/.md, per_class.csv, confusion_matrices.md, paired_tests.csv/.md
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_experiments.common import CLASSES, PLOTS, RES, load_meta, load_run_probs, paired_bootstrap  # noqa: E402

GROUP = [  # (prefix, group label) - order of the main table
    ("B0_", "Baseline (existing)"), ("R0_", "Baseline retrain"), ("A1_ecgf_aligned_deep_multinomial", "A1 frozen ECGFounder"),
    ("A2_", "A2 fine-tuned ECGFounder"), ("AB_", "B ordinal / loss (ECGFounder frozen)"),
    ("B_header", "B ordinal / loss (header)"), ("B_resnet18", "B ordinal (image CNN)"),
    ("HA_", "Header ablation"), ("FU_", "Fusion"), ("FUacc_", "Fusion, weights by val accuracy (secondary)"),
    ("A1_ecgf_", "A1 input/feature ablation"),
]

# pre-specified comparisons: (question, reference A, candidate B)
PAIRS = [
    ("A: frozen ECGFounder vs image CNN", "B0_resnet18_image_ce", "A1_ecgf_aligned_deep_multinomial"),
    ("A: fine-tuned ECGFounder (3-seed ens.) vs image CNN", "B0_resnet18_image_ce", "A2_ecgf_ft56_ce_3seed_ens"),
    ("A: fine-tuned vs frozen ECGFounder", "A1_ecgf_aligned_deep_multinomial", "A2_ecgf_ft56_ce_3seed_ens"),
    ("A: header+ECGFounder(A1) vs default model", "B0_default_fusion_header0.6_resnet18", "FU_header+ecgfA1"),
    ("A: header+ECGFounder(A2) vs default model", "B0_default_fusion_header0.6_resnet18", "FU_header+ecgfA2"),
    ("B: ordinal vs multinomial - header", "B_header_multinomial", "B_header_cumlink"),
    ("B: ordinal vs multinomial - ECGFounder frozen", "A1_ecgf_aligned_deep_multinomial", "AB_ecgf_aligned_deep_cumlink"),
    ("B: ordinal vs multinomial - ECGFounder fine-tuned", "A2_ecgf_ft56_ce_3seed_ens", "A2_ecgf_ft56_cumlink_3seed_ens"),
    ("B: ordinal vs multinomial - image CNN", "B0_resnet18_image_ce", "B_resnet18_image_cumlink_seed42"),
    ("B: ordinal fusion vs default model", "B0_default_fusion_header0.6_resnet18", "FU_ordinal_header+ecgfA2"),
    ("Ablation: header numbers only vs full header", "B0_header_logreg_multinomial", "HA_H3_demographics+measurements"),
]


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0
    for r, i in enumerate(order):
        run = max(run, (n - r) * p[i]); adj[i] = min(1, run)
    return adj


def load_all():
    runs = {}
    for f in sorted(RES.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if "test" in m and "auroc_ovr" in m["test"]:
            runs[m["tag"]] = m
    return runs


def group_of(tag):
    for p, g in GROUP:
        if tag.startswith(p):
            return g
    return "other"


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.2f}-{ci[1]:.2f}]"


def main():
    runs = load_all()
    rows, pcr = [], []
    for tag, m in runs.items():
        t, v = m["test"], m["val"]
        pc = t["per_class"]
        rows.append(dict(
            group=group_of(tag), model=tag, input=m.get("input", ""), decoding=m.get("decoding", "argmax"),
            val_selection_score=m.get("val_selection_score", np.nan),
            accuracy=t["accuracy"], accuracy_ci=t["ci95"]["accuracy"],
            balanced_accuracy=t["balanced_accuracy"], balanced_accuracy_ci=t["ci95"]["balanced_accuracy"],
            macro_f1=t["macro_f1"], macro_f1_ci=t["ci95"]["macro_f1"],
            normal_f1=pc["Normal"]["f1"], mild_f1=pc["Mild"]["f1"], moderate_f1=pc["Moderate"]["f1"],
            severe_f1=pc["Severe"]["f1"], mild_f1_ci=t["ci95"]["mild_f1"],
            macro_auroc=t["auroc_ovr"]["macro"], macro_auroc_ci=t["ci95"]["macro_auroc"],
            mild_auroc=t["auroc_ovr"]["Mild"], qwk=t["qwk"], qwk_ci=t["ci95"]["qwk"],
            hf_auc=t["hf_screen"]["auc"], hf_sens=t["hf_screen"]["sensitivity"], hf_spec=t["hf_screen"]["specificity"],
            within_one=t["within_one_category"], off_by_2plus=t["error_severity"]["off_by_2"] + t["error_severity"]["off_by_3"],
            mean_abs_class_error=t["error_severity"]["mean_abs_class_error"],
            val_balanced_accuracy=v["balanced_accuracy"], val_macro_f1=v["macro_f1"],
            params_trainable=m.get("params_trainable", m.get("head_trainable_params", m.get("params"))),
            params_total=m.get("params_total", m.get("params")),
            train_minutes=m.get("train_minutes", (m.get("head_train_seconds") or np.nan) / 60 if m.get("head_train_seconds") else np.nan),
            inference_s_per_ecg_model=m.get("inference_seconds_per_ecg_model_only", m.get("head_inference_seconds_per_ecg")),
            best_epoch=m.get("best_epoch")))
        for c in CLASSES:
            pcr.append(dict(model=tag, cls=c, precision=pc[c]["precision"], recall=pc[c]["recall"], f1=pc[c]["f1"],
                            support=pc[c]["support"], auroc_ovr=t["auroc_ovr"][c]))
    order = {g: i for i, (_, g) in enumerate(GROUP)}
    df = pd.DataFrame(rows).sort_values(["group", "model"], key=lambda s: s.map(order) if s.name == "group" else s)
    df.to_csv(RES / "comparison_main.csv", index=False)
    pd.DataFrame(pcr).to_csv(RES / "per_class.csv", index=False)

    md = ["| Group | Model | Input | Accuracy [95% CI] | Balanced acc. [95% CI] | Macro-F1 [95% CI] | Normal F1 | Mild F1 | "
          "Moderate F1 | Severe F1 | Macro AUROC | QWK | HF AUC |", "|" + "---|" * 13]
    for r in df.itertuples():
        md.append(f"| {r.group} | {r.model} | {r.input} | {fmt_ci(r.accuracy, r.accuracy_ci)} | "
                  f"{fmt_ci(r.balanced_accuracy, r.balanced_accuracy_ci)} | {fmt_ci(r.macro_f1, r.macro_f1_ci)} | "
                  f"{r.normal_f1:.3f} | {r.mild_f1:.3f} | {r.moderate_f1:.3f} | {r.severe_f1:.3f} | "
                  f"{r.macro_auroc:.3f} | {r.qwk:.3f} | {r.hf_auc:.3f} |")
    (RES / "comparison_main.md").write_text("\n".join(md) + "\n")

    cm_md = []
    for tag, m in runs.items():
        cm = np.array(m["test"]["confusion_matrix"])
        cm_md.append(f"**{tag}** (rows = true N/Mi/Mo/S, cols = predicted)\n\n```\n" +
                     "\n".join(" ".join(f"{x:4d}" for x in row) for row in cm) + "\n```\n")
    (RES / "confusion_matrices.md").write_text("\n".join(cm_md))

    # paired tests on the test set (patient-cluster bootstrap)
    meta = load_meta(); mt = meta[meta.split == "test"].reset_index(drop=True)
    y, g = mt.label_idx.values, mt.patient.values

    def get(tag):
        ids, p = load_run_probs(tag, "test")
        dfp = pd.read_csv(RES / tag / "test_predictions.csv")
        order_ = pd.Series(range(len(ids)), index=ids).loc[mt.ecg_id].values
        pred = dfp.pred.map({c: i for i, c in enumerate(CLASSES)}).values[order_]
        return p[order_], pred

    trows = []
    for q, a, b in PAIRS:
        if a not in runs or b not in runs:
            trows.append(dict(question=q, reference=a, candidate=b, status="not run")); continue
        (pa, da), (pb, db) = get(a), get(b)
        r = paired_bootstrap(y, pa, pb, g, da, db)
        row = dict(question=q, reference=a, candidate=b, status="ok")
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk", "mild_f1", "mild_recall", "macro_auroc", "hf_auc"):
            row[f"{k}_diff"] = r[k]["diff"]; row[f"{k}_ci"] = r[k]["ci95"]; row[f"{k}_p"] = r[k]["p_two_sided"]
        row["mcnemar_p"] = r["mcnemar_accuracy"]["p_exact"]
        row["b_right_a_wrong"] = r["mcnemar_accuracy"]["b_right_a_wrong"]; row["a_right_b_wrong"] = r["mcnemar_accuracy"]["a_right_b_wrong"]
        trows.append(row)
    T = pd.DataFrame(trows)
    ok = T.status == "ok"
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "mild_f1"):
        T.loc[ok, f"{k}_p_holm"] = holm(T.loc[ok, f"{k}_p"].values)
    T.to_csv(RES / "paired_tests.csv", index=False)
    md = ["| Question | Δ Accuracy [95% CI] (p, Holm p) | Δ Balanced acc. [95% CI] (p, Holm p) | Δ Macro-F1 [95% CI] (p, Holm p) | "
          "Δ Mild F1 [95% CI] (p, Holm p) | Δ Macro AUROC [95% CI] | McNemar p |", "|---|---|---|---|---|---|---|"]
    for r in T[ok].itertuples():
        cell = lambda k: (f"{getattr(r, k + '_diff'):+.3f} [{getattr(r, k + '_ci')[0]:+.2f}, {getattr(r, k + '_ci')[1]:+.2f}] "
                          f"(p={getattr(r, k + '_p'):.3f}, {getattr(r, k + '_p_holm'):.3f})")
        md.append(f"| {r.question} | {cell('accuracy')} | {cell('balanced_accuracy')} | {cell('macro_f1')} | {cell('mild_f1')} | "
                  f"{r.macro_auroc_diff:+.3f} [{r.macro_auroc_ci[0]:+.2f}, {r.macro_auroc_ci[1]:+.2f}] | {r.mcnemar_p:.3f} |")
    (RES / "paired_tests.md").write_text("\n".join(md) + "\n")

    # summary figure: balanced accuracy with CIs for the key models
    key = [t for t in ["B0_resnet18_image_ce", "R0_resnet18_filtered_retrain_seed42", "B0_header_logreg_multinomial",
                       "B0_default_fusion_header0.6_resnet18", "A1_ecgf_aligned_deep_multinomial",
                       "A2_ecgf_ft56_ce_3seed_ens", "A2_ecgf_ft56_cumlink_3seed_ens", "B_header_cumlink",
                       "AB_ecgf_aligned_deep_cumlink", "B_resnet18_image_cumlink_seed42", "FU_header+ecgfA1",
                       "FU_header+ecgfA2", "FU_ordinal_header+ecgfA2", "FU_header+image+ecgfA2"] if t in runs]
    d = df.set_index("model").loc[key]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(key) + 1.2), sharey=True)
    for ax, (col, lab) in zip(axes, (("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy"))):
        lo = d[col] - d[col + "_ci"].map(lambda c: c[0]); hi = d[col + "_ci"].map(lambda c: c[1]) - d[col]
        ax.errorbar(d[col], range(len(d)), xerr=[lo, hi], fmt="o", color="#1f4e79", capsize=3)
        ax.axvline(d.loc["B0_default_fusion_header0.6_resnet18", col] if "B0_default_fusion_header0.6_resnet18" in d.index else np.nan,
                   ls=":", c="gray")
        ax.set_xlabel(lab + " (test, 95% patient-bootstrap CI)"); ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(range(len(d)), d.index, fontsize=8); axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(PLOTS / "summary_key_models.png", dpi=130); plt.close(fig)
    print(df[["model", "accuracy", "balanced_accuracy", "macro_f1", "mild_f1", "qwk", "macro_auroc", "hf_auc",
              "val_selection_score"]].round(3).to_string(index=False))
    print(T[ok][["question", "accuracy_diff", "accuracy_p", "balanced_accuracy_diff", "balanced_accuracy_p",
                 "balanced_accuracy_p_holm", "mild_f1_diff", "mild_f1_p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
