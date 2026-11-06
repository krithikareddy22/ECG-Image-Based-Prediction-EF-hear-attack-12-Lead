"""5E statistical validation + 5F model selection (pre-registered rules, PHASE5_PLAN.md §3-4).

1. Every Phase-5 candidate + the baselines: seed-combined per repeat, full metric suite, admissibility.
2. Planned family comparisons, each family vs its stated baseline, Holm within family and metric.
3. Global: every candidate vs C1 (vote) and vs C2 (calibrated MLP), Holm over all candidates.
4. Patient-cluster paired bootstrap + McNemar (repeat 0) for the five most accurate admissible candidates.
5. Admissibility sensitivity (recall floor 0.10 / 0.15 / 0.20), decision gate, selection.
Outputs: results/phase5_all_models.csv, phase5_family_comparisons.csv, phase5_selection.json, plots/phase5_*.png
    python -m research_experiments.phase_5.validation.final_stats
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_experiments.phase_5.baseline.report import C1, C2, groups  # noqa: E402
from research_experiments.phase_5.p5common import CLASSES, RES5, Y, confusion, guard5  # noqa: E402
from research_experiments.phase_5.validation.stats import family_table, model_summary  # noqa: E402

MLP = "P5A-import-mlpfusion-v1"
BASELINES = [C1, C2, MLP, "P5A-import-p4-fusvotemlpreg-v1", "P5A-import-stack-v1", "P5A-calmlp-accuracy-v1", "P5A-calmlp-floor-v1",
             "P5A-import-hdrclinlr-v1"]


def complete(e, G):
    """All registered runs of the experiment exist (partially finished experiments are never summarised)."""
    from research_experiments.phase_5.registry import EXPERIMENTS
    want = EXPERIMENTS.get(e, {}).get("runs")
    have = {r for rr in G.get(e, {}).values() for r in rr}
    return bool(want) and set(want) <= have


def families(G):
    ids = sorted(e for e in G if e.startswith("P5D-") and complete(e, G))
    pick = lambda pre: [e for e in ids if e.startswith(pre)]
    return {
        "E1 feature-enhanced MLP": (pick("P5D-E1-"), MLP),
        "E2 gradient boosting": (pick("P5D-E2-"), MLP),
        "E3 nested stacking": (pick("P5D-E3-"), C1),
        "E4 hierarchical": (pick("P5D-E4-"), MLP),
        "E5 decision rules on the vote": (pick("P5D-E5-vote-"), C1),
        "E5 decision rules on the MLP fusion": (pick("P5D-E5-mlpfusion-"), MLP),
        "E6 quality-aware training": ([e for e in pick("P5D-E6-") if "qgate" not in e], MLP),
        "E6 quality-gated blend": (pick("P5D-E6-qgate"), C1),
        "E7 training strategies (new)": ([e for e in pick("P5D-E7-") if "-p2-" not in e], MLP),
        "E7 phase-2 strategies (repeat 0 only)": (pick("P5D-E7-p2-"), MLP),
    }


def main():
    G = groups()
    fam = families(G)
    cands = [e for f in fam.values() for e in f[0]]
    allm = list(dict.fromkeys(BASELINES + cands))
    rows = []
    for e in allm:
        s, rr = model_summary(e, G)
        cm = np.mean([confusion(Y, rr[r][1]) for r in rr], 0)
        rows.append(dict(model=e, family=next((n for n, (ids, _) in fam.items() if e in ids), "baseline"), repeats=s["repeats"], runs=s["runs"],
                         **{k: s[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "qwk", "min_class_recall_mean")},
                         **{f"{m}_{c}": s[f"{m}_{c}"] for c in CLASSES for m in ("precision", "recall", "f1")},
                         admissible_0_10=s["min_class_recall_mean"] >= 0.10, admissible_0_15=s["admissible"],
                         admissible_0_20=s["min_class_recall_mean"] >= 0.20, confusion_mean=json.dumps(cm.round(1).tolist())))
    A = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    # eligibility (pre-registered: mean over repeats 0-2): models evaluated on repeat 0 only are reported, never selected
    A["eligible"] = A.repeats == 3
    # planned family comparisons
    F = []
    for name, (ids, ref) in fam.items():
        if ids:
            T = family_table(ids, ref, G); T.insert(0, "family", name); F.append(T)
    F = pd.concat(F)
    # global comparisons vs C1 and C2 (Holm over all candidates)
    g1 = family_table(cands, C1, G).add_prefix("vsC1_").rename(columns={"vsC1_model": "model"})
    g2 = family_table(cands, C2, G).add_prefix("vsC2_").rename(columns={"vsC2_model": "model"})
    keep1 = ["model", "vsC1_d_accuracy", "vsC1_p_accuracy", "vsC1_p_holm_accuracy", "vsC1_same_sign_accuracy", "vsC1_ci_accuracy",
             "vsC1_genuine_accuracy_gain", "vsC1_d_balanced_accuracy", "vsC1_d_macro_f1", "vsC1_d_recall_Severe"]
    keep2 = ["model", "vsC2_d_accuracy", "vsC2_p_accuracy", "vsC2_p_holm_accuracy", "vsC2_same_sign_accuracy", "vsC2_genuine_accuracy_gain"]
    A = A.merge(g1[keep1], on="model", how="left").merge(g2[keep2], on="model", how="left")
    # bootstrap for the top admissible candidates
    top = A[A.admissible_0_15 & A.eligible & A.model.isin(cands)].head(5).model.tolist()
    boot = {}
    if top:
        b1 = family_table(top, C1, G, boot=True).set_index("model"); b2 = family_table(top, C2, G, boot=True).set_index("model")
        for m in top:
            boot[m] = dict(vs_C1=dict(d_acc_ci95_cluster=b1.loc[m, "boot_r0_d_accuracy_ci95"], mcnemar_p=b1.loc[m, "mcnemar_r0_p"]),
                           vs_C2=dict(d_acc_ci95_cluster=b2.loc[m, "boot_r0_d_accuracy_ci95"], mcnemar_p=b2.loc[m, "mcnemar_r0_p"]))
    # selection
    sel = {}
    for fl in ("0_10", "0_15", "0_20"):
        ok = A[A[f"admissible_{fl}"] & A.eligible]
        sel[f"best_admissible_floor_{fl}"] = ok.iloc[0][["model", "accuracy", "balanced_accuracy", "macro_f1", "min_class_recall_mean"]].to_dict() if len(ok) else None
    best = A[A.admissible_0_15 & A.eligible].iloc[0]
    acc = float(best.accuracy)
    gate = ("< 58 %: investigate feature and data limitations" if acc < 0.58 else "58-60 %: assess complementary models" if acc < 0.60 else
            "60-62 %: refine" if acc < 0.62 else "62-65 %: consolidate" if acc < 0.65 else ">= 65 %: nominate for final validation")
    sel.update(selected_model=best.model, selected_accuracy=acc, decision_gate=gate, target_65_reached=bool(acc >= 0.65),
               best_any_model=A.iloc[0][["model", "accuracy", "min_class_recall_mean", "admissible_0_15", "eligible"]].to_dict(),
               best_eligible_any_admissibility=A[A.eligible].iloc[0][["model", "accuracy", "min_class_recall_mean"]].to_dict(),
               eligibility_rule="selection requires all 3 CV repeats (pre-registered mean over repeats 0-2); repeat-0-only models are reported only",
               genuine_improvement_vs_C1=A[A.vsC1_genuine_accuracy_gain.fillna(False).astype(bool)].model.tolist(),
               genuine_improvement_vs_C2=A[A.vsC2_genuine_accuracy_gain.fillna(False).astype(bool)].model.tolist(),
               bootstrap_top_admissible=boot, n_candidates=len(cands))
    guard5(RES5).mkdir(exist_ok=True)
    A.to_csv(RES5 / "phase5_all_models.csv", index=False, float_format="%.4f")
    F.to_csv(RES5 / "phase5_family_comparisons.csv", index=False, float_format="%.4f")
    (RES5 / "phase5_selection.json").write_text(json.dumps(sel, indent=1, default=str))
    # figures: accuracy vs min class recall, and the selected model's confusion matrix
    P = RES5 / "plots"; P.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in A.iterrows():
        c = "#7f7f7f" if r.family == "baseline" else "#4c78a8"
        ax.scatter(r.min_class_recall_mean, r.accuracy, c=c, s=22, alpha=0.8)
    for m, lab in ((C1, "C1 vote"), (C2, "C2 calibrated MLP"), (best.model, "selected")):
        r = A[A.model == m].iloc[0]; ax.annotate(lab, (r.min_class_recall_mean, r.accuracy), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.15, ls=":", c="k", lw=0.8); ax.axhline(0.65, ls="--", c="#e45756", lw=0.8)
    ax.set_xlabel("minimum per-class recall (mean over repeats)"); ax.set_ylabel("four-class accuracy (837 ECGs, dev CV)")
    ax.set_title("Phase 5 candidates (blue) and baselines (grey); dotted = recall floor, dashed = 65 % target", fontsize=8)
    fig.tight_layout(); fig.savefig(P / "phase5_accuracy_vs_min_recall.png", dpi=130); plt.close(fig)
    cm = np.array(json.loads(best.confusion_mean)); fr = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4)); ax.imshow(fr, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.0f}\n{fr[i, j]:.0%}", ha="center", va="center", fontsize=7, color="white" if fr[i, j] > .55 else "black")
    ax.set_xticks(range(4), CLASSES, fontsize=7); ax.set_yticks(range(4), CLASSES, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("folder label"); ax.set_title(f"Selected: {best.model}\n(mean counts over 3 repeats)", fontsize=7)
    fig.tight_layout(); fig.savefig(P / "phase5_selected_confusion.png", dpi=130); plt.close(fig)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 120)
    print(A[["model", "family", "accuracy", "balanced_accuracy", "macro_f1", "recall_Severe", "precision_Severe", "min_class_recall_mean",
             "admissible_0_15", "vsC1_d_accuracy", "vsC1_p_accuracy", "vsC1_p_holm_accuracy"]].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in sel.items() if k != "bootstrap_top_admissible"}, indent=1, default=str))


if __name__ == "__main__":
    main()
