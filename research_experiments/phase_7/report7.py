"""Phase 7 deliverables: model comparison CSV, statistics, plots (accuracy / macro-F1, Mild P/R/F1, confusion matrices,
learning curve) and the experiment inventory. Aggregates only.
    python -m research_experiments.phase_7.report7
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_experiments.phase_7 import p7common as C

R3 = (0, 1, 2)
TARGETS = dict(accuracy=0.70, recall_Mild=0.60, precision_Mild=0.50, f1_Mild=0.55, macro_f1=0.60, balanced_accuracy=0.60)


def models():
    """name -> (family, phase, experiment, list over repeats of run-id lists, gate key)."""
    M = {
        "C3 (C1 vote + nested prior rule) [reference]": ("ensemble", "5/6A", "reference", [[C.REF_C3.format(r=r)] for r in R3], None),
        "C2 calibrated MLP fusion (protected rule)": ("tabular + ECGFounder", "5", "existing", [[f"P5A-calmlp-protected-v1__s0_r{r}"] for r in R3], None),
        "C1 vote (MLP fusion x3, ECGF linear, header LR)": ("ensemble", "3/6A", "existing", [[f"P6A-vote-B1-v1__s0_r{r}"] for r in R3], None),
        "MLP fusion [header || f6], 3 seeds": ("tabular + ECGFounder", "2/6A", "existing", [[f"P6A-mlpfusion-B1-v1__s{s}_r{r}" for s in C.SEEDS] for r in R3], None),
        "MLP fusion [header || f6], seed 42 [pilot reference]": ("tabular + ECGFounder", "2/6A", "existing", [[C.REF_MEMBER.format(r=r)] for r in R3], None),
        "ECGFounder f6 linear (distance-CE)": ("ECGFounder", "2/6A", "existing", [[f"P6A-ecgflindist-B1-v1__s42_r{r}"] for r in R3], None),
        "Header + clinical logistic regression": ("tabular", "3/6A", "existing", [[f"P6A-hdrclinlr-B1-v1__s0_r{r}"] for r in R3], None),
    }
    for n, lab in (("f6", "E1-f6 (reproduction)"), ("msp", "E1-msp stage-5 mean||max||std"), ("col", "E1-col per-column stage-5 mean"),
                   ("s45", "E1-s45 stage-4 mean || f6"), ("dx", "E1-dx 150 pre-trained diagnostic outputs")):
        M[lab] = ("ECGFounder representation", "7", "E1 pilot", [[f"P7-E1-{n}-v1__s42_r{r}"] for r in R3], ("e1", n))
    for n, lab in (("ce", "E2-ce CE control"), ("mt", "E2-mt multitask CE + EF"), ("ldl", "E2-ldl boundary-aware soft labels"),
                   ("scl", "E2-scl EF-weighted supervised contrastive")):
        M[lab] = ("EF supervision head", "7", "E2 pilot", [[f"P7-E2-{n}-v1__s42_r{r}"] for r in R3], ("e2", n))
    for n, lab in (("cnn", "E3-cnn multi-lead 1D ResNet (repeat 0)"), ("cnnhdr", "E3-cnnhdr 1D ResNet + header (repeat 0)")):
        M[lab] = ("waveform CNN", "7", "E3 pilot", [[f"P7-E3-{n}-v1__s42_r0"]], ("e3", n))
    return M


def gate_result(g):
    if g is None:
        return ""
    exp, n = g
    d = json.loads((C.RES7 / f"{exp}_pilot.json").read_text())
    x = d["pilots"][n] if exp != "e3" else d[n]
    return "pass" if x.get("passes", x.get("gate", {}).get("passes")) else "fail"


def main():
    M = models(); rows, stats = [], []
    ref_member = M["MLP fusion [header || f6], seed 42 [pilot reference]"][3]; ref_c3 = M["C3 (C1 vote + nested prior rule) [reference]"][3]
    for name, (fam, ph, exp, rids, g) in M.items():
        s = C.summary(rids)
        rows.append(dict(model=name, family=fam, phase=ph, experiment=exp, repeats=len(rids), gate=gate_result(g),
                         accuracy=s["accuracy"], accuracy_ci95_corrected=f"[{s['accuracy_ci95_corrected'][0]:.3f}, {s['accuracy_ci95_corrected'][1]:.3f}]",
                         balanced_accuracy=s["balanced_accuracy"], macro_f1=s["macro_f1"],
                         **{f"{m}_{c}": s[f"{m}_{c}"] for c in C.CLASSES for m in ("precision", "recall", "f1")},
                         min_recall=s["min_recall"], admissible=s["admissible"],
                         **{f"meets_{k}": bool(s[k] >= v) for k, v in TARGETS.items()}, confusion_mean=json.dumps(s["confusion_mean"])))
        if exp.startswith("E"):
            ref = [ref_member[0]] if len(rids) == 1 else ref_member
            for metric in ("accuracy", "macro_f1", "f1_Mild"):
                stats.append(dict(model=name, comparator="MLP fusion seed 42 (same folds)", **C.paired(rids, ref, metric)))
            refc = [ref_c3[0]] if len(rids) == 1 else ref_c3
            stats.append(dict(model=name, comparator="C3", **C.paired(rids, refc, "accuracy")))
    adj = C.holm([x["p_corrected_t"] for x in stats])
    for x, a in zip(stats, adj):
        x["p_holm"] = float(a); x["ci95_corrected_delta"] = f"[{x['ci95_corrected_delta'][0]:+.3f}, {x['ci95_corrected_delta'][1]:+.3f}]"
    D = pd.DataFrame(rows); S = pd.DataFrame(stats)
    D.to_csv(C.guard7(C.P7 / "PHASE7_MODEL_COMPARISON.csv"), index=False, float_format="%.4f")
    S.to_csv(C.guard7(C.RES7 / "phase7_comparisons.csv"), index=False, float_format="%.4f")
    plots(D)
    inventory(D)
    print(D[["model", "accuracy", "macro_f1", "precision_Mild", "recall_Mild", "f1_Mild", "admissible", "gate"]].to_string(index=False))
    print(S[["model", "comparator", "metric", "delta", "p_corrected_t", "p_holm"]].to_string(index=False))


def plots(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.guard7(C.PLOTS7).mkdir(parents=True, exist_ok=True)
    col = D.experiment.map(lambda e: "#9aa5b1" if not e.startswith("E") else {"E1 pilot": "#4c78a8", "E2 pilot": "#f58518", "E3 pilot": "#54a24b"}[e])
    xs = np.arange(len(D))
    short = [m.replace(" [pilot reference]", "").replace(" [reference]", "").replace("[header || f6], ", "")[:38] for m in D.model]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, k, t in ((axes[0], "accuracy", 0.70), (axes[1], "macro_f1", 0.60)):
        v = D[k].values; ax.bar(xs, v, color=col)
        if k == "accuracy":
            ci = np.array([json.loads(c.replace("[", "[").replace("]", "]")) for c in D.accuracy_ci95_corrected])
            ax.errorbar(xs, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", ecolor="k", capsize=2, lw=0.8)
        for i, y in enumerate(v):
            ax.text(i, y + 0.012, f"{y:.3f}", ha="center", fontsize=7)
        ax.axhline(t, ls=":", c="r", lw=0.9); ax.text(len(D) - 0.5, t + 0.01, f"target {t}", color="r", fontsize=7, ha="right")
        ax.set_ylabel(k.replace("_", "-")); ax.set_ylim(0, 0.85)
    axes[0].set_title("Phase 7: four-class accuracy (corrected 95 % CI) and macro-F1, all 837 ECGs (grey existing, blue E1, orange E2, green E3)", fontsize=9)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(short, rotation=55, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(C.PLOTS7 / "phase7_accuracy_macro_f1.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 5))
    for j, (k, c, t) in enumerate((("precision_Mild", "#4c78a8", 0.50), ("recall_Mild", "#f58518", 0.60), ("f1_Mild", "#54a24b", 0.55))):
        ax.bar(xs + (j - 1) * 0.27, D[k], 0.26, color=c, label=f"Mild {k.split('_')[0]} (target {t})")
        ax.axhline(t, ls=":", c=c, lw=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(short, rotation=55, ha="right", fontsize=7); ax.set_ylim(0, 0.8); ax.legend(fontsize=7)
    ax.set_title("Mild-class precision, recall and F1, all 837 ECGs (dotted lines: owner targets)", fontsize=9)
    fig.tight_layout(); fig.savefig(C.PLOTS7 / "phase7_mild_performance.png", dpi=150); plt.close(fig)
    pick = ["C3 (C1 vote + nested prior rule) [reference]", "MLP fusion [header || f6], seed 42 [pilot reference]", "E1-col per-column stage-5 mean",
            "E2-mt multitask CE + EF", "E2-ldl boundary-aware soft labels", "E3-cnn multi-lead 1D ResNet (repeat 0)"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, name in zip(axes.ravel(), pick):
        cm = np.array(json.loads(D.set_index("model").loc[name, "confusion_mean"])); rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.0f}\n{rn[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if rn[i, j] > 0.5 else "black")
        ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(C.CLASSES, fontsize=7); ax.set_yticklabels(C.CLASSES, fontsize=7)
        ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("true", fontsize=7)
        ax.set_title(name.replace(" [pilot reference]", " (seed 42)").replace(" [reference]", "")[:48], fontsize=8)
    fig.suptitle("Confusion matrices (mean counts over repeats; row %), all 837 ECGs", fontsize=10)
    fig.tight_layout(); fig.savefig(C.PLOTS7 / "phase7_confusion_matrices.png", dpi=150); plt.close(fig)
    lc = C.RES7 / "e5_learning_curve.csv"
    if lc.exists():
        L = pd.read_csv(lc); fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); n = L.mean_inner_train_ecgs
        ax[0].plot(n, L.accuracy, "o-", label="accuracy"); ax[0].plot(n, L.macro_f1, "s-", label="macro-F1"); ax[0].plot(n, L.f1_Mild, "^-", label="Mild F1")
        ax[0].axhline(0.70, ls=":", c="r", lw=0.8); ax[0].set_xscale("log", base=2); ax[0].set_xlabel("inner-train ECGs (log2)"); ax[0].legend(fontsize=7)
        ax[0].set_title("Reference member vs training size", fontsize=9)
        ax[1].plot(n, L.ef_sd_all, "o-", label="all ECGs"); ax[1].plot(n, L.ef_sd_verified, "s-", label="verified-EF ECGs")
        ax[1].axhline(9, ls=":", c="r", lw=0.8); ax[1].set_xscale("log", base=2); ax[1].set_xlabel("training ECGs (log2)")
        ax[1].set_ylabel("EF residual SD (points)"); ax[1].legend(fontsize=7); ax[1].set_title("Ridge EF error vs training size", fontsize=9)
        fig.tight_layout(); fig.savefig(C.PLOTS7 / "phase7_learning_curve.png", dpi=150); plt.close(fig)


def inventory(D):
    lines = ["# Phase 7 experiment inventory", "",
             "Generated by `report7.py`. Development set (837 ECGs), Phase 2 patient-grouped folds; aggregates only. Run ids:",
             "`results/experiments/P7-*.json`; per-ECG predictions in the git-ignored `data/private/runs/`.", "",
             "| Experiment | Model | Seeds × repeats | Gate | Accuracy | Mild F1 | Status |", "|---|---|---|---|---|---|---|"]
    for _, x in D[D.experiment.str.startswith("E")].iterrows():
        lines.append(f"| {x.experiment} | {x.model} | 1 × {x.repeats} | {x.gate} | {x.accuracy:.3f} | {x.f1_Mild:.3f} | complete; not advanced |")
    lines += ["| E6 | Header-free two-stage cascades (2 stage-1 x 5 stage-2), all 837 ECGs | 1 x 3 | n/a | 0.538-0.568 | 0.177-0.232 | complete; `results/e6_cascades.csv` |",
              "| E7 | Header-free selective prediction with >= 0.70 accuracy and Mild detected | 1 x 3 | n/a | infeasible in all 15 training folds | | complete; `results/e7_selective_mild.csv` |",
              "| E8 | Doctor-style hierarchy with refer zones (12 selective variants + no-refer), header-free | 1 x 3 | n/a | 0.544 all / 0.633-0.677 answered | Mild <= 3 % caught | complete; `results/e8_doctor.csv` |",
              "| E5-lc | Learning curve: reference member and ridge EF at 25/50/75/100 % training | 1 × 3 | n/a | see `results/e5_learning_curve.csv` | | complete |",
              "", "## Not run, and why", "",
              "| Item | Reason |", "|---|---|",
              "| Full runs (3 seeds) of any E1-E3 candidate | No pilot passed its pre-registered gate (PHASE7_PLAN.md section 3) |",
              "| E4 fusion | Only for candidates that pass their full run; none did |",
              "| ECGFounder fine-tuning (LoRA / partial) | Phase 6 A1 / A2 fine-tuning failed its screen (r 0.650 vs 0.652); not repeated |",
              "| LSTM / BiLSTM / GRU / CNN-LSTM / ECGFounder-LSTM | Discontinued by the owner; none running, none launched |",
              "| Image (page) ResNet | Phases 1-3: 0.40-0.49; not repeated |",
              "| Raw digital ECG models | No raw digital ECG export is available (DATA_ACCESS.md); waveforms are digitised from PDFs |",
              "| Locked 210-ECG test cohort | Not authorised; never loaded or scored |"]
    C.guard7(C.P7 / "PHASE7_EXPERIMENT_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
