"""Figures for the final report (aggregate results only; no per-ECG data).
    python -m research_experiments.final_report.make_figures
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
R6B = HERE.parent / "phase_6b_threshold_sampling_lstm" / "results"
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a1", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})

# Four-class accuracy on all 837 development ECGs (patient-grouped 5-fold CV x 3 repeats; corrected 95 % CI where available).
# (name, accuracy, lo, hi, group) ; group: "img" image-only, "hdr" uses printed header, "final" highlighted
MODELS = [
    ("1-D ResNet on digitised waveforms (P7 E3, rep. 0)", 0.496, None, None, "img"),
    ("LSTM-sel, nested (P6A)", 0.470, 0.424, 0.515, "img"),
    ("CNN-LSTM L5 (P6B)", 0.484, 0.435, 0.532, "img"),
    ("EchoNext-Mini outputs + LR (external model)", 0.486, 0.436, 0.536, "img"),
    ("Median-beat shape LR (ENH M3)", 0.467, None, None, "img"),
    ("MiniRocket LR (ENH M4)", 0.504, None, None, "img"),
    ("ECGFounder frozen, linear", 0.528, 0.484, 0.573, "img"),
    ("ECGFounder fine-tuned (P6 A1)", 0.536, 0.492, 0.581, "img"),
    ("IMG-vote (ECGFounder MLP + linear)", 0.541, 0.502, 0.580, "img"),
    ("Header + ECGFounder MLP fusion", 0.544, 0.505, 0.583, "hdr"),
    ("C1 vote ensemble (P3)", 0.547, 0.504, 0.590, "hdr"),
    ("IMG-C3 (image-only C3)", 0.559, 0.506, 0.613, "img"),
    ("C3 selected model (P5)", 0.561, 0.520, 0.603, "hdr"),
    ("HIER-ext two-stage", 0.570, 0.527, 0.612, "img"),
    ("ENH-vote (6 image-only members)", 0.569, 0.527, 0.610, "img"),
    ("ULT-stack (10 image-only members)", 0.571, 0.542, 0.600, "final"),
    ("Original: ResNet-18 page image (CV, rep. 0)", 0.471, None, None, "img"),
    ("Original: header + ResNet-18 fusion (CV, rep. 0)", 0.495, None, None, "hdr"),
    ("Header numbers only, LR (P2)", 0.489, None, None, "hdr"),
    ("Header + clinical LR (P3)", 0.501, None, None, "hdr"),
    ("MLP + all engineered features (P5 E1)", 0.554, None, None, "hdr"),
    ("LightGBM (P5 E2)", 0.544, None, None, "hdr"),
]


def fig_models(extra=()):
    rows = sorted(list(MODELS) + list(extra), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.6, 0.26 * len(rows) + 0.9))
    for i, (n, a, lo, hi, g) in enumerate(rows):
        c = BLUE if g == "final" else (ORANGE if g == "hdr" else GRAY)
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([a], [i], "o", ms=6, color=c, mec="white", mew=1.2, zorder=3)
        ax.text(0.668, i, f"{a:.3f}", va="center", ha="right", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.axvline(0.459, color=INK2, lw=0.8, ls=":"); ax.text(0.459, len(rows) - 0.3, "majority class\n(always Normal) 0.459", fontsize=6.5, color=INK2, ha="center", va="bottom")
    ax.set_xlim(0.40, 0.67); ax.set_xlabel("Four-class accuracy, all 837 development ECGs (95% CI)")
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in ((BLUE, "final image-only model"), (GRAY, "image only"), (ORANGE, "uses printed header"))]
    ax.legend(handles=h, loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_model_landscape.pdf"); fig.savefig(FIG / "fig_model_landscape.png", dpi=200); plt.close(fig)


def fig_boundary():
    d = json.loads((R6B / "enh_exploratory.json").read_text())["error_analysis"]
    bins = ["<3", "3-6", "6-10", ">=10"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(bins)); w = 0.38
    for j, (name, c, lab) in enumerate((("C3 (with header)", ORANGE, "C3 (with header)"), ("ENH-stack", BLUE, "image-only ensemble"))):
        v = [d[name]["by_ef_distance_to_boundary"][b]["error_rate"] for b in bins]
        ax.bar(x + (j - 0.5) * w, v, w - 0.03, color=c, label=lab)
    n = [d["ENH-stack"]["by_ef_distance_to_boundary"][b]["n"] for b in bins]
    ax.set_xticks(x); ax.set_xticklabels([f"{b} points\n(n={k})" for b, k in zip(["< 3", "3-6", "6-10", "≥ 10"], n)], fontsize=8)
    ax.set_ylabel("Error rate"); ax.set_ylim(0, 0.8); ax.set_xlabel("Distance of echo EF from the nearest class boundary")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig_error_vs_boundary.pdf"); fig.savefig(FIG / "fig_error_vs_boundary.png", dpi=200); plt.close(fig)


def fig_cv_vs_test():
    rows = [("Triage: accuracy\non answered ECGs", (0.694, 0.655, 0.736), (0.747, 0.660, 0.837)),
            ("Triage: share\nof ECGs answered", (0.56, 0.48, 0.63), (0.497, 0.429, 0.567)),
            ("Four-class accuracy,\nall ECGs (IMG-C3)", (0.558, 0.504, 0.611), (0.553, 0.485, 0.621))]
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for i, (lab, cv, te) in enumerate(rows):
        for dy, (v, lo, hi), c in ((0.15, cv, BLUE), (-0.15, te, ORANGE)):
            ax.plot([lo, hi], [i + dy] * 2, color=c, lw=2, solid_capstyle="round"); ax.plot([v], [i + dy], "o", color=c, ms=6, mec="white", mew=1.2)
            ax.text(hi + 0.01, i + dy, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.set_xlim(0.4, 0.92)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=BLUE, label="Cross-validation (837 development ECGs)"),
                       plt.Line2D([], [], marker="o", color=ORANGE, label="Locked test, scored once (199 ECGs)")], fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2)
    ax.set_xlabel("Proportion (95% CI)")
    fig.tight_layout(); fig.savefig(FIG / "fig_cv_vs_locked_test.pdf"); fig.savefig(FIG / "fig_cv_vs_locked_test.png", dpi=200); plt.close(fig)


def fig_locked_confusion():
    M = np.array([[76, 11, 6, 2], [13, 13, 13, 2], [13, 13, 15, 2], [6, 4, 4, 6]])
    P = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.imshow(P, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#1c5cab"]), vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]}\n{P[i, j]:.0%}", ha="center", va="center", fontsize=7.5, color="white" if P[i, j] > 0.55 else INK)
    cl = ["Normal", "Mild", "Moderate", "Severe"]
    ax.set_xticks(range(4)); ax.set_xticklabels(cl, fontsize=7.5); ax.set_yticks(range(4)); ax.set_yticklabels([f"{c} ({n})" for c, n in zip(cl, M.sum(1))], fontsize=7.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (n)"); ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "fig_locked_confusion.pdf"); fig.savefig(FIG / "fig_locked_confusion.png", dpi=200); plt.close(fig)


def fig_bands():
    d = json.loads((R6B / "bands_exploratory.json").read_text())["one_dimensional"]["c_Mi (Mild/Moderate)"]
    x = [r["c_Mi"] for r in d]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for key, c, lab in (("accuracy", BLUE, "accuracy"), ("recall_Mild", ORANGE, "Mild detected (recall)"), ("recall_Moderate", AQUA, "Moderate detected (recall)")):
        ax.plot(x, [r[key] for r in d], "-o", color=c, lw=2, ms=4, label=lab)
    ax.axvline(41, color=INK2, lw=0.8, ls=":"); ax.text(41.15, 0.05, "original\nlimit 41", fontsize=6.5, color=INK2)
    ax.set_xlabel("Mild / Moderate EF limit (%)"); ax.set_ylabel("Proportion"); ax.set_ylim(0, 0.7)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "fig_band_sweep.pdf"); fig.savefig(FIG / "fig_band_sweep.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_models(); fig_boundary(); fig_cv_vs_test(); fig_locked_confusion(); fig_bands()
    print("figures written to", FIG)
