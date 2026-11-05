"""LSTM architecture diagram (no data): plots/lstm_architecture.png
    python -m research_experiments.phase_6a_normal_mild_lstm.diagram
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from research_experiments.phase_6a_normal_mild_lstm.p6acommon import PLOTS, guard  # noqa: E402


def box(ax, x, y, w, h, text, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1))


def main():
    fig = plt.figure(figsize=(14, 7.2))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.62]); ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 1.4, 2.5, 1.8, "Input (100 Hz, 10 s = 1,000 steps)\n\naligned: 26 channels\n12 leads (own 2.5-s window,\nzero elsewhere) + rhythm II\n+ 13 recorded-sample masks\n\nrhythm (L0): lead II only, 1 ch", "#e8f1fa")
    box(ax, 3.1, 1.4, 2.3, 1.8, "CNN front-end\n(L5 CNN-LSTM only)\n3 x [Conv1d k7, stride 2, GELU]\n26 -> 32 -> 64 -> 64 ch\n1,000 -> 125 steps", "#f3f3f3")
    box(ax, 5.9, 1.4, 2.6, 1.8, "LSTM, hidden 64\nL0 / L1: 1 layer, uni\nL2: 2 layers, uni\nL3: 1 layer, bidirectional\nL4: 2 layers, bi, dropout 0.3\nL5: 1 layer, bi (after CNN)", "#fde9d9")
    box(ax, 9.0, 2.45, 2.1, 0.75, "last hidden state\n(all directions)", "#eaf5ea")
    box(ax, 9.0, 1.4, 2.1, 0.75, "mean of outputs over\nrecorded steps (mask)", "#eaf5ea")
    box(ax, 11.6, 1.4, 2.25, 1.8, "concat -> Dropout 0.3\n-> Linear -> 4 logits\n\nNormal / Mild /\nModerate / Severe", "#f5e9f5")
    for (x0, y0, x1, y1) in ((2.6, 2.3, 3.1, 2.3), (5.4, 2.3, 5.9, 2.3), (8.5, 2.8, 9.0, 2.8), (8.5, 1.8, 9.0, 1.8),
                             (11.1, 2.8, 11.6, 2.5), (11.1, 1.8, 11.6, 2.1)):
        arrow(ax, x0, y0, x1, y1)
    ax.text(0.1, 0.75, "Training: class-balanced cross-entropy, AdamW (lr 1e-3, weight decay 1e-2), batch 32, gradient clipping 1.0, "
            "<= 60 epochs, early stopping on inner-validation S3 (patience 12, min 10 epochs); seeds 42/43/44 x CV repeats 0-2.\n"
            "LSTM-sel: per outer fold, the variant L0-L4 with the best mean inner-validation S3 (outer fold never used). "
            "No ECGFounder, ResNet or header features are used by the LSTM.", fontsize=8.5, va="top")
    ax2 = fig.add_axes([0.06, 0.70, 0.88, 0.26]); ax2.set_xlim(0, 10); ax2.set_ylim(-0.5, 13.5); ax2.invert_yaxis()
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "rhythm II"]
    col = {n: c for c, grp in enumerate((["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"])) for n in grp}
    for i, n in enumerate(names):
        if n == "rhythm II":
            ax2.add_patch(Rectangle((0, i - 0.4), 10, 0.8, color="#4C78A8"))
        else:
            ax2.add_patch(Rectangle((2.5 * col[n], i - 0.4), 2.5, 0.8, color="#F58518"))
    ax2.set_yticks(range(13)); ax2.set_yticklabels(names, fontsize=7); ax2.set_xlabel("time (s)", fontsize=8)
    ax2.set_title("Recorded windows on the common 10-s axis (coloured = recorded, mask 1; white = not recorded, zero input, mask 0)", fontsize=9)
    guard(PLOTS).mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "lstm_architecture.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
