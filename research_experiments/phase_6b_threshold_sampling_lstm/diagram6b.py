"""Phase 6B architecture diagram (no data): LSTM-fixed / CNN-LSTM (L5), CNN-LSTM + header (L6), early fusion with
ECGFounder f6 (F1 / F2) and late fusion (F3 / F4). Output: plots/architecture_lstm_cnn_fusion.png
    python -m research_experiments.phase_6b_threshold_sampling_lstm.diagram6b
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research_experiments.phase_6a_normal_mild_lstm.diagram import arrow, box  # noqa: E402
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import CONF as CONF6A  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm.p6bcommon import PLOTS, guard  # noqa: E402


def main():
    f = CONF6A / "lstm_fixed.json"
    fx = json.loads(f.read_text())["lstm_fixed"] if f.exists() else "LSTM-fixed"
    fig, ax = plt.subplots(figsize=(15, 9.5)); ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis("off")
    # --- row A: encoders
    ax.text(0.1, 9.7, "A. Waveform encoders (input: 26 x 1,000 = 12 leads in their recorded 2.5-s windows + rhythm II + 13 masks, 100 Hz)",
            fontsize=10, weight="bold", va="top")
    box(ax, 0.2, 7.4, 2.4, 1.6, "Input\n26 ch x 1,000 steps\n(zeros + mask 0 where\na lead was not recorded)", "#e8f1fa")
    box(ax, 3.2, 8.3, 3.0, 0.9, f"LSTM-fixed = {fx}\n(variant chosen most often by the\nnested selection over L0-L4)", "#fde9d9", 8)
    box(ax, 3.2, 7.2, 3.0, 0.9, "L5 CNN-LSTM: 3 x [Conv1d k7 s2, GELU]\n26->32->64->64 ch, 1,000->125 steps\n-> BiLSTM, hidden 64", "#fde9d9", 8)
    box(ax, 6.8, 7.4, 2.8, 1.6, "pooled features\n[last hidden state ||\nmask-weighted mean of outputs]\n(256-d for a BiLSTM-64)", "#eaf5ea")
    box(ax, 10.2, 7.4, 2.5, 1.6, "6A head:\nDropout 0.3 -> Linear -> 4\n(LSTM-fixed, L5)", "#f5e9f5")
    for a in ((2.6, 8.2, 3.2, 8.7), (2.6, 8.2, 3.2, 7.65), (6.2, 8.75, 6.8, 8.3), (6.2, 7.65, 6.8, 8.1), (9.6, 8.2, 10.2, 8.2)):
        arrow(ax, *a)
    # --- row B: early fusion
    ax.text(0.1, 6.7, "B. Early fusion (trained jointly, end to end)", fontsize=10, weight="bold", va="top")
    box(ax, 0.2, 4.5, 2.4, 1.0, "pooled waveform features\n(L5 encoder for L6 / F2,\nLSTM-fixed encoder for F1)", "#eaf5ea", 8)
    box(ax, 0.2, 3.2, 2.4, 1.0, "header numeric (16):\nage, sex, 14 machine\nmeasurements", "#e8f1fa", 8)
    box(ax, 0.2, 1.9, 2.4, 1.0, "ECGFounder f6\n(frozen embedding;\nF1 / F2 only)", "#e8f1fa", 8)
    box(ax, 3.3, 3.2, 2.8, 2.3, "concatenate\nL6: header z-scored\n(inner-train statistics)\nF1 / F2: [header || f6]\nblock-standardised\n(Phase 2 recipe, inner-train)", "#f3f3f3", 8)
    box(ax, 6.8, 4.3, 3.0, 1.2, "L6 head:\nDropout 0.3 -> Linear -> 4", "#f5e9f5", 8)
    box(ax, 6.8, 2.6, 3.0, 1.4, "F1 / F2 body (Phase 2 MLP fusion):\nDropout 0.5 -> Linear 64 -> GELU\n-> Dropout 0.5 -> Linear -> 4", "#f5e9f5", 8)
    for a in ((2.6, 5.0, 3.3, 4.7), (2.6, 3.7, 3.3, 4.2), (2.6, 2.4, 3.3, 3.6), (6.1, 4.6, 6.8, 4.9), (6.1, 3.9, 6.8, 3.3)):
        arrow(ax, *a)
    # --- row B right: late fusion
    ax.text(10.4, 6.7, "C. Late fusion (no joint training)", fontsize=10, weight="bold", va="top")
    box(ax, 10.4, 4.6, 4.4, 1.6, "C1 members (probabilities):\nMLP fusion (header || f6; 3 seeds)\nECGFounder linear (distance-penalised CE)\nheader + clinical logistic regression", "#e8f1fa", 8)
    box(ax, 10.4, 3.5, 4.4, 0.8, "LSTM-sel (nested selection, 3 seeds)", "#fde9d9", 8)
    box(ax, 10.4, 1.7, 4.4, 1.4, "F3: equal-weight mean of the 4 -> argmax\nF4: F3 + nested class-prior rule\n(alpha on inner OOF incl. LSTM-fixed s42;\nprotected objective = C3 recipe)", "#f5e9f5", 8)
    arrow(ax, 12.6, 4.6, 12.6, 4.3); arrow(ax, 12.6, 3.5, 12.6, 3.1)
    ax.text(0.2, 1.2, "Training of every network: class-balanced cross-entropy, AdamW (lr 1e-3, weight decay 1e-2), batch 32, gradient clipping 1.0, "
            "<= 60 epochs, early stopping on the fold's inner-validation S3 (patience 12, min 10 epochs), best epoch restored;\n"
            "seeds 42/43/44 x CV repeats 0-2 (patient-grouped 5-fold); all standardisation statistics from the inner-training rows only; "
            "the outer test fold is predicted once. Development set only (locked test never used).", fontsize=8, va="top")
    guard(PLOTS).mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "architecture_lstm_cnn_fusion.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
