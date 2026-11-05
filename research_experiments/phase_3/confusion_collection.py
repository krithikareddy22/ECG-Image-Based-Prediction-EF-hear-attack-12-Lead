"""Confusion-matrix collection for the leading models (development CV, summed over the available CV repeats;
seed-ensembled where seeds exist) + one PNG per model.
    python -m research_experiments.phase_3.confusion_collection
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from research_experiments.common import CLASSES  # noqa: E402
from research_experiments.phase_2.p2common import PCOLS  # noqa: E402
from research_experiments.phase_3.p3_robust import D, find, seeds_repeats  # noqa: E402
from research_experiments.phase_3.p3common import PLOT3, RES3  # noqa: E402

Y = D.label_idx.values
LEADING = {
    "Header + ResNet-18 (current default, R)": [["R_hdrall+img_wavg"]],
    "ECGFounder frozen (linear + distance penalty)": [["ecgf_f6_lin_cedist1"], ["ecgf_f6_lin_cedist1_r1"], ["ecgf_f6_lin_cedist1_r2"]],
    "Header numeric + clinical (LR)": [["clin_hdrnum+clin_skLR"], ["clin_hdrnum+clin_skLR_r1"], ["clin_hdrnum+clin_skLR_r2"]],
    "2-way MLP fusion (3-seed ens.)": [seeds_repeats("ff_hdrnum+ecgf_mlp", repeats=(r,)) for r in (0, 1, 2)],
    "3-way MLP fusion (3-seed ens.)": [seeds_repeats("clin_mlpfusion_hdr+clin+ecgf", repeats=(r,)) for r in (0, 1, 2)],
    "Vote ensemble (2-way MLP + ECGFounder lin + header+clin LR)": [["ens_mlp2+ecgflin+hdrclin__vote"], ["ens_mlp2+ecgflin+hdrclin__vote_r1"], ["ens_mlp2+ecgflin+hdrclin__vote_r2"]],
}


def main():
    PLOT3.mkdir(exist_ok=True); md = ["# Confusion matrices (development CV; rows = true, columns = predicted; summed over CV repeats)\n"]
    for name, per_repeat in LEADING.items():
        cm = 0; n_rep = 0
        for tags in per_repeat:
            Ps = [pd.read_csv(find(t) / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS].values for t in tags if find(t)]
            if not Ps:
                continue
            cm = cm + confusion_matrix(Y, np.mean(Ps, 0).argmax(1), labels=range(4)); n_rep += 1
        if not n_rep:
            continue
        frac = cm / cm.sum(1, keepdims=True)
        md.append(f"**{name}** ({n_rep} repeat(s) x 837 ECGs)\n\n| true \\ pred | " + " | ".join(CLASSES) + " | recall |\n|---|---|---|---|---|---|")
        for i, c in enumerate(CLASSES):
            md.append(f"| {c} | " + " | ".join(f"{cm[i, j]} ({frac[i, j]:.0%})" for j in range(4)) + f" | {frac[i, i]:.3f} |")
        md.append("")
        fig, ax = plt.subplots(figsize=(4.6, 4.0)); ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES); ax.set_xlabel("Predicted"); ax.set_ylabel("True (echo)")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if frac[i, j] > 0.55 else "black")
        ax.set_title(name[:48], fontsize=8); fig.tight_layout()
        fig.savefig(PLOT3 / ("cm_" + "".join(ch if ch.isalnum() else "_" for ch in name)[:60] + ".png"), dpi=130); plt.close(fig)
    (RES3 / "confusion_matrices_leading.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
