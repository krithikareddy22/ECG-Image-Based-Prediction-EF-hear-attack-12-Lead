"""Confusion-matrix figures (aggregates only): cells = count and row %, colour = row % (single-hue blue ramp),
recall on the right, precision underneath; triage figures add a grey 'Referred to echo' column so every patient
in the evaluated set is accounted for."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

for f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic"):
    font_manager.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf")
plt.rcParams.update({"font.family": "Latin Modern Roman", "font.size": 10})
INK, MUTED, GRID = "#1f2328", "#5b6470", "#d8dde3"
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CL = ["Normal", "Mild", "Moderate", "Severe"]


def draw(M, out, title, subtitle, referred=None, fmt="{:.0f}", note=None):
    M = np.asarray(M, float)
    n = M.sum(1) + (np.asarray(referred, float) if referred is not None else 0)
    answered = M.sum(1)
    P = np.divide(M, answered[:, None], out=np.zeros_like(M), where=answered[:, None] > 0)
    cols = 5 if referred is not None else 4
    import textwrap
    W = 6.9 if referred is not None else 6.0
    sub = "\n".join(textwrap.fill(par, width=int(W * 15.5)) for par in subtitle.split("\n"))
    nl = sub.count("\n") + 1
    H = 5.0 + 0.2 * nl + (0.3 if note else 0)
    fig, ax = plt.subplots(figsize=(W, H))
    ax.imshow(P, cmap=CMAP, vmin=0, vmax=1, extent=(-0.5, 3.5, 3.5, -0.5))
    for i in range(4):
        for j in range(4):
            dark = P[i, j] > 0.5
            ax.text(j, i - 0.12, fmt.format(M[i, j]), ha="center", va="center", fontsize=13,
                    fontweight="bold" if i == j else "normal", color="white" if dark else INK)
            ax.text(j, i + 0.2, f"{P[i, j]:.0%}", ha="center", va="center", fontsize=8.5, color="white" if dark else MUTED)
    # cell gaps (2 px surface lines)
    for k in range(5):
        ax.axhline(k - 0.5, color="white", lw=2); ax.axvline(k - 0.5, color="white", lw=2)
    if referred is not None:                                    # referred column, outside the colour scale
        for i in range(4):
            ax.add_patch(plt.Rectangle((3.62, i - 0.46), 0.92, 0.92, facecolor="#eef0f2", edgecolor="none"))
            ax.text(4.08, i - 0.12, fmt.format(referred[i]), ha="center", va="center", fontsize=12, color=MUTED)
            ax.text(4.08, i + 0.2, f"{referred[i] / n[i]:.0%} of row", ha="center", va="center", fontsize=8, color=MUTED)
    rec = np.divide(np.diag(M), answered, out=np.zeros(4), where=answered > 0)
    colsum = M.sum(0)
    prec = np.divide(np.diag(M), colsum, out=np.full(4, np.nan), where=colsum > 0)
    xr = 4.75 if referred is not None else 3.75
    ax.text(xr, -0.85, "Recall", ha="left", va="center", fontsize=9, color=MUTED, fontstyle="italic")
    for i in range(4):
        ax.text(xr, i, f"{rec[i]:.0%}", ha="left", va="center", fontsize=10.5, color=INK)
    ax.text(-0.5, 4.05, "Precision", ha="right", va="center", fontsize=9, color=MUTED, fontstyle="italic")
    for j in range(4):
        ax.text(j, 4.05, "–" if np.isnan(prec[j]) else f"{prec[j]:.0%}", ha="center", va="center", fontsize=10.5, color=INK)
    ax.set_xticks(range(cols)); ax.set_xticklabels(CL + (["Referred\nto echo"] if referred is not None else []), fontsize=10)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{c}  (n = {fmt.format(k)})" for c, k in zip(CL, n)], fontsize=10)
    ax.set_xlabel("Predicted class", fontsize=10, color=MUTED, labelpad=8); ax.set_ylabel("True class (echo)", fontsize=10, color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, (4.6 if referred is not None else 3.5) + 0.6); ax.set_ylim(4.35, -0.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.02, 1 - 0.55 / H, sub, ha="left", va="top", fontsize=9.5, color=MUTED, linespacing=1.35)
    if note:
        fig.text(0.02, 0.012, note, ha="left", va="bottom", fontsize=8.5, color=MUTED, fontstyle="italic")
    top = 1 - (0.62 + 0.2 * nl + (0.95 if referred is not None else 0.75)) / H
    fig.subplots_adjust(left=0.25, right=0.93, top=top, bottom=(0.3 if note else 0.05) / H + 0.02)
    fig.savefig(out + ".png", dpi=220, facecolor="white"); fig.savefig(out + ".pdf", facecolor="white")
    plt.close(fig)
    print(out, "accuracy (answered)", round(np.trace(M) / M.sum(), 3), "answered", M.sum(), "total", n.sum())
