"""Phase 4C: experimental cohorts + audit plots (no identifiers or dates in any output except the git-ignored private table).

EF target for every ECG = EF of the NEAREST-IN-TIME linked echo (chosen without looking at the folder label, so the
folder label cannot leak into the regression target); ties at the same distance -> mean EF (flagged).
Cohorts (flags; all ECGs keep their original phase-2 patient-grouped folds, so cohorts are nested subsets of D):
  A_all       every ECG (folder label + nearest-link EF)
  B_7d .. F_90d  nearest link within 7 / 14 / 30 / 60 / 90 days
  G_hc        high confidence: nearest link has exact name score 100, |interval| <= 30 d, sex agrees where recorded,
              and no other link of a different EF class within 30 days of the ECG
    python -m research_experiments.phase_4.cohorts
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_experiments.phase_4.audit_matching import CLASSES, classify  # noqa: E402

P4 = Path(__file__).resolve().parent
PRIV = P4 / "data" / "private"


def build():
    L = pd.read_csv(PRIV / "ecg_echo_links.csv")
    hdr = pd.read_csv(P4.parents[1] / "data" / "header_features.csv")[["ecg_id", "male"]]
    L["mind"] = L.groupby("ecg_id").abs_days.transform("min")
    near = L[L.abs_days == L.mind]
    agg = near.groupby("ecg_id").agg(ef=("ef", "mean"), n_tied=("ef", "size"), abs_days=("abs_days", "first"),
                                     interval_days=("interval_days", "first"), score=("score", "min"),
                                     sex_ecg=("sex_ecg", "first"), sex_echo=("sex_echo", "first"))
    within30 = L[L.abs_days <= 30].assign(cls=lambda d: [classify(e, s) for e, s in zip(d.ef, d.sex_ecg)])
    conflict30 = within30.groupby("ecg_id").cls.nunique() > 1
    meta = L.groupby("ecg_id").agg(label=("label", "first"), patient=("patient", "first"), split=("split", "first"),
                                   year=("year", "first"), n_links=("ef", "size"),
                                   ef_spread_all_links=("ef", lambda s: s.max() - s.min()))
    C = meta.join(agg).join(conflict30.rename("conflict_30d")).reset_index().merge(hdr, on="ecg_id", how="left")
    C["conflict_30d"] = C.conflict_30d.fillna(False).astype(bool)
    C["derived"] = [classify(e, s) for e, s in zip(C.ef, C.sex_ecg)]
    sex_ok = C.sex_echo.isna() | (C.sex_echo == C.sex_ecg)
    for name, d in (("B_7d", 7), ("C_14d", 14), ("D_30d", 30), ("E_60d", 60), ("F_90d", 90)):
        C[name] = C.abs_days <= d
    C["A_all"] = True
    C["G_hc"] = (C.score >= 100) & (C.abs_days <= 30) & sex_ok & ~C.conflict_30d
    C["label_idx"] = C.label.map({c: i for i, c in enumerate(CLASSES)})
    C["derived_idx"] = C.derived.map({c: i for i, c in enumerate(CLASSES)})
    C.to_csv(PRIV / "cohorts.csv", index=False)
    return C


def summarise(C):
    rows = []
    for co in ["A_all", "B_7d", "C_14d", "D_30d", "E_60d", "F_90d", "G_hc"]:
        for part in ("development", "locked test"):
            s = C[C[co] & ((C.split != "test") if part == "development" else (C.split == "test"))]
            rows.append(dict(cohort=co, partition=part, ecgs=len(s), patients=s.patient.nunique(),
                             **{f"n_{c}": int((s.label == c).sum()) for c in CLASSES},
                             ef_mean=round(s.ef.mean(), 1), ef_sd=round(s.ef.std(), 1),
                             folder_vs_derived_agreement=round(float(np.mean(s.label == s.derived)), 3),
                             excluded_vs_A=int(((C.split != "test") if part == "development" else (C.split == "test")).sum() - len(s))))
    S = pd.DataFrame(rows)
    S.to_csv(P4 / "results" / "cohort_summary.csv", index=False)
    return S


def plots(C):
    P = P4 / "plots"; P.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.arange(0, 95, 2.5)
    for c, col in zip(CLASSES, ["#4c78a8", "#f58518", "#54a24b", "#e45756"]):
        ax[0].hist(C[C.label == c].ef, bins=bins, alpha=0.55, label=f"folder {c}", color=col)
    for x in (31, 41, 52, 54):
        ax[0].axvline(x, ls=":", c="k", lw=0.8)
    ax[0].set_xlabel("EF (Teichholz, %) of nearest-in-time linked echo"); ax[0].set_ylabel("ECGs"); ax[0].legend(fontsize=7)
    ax[0].set_title("EF distribution by folder class (dotted: 31 / 41 / 52 (M) / 54 (F))", fontsize=8)
    ax[1].hist(C.interval_days.clip(-200, 200), bins=np.arange(-200, 205, 5), color="#4c78a8")
    ax[1].set_xlabel("echo date - ECG date (days; clipped at +-200)"); ax[1].set_ylabel("ECGs")
    ax[1].set_title("ECG-to-echo interval (nearest link)", fontsize=8)
    fig.tight_layout(); fig.savefig(P / "ef_distribution_and_intervals.png", dpi=130); plt.close(fig)
    ct = pd.crosstab(C.label, C.derived).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.6, 4)); fr = ct.values / ct.values.sum(1, keepdims=True)
    ax.imshow(fr, cmap="Blues", vmin=0, vmax=1); ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("class from nearest-echo EF"); ax.set_ylabel("folder label")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{ct.values[i, j]}", ha="center", va="center", fontsize=8, color="white" if fr[i, j] > .55 else "black")
    ax.set_title("Label disagreement (1,047 ECGs)", fontsize=9); fig.tight_layout(); fig.savefig(P / "label_disagreement.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    multi = C[C.n_links > 1]
    ax.hist(multi.ef_spread_all_links, bins=30, color="#e45756"); ax.set_xlabel("max - min EF across linked echoes (points)")
    ax.set_ylabel("ECGs with >1 linked echo"); ax.set_title(f"EF spread across matched echoes (n = {len(multi)})", fontsize=9)
    fig.tight_layout(); fig.savefig(P / "ef_spread_multiple_links.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    C = build(); S = summarise(C); plots(C)
    pd.set_option("display.width", 220)
    print(S.to_string(index=False))
    mild = C[C.label == "Mild"]
    print("\nMild folder ECGs: EF quantiles", mild.ef.quantile([0, .1, .25, .5, .75, .9, 1]).round(1).to_dict())
    print("Mild within 2 points of 41:", int((mild.ef - 41).abs().le(2).sum()), "| within 2 of upper boundary (52 M / 54 F):",
          int(((mild.sex_ecg == "M") & (mild.ef - 52).abs().le(2) | (mild.sex_ecg == "F") & (mild.ef - 54).abs().le(2)).sum()))
