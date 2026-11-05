"""Phase 6A sections 1-4 (analysis only, no model): label rule, cohort facts, borderline Mild audit, alternative
boundary counts, the B2 / B3 label configurations and the A270 training cohorts. Development ECGs only.
Committed outputs are aggregates; counts 1-4 are suppressed ("<5") in the borderline audit.
  results/label_configurations.json, results/boundary_sensitivity.csv, results/a270_cohorts.csv,
  plots/ef_boundary_sensitivity.png
Private (git-ignored): data/private/labels.csv (per-ECG configuration flags), its SHA-256 recorded in the json.
    python -m research_experiments.phase_6a_normal_mild_lstm.labels
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_experiments.phase_6a_normal_mild_lstm.p6acommon import (  # noqa: E402
    A_TARGET, B2_BOUNDARY, B3_BAND, CLASSES, PLOTS, PRIV, RES, Y, a270_split, guard, label_table, sha256_file, split_indices)


def sup(n):
    n = int(n)
    return "<5" if 0 < n < 5 else n


def main():
    T = label_table()
    guard(PRIV).mkdir(parents=True, exist_ok=True)
    T.drop(columns=["patient"]).to_csv(PRIV / "labels.csv")
    out = dict(rule="Normal M>52 / F>=54; Mild 41<=EF<=52 (M) / 41<=EF<54 (F); Moderate 31<=EF<41; Severe EF<31 (Teichholz)",
               labels_csv_sha256=sha256_file(PRIV / "labels.csv"))
    # ---- class / patient counts per class
    out["classes"] = {c: dict(ecgs=int((Y == i).sum()), patients=int(T[T.y == i].patient.nunique())) for i, c in enumerate(CLASSES)}
    # ---- borderline Mild audit
    mild = T[T.label == "Mild"]
    near = mild[(mild.derived == "Mild") & (mild.ef >= B2_BOUNDARY)]
    rep = T.groupby("patient").size()
    def audit(S):
        return dict(ecgs=sup(len(S)), patients=sup(S.patient.nunique()),
                    ef_quartiles=[round(float(q), 1) for q in np.nanpercentile(S.ef, [25, 50, 75])] if len(S) >= 5 else "suppressed",
                    name_score_100=sup((S.score >= 100).sum()), name_score_below_100=sup((S.score < 100).sum()),
                    same_day_echo=sup((S.abs_days == 0).sum()), echo_within_30d=sup((S.abs_days <= 30).sum()),
                    echo_over_30d=sup((S.abs_days > 30).sum()), single_link=sup((S.n_links == 1).sum()),
                    multiple_links=sup((S.n_links > 1).sum()), high_confidence_G=sup(S.G.sum()),
                    patient_key_with_repeated_ecgs=sup(S.patient.map(rep).gt(1).sum()), verified_link=0)
    out["borderline_audit"] = {
        "all_folder_Mild": audit(mild),
        "Mild_nearest_echo_outside_Mild_band": audit(mild[mild.derived != "Mild"]),
        f"Mild_consistent_EF_ge_{B2_BOUNDARY:g} (B2 movers)": audit(near),
        f"Mild_within_{B3_BAND:g}_of_sex_boundary": audit(mild[np.abs(mild.ef - mild.sex_boundary) <= B3_BAND]),
        "note": "EF values come from fuzzy NAME matches (no patient ID): none is verified (verified_link = 0)."}
    # ---- alternative boundaries (counts only; never model output)
    rows = []
    nm = T.label.isin(["Normal", "Mild"])
    for thr in np.arange(48, 55.5, 0.5):
        to_n = (T.label == "Mild") & (T.ef >= thr); to_m = (T.label == "Normal") & (T.ef < thr)
        rows.append(dict(rule=f"sex-independent EF >= {thr:g} is Normal", threshold=float(thr), sex_specific=False,
                         mild_to_normal_ecgs=int(to_n.sum()), mild_to_normal_patients=int(T[to_n].patient.nunique()),
                         mild_to_normal_consistent_ecgs=int((to_n & (T.derived == "Mild")).sum()),
                         normal_to_mild_ecgs=int(to_m.sum()), normal_to_mild_patients=int(T[to_m].patient.nunique())))
    for sh in (-4, -3, -2, -1, 0, 1, 2):
        b = T.sex_boundary + sh
        isn = np.where(T.male, T.ef > b, T.ef >= b)
        to_n = (T.label == "Mild") & isn; to_m = (T.label == "Normal") & ~isn & T.ef.notna()
        rows.append(dict(rule=f"original sex-specific boundary {sh:+d} points", threshold=float(sh), sex_specific=True,
                         mild_to_normal_ecgs=int(to_n.sum()), mild_to_normal_patients=int(T[to_n].patient.nunique()),
                         mild_to_normal_consistent_ecgs=int((to_n & (T.derived == "Mild")).sum()),
                         normal_to_mild_ecgs=int(to_m.sum()), normal_to_mild_patients=int(T[to_m].patient.nunique())))
    B = pd.DataFrame(rows)
    Bc = B.copy()
    for col in [c for c in Bc.columns if c.endswith(("_ecgs", "_patients"))]:
        Bc[col] = Bc[col].map(sup)                   # counts 1-4 suppressed in the committed table
    Bc.to_csv(guard(RES) / "boundary_sensitivity.csv", index=False)
    # ---- configurations
    out["B2"] = dict(boundary=f"sex-independent EF >= {B2_BOUNDARY:g} % is Normal (only clearly defined Mild movers)",
                     moved_ecgs=int(T.b2_move.sum()), moved_patients=int(T[T.b2_move].patient.nunique()),
                     class_counts_B2={c: int((T.y_b2 == i).sum()) for i, c in enumerate(CLASSES)})
    out["B3"] = dict(definition=f"Normal/Mild with nearest-echo EF within +-{B3_BAND:g} of the sex-specific boundary, or a conflicting Normal/Mild link",
                     uncertain_ecgs=int(T.b3_uncertain.sum()), uncertain_patients=int(T[T.b3_uncertain].patient.nunique()),
                     by_band=int(T.b3_band.sum()), by_conflicting_link=int(T.b3_conflict.sum()),
                     uncertain_by_class={c: int((T.b3_uncertain & (T.y == i)).sum()) for i, c in enumerate(CLASSES)},
                     removed_from_inner_train_per_fold="see a270 / b3 fold table")
    # ---- A270 cohorts and B3 fold sizes
    fr = []
    for r in (0, 1, 2):
        for k in range(5):
            itr, iva, ote = split_indices(r, k)
            _, _, _, info = a270_split(r, k)
            fr.append(dict(repeat=r, fold=k, inner_train=len(itr), inner_train_normal=int((Y[itr] == 0).sum()),
                           inner_val_normal=int((Y[iva] == 0).sum()), a270_inner_train_normal=info["normal_inner_train"],
                           a270_moved_ecgs=info["moved_ecgs"], a270_inner_val_normal=info["normal_inner_val"],
                           b3_removed_from_inner_train=int(label_table().b3_uncertain.values[itr].sum())))
    F = pd.DataFrame(fr); F.to_csv(RES / "a270_b3_fold_cohorts.csv", index=False)
    out["A270"] = dict(target=A_TARGET, all_folds_exactly_270=bool((F.a270_inner_train_normal == A_TARGET).all()),
                       moved_ecgs_per_fold=[int(F.a270_moved_ecgs.min()), int(F.a270_moved_ecgs.max())],
                       inner_val_normal_after=[int(F.a270_inner_val_normal.min()), int(F.a270_inner_val_normal.max())])
    (RES / "label_configurations.json").write_text(json.dumps(out, indent=1, default=str))
    # ---- plot: EF near the Normal/Mild boundary + affected counts
    guard(PLOTS).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.arange(40, 62.5, 1.0)                     # no clipping: ECGs outside 40-62 % are not drawn
    for lab, col in (("Normal", "#4C78A8"), ("Mild", "#F58518")):
        ax[0].hist(T[T.label == lab].ef, bins=bins, alpha=0.6, label=f"folder {lab}", color=col)
    for x, ls, lb in ((52, "--", "original M (>52)"), (54, ":", "original F (>=54)"), (50, "-", "B2: 50 %")):
        ax[0].axvline(x, color="k", ls=ls, lw=1, label=lb)
    ax[0].set_xlabel("nearest-echo EF (%)"); ax[0].set_ylabel("development ECGs"); ax[0].legend(fontsize=8)
    ax[0].set_title("Folder Normal / Mild ECGs by nearest-echo EF, 40-62 % shown")
    S = B[~B.sex_specific]
    ax[1].plot(S.threshold, S.mild_to_normal_ecgs, "o-", label="Mild -> Normal (all)")
    ax[1].plot(S.threshold, S.mild_to_normal_consistent_ecgs, "s--", label="Mild -> Normal (consistent link)")
    ax[1].plot(S.threshold, S.normal_to_mild_ecgs, "^-", label="Normal -> Mild")
    ax[1].axvline(B2_BOUNDARY, color="k", lw=1)
    ax[1].set_xlabel("sex-independent Normal/Mild boundary (EF %)"); ax[1].set_ylabel("ECGs relabelled")
    ax[1].set_title("ECGs affected by an alternative boundary (counts only)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "ef_boundary_sensitivity.png", dpi=150); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "borderline_audit"}, indent=1, default=str))
    print(B.to_string(index=False))


if __name__ == "__main__":
    main()
