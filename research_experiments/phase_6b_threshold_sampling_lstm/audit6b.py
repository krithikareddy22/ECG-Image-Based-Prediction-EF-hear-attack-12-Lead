"""Phase 6B step 1 (PHASE6B_PLAN.md section 1): data / label audit compiled from result files and the private label
tables - no model is trained. Development ECGs only (837 / 794 keys); the locked test is not loaded.
Outputs (aggregates only; counts 1-4 suppressed as "<5"): PHASE6B_AUDIT.md, results/audit6b.json.
Sources: phase_6a p6acommon.label_table (Phase 4 private cohorts / links), p6bcommon.table (T50 / T55), the matching
sheets in phase_4/data/private/ecg_echo (true acquisition dates for intervals, via phase_6a data_check.load), the 6A
matching check (results/matching_data_check.json) and the Phase 5 selected model C3 (regenerated baseline predictions)
for the descriptive "genuinely difficult" category.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.audit6b
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_experiments.phase_6.p6common import RUNS6
from research_experiments.phase_6a_normal_mild_lstm import data_check as DC
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import RES as RES6A
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import _links_dev, a270_split
from research_experiments.phase_6b_threshold_sampling_lstm.p6bcommon import (CLASSES, IDS, P6B, PAT, RES, SIZES, Y, guard,
                                                                             load_oof_csv, split, split_indices, table)

PRIV4 = DC.PRIV4 / "ecg_echo"
BAND = 3.0          # borderline: nearest-echo EF within 3 points of the sex-specific Normal/Mild boundary
MEAS = 2.0          # measurement band (potentially unreliable EF)
SPREAD = 10.0       # EF spread across links (potentially unreliable EF)


def sup(n):
    n = int(n)
    return "<5" if 0 < n < 5 else n


def rng(v):
    v = [int(x) for x in v]
    return f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}"


def true_interval():
    """|ECG - nearest echo| in days with the TRUE acquisition date (file-name timestamp; C6A-1), per development ECG."""
    rows = [DC.load(PRIV4, y)[1] for y in DC.YEARS]
    Lk = pd.concat(rows, ignore_index=True)
    ts = pd.Series(IDS).str.split("_").str[1]
    Lk = Lk[Lk.ts.isin(set(ts))].merge(pd.DataFrame(dict(ts=ts, ecg_id=IDS)), on="ts")
    Lk = Lk[Lk.ef_raw.notna()].copy()
    Lk["d"] = (Lk.echo_date.dt.normalize() - pd.to_datetime(Lk.ts.str[:8], format="%Y%m%d")).dt.days.abs()
    return Lk.groupby("ecg_id").d.min().reindex(IDS).values


def c3_wrong_all_repeats():
    W = []
    for r in (0, 1, 2):
        _, pred = load_oof_csv(RUNS6 / f"P5D-E5-vote-prior-protected-v1__s0_r{r}" / "oof.csv")
        W.append(pred != Y)
    return np.all(W, 0)


def main():
    T = table()
    lab, male, ef = T.label.values, T.male.values, T.ef.values
    bnd = np.where(male, 52.0, 54.0)
    out, md = {}, []
    # ---- 1. thresholds
    out["thresholds"] = dict(
        ORIG="Normal: male EF > 52, female EF >= 54 | Mild: 41 <= EF <= 52 (M) / 41 <= EF < 54 (F) | Moderate: 31 <= EF < 41 | Severe: EF < 31",
        borderline_assignment="male 52.0 -> Mild, 52.1 -> Normal; female 53.9 -> Mild, 54.0 -> Normal; 41.0 -> Mild, 40.9 -> Moderate; 31.0 -> Moderate, 30.9 -> Severe",
        T50="sex-independent Normal >= 50 (consistently linked folder-Mild ECGs with nearest-echo EF >= 50 -> Normal)",
        T55="sex-independent Normal >= 55 (consistently linked folder-Normal ECGs with nearest-echo EF < 55 -> Mild)",
        source="classify_lv in the original matching notebook (code cells only) = ecg_ef/config.py; Teichholz EF")
    # ---- 2. classes
    cls = {c: dict(ecgs=int((Y == i).sum()), patient_keys=int(pd.unique(PAT[Y == i]).size)) for i, c in enumerate(CLASSES)}
    out["classes"] = cls
    out["definitions"] = {d: {c: int((T[f"y_{d.lower()}"].values == i).sum()) for i, c in enumerate(CLASSES)} for d in ("T50", "T55")}
    out["definitions"]["moved"] = dict(T50_Mild_to_Normal=int(T.b2_move.sum()), T55_Normal_to_Mild=int(T.t55_move.sum()))
    # ---- 3. folds (patient-grouped StratifiedGroupKFold(5) x 3 repeats)
    fold = {role: {c: [] for c in CLASSES} for role in ("inner_train", "inner_val", "outer_test")}
    for r in (0, 1, 2):
        for k in range(5):
            for role, idx in zip(fold, split_indices(r, k)):
                for i, c in enumerate(CLASSES):
                    fold[role][c].append(int((Y[idx] == i).sum()))
    out["folds"] = {role: {c: rng(v) for c, v in d.items()} for role, d in fold.items()}
    cfg = {}
    for name in ("ORIG", "XUNC", "N300", "N270", "N240", "N200"):
        n_it, n_out, rem = [], [], []
        for r in (0, 1, 2):
            for k in range(5):
                itr, iva, _ = split(name, r, k); o_itr, _, _ = split_indices(r, k)
                n_it.append(int((Y[itr] == 0).sum())); n_out.append(int((Y[itr] == 0).sum() + (Y[iva] == 0).sum()))
                rem.append(int(len(o_itr) - len(itr)))
        cfg[name] = dict(normal_inner_train=rng(n_it), normal_outer_training=rng(n_out), removed_training_ecgs=rng(rem))
    n_it, n_va = [], []
    for r in (0, 1, 2):
        for k in range(5):
            _, _, _, info = a270_split(r, k); n_it.append(info["normal_inner_train"]); n_va.append(info["normal_inner_val"])
    cfg["A270X"] = dict(normal_inner_train=rng(n_it), normal_inner_val=rng(n_va), note="Normal patients moved from inner-val")
    out["training_configurations"] = cfg
    # ---- 4. patient grouping
    per = pd.Series(PAT).value_counts()
    multi_cls = pd.DataFrame(dict(p=PAT, y=Y)).groupby("p").y.nunique()
    out["patient_grouping"] = dict(patient_keys=int(per.size), ecgs=len(IDS), keys_with_1_ecg=int((per == 1).sum()),
                                   keys_with_2_ecgs=sup((per == 2).sum()), keys_with_3plus_ecgs=sup((per >= 3).sum()),
                                   keys_with_ecgs_in_more_than_one_class=sup((multi_cls > 1).sum()),
                                   key="pseudonymous name-derived hash (data/manifest.csv:patient); no patient ID exists",
                                   folds="patient-disjoint between inner-train, inner-val and outer test (asserted by phase_6 tests)")
    # ---- 5. matching quality (6A independent check)
    mc = json.loads((RES6A / "matching_data_check.json").read_text())
    sw = mc["finding_2021_ecg_date_swap"]
    out["matching_quality"] = dict(
        sheets_identical_to_phase4_copies=all(v["identical_to_phase4_private_copy"] for v in mc["files"].values()),
        dev_ecgs_with_a_link=mc["linking"]["with_at_least_one_link"], links_per_ecg=mc["linking"]["links_per_ecg"],
        linked_rows=mc["data_quality_linked_rows"]["linked_rows"], links_without_ef=mc["data_quality_linked_rows"]["ef_missing"],
        exact_duplicate_rows=mc["data_quality_linked_rows"]["exact_duplicate_rows"],
        ef_range=[mc["data_quality_linked_rows"]["ef_min"], mc["data_quality_linked_rows"]["ef_max"]],
        rule_vs_sheet_LV_agreement=mc["rule_vs_sheet_LV_column_agreement"], rule_vs_sheet_LV_disagreements=mc["rule_vs_sheet_LV_disagreements"],
        folder_vs_nearest_echo_class_agreement=mc["folder_vs_nearest_echo_class"]["agreement"],
        date_swap_2021=dict(rows_with_wrong_sheet_ecg_date=sw["linked_rows_with_wrong_sheet_ecg_date"],
                            explained_by_day_month_swap=sw["explained_by_day_month_swap"],
                            dev_ecgs_with_changed_interval=sw["dev_ecgs_with_changed_interval"],
                            dev_ecgs_with_changed_nearest_ef=sup(sw["dev_ecgs_with_changed_nearest_ef"]),
                            max_abs_change_nearest_ef=sw["max_abs_change_nearest_ef"],
                            changed_nearest_class=sup(sw["dev_ecgs_with_changed_nearest_class"]),
                            interval_counts_sheet_vs_true=sw["interval_counts_sheet_vs_true"],
                            labels_and_configurations_affected=False))
    # ---- 6. repeated patients and multiple links
    L = _links_dev()
    out["multiple_links"] = dict(
        ecgs_with_more_than_one_link=int((T.n_links.values > 1).sum()),
        ef_spread_gt_10_points=sup((T.ef_spread.values > SPREAD).sum()),
        ecgs_with_a_link_of_another_class=int(L.assign(lab=L.ecg_id.map(pd.Series(lab, index=IDS)))
                                              .query("derived != 'Unknown' and derived != lab").ecg_id.nunique()),
        repeated_patient_ecgs=int((pd.Series(PAT).map(per).values > 1).sum()))
    # ---- 7. borderline folder-Mild ECGs: matching uncertainty > unreliable EF > high confidence; difficult (descriptive)
    days_true = true_interval()
    other = set(L.assign(lab=L.ecg_id.map(pd.Series(lab, index=IDS))).query("derived != 'Unknown' and derived != lab").ecg_id)
    border = (lab == "Mild") & (np.abs(ef - bnd) <= BAND)
    match_unc = border & ((T.score.values < 100) | (days_true > 30) | np.isin(IDS, list(other)))
    unrel = border & ~match_unc & ((np.abs(ef - bnd) <= MEAS) | (T.ef_spread.values > SPREAD))
    hc = border & ~match_unc & ~unrel
    hard = hc & c3_wrong_all_repeats()
    def cnt(m):
        return dict(ecgs=sup(m.sum()), patient_keys=sup(pd.unique(PAT[m]).size))
    out["borderline_mild"] = dict(
        definition=f"folder label Mild and nearest-echo EF within {BAND:g} points of the sex-specific Normal/Mild boundary (52 M / 54 F)",
        all=cnt(border), above_boundary=cnt(border & (ef > bnd)), at_or_below_boundary=cnt(border & (ef <= bnd)),
        **{"1_matching_uncertainty": cnt(match_unc), "2_potentially_unreliable_ef": cnt(unrel), "3_high_confidence": cnt(hc),
           "4_genuinely_difficult_of_3": cnt(hard)},
        matching_uncertainty_reasons=dict(name_score_lt_100=sup((border & (T.score.values < 100)).sum()),
                                          interval_gt_30d_true_dates=sup((border & (days_true > 30)).sum()),
                                          linked_echo_of_another_class=sup((border & np.isin(IDS, list(other))).sum())),
        in_T50_movers=sup((border & T.b2_move.values).sum()), in_XUNC=sup((border & T.b3_uncertain.values).sum()),
        mild_with_nearest_ef_above_boundary=dict(all=sup(((lab == "Mild") & (ef > bnd)).sum()),
                                                 more_than_3_points_above=sup(((lab == "Mild") & (ef > bnd + BAND)).sum()),
                                                 nearest_echo_class_normal=sup(((lab == "Mild") & (T.derived.values == "Normal")).sum())),
        rules="precedence 1 > 2 > 3; 4 = category-3 ECGs misclassified by C3 (Phase 5 selected) in all three repeats "
              "(descriptive only, never used for training or evaluation choices); no EF is verified (name matching only)")
    guard(RES).mkdir(parents=True, exist_ok=True)
    (RES / "audit6b.json").write_text(json.dumps(out, indent=1, default=str))
    write_md(out)
    print(json.dumps(out["borderline_mild"], indent=1, default=str))


def write_md(o):
    b, mq, sw = o["borderline_mild"], o["matching_quality"], o["matching_quality"]["date_swap_2021"]
    lines = [
        "# Phase 6B audit (step 1; PHASE6B_PLAN.md section 1)",
        "",
        "Development set only (837 ECGs / 794 patient keys); the 210-ECG locked test is not loaded. Aggregates only; counts 1-4",
        "are suppressed as \"<5\". Generated by `audit6b.py` (`results/audit6b.json`). No model is trained here.",
        "",
        "## 1. Exact thresholds (four classes)",
        "| Class | Male | Female |", "|---|---|---|",
        "| Normal | EF > 52 | EF >= 54 |", "| Mild | 41 <= EF <= 52 | 41 <= EF < 54 |",
        "| Moderate | 31 <= EF < 41 | 31 <= EF < 41 |", "| Severe | EF < 31 | EF < 31 |",
        "",
        f"Borderline assignment: {o['thresholds']['borderline_assignment']}. Source: {o['thresholds']['source']}.",
        f"Alternative definitions (sensitivity analyses only): **T50** {o['thresholds']['T50']}; **T55** {o['thresholds']['T55']}.",
        "",
        "## 2. EF provenance",
        "EF is Teichholz (M-mode) from the echo report, linked to each ECG by **fuzzy name matching** (no patient identifier).",
        "Folder labels came from name matches without a date window; the continuous EF used for T50 / T55 / XUNC is the",
        "label-agnostic **nearest-in-time linked echo** (Phase 4). **No EF value is verified.** Guideline thresholds refer to",
        "biplane (Simpson) EF, so no threshold here is claimed to be clinically validated for this dataset.",
        "",
        "## 3. ECGs and patient keys per class",
        "| | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * 4,
        "| ECGs (ORIG) | " + " | ".join(str(o["classes"][c]["ecgs"]) for c in CLASSES) + " |",
        "| Patient keys | " + " | ".join(str(o["classes"][c]["patient_keys"]) for c in CLASSES) + " |",
        "| ECGs under T50 | " + " | ".join(str(o["definitions"]["T50"][c]) for c in CLASSES) + " |",
        "| ECGs under T55 | " + " | ".join(str(o["definitions"]["T55"][c]) for c in CLASSES) + " |",
        "",
        f"T50 moves {o['definitions']['moved']['T50_Mild_to_Normal']} Mild -> Normal; T55 moves "
        f"{o['definitions']['moved']['T55_Normal_to_Mild']} Normal -> Mild (consistently linked ECGs only).",
        "",
        "## 4. Fold distributions (range over 3 repeats x 5 outer folds)",
        "| Role | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * 4,
    ]
    for role, d in o["folds"].items():
        lines.append(f"| {role} | " + " | ".join(d[c] for c in CLASSES) + " |")
    lines += ["", "Training configurations (Normal counts; inner-val and outer test unchanged except A270X):", "",
              "| Config | Normal in inner-train | Normal in outer-training portion | Training ECGs removed |", "|---|---|---|---|"]
    for k, v in o["training_configurations"].items():
        lines.append(f"| {k} | {v['normal_inner_train']} | {v.get('normal_outer_training', '-')} | {v.get('removed_training_ecgs', '-')} |")
    pg = o["patient_grouping"]
    lines += ["", "A270X moves Normal patients from inner-val (inner-val Normal " + o["training_configurations"]["A270X"]["normal_inner_val"] + ").",
              "", "## 5. Patient grouping",
              f"{pg['patient_keys']} patient keys ({pg['key']}); {pg['keys_with_1_ecg']} keys with one ECG, {pg['keys_with_2_ecgs']} with two, "
              f"{pg['keys_with_3plus_ecgs']} with three or more; {pg['keys_with_ecgs_in_more_than_one_class']} keys have ECGs in more than one "
              f"class. Folds are {pg['folds']}.",
              "", "## 6. Matching quality (owner-supplied sheets; Phase 6A independent check)",
              f"- Sheets identical to the copies used in Phases 4-6A: {mq['sheets_identical_to_phase4_copies']}; every development ECG has "
              f">= 1 link ({mq['dev_ecgs_with_a_link']} / 837); links per ECG: {mq['links_per_ecg']}.",
              f"- Linked rows {mq['linked_rows']}; without EF {mq['links_without_ef']}; exact duplicate rows {mq['exact_duplicate_rows']}; "
              f"EF range {mq['ef_range'][0]}-{mq['ef_range'][1]} %.",
              f"- Recovered rule vs the sheets' LV column: agreement {mq['rule_vs_sheet_LV_agreement']:.3f} ({mq['rule_vs_sheet_LV_disagreements']} "
              "disagreements, all but one within 1 EF point of a boundary).",
              f"- Folder label vs class of the nearest echo: agreement {mq['folder_vs_nearest_echo_class_agreement']:.3f}.",
              f"- **2021 ECG-date swap (C6A-1):** {sw['rows_with_wrong_sheet_ecg_date']} linked rows have a wrong sheet ECG date, "
              f"{sw['explained_by_day_month_swap']} explained by a day/month swap; {sw['dev_ecgs_with_changed_interval']} development ECGs "
              f"change interval, {sw['dev_ecgs_with_changed_nearest_ef']} change nearest EF (max {sw['max_abs_change_nearest_ef']:.1f} points), "
              f"{sw['changed_nearest_class']} change class. Same-day / <= 7 / <= 14 / <= 30 / <= 90 d, sheet vs true dates: "
              + ", ".join(f"{k}: {v[0]} vs {v[1]}" for k, v in sw["interval_counts_sheet_vs_true"].items())
              + ". Labels, EF targets and the T50 / XUNC configurations are unaffected; intervals below use true dates.",
              "", "## 7. Repeated patients and multiple links",
              f"- ECGs with more than one linked echo: {o['multiple_links']['ecgs_with_more_than_one_link']}; EF spread > 10 points across links: "
              f"{o['multiple_links']['ef_spread_gt_10_points']}; ECGs with a linked echo of another class than the folder label: "
              f"{o['multiple_links']['ecgs_with_a_link_of_another_class']}.",
              f"- ECGs of patient keys with more than one ECG: {o['multiple_links']['repeated_patient_ecgs']} (kept together in every fold).",
              "", "## 8. Borderline Mild ECGs", f"Definition: {b['definition']}.", "",
              "| Category (precedence 1 > 2 > 3) | ECGs | Patient keys |", "|---|---|---|",
              f"| All borderline Mild | {b['all']['ecgs']} | {b['all']['patient_keys']} |",
              f"| - EF above the boundary (would be Normal by EF) | {b['above_boundary']['ecgs']} | {b['above_boundary']['patient_keys']} |",
              f"| - EF at / below the boundary | {b['at_or_below_boundary']['ecgs']} | {b['at_or_below_boundary']['patient_keys']} |",
              f"| 1. Matching uncertainty (name score < 100, or interval > 30 d with true dates, or a linked echo of another class) | "
              f"{b['1_matching_uncertainty']['ecgs']} | {b['1_matching_uncertainty']['patient_keys']} |",
              f"| 2. Potentially unreliable EF (within +-2 points of the boundary, or EF spread > 10 points) | "
              f"{b['2_potentially_unreliable_ef']['ecgs']} | {b['2_potentially_unreliable_ef']['patient_keys']} |",
              f"| 3. High-confidence borderline (none of the above; best proxy for verified) | {b['3_high_confidence']['ecgs']} | "
              f"{b['3_high_confidence']['patient_keys']} |",
              f"| 4. Genuinely difficult (category 3, misclassified by C3 in all 3 repeats; descriptive) | "
              f"{b['4_genuinely_difficult_of_3']['ecgs']} | {b['4_genuinely_difficult_of_3']['patient_keys']} |",
              "", f"Matching-uncertainty reasons (overlapping): name score < 100: {b['matching_uncertainty_reasons']['name_score_lt_100']}; "
              f"interval > 30 d: {b['matching_uncertainty_reasons']['interval_gt_30d_true_dates']}; linked echo of another class: "
              f"{b['matching_uncertainty_reasons']['linked_echo_of_another_class']}. Borderline ECGs among the T50 movers: {b['in_T50_movers']}; "
              f"among the XUNC exclusions: {b['in_XUNC']}.", "",
              f"Folder-Mild ECGs whose nearest echo lies ABOVE the boundary: {b['mild_with_nearest_ef_above_boundary']['all']}, of which "
              f"{b['mild_with_nearest_ef_above_boundary']['more_than_3_points_above']} by more than 3 points (nearest-echo class Normal: "
              f"{b['mild_with_nearest_ef_above_boundary']['nearest_echo_class_normal']}). These are link inconsistencies (the folder label "
              "came from another name-matched echo), not borderline EF values, so none falls in the borderline band above.", "",
              "Category 4 is descriptive only: it never informs training, configurations or evaluation, and a model error is never",
              "treated as a label error.", ""]
    guard(P6B / "PHASE6B_AUDIT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
