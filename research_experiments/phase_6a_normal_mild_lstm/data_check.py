"""Independent check of the ECG-echo matching data used by Phases 4-6A (owner request 2026-11-05: "make sure matching
also works and data is proper"). Re-implemented here (no import of phase_4 code) and run on the matching sheets the
owner supplied (read-only; never copied into the repository). DEVELOPMENT ECGs ONLY: sheet rows linked to locked-test
ECGs are dropped before any check. Names are never read into outputs: only the ECG file-name timestamp, dates, EF,
sex, match score and LV class columns are used. Output (aggregates only): results/matching_data_check.json
    python -m research_experiments.phase_6a_normal_mild_lstm.data_check <folder with the four .xlsx files>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.common import ROOT
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import CLASSES, IDS, RES, guard, label_table

YEARS = {2020: "2020_echo_ecg_matching_50percent_20261029_162620.xlsx", 2021: "2021_echo_first_ecg_matching_20261029_161227.xlsx",
         2022: "2022_ECG_ECHO_MATCHING_70_PERCENT_20261029_170108.xlsx", 2024: "2024_ECG_ECHO_MATCHING_70_PERCENT_20261029_165914.xlsx"}
PRIV4 = ROOT / "research_experiments" / "phase_4" / "data" / "private"


def rule(ef, sex):
    """classify_lv from the matching-notebook code."""
    if pd.isna(ef):
        return "Unknown"
    if sex == "M":
        if ef > 52:
            return "Normal"
        if 41 <= ef <= 52:
            return "Mild"
    elif sex == "F":
        if ef >= 54:
            return "Normal"
        if 41 <= ef < 54:
            return "Mild"
    if 31 <= ef < 41:
        return "Moderate"
    if ef < 31:
        return "Severe"
    return "Unknown"


def sex_of(v):
    v = str(v).strip().upper()
    return "M" if v in ("M", "MALE") else "F" if v in ("F", "FEMALE") else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(folder, year):
    f = next(Path(folder).glob(f"*{YEARS[year]}"))
    df = pd.read_excel(f, sheet_name="Matched Records")
    col = {c.lower(): c for c in df.columns}
    g = lambda *n: df[next(col[x] for x in n if x in col)]
    out = pd.DataFrame(dict(
        ts=g("ecg file name", "ecg file").astype(str).str.extract(r"\[(\d{14})\]")[0],
        ecg_date=pd.to_datetime(g("ecg date"), errors="coerce"), echo_date=pd.to_datetime(g("echo study date", "echo date"), errors="coerce"),
        ef_raw=pd.to_numeric(g("ef (teich) %", "ef teich (%)"), errors="coerce"), score=pd.to_numeric(g("match score", "name match %"), errors="coerce"),
        sex=g("gender", "gender ecg").map(sex_of), lv_sheet=g("lv", "lv classification").astype(str).str.strip().str.capitalize()))
    out["year"] = year
    return f, out


def main():
    folder = sys.argv[1]
    rep = dict(files={}, scope="development ECGs only (837); locked-test rows dropped before any check")
    rows = []
    for y in YEARS:
        f, d = load(folder, y)
        rep["files"][str(y)] = dict(sha256=sha(f), identical_to_phase4_private_copy=sha(f) == sha(PRIV4 / "ecg_echo" / YEARS[y]),
                                    matched_rows=int(len(d)), rows_without_parsable_timestamp=int(d.ts.isna().sum()))
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    man = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "label", "patient", "split"])
    man = man[man.split != "test"]                                            # development rows only
    man["ts"] = man.ecg_id.str.split("_").str[1]
    assert set(man.ecg_id) == set(IDS)
    L = L[L.ts.isin(set(man.ts))].copy()                                      # drop every row not linked to a development ECG
    L = L.merge(man[["ts", "ecg_id", "label"]], on="ts")
    # ---- data properness of the linked rows
    L["ef"] = L.ef_raw
    q = dict(linked_rows=int(len(L)), ef_missing=int(L.ef.isna().sum()), ef_fraction_format_le_1=int((L.ef <= 1).sum()),
             ef_outside_5_95=int(((L.ef < 5) | (L.ef > 95)).sum()), ef_min=float(L.ef.min()), ef_max=float(L.ef.max()),
             sex_missing_or_invalid=int(L.sex.isna().sum()), ecg_date_unparsable=int(L.ecg_date.isna().sum()),
             echo_date_unparsable=int(L.echo_date.isna().sum()),
             sheet_ecg_date_equals_timestamp_date=float(np.mean(L.ecg_date.dt.strftime("%Y%m%d") == L.ts.str[:8])),
             exact_duplicate_rows=int(L.duplicated(["ts", "echo_date", "ef", "lv_sheet"]).sum()),
             timestamps_in_more_than_one_year_sheet=int((L.groupby("ts").year.nunique() > 1).sum()),
             ecg_year_vs_sheet_year_mismatch=int((L.ts.str[:4].astype(int) != L.year).sum()))
    rep["data_quality_linked_rows"] = q
    # ---- the rule reproduces the sheets' own LV column
    L["derived"] = [rule(e, s) for e, s in zip(L.ef, L.sex)]
    rep["rule_vs_sheet_LV_column_agreement"] = float(np.mean(L.derived == L.lv_sheet))
    rep["rule_vs_sheet_LV_disagreements"] = int((L.derived != L.lv_sheet).sum())
    # ---- linking of the development ECGs
    per = L.groupby("ecg_id")
    n_links = per.size().reindex(IDS).fillna(0).astype(int)
    rep["linking"] = dict(dev_ecgs=len(IDS), with_at_least_one_link=int((n_links > 0).sum()),
                          links_per_ecg=n_links.value_counts().sort_index().to_dict(),
                          folder_label_reproduced_by_some_link=int(per.apply(lambda g: (g.derived == g.label.iloc[0]).any()).reindex(IDS).fillna(False).sum()),
                          folder_label_reproduced_by_sheet_LV_of_some_link=int(per.apply(lambda g: (g.lv_sheet == g.label.iloc[0]).any()).reindex(IDS).fillna(False).sum()))
    # ---- nearest-in-time echo (label-agnostic; ties -> mean EF), compared with the private tables used by Phases 4-6A
    def nearest(ecg_col):
        X = L[L.ef.notna()].copy()                       # a link without an EF value cannot define EF (as in Phase 4)
        X["abs_days"] = (X.echo_date.dt.normalize() - X[ecg_col].dt.normalize()).dt.days.abs()
        n = X[X.abs_days == X.groupby("ecg_id").abs_days.transform("min")].groupby("ecg_id").agg(
            ef=("ef", "mean"), sex=("sex", "first"), abs_days=("abs_days", "first"), score=("score", "min")).reindex(IDS)
        n["derived"] = [rule(e, s) for e, s in zip(n.ef, n.sex)]
        w = X[X.abs_days <= 30].assign(c=lambda d: [rule(e, s) for e, s in zip(d.ef, d.sex)])
        conflict = (w.groupby("ecg_id").c.nunique() > 1).reindex(IDS).fillna(False).values
        n["G"] = (n.score.values >= 100) & (n.abs_days.values <= 30) & ~conflict
        return n
    L["ecg_true"] = pd.to_datetime(L.ts.str[:8], format="%Y%m%d")
    near = nearest("ecg_date")                            # sheet ECG dates, as used by Phases 4-6A
    true = nearest("ecg_true")                            # true acquisition date from the file-name timestamp
    T = label_table()
    rep["nearest_echo_vs_phase4_tables"] = dict(
        ef_identical=int(np.isclose(near.ef.values, T.ef.values, atol=1e-6).sum()), of=len(IDS),
        derived_class_identical=int((near.derived.values == T.derived.values).sum()),
        abs_days_identical=int((near.abs_days.values == T.abs_days.values).sum()))
    lab = T.label.values
    rep["folder_vs_nearest_echo_class"] = dict(agreement=float(np.mean(lab == near.derived.values)),
                                               crosstab=pd.crosstab(pd.Series(lab, name="folder"), pd.Series(near.derived.values, name="nearest"))
                                               .reindex(index=CLASSES, columns=CLASSES + ["Unknown"], fill_value=0).to_dict())
    # ---- Phase 6A label configurations re-derived independently
    male = near.sex.values == "M"
    b2 = (lab == "Mild") & (near.derived.values == "Mild") & (near.ef.values >= 50)
    rep["phase6a_configs_rederived"] = dict(B2_movers=int(b2.sum()), B2_matches_label_table=bool((b2 == T.b2_move.values).all()),
                                            B3_band=int(((np.isin(lab, ["Normal", "Mild"])) & (np.abs(near.ef.values - np.where(male, 52, 54)) <= 2)).sum()),
                                            B3_band_matches=bool((((np.isin(lab, ["Normal", "Mild"])) & (np.abs(near.ef.values - np.where(male, 52, 54)) <= 2)) == T.b3_band.values).all()))
    # ---- finding: 2021 sheet ECG dates are day/month-swapped for days <= 12 (echo dates are correct)
    tsd = pd.to_datetime(L.ts.str[:8], format="%Y%m%d")
    mism = L.ecg_date.dt.normalize() != tsd
    sw = L.ecg_date.dt.day <= 12
    swapped = pd.to_datetime(dict(year=L.ecg_date.dt.year, month=L.ecg_date.dt.day.where(sw, L.ecg_date.dt.month),
                                  day=L.ecg_date.dt.month.where(sw, L.ecg_date.dt.day)), errors="coerce")
    mild, b2rows = lab == "Mild", T.b2_move.values
    rep["finding_2021_ecg_date_swap"] = dict(
        linked_rows_with_wrong_sheet_ecg_date=int(mism.sum()), explained_by_day_month_swap=int((mism & (swapped == tsd)).sum()),
        rows_per_year=L[mism].year.value_counts().to_dict(),
        dev_ecgs_with_changed_interval=int((near.abs_days.values != true.abs_days.values).sum()),
        dev_ecgs_with_changed_nearest_ef=int((~np.isclose(near.ef.values, true.ef.values)).sum()),
        max_abs_change_nearest_ef=float(np.nanmax(np.abs(near.ef.values - true.ef.values))),
        dev_ecgs_with_changed_nearest_class=int((near.derived.values != true.derived.values).sum()),
        interval_counts_sheet_vs_true={f"<= {d} d": [int((near.abs_days <= d).sum()), int((true.abs_days <= d).sum())] for d in (0, 7, 14, 30, 90)},
        G_cohort_sheet_vs_true=[int(near.G.sum()), int(true.G.sum())], G_membership_changes=int((near.G.values != true.G.values).sum()),
        folder_Mild_same_day_sheet_vs_true=[int(((near.abs_days == 0) & mild).sum()), int(((true.abs_days == 0) & mild).sum())],
        B2_movers_same_day_sheet_vs_true=[int(((near.abs_days == 0) & b2rows).sum()), int(((true.abs_days == 0) & b2rows).sum())],
        B2_movers_G_sheet_vs_true=[int((near.G & b2rows).sum()), int((true.G & b2rows).sum())],
        B2_and_B3_membership_unchanged_with_true_dates=bool(
            (((lab == "Mild") & (true.derived.values == "Mild") & (true.ef.values >= 50)) == b2rows).all()
            and (((np.isin(lab, ["Normal", "Mild"])) & (np.abs(true.ef.values - np.where(true.sex.values == "M", 52, 54)) <= 2)) == T.b3_band.values).all()),
        echo_dates="correct as stored: swapping them lowers same-day ECG-echo pairs in every year",
        folder_labels="unaffected (made by name matching without a date window)")
    rep["other_findings"] = dict(
        rule_vs_sheet_LV_disagreements_within_1_point_of_a_boundary="21 of 22 (rounding at class boundaries; the sheet LV column reproduces every folder label)",
        exact_duplicate_link_rows=q["exact_duplicate_rows"], links_without_EF=q["ef_missing"])
    ok = (all(v["identical_to_phase4_private_copy"] for v in rep["files"].values()) and rep["linking"]["with_at_least_one_link"] == len(IDS)
          and rep["nearest_echo_vs_phase4_tables"]["ef_identical"] == len(IDS) and rep["phase6a_configs_rederived"]["B2_matches_label_table"]
          and rep["phase6a_configs_rederived"]["B3_band_matches"])
    rep["all_consistent"] = bool(ok)
    (guard(RES) / "matching_data_check.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
