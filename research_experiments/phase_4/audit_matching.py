"""Phase 4A/4B: record-level audit of ECG-echo matching, intervals and label quality.

Reads the ECG_ECHO matching spreadsheets (extracted read-only from git history into the git-ignored
phase_4/data/private/ecg_echo/). NAMES ARE NEVER READ INTO OUTPUTS: the only columns used are the ECG
file-name timestamp, dates, EF, match score, sex and age. Outputs:
  data/private/ecg_echo_links.csv   one row per (dataset ECG, matched echo) - ecg_id, pseudonymous patient
                                    hash, split, folder label, EF, signed interval, score, derived class
                                    (git-ignored: contains acquisition timestamps via ecg_id)
  results/audit_matching.json       aggregate statistics only (no identifiers, no dates)
    python -m research_experiments.phase_4.audit_matching
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / "research_experiments" / "phase_4"
SRC = P4 / "data" / "private" / "ecg_echo"
CLASSES = ["Normal", "Mild", "Moderate", "Severe"]
FILES = {2020: "2020_echo_ecg_matching_50percent_20261029_162620.xlsx", 2021: "2021_echo_first_ecg_matching_20261029_161227.xlsx",
         2022: "2022_ECG_ECHO_MATCHING_70_PERCENT_20261029_170108.xlsx", 2024: "2024_ECG_ECHO_MATCHING_70_PERCENT_20261029_165914.xlsx"}


def classify(ef, sex):
    """Exact rule recovered from ecg_echo_matching.ipynb (classify_lv)."""
    if pd.isna(ef):
        return "Unknown"
    if sex == "M":
        if ef > 52:
            return "Normal"
        if ef >= 41:
            return "Mild"
    elif sex == "F":
        if ef >= 54:
            return "Normal"
        if ef >= 41:
            return "Mild"
    if 31 <= ef < 41:
        return "Moderate"
    if ef < 31:
        return "Severe"
    return "Unknown"


def sexkey(v):
    v = str(v).strip().upper()
    return "M" if v in ("M", "MALE") else ("F" if v in ("F", "FEMALE") else None)


def load_year(y):
    df = pd.read_excel(SRC / FILES[y], sheet_name="Matched Records")
    c = {k.lower(): k for k in df.columns}
    pick = lambda *names: next((c[n] for n in names if n in c), None)
    out = pd.DataFrame({
        "ts": df[pick("ecg file name", "ecg file")].astype(str).str.extract(r"\[(\d{14})\]")[0],
        "ecg_date": pd.to_datetime(df[pick("ecg date")], errors="coerce"),
        "echo_date": pd.to_datetime(df[pick("echo study date", "echo date")], errors="coerce"),
        "ef": pd.to_numeric(df[pick("ef (teich) %", "ef teich (%)")], errors="coerce"),
        "score": pd.to_numeric(df[pick("match score", "name match %")], errors="coerce"),
        "sex_ecg": df[pick("gender", "gender ecg")].map(sexkey),
        "sex_echo": df[pick("gender echo")].map(sexkey) if pick("gender echo") else None,
        "lv_sheet": df[pick("lv", "lv classification")].astype(str),
        "match_reason": df[pick("match reason")].astype(str),
    })
    out["year"] = y
    out["sheet_rows"] = len(df)
    return out


def main():
    man = pd.read_csv(ROOT / "data" / "manifest.csv")[["ecg_id", "year", "label", "patient", "split"]]
    man["ts"] = man.ecg_id.str.split("_").str[1]
    L = pd.concat([load_year(y) for y in FILES], ignore_index=True)
    rep = {"matched_sheet_rows_per_year": L.groupby("year").size().to_dict()}
    unmatched = {y: {s: len(pd.read_excel(SRC / FILES[y], sheet_name=s)) for s in ("Unmatched Echo", "Unmatched ECG")} for y in FILES}
    rep["unmatched_rows_per_year"] = unmatched
    rep["matched_rows_unique_ecgs_per_year"] = L.groupby("year").ts.nunique().to_dict()
    rep["ef_rounding"] = {"integer_valued_share": float(np.mean(np.isclose(L.ef.dropna() % 1, 0))),
                          "one_decimal_share": float(np.mean(~np.isclose(L.ef.dropna() % 1, 0)))}
    rep["sheet_lv_vs_recomputed_rule_agreement"] = float(np.mean(
        L.apply(lambda r: classify(r.ef, r.sex_ecg), axis=1).str.lower() == L.lv_sheet.str.lower()))

    D = man.merge(L, on=["ts"], how="left", suffixes=("", "_sheet"))
    D["has_link"] = D.ef.notna()
    D["derived"] = D.apply(lambda r: classify(r.ef, r.sex_ecg) if r.has_link else "NoLink", axis=1)
    D["interval_days"] = (D.echo_date - D.ecg_date).dt.days            # signed: + = echo after ECG
    D["abs_days"] = D.interval_days.abs()
    D["sex_agree"] = np.where(D.sex_echo.isna() | D.sex_ecg.isna(), np.nan, (D.sex_echo == D.sex_ecg).astype(float))
    per_ecg = D.groupby("ecg_id").agg(n_links=("has_link", "sum"), n_classes=("derived", lambda s: s[s != "NoLink"].nunique()),
                                      any_agree=("derived", lambda s: False), label=("label", "first"))
    agree = D[D.derived == D.label].groupby("ecg_id").size()
    per_ecg["n_links_agreeing_with_folder_label"] = agree.reindex(per_ecg.index).fillna(0).astype(int)
    rep["dataset_ecgs"] = int(len(per_ecg))
    rep["dataset_ecgs_with_any_echo_link"] = int((per_ecg.n_links > 0).sum())
    rep["links_per_ecg_distribution"] = per_ecg.n_links.value_counts().sort_index().to_dict()
    rep["ecgs_with_links_of_conflicting_derived_class"] = int((per_ecg.n_classes > 1).sum())
    rep["ecgs_with_no_link_agreeing_with_folder_label"] = int(((per_ecg.n_links > 0) & (per_ecg.n_links_agreeing_with_folder_label == 0)).sum())

    # the echo that most plausibly produced the folder label: label-consistent link nearest in time
    C = D[D.has_link].copy()
    C["consistent"] = C.derived == C.label
    C = C.sort_values(["ecg_id", "consistent", "abs_days"], ascending=[True, False, True])
    best = C.groupby("ecg_id").head(1).set_index("ecg_id")
    rep["primary_link"] = {
        "ecgs": int(len(best)), "consistent_with_folder_label": int(best.consistent.sum()),
        "abs_interval_days_quantiles": best.abs_days.quantile([0, .25, .5, .75, .9, 1]).round(1).to_dict(),
        "echo_before_ecg": int((best.interval_days < 0).sum()), "same_day": int((best.interval_days == 0).sum()),
        "echo_after_ecg": int((best.interval_days > 0).sum()), "missing_dates": int(best.interval_days.isna().sum()),
        "score_100_exact_name": int((best.score >= 100).sum()), "score_lt_100": int((best.score < 100).sum()),
        "score_quantiles": best.score.quantile([0, .1, .25, .5]).to_dict(),
        "sex_echo_available": int(best.sex_agree.notna().sum()), "sex_agree_where_available": float(np.nanmean(best.sex_agree)),
    }
    win = {}
    for lo, hi, name in ((0, 7, "<=7d"), (0, 14, "<=14d"), (0, 30, "<=30d"), (0, 60, "<=60d"), (0, 90, "<=90d"), (91, 10 ** 6, ">90d")):
        m = (best.abs_days >= lo) & (best.abs_days <= hi) if name != ">90d" else best.abs_days > 90
        sub = best[m]
        win[name] = {"ecgs": int(len(sub)), "patients": int(sub.patient.nunique()), "classes": sub.label.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
                     "dev": int((sub.split != "test").sum()), "test": int((sub.split == "test").sum()),
                     "echo_before": int((sub.interval_days < 0).sum()), "echo_after_or_same": int((sub.interval_days >= 0).sum()),
                     "ef_median": float(sub.ef.median()) if len(sub) else None}
    rep["windows_primary_link"] = win
    # label audit: EF range per folder class + borderline (within +-2 EF points of the class's sex-specific boundaries)
    def borderline(r):
        b = [31, 41] + ([52] if r.sex_ecg == "M" else [54] if r.sex_ecg == "F" else [52, 54])
        return any(abs(r.ef - x) <= 2 for x in b)
    best["borderline_2pt"] = best.apply(borderline, axis=1)
    rep["label_audit"] = {c: {"n": int((best.label == c).sum()), "patients": int(best[best.label == c].patient.nunique()),
                              "ef_min": float(best[best.label == c].ef.min()), "ef_max": float(best[best.label == c].ef.max()),
                              "ef_median": float(best[best.label == c].ef.median()),
                              "borderline_within_2_points": int(best[(best.label == c)].borderline_2pt.sum()),
                              "derived_from_primary_ef": best[best.label == c].derived.value_counts().to_dict()} for c in CLASSES}
    # alternative label versions: (1) folder, (2) derived from the NEAREST-in-time link, (3) high-confidence nearest link
    near = D[D.has_link].sort_values(["ecg_id", "abs_days"]).groupby("ecg_id").head(1).set_index("ecg_id")
    near["hc"] = (near.score >= 100) & (near.abs_days <= 30) & (near.sex_agree.fillna(1) == 1)
    lab = per_ecg[["label"]].join(near[["derived", "ef", "abs_days", "interval_days", "score", "hc", "sex_ecg"]], how="left")
    lab["derived"] = lab.derived.fillna("NoLink")
    ct = pd.crosstab(lab.label, lab.derived)
    rep["folder_vs_nearest_link_derived"] = {"agreement": float(np.mean(lab.label == lab.derived)), "crosstab": ct.to_dict()}
    hc = lab[lab.hc == True]
    rep["high_confidence_nearest_link"] = {"rule": "exact name score 100, |ECG-echo| <= 30 d, sex agrees where recorded",
                                           "ecgs": int(len(hc)), "agreement_with_folder": float(np.mean(hc.label == hc.derived)) if len(hc) else None,
                                           "classes_folder": hc.label.value_counts().to_dict()}
    out = P4 / "data" / "private" / "ecg_echo_links.csv"
    keep = ["ecg_id", "patient", "split", "year", "label", "ef", "sex_ecg", "sex_echo", "interval_days", "abs_days", "score", "derived", "lv_sheet"]
    D[D.has_link][keep].to_csv(out, index=False)
    lab.reset_index().merge(man[["ecg_id", "patient", "split"]], on="ecg_id").to_csv(P4 / "data" / "private" / "ecg_label_versions.csv", index=False)
    (P4 / "results" / "audit_matching.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
