"""Record-level dataset accounting (no names printed or stored; counts and anonymous ids only).
    python -m research_experiments.phase_3.dataset_accounting
"""
import hashlib, json, re
from collections import Counter
from pathlib import Path
import pandas as pd
from research_experiments.common import ROOT

OUT = ROOT / "research_experiments" / "phase_3" / "results"
m = pd.read_csv(ROOT / "data" / "manifest.csv")
pdfs = sorted(p for p in ROOT.glob("LV_ECG_Dataset_*/*/*.pdf"))
xlsx = sorted(ROOT.glob("LV_ECG_Dataset_*/*.xlsx"))
rel = {str(p.relative_to(ROOT)) for p in pdfs}
in_manifest = set(m.source)
sha = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in pdfs}
dup_groups = [v for v in pd.Series(sha).groupby(pd.Series(sha)).groups.values() if len(v) > 1]
src2id = dict(zip(m.source, m.ecg_id)); src2split = dict(zip(m.source, m.split)); src2pat = dict(zip(m.source, m.patient))
dups = [{"ecg_ids": [src2id[s] for s in g], "splits": sorted({src2split[s] for s in g}), "same_patient_hash": len({src2pat[s] for s in g}) == 1,
         "same_label": len({m.set_index('source').label[s] for s in g}) == 1} for g in dup_groups]
copy_suffix = sum(bool(re.search(r"\(\d+\)\.pdf$", s)) for s in m.source)
ts = m.ecg_id.str.split("_").str[1]
rep = {
  "pdf_files_in_label_folders": len(pdfs), "pdfs_in_manifest": len(rel & in_manifest),
  "pdfs_not_in_manifest (excluded)": len(rel - in_manifest), "manifest_rows_without_pdf": len(in_manifest - rel),
  "spreadsheets_present_not_opened": len(xlsx),
  "per_year_per_class": m.groupby(["year", "label"]).size().unstack().to_dict(orient="index"),
  "class_totals": m.label.value_counts().to_dict(),
  "byte_identical_pdf_groups": len(dup_groups), "byte_identical_detail": dups,
  "files_with_copy_suffix_like_(1).pdf": int(copy_suffix),
  "duplicate_ecg_ids": int(m.ecg_id.duplicated().sum()), "duplicate_acquisition_timestamps": int(ts.duplicated().sum()),
  "patients_total": int(m.patient.nunique()), "ecgs_from_patients_with_multiple_ecgs": int(m.patient.duplicated(keep=False).sum()),
  "patients_with_multiple_ecgs_spanning_years": int((m.groupby("patient").year.nunique() > 1).sum()),
  "patients_with_multiple_ecgs_different_labels": int((m.groupby("patient").label.nunique() > 1).sum()),
  "split_counts": {s: {"ecgs": int((m.split == s).sum()), "patients": int(m[m.split == s].patient.nunique())} for s in ("train", "val", "test")},
  "development_D": {"ecgs": int((m.split != "test").sum()), "patients": int(m[m.split != "test"].patient.nunique())},
  "patients_shared_between_D_and_test": len(set(m[m.split != "test"].patient) & set(m[m.split == "test"].patient)),
}
(OUT / "dataset_accounting.json").write_text(json.dumps(rep, indent=1, default=int))
print(json.dumps({k: v for k, v in rep.items() if k != "byte_identical_detail"}, indent=1, default=int))
print("byte-identical groups detail:", dups[:10])
