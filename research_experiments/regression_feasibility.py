"""Experiment C feasibility: is a continuous LVEF value available in the AUTHORISED data?
Checks (counts only; no names, no free text printed or stored):
  1. columns of every authorised table (manifest, header/ecg features, split)
  2. PDF text layer of all 1,047 ECGs for EF-like tokens (EF, LVEF, EJECTION, FRACTION)
  3. the bracketed fields of the file names (count of non-empty / numeric-looking only)
The matching spreadsheets (LV_ECG_Dataset_*/*.xlsx) are NOT opened (no permission).
    python -m research_experiments.regression_feasibility
"""
import json
import re
from pathlib import Path

import pandas as pd
import pymupdf

from research_experiments.common import RES, ROOT

out = {}
for f in ("data/manifest.csv", "data/header_features.csv", "data/ecg_features.csv", "results/split.csv"):
    cols = pd.read_csv(ROOT / f, nrows=1).columns.tolist()
    out[f] = {"n_columns": len(cols), "ef_like_columns": [c for c in cols if re.search(r"(^|_)(lv)?ef($|_)|eject|fraction", c, re.I)]}
m = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["source"])
tok = re.compile(r"\b(LV\s*EF|EF|EJECTION|FRACTION)\b", re.I)
hits = 0
for src in m.source:
    with pymupdf.open(str(ROOT / src)) as doc:
        if tok.search(doc[0].get_text()):
            hits += 1
br = m.source.map(lambda s: re.findall(r"\[([^\]]*)\]", Path(s).name))
second = br.map(lambda L: L[1] if len(L) > 1 else "")
out["pdf_text_ef_token_files"] = hits
out["filename_second_bracket_nonempty"] = int((second != "").sum())
out["filename_second_bracket_numeric_like"] = int(second.str.fullmatch(r"\s*\d{1,2}(\.\d+)?\s*%?\s*").sum())
out["xlsx_spreadsheets_opened"] = False
out["conclusion"] = ("No continuous LVEF value exists in the authorised data (folder labels only)."
                     if hits == 0 and out["filename_second_bracket_numeric_like"] == 0
                     and not any(v.get("ef_like_columns") for v in out.values() if isinstance(v, dict))
                     else "Possible EF information found - inspect manually with permission.")
(RES / "regression_feasibility.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
