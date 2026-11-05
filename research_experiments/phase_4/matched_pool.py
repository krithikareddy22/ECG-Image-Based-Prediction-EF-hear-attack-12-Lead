"""How many ECG-echo matched ECGs exist in the authorised ECG_ECHO matching sheets beyond the 1,047-ECG dataset?
Aggregate counts only (per EF class of the nearest-in-time link, label-agnostic; by interval / exact-name subsets),
to size the data that a future study could add. Read-only on the private sheets. -> results/matched_pool_summary.json
    python -m research_experiments.phase_4.matched_pool
"""
import json

import numpy as np
import pandas as pd

from research_experiments.phase_4.audit_matching import CLASSES, FILES, ROOT, classify, load_year
from research_experiments.phase_4.p4common import RES4


def main():
    L = pd.concat([load_year(y) for y in FILES], ignore_index=True).dropna(subset=["ts", "ef"])
    L["abs_days"] = (L.echo_date - L.ecg_date).dt.days.abs()
    L["mind"] = L.groupby("ts").abs_days.transform("min")
    near = L[(L.abs_days == L.mind) | L.mind.isna()].groupby("ts").agg(ef=("ef", "mean"), sex=("sex_ecg", "first"),
                                                                       abs_days=("abs_days", "first"), score=("score", "min"),
                                                                       year=("year", "first"))
    near["cls"] = [classify(e, s) for e, s in zip(near.ef, near.sex)]
    ds = set(pd.read_csv(ROOT / "data" / "manifest.csv").ecg_id.str.split("_").str[1])
    near["in_dataset"] = near.index.isin(ds)
    out = {}
    for name, m in (("all_matched_ecgs", np.ones(len(near), bool)), ("nearest_link_within_14d", near.abs_days.le(14).values),
                    ("exact_name_and_within_30d", (near.score.ge(100) & near.abs_days.le(30)).values)):
        s = near[m]
        out[name] = {"total": int(len(s)), "in_dataset": int(s.in_dataset.sum()), "not_in_dataset": int((~s.in_dataset).sum()),
                     "by_class_not_in_dataset": {c: int(((s.cls == c) & ~s.in_dataset).sum()) for c in CLASSES},
                     "by_class_in_dataset": {c: int(((s.cls == c) & s.in_dataset).sum()) for c in CLASSES}}
    out["note"] = ("Counts of unique ECG acquisition timestamps in the 'Matched Records' sheets, class of the nearest-in-time "
                   "linked echo (fuzzy name match, not verified). Whether the PDFs of the ECGs not in the dataset are still "
                   "available is unknown. Unmatched-ECG rows (no echo) are not counted.")
    (RES4 / "matched_pool_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
