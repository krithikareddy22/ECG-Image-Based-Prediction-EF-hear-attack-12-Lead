"""Phase-5 data audit: patient grouping, repeated / duplicate recordings, possible patient overlap.

The patient key used for grouping (data/manifest.csv:patient) is sha1(letters-only upper-case name)[:12] -
there is NO medical-record number. This script checks how reliable that grouping is:
  1. ECGs per patient key (development and locked test, counts only);
  2. near-identical NAMES under different keys (typos / extra name parts would split one person into two keys ->
     possible leakage across CV folds or into the locked test). Names are read into memory from the file paths,
     compared, and discarded; only pair COUNTS are written;
  3. duplicate / near-duplicate recordings (identical or almost identical 10-s rhythm strips) under different ids;
  4. how often such pairs fall into different outer CV folds (repeats 0-2) or straddle development / locked test.
Locked-test rows are used ONLY for overlap counting (names and signals); their labels are never read.
Output (aggregates only): phase_5/data_audit/grouping_audit.json
    python -m research_experiments.phase_5.data_audit.grouping_audit
"""
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.common import ROOT
from research_experiments.phase_5.p5common import P5, folds, guard5

OUT = P5 / "data_audit" / "grouping_audit.json"


def norm_name(src):
    m = re.match(r"EK__\[(\d+)\]_(.*?)_*\[\]_", Path(src).name)
    return re.sub(r"[^A-Z]", "", m.group(2).upper()) if m else ""


def main():
    man = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "patient", "split", "source"])
    names = man.source.map(norm_name).values              # in memory only
    man = man.drop(columns=["source"])
    dev = (man.split != "test").values
    out = {"patient_key": "sha1(normalised name)[:12]; no medical-record number available"}
    for nm, m in (("development", dev), ("locked_test", ~dev)):
        c = man[m].groupby("patient").size()
        out[f"ecgs_per_patient_{nm}"] = {str(k): int(v) for k, v in c.value_counts().sort_index().items()}
        out[f"{nm}_ecgs_patients"] = [int(m.sum()), int(c.size)]
    # 2. near-identical names under different patient keys
    key_name = pd.Series(names, index=man.patient).groupby(level=0).first()
    keys, nms = key_name.index.values, key_name.values
    split_of = man.groupby("patient").split.first().map(lambda s: "test" if s == "test" else "dev")
    fold_of = {r: folds(r).query("role == 'outer_test'").set_index("ecg_id").fold for r in (0, 1, 2)}
    pat_folds = {r: man[dev].assign(f=lambda d: d.ecg_id.map(fold_of[r])).groupby("patient").f.agg(set) for r in (0, 1, 2)}
    pairs = []
    for i in range(len(keys)):
        a = nms[i]
        for j in range(i + 1, len(keys)):
            b = nms[j]
            if not a or not b:
                continue
            sm = difflib.SequenceMatcher(None, a, b)
            if sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9:
                contained = min(len(a), len(b)) >= 8 and (a in b or b in a)
                if not contained:
                    continue
            r = sm.ratio(); contained = min(len(a), len(b)) >= 8 and (a in b or b in a)
            if r >= 0.9 or contained:
                pairs.append((keys[i], keys[j], r, contained))
    del nms, names, key_name
    rows = []
    for ka, kb, r, cont in pairs:
        sa, sb = split_of[ka], split_of[kb]
        row = dict(ratio_ge_0_95=r >= 0.95, contained=bool(cont), straddles_dev_test=sa != sb, both_dev=sa == sb == "dev")
        if row["both_dev"]:
            row["different_outer_fold_in_repeats"] = int(sum(bool(pat_folds[q][ka] ^ pat_folds[q][kb] or pat_folds[q][ka] != pat_folds[q][kb])
                                                             for q in (0, 1, 2)))
        rows.append(row)
    R = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ratio_ge_0_95", "contained", "straddles_dev_test", "both_dev"])
    out["similar_name_pairs_under_different_keys"] = dict(
        criterion="difflib ratio >= 0.90, or one normalised name (>= 8 letters) contained in the other",
        pairs=int(len(R)), ratio_ge_0_95=int(R.ratio_ge_0_95.sum()) if len(R) else 0,
        containment=int(R.contained.sum()) if len(R) else 0,
        straddling_development_and_locked_test=int(R.straddles_dev_test.sum()) if len(R) else 0,
        both_in_development=int(R.both_dev.sum()) if len(R) else 0,
        dev_pairs_split_across_outer_folds_in_at_least_one_repeat=int((R[R.both_dev].different_outer_fold_in_repeats > 0).sum())
        if len(R) and R.both_dev.any() else 0,
        note="a pair is a POSSIBLE same person; names cannot be verified without an identifier")
    # 3. duplicate recordings (rhythm strip)
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[man.ecg_id].values
    rh = z["rhythm"][pos].astype(np.float64)
    hashes = pd.Series([hash(x.tobytes()) for x in rh])
    out["exact_duplicate_rhythm_strips"] = int(hashes.duplicated(keep=False).sum())
    Zs = rh - rh.mean(1, keepdims=True); Zs /= (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-9)
    C = Zs @ Zs.T; np.fill_diagonal(C, 0)
    ii, jj = np.where(np.triu(C >= 0.98))
    same_pat = man.patient.values[ii] == man.patient.values[jj]
    straddle = dev[ii] != dev[jj]
    out["near_duplicate_rhythm_strips"] = dict(criterion="Pearson r >= 0.98 between 10-s lead-II strips", pairs=int(len(ii)),
                                               same_patient_key=int(same_pat.sum()), different_patient_key=int((~same_pat).sum()),
                                               straddling_development_and_locked_test=int(straddle.sum()))
    guard5(OUT).parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
