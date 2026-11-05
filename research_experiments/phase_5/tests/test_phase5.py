"""Phase-5 safeguard tests (plain python; pytest is not installed).
    python -m research_experiments.phase_5.tests.test_phase5
Checks: no locked-test loading modules imported; development-only data and folds; patient-disjoint folds; no Phase-5
code path that reads the locked-test prediction folders; saved predictions contain exactly the 837 development ECGs;
aggregate <-> prediction hashes; privacy scan of every Phase-5 file that git would commit.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_experiments.phase_5 import p5common as P

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        FAIL.append(name)


def committable_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "research_experiments/phase_5"],
                         cwd=P.ROOT, capture_output=True, text=True).stdout.split()
    return [P.ROOT / f for f in out]


def main():
    import importlib
    for py in sorted((P.P5).rglob("*.py")):
        if "tests" in py.parts or py.name == "__init__.py":
            continue
        mod = "research_experiments." + ".".join(py.relative_to(P.ROOT / "research_experiments").with_suffix("").parts)
        try:
            importlib.import_module(mod)
        except SystemExit:
            pass
    check("no module that loads the locked test is imported by phase-5 code", not P.forbidden_modules_loaded(),
          str(P.forbidden_modules_loaded()))
    check("development set = 837 ECGs / 794 keys, no test rows", P.nD == 837 and len(set(P.PAT)) == 794 and (P.D.split != "test").all())
    for r in P.REPEATS:
        f = P.folds(r)
        pat = P.D.set_index("ecg_id").patient
        ok = True
        for k in range(5):
            g = f[f.fold == k]
            s = {role: set(pat.loc[g[g.role == role].ecg_id]) for role in ("inner_train", "inner_val", "outer_test")}
            ok &= not (s["inner_train"] & s["outer_test"]) and not (s["inner_val"] & s["outer_test"]) and not (s["inner_train"] & s["inner_val"])
        check(f"repeat {r}: folds patient-disjoint and development-only", ok and set(f.ecg_id) <= set(P.IDS))
    pat_re = re.compile(r"""/\s*["']locked_test|locked_test/|LOCK3|P2LOCK|\bLOCK\b""")   # path usage, not labels
    bad = []
    for py in (P.P5).rglob("*.py"):
        if "tests" in py.parts:
            continue
        for i, line in enumerate(py.read_text().splitlines()):
            if pat_re.search(line) and not line.lstrip().startswith("#"):
                bad.append(f"{py.name}:{i + 1}")
    check("no phase-5 code references the locked-test prediction folders", not bad, ", ".join(bad))
    ids = set(P.IDS); bad = []
    for d in (P.RUNS5).glob("*/oof.csv") if P.RUNS5.exists() else []:
        if set(pd.read_csv(d, usecols=["ecg_id"]).ecg_id) != ids:
            bad.append(d.parent.name)
    check("every saved prediction file covers exactly the 837 development ECGs", not bad, ", ".join(bad[:5]))
    bad = []
    import json
    for a in (P.EXP5).glob("*.json") if P.EXP5.exists() else []:
        o = P.RUNS5 / a.stem / "oof.csv"
        if o.exists() and json.loads(a.read_text()).get("oof_sha256") not in (None, P.sha256_file(o)):
            bad.append(a.stem)
    check("aggregate <-> prediction hashes match", not bad, ", ".join(bad[:5]))
    files = committable_files()
    priv = [str(f) for f in files if "data/private" in str(f) or f.suffix in (".pt", ".pth", ".npz")]
    check("nothing private or binary-per-ECG is committable", not priv, ", ".join(priv[:5]))
    flagged = P.privacy_scan(files)
    check(f"privacy scan of {len(files)} committable phase-5 files", not flagged, ", ".join(flagged[:5]))
    print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
