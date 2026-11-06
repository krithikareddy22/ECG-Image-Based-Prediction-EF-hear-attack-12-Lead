"""Phase-6 safeguard tests (plain python; pytest is not installed).
    python -m research_experiments.phase_6.tests.test_phase6
Checks:
  * no module that loads the locked test is imported by any Phase-6 module;
  * development set 837 / 794, no test rows; folds patient-disjoint and development-only;
  * no Phase-6 code references the locked-test prediction folders, and none selects test rows;
  * echo EF is never used to build model inputs (static check of the input-building modules);
  * every saved prediction / embedding file covers exactly the 837 development ECGs; aggregate <-> prediction hashes;
  * the pre-registration hash matches configs/PLAN.sha256;
  * regenerated baselines reproduce the committed Phase-5 aggregates within 0.005;
  * git-level integrity (no change outside phase_6/ since 27cbdbd);
  * nothing private or binary is committable; privacy scan of every committable Phase-6 file.
"""
import hashlib
import importlib
import json
import re
import subprocess
import sys

import pandas as pd

from research_experiments.phase_6 import p6common as P

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        FAIL.append(name)


def committable_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "research_experiments/phase_6"],
                         cwd=P.ROOT, capture_output=True, text=True).stdout.split()
    return [P.ROOT / f for f in out]


def main():
    for py in sorted(P.P6.rglob("*.py")):
        if "tests" in py.parts or py.name == "__init__.py" or "data" in py.relative_to(P.P6).parts:
            continue
        mod = "research_experiments." + ".".join(py.relative_to(P.ROOT / "research_experiments").with_suffix("").parts)
        try:
            importlib.import_module(mod)
        except SystemExit:
            pass
    check("no module that loads the locked test is imported by phase-6 code", not P.forbidden_modules_loaded(),
          str(P.forbidden_modules_loaded()))
    check("development set = 837 ECGs / 794 keys, no test rows", P.nD == 837 and len(set(P.PAT)) == 794 and (P.D.split != "test").all())
    pat = P.D.set_index("ecg_id").patient
    for r in P.REPEATS:
        f = P.folds(r); ok = True
        for k in range(5):
            g = f[f.fold == k]
            s = {role: set(pat.loc[g[g.role == role].ecg_id]) for role in ("inner_train", "inner_val", "outer_test")}
            ok &= not (s["inner_train"] & s["outer_test"]) and not (s["inner_val"] & s["outer_test"]) and not (s["inner_train"] & s["inner_val"])
        check(f"repeat {r}: folds patient-disjoint and development-only", ok and set(f.ecg_id) <= set(P.IDS))
    lock_re = re.compile(r"""/\s*["']locked_test|locked_test/|LOCK3|P2LOCK|\bLOCK\b|split\s*==\s*["']test""")
    bad = []
    for py in P.P6.rglob("*.py"):
        if "tests" in py.parts:
            continue
        for i, line in enumerate(py.read_text().splitlines()):
            if lock_re.search(line) and not line.lstrip().startswith("#"):
                bad.append(f"{py.name}:{i + 1}")
    check("no phase-6 code references locked-test folders or selects test rows", not bad, ", ".join(bad))
    inputs = [p for p in list((P.P6 / "features").glob("*.py")) + list((P.P6 / "experiments").glob("embed*.py"))]
    bad = [p.name for p in inputs if re.search(r"ef_target|cohorts\.csv|\becho\b", p.read_text())]
    check("echo EF is not referenced by the input-building modules", not bad, ", ".join(bad))
    ids = set(P.IDS); bad = []
    for d in (P.RUNS6).glob("*/oof.csv") if P.RUNS6.exists() else []:
        if set(pd.read_csv(d, usecols=["ecg_id"]).ecg_id) != ids:
            bad.append(d.parent.name)
    check("every saved prediction file covers exactly the 837 development ECGs", not bad, ", ".join(bad[:5]))
    bad = []
    for a in (P.EXP6).glob("*.json") if P.EXP6.exists() else []:
        o = P.RUNS6 / a.stem / "oof.csv"
        agg = json.loads(a.read_text())
        if o.exists() and agg.get("oof_sha256") is not None and P.sha256_file(o) not in P.accepted_sha(a.stem, agg):
            bad.append(a.stem)
    check("aggregate <-> prediction hashes match (or a recorded within-tolerance regeneration)", not bad, ", ".join(bad[:5]))
    bad = [f.stem for f in (P.REGEN6.glob("*.json") if P.REGEN6.exists() else [])
           if not ((P.EXP6 / f"{f.stem}.json").exists() and json.loads(f.read_text())["within_tolerance"]
                   and json.loads(f.read_text())["committed_sha256"] == json.loads((P.EXP6 / f"{f.stem}.json").read_text()).get("oof_sha256"))]
    check("regeneration records: within tolerance and tied to the committed aggregate", not bad, ", ".join(bad[:5]))
    plan, shaf = P.P6 / "PHASE6_PLAN.md", P.CONF6 / "PLAN.sha256"
    if shaf.exists():
        body = plan.read_text()
        cut = body.find("\n## Amendments")
        h = hashlib.sha256(body[:cut].encode()).hexdigest()
        check("pre-registration (text above '## Amendments') matches configs/PLAN.sha256", shaf.read_text().split()[0] == h)
    else:
        check("configs/PLAN.sha256 exists", False)
    reg = P.RES6 / "baseline_regeneration.json"
    if reg.exists():
        rep = json.loads(reg.read_text())
        check("regenerated C1 / C2 / C3 and their members reproduce the committed aggregates within 0.005",
              rep["all_within_0_005"], f"max |diff| {rep['max_abs_diff']}")
    from research_experiments.phase_6.integrity6 import git_check
    g = git_check()
    check("git integrity: nothing outside phase_6/ changed since 27cbdbd", g["ok"], json.dumps(g))
    files = committable_files()
    priv = [str(f) for f in files if "data/private" in str(f) or "/data/" in str(f) or
            f.suffix in (".pt", ".pth", ".npz", ".npy", ".pdf", ".xlsx", ".ipynb")]
    check("nothing private or binary-per-ECG is committable", not priv, ", ".join(priv[:5]))
    flagged = P.privacy_scan(files)
    check(f"privacy scan of {len(files)} committable phase-6 files", not flagged, ", ".join(flagged[:5]))
    print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
