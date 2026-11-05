"""Phase-6A safeguard tests (plain python).
    python -m research_experiments.phase_6a_normal_mild_lstm.tests.test_phase6a
Checks: no locked-test-loading module imported; no locked-test references; development-only labels; label
configurations as pre-registered (B2 only moves Mild -> Normal; B3 never changes labels; A270 inner-train Normal = 270,
patient-disjoint, outer folds unchanged); echo EF not referenced by the waveform-input module; saved predictions cover
exactly the 837 development ECGs and match their hashes; plan hash; git integrity; nothing private committable;
privacy scan of every committable Phase-6A file.
"""
import hashlib
import importlib
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from research_experiments.phase_6a_normal_mild_lstm import p6acommon as P

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        FAIL.append(name)


def main():
    for py in sorted(P.P6A.rglob("*.py")):
        if "tests" in py.parts or py.name == "__init__.py" or "data" in py.relative_to(P.P6A).parts:
            continue
        importlib.import_module("research_experiments." + ".".join(py.relative_to(P.ROOT / "research_experiments").with_suffix("").parts))
    check("no module that loads the locked test is imported", not P.forbidden_modules_loaded(), str(P.forbidden_modules_loaded()))
    lock_re = re.compile(r"""locked_test|LOCK3|P2LOCK|\bLOCK\b|split\s*==\s*["']test""")
    bad = [f"{py.name}:{i + 1}" for py in P.P6A.rglob("*.py") if "tests" not in py.parts
           for i, line in enumerate(py.read_text().splitlines()) if lock_re.search(line) and not line.lstrip().startswith("#")]
    check("no phase-6A code references locked-test folders or selects test rows", not bad, ", ".join(bad))
    T = P.label_table()
    check("label table = 837 development ECGs, labels identical to the development labels", len(T) == 837 and (T.y.values == P.Y).all())
    moved = T.y_b2.values != T.y.values
    check("B2 only moves Mild -> Normal, exactly the pre-registered movers",
          (T.y.values[moved] == 1).all() and (T.y_b2.values[moved] == 0).all() and (moved == T.b2_move.values).all())
    check("B3 flags only Normal / Mild ECGs", set(T.y.values[T.b3_uncertain.values]) <= {0, 1})
    ok = True
    for r in (0, 1, 2):
        for k in range(5):
            itr, iva, ote = P.split("A270", r, k); o_itr, o_iva, o_ote = P.split_indices(r, k)
            ok &= (P.Y[itr] == 0).sum() == 270 and np.array_equal(ote, o_ote) and not set(P.PAT[itr]) & set(P.PAT[iva])
            ok &= set(itr) | set(iva) == set(o_itr) | set(o_iva) and np.array_equal(np.sort(itr[P.Y[itr] != 0]), np.sort(o_itr[P.Y[o_itr] != 0]))
            b_itr, b_iva, b_ote = P.split("B3", r, k)
            ok &= np.array_equal(b_iva, o_iva) and np.array_equal(b_ote, o_ote) and set(b_itr) <= set(o_itr)
    check("A270: 270 Normal in inner-train, patient-disjoint, only Normal moved, outer folds unchanged; B3: only inner-train reduced", bool(ok))
    sig = P.P6A / "signals.py"
    if sig.exists():
        check("echo EF / labels not referenced by the waveform-input module", not re.search(r"label_table|cohorts|ef_target|\bef\b", sig.read_text()))
    ids = set(P.IDS); bad = []
    for d in P.RUNS.glob("*/oof.csv") if P.RUNS.exists() else []:
        if set(pd.read_csv(d, usecols=["ecg_id"]).ecg_id) != ids:
            bad.append(d.parent.name)
    check("every saved prediction file covers exactly the 837 development ECGs", not bad, ", ".join(bad[:5]))
    bad = [a.stem for a in (P.EXP.glob("*.json") if P.EXP.exists() else [])
           if (P.RUNS / a.stem / "oof.csv").exists()
           and P.sha256_file(P.RUNS / a.stem / "oof.csv") not in P.accepted_sha(a.stem, json.loads(a.read_text()))]
    check("aggregate <-> prediction hashes match (or a recorded within-tolerance regeneration)", not bad, ", ".join(bad[:5]))
    bad = [f.stem for f in (P.REGEN.glob("*.json") if P.REGEN.exists() else [])
           if not (P.agg_path(f.stem).exists() and json.loads(f.read_text())["within_tolerance"]
                   and json.loads(f.read_text())["committed_sha256"] == json.loads(P.agg_path(f.stem).read_text()).get("oof_sha256"))]
    check("regeneration records: within tolerance and tied to the committed aggregate", not bad, ", ".join(bad[:5]))
    body = (P.P6A / "PHASE6A_PLAN.md").read_text()
    h = hashlib.sha256(body[:body.find("\n## Amendments")].encode()).hexdigest()
    shaf = P.CONF / "PLAN.sha256"
    check("pre-registration body matches configs/PLAN.sha256", shaf.exists() and shaf.read_text().split()[0] == h)
    from research_experiments.phase_6.integrity6 import git_check
    g = git_check()
    check("git integrity: nothing outside phase_6/ and phase_6a/ changed since 27cbdbd", g["ok"], json.dumps(g))
    files = [P.ROOT / f for f in subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                                                 "research_experiments/phase_6a_normal_mild_lstm"], cwd=P.ROOT,
                                                capture_output=True, text=True).stdout.split()]
    priv = [str(f) for f in files if "/data/" in str(f) or f.suffix in (".pt", ".pth", ".npz", ".npy", ".pdf", ".xlsx", ".ipynb")]
    check("nothing private or binary is committable", not priv, ", ".join(priv[:5]))
    flagged = P.privacy_scan(files)
    check(f"privacy scan of {len(files)} committable phase-6A files", not flagged, ", ".join(flagged[:5]))
    print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
