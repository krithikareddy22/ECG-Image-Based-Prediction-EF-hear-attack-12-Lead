"""Phase 7 checks: pre-registration present, writes confined to phase_7/, no recurrent model code, and no development
ECG identifier (or patient key) in any tracked-candidate Phase 7 file (results, reports, CSV / JSON). Identifiers are
never printed.
    python -m research_experiments.phase_7.tests.test_phase7
"""
import re
import sys

from research_experiments.phase_7 import p7common as C


def main():
    ok = True
    plan = C.P7 / "PHASE7_PLAN.md"
    ok &= plan.exists() and "Recurrent models are discontinued" in plan.read_text(encoding="utf-8")
    try:
        C.guard7(C.ROOT / "data" / "x.npz"); ok = False
    except PermissionError:
        pass
    for py in C.P7.rglob("*.py"):
        if "tests" in py.parts:
            continue
        if re.search(r"nn\.(LSTM|GRU|RNN)\b", py.read_text(encoding="utf-8")):
            print("recurrent layer found in", py.name); ok = False
    ids = set(map(str, C.IDS)) | set(map(str, C.PAT))
    files = [p for p in C.P7.rglob("*") if p.is_file() and "private" not in p.parts and "__pycache__" not in p.parts
             and p.suffix in (".md", ".csv", ".json", ".txt")]
    bad = 0
    for p in files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        bad += sum(1 for i in ids if i and i in txt)
    if bad:
        print(f"identifier hits in tracked-candidate files: {bad}"); ok = False
    print(f"checked {len(files)} files; RESULT:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
