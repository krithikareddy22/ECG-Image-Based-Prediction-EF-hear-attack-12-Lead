"""Test-kit checks (PHASE7_PLAN.md Amendment 5): label rule, metric code, refusal guards, private-location rule.
In memory only; no model, no PDF opened; identifiers and timestamps are never printed.
    python -m research_experiments.phase_7.tests.test_testkit
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.testkit import score as S


def refused(fn):
    try:
        fn()
    except SystemExit as e:
        return e.code == 2
    return False


def main():
    ok = True
    # label rule (sex-specific Normal / Mild cut)
    lab = [S.rule(e, s) for e, s in ((58, "M"), (53, "M"), (53, "F"), (54, "F"), (41, "M"), (40.5, "F"), (31, "M"), (30, "F"))]
    ok &= lab == ["Normal", "Normal", "Mild", "Normal", "Mild", "Moderate", "Moderate", "Severe"]
    # metrics
    y = np.array([0, 0, 1, 2, 3, 3]); fc = S.fourclass(y, y)
    ok &= fc["accuracy"] == 1.0 and fc["macro_f1"] == 1.0 and fc["balanced_accuracy"] == 1.0
    tr = S.triage_stats(y, np.full(6, -1)); ok &= tr["answered"] == 0 and tr["accuracy_answered"] is None
    tr = S.triage_stats(y, np.array([0, -1, 0, 2, -1, 3]))
    ok &= tr["answered"] == 4 and abs(tr["accuracy_answered"] - 0.75) < 1e-12 and tr["all_mild_given_normal"] == 1.0
    ok &= abs(tr["normal_calls_without_ef_below_41"] - 1.0) < 1e-12 and tr["mild_caught_of_all_mild"] == 0.0
    lo, hi = S.wilson(70, 100); ok &= 0.60 < lo < 0.70 < hi < 0.79
    b = S.bootstrap(np.tile(y, 5), np.tile(y, 5), np.tile(y, 5), np.repeat(np.arange(10), 3), nboot=50)
    ok &= b["imgc3.accuracy"] == [1.0, 1.0]
    # guards: in-memory manifests whose file names carry a development / locked-test / new timestamp
    man = pd.read_csv(C.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "split"])
    ts = man.ecg_id.str.split("_").str[1]
    dev_ts, test_ts = ts[man.split != "test"].iloc[0], ts[man.split == "test"].iloc[0]
    new_ts = "19990101000000"
    assert new_ts not in set(ts)
    frame = lambda t: pd.DataFrame(dict(path=[Path(f"C:/nowhere/EK__[{t}]_x.pdf")]))           # noqa: E731
    ok &= refused(lambda: S.guards(frame(dev_ts), False))
    ok &= refused(lambda: S.guards(frame(test_ts), False))
    ok &= bool(S.guards(frame(test_ts), True)[0])
    ok &= not S.guards(frame(new_ts), False).any()
    # manifests must live outside the repository or under a git-ignored data/ folder
    ok &= not S.in_ignored_place(C.P7 / "testkit" / "m.csv")
    ok &= S.in_ignored_place(C.P7 / "data" / "private" / "m.csv") and S.in_ignored_place(Path("D:/elsewhere/m.csv"))
    print("RESULT:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
