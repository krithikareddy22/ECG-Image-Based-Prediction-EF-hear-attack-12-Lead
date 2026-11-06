"""Phase-5 resume tool: reads the registry, checks every artefact on disk, classifies each run and milestone, writes
checkpoint.json and the auto-generated part of PHASE5_STATUS.md, and prints the exact next command.

    python -m research_experiments.phase_5.resume5              # status + write checkpoint.json / PHASE5_STATUS.md
    python -m research_experiments.phase_5.resume5 --run-next   # also execute the next incomplete experiment's command

Run states:
  DONE            aggregate + private predictions present and the prediction SHA-256 matches the aggregate
  MISSING_PRIVATE aggregate present (committed) but per-ECG predictions absent (fresh clone): the command regenerates
                  them deterministically; the completed result is NOT lost and is NOT recomputed for its numbers
  CORRUPT         prediction hash does not match the aggregate -> investigate, never silently overwrite
  RUNNING         run marker with a live process
  INTERRUPTED     run marker whose process is gone -> re-run the printed command (completed runs are skipped)
  PENDING         nothing on disk
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from research_experiments.phase_5.p5common import (EXP5, P5, PRIV5, RUN_MARK, RUNS5, ROOT, git_head, sha256_file)
from research_experiments.phase_5.registry import EXPERIMENTS, MILESTONES

FINGERPRINT_FILES = ["data/manifest.csv", "results/split.csv", "research_experiments/phase_2/configs/folds_D.csv",
                     "data/header_features.csv", "research_experiments/phase_2/models/ecgf_aligned_features.npz",
                     "data/signals.npz", "research_experiments/phase_4/data/private/cohorts.csv"]
MANUAL = P5 / "configs" / "manual_status.json"     # BLOCKED milestones / known issues set by hand


def pid_alive(pid):
    try:
        os.kill(pid, 0); return True
    except (OSError, TypeError):
        return False


def run_state(rid):
    mk = RUN_MARK / f"{rid}.json"
    if mk.exists():
        m = json.loads(mk.read_text())
        return ("RUNNING" if pid_alive(m.get("pid")) else "INTERRUPTED"), m.get("command")
    agg, oof = EXP5 / f"{rid}.json", RUNS5 / rid / "oof.csv"
    if agg.exists() and oof.exists():
        want = json.loads(agg.read_text()).get("oof_sha256")
        return ("DONE" if want in (None, sha256_file(oof)) else "CORRUPT"), None
    if agg.exists():
        return "MISSING_PRIVATE", None
    return "PENDING", None


def exp_state(e):
    if "runs" in e:
        st = {r: run_state(r) for r in e["runs"]}
        states = [s for s, _ in st.values()]
    else:
        states = ["DONE" if (P5 / a).exists() else "PENDING" for a in e.get("artifacts", [])]; st = {}
    for s in ("CORRUPT", "INTERRUPTED", "RUNNING"):
        if s in states:
            return s, st
    if all(s == "DONE" for s in states):
        return "DONE", st
    if all(s in ("DONE", "MISSING_PRIVATE") for s in states):
        return "MISSING_PRIVATE", st
    return ("IN PROGRESS" if any(s == "DONE" for s in states) else "PENDING"), st


def main():
    manual = json.loads(MANUAL.read_text()) if MANUAL.exists() else {"blocked": {}, "known_issues": [], "milestone_overrides": {}}
    exps = {eid: exp_state(e) for eid, e in EXPERIMENTS.items()}
    ms = {}
    for m in MILESTONES:
        mine = [s for eid, (s, _) in exps.items() if EXPERIMENTS[eid]["milestone"] == m]
        if m in manual.get("blocked", {}):
            ms[m] = "BLOCKED"
        elif m in manual.get("milestone_overrides", {}):
            ms[m] = manual["milestone_overrides"][m]
        elif not mine:
            ms[m] = "NOT STARTED"
        elif "INTERRUPTED" in mine or "CORRUPT" in mine:
            ms[m] = "INTERRUPTED"
        elif all(s in ("DONE", "MISSING_PRIVATE") for s in mine):
            ms[m] = "DONE"
        elif all(s == "PENDING" for s in mine):
            ms[m] = "NOT STARTED"
        else:
            ms[m] = "IN PROGRESS"
    nxt = next(((eid, EXPERIMENTS[eid]["command"]) for eid, (s, _) in exps.items() if s not in ("DONE",)), None)
    runs_done = sorted(r for _, (s, st) in exps.items() for r, (rs, _) in st.items() if rs == "DONE")
    last = max(((EXP5 / f"{r}.json").stat().st_mtime, r) for r in runs_done) if runs_done else None
    ck = dict(
        phase="5", updated=time.strftime("%F %T"), git_commit=git_head(),
        current_milestone=next((m for m in MILESTONES if ms[m] != "DONE"), "complete"), milestones=ms,
        experiments={eid: dict(status=s, milestone=EXPERIMENTS[eid]["milestone"], command=EXPERIMENTS[eid]["command"],
                               runs={r: rs for r, (rs, _) in st.items()} or None,
                               artifacts=EXPERIMENTS[eid].get("artifacts"))
                     for eid, (s, st) in exps.items()},
        completed=[eid for eid, (s, _) in exps.items() if s == "DONE"],
        running=[eid for eid, (s, _) in exps.items() if s == "RUNNING"],
        interrupted=[eid for eid, (s, _) in exps.items() if s == "INTERRUPTED"],
        pending=[eid for eid, (s, _) in exps.items() if s in ("PENDING", "IN PROGRESS")],
        missing_private=[eid for eid, (s, _) in exps.items() if s == "MISSING_PRIVATE"],
        last_successful_run=dict(run_id=last[1], at=time.strftime("%F %T", time.localtime(last[0]))) if last else None,
        n_runs_done=len(runs_done),
        aggregate_results="research_experiments/phase_5/results/experiments/<run_id>.json",
        private_predictions="research_experiments/phase_5/data/private/runs/<run_id>/oof.csv (git-ignored)",
        scripts="research_experiments/phase_5/*.py (registry.py lists every experiment and its command)",
        configs="research_experiments/phase_5/configs/ (+ the 'config' block inside every aggregate json)",
        seeds=[42, 43, 44], repeats=[0, 1, 2],
        fingerprints={f: (sha256_file(ROOT / f) if (ROOT / f).exists() else "MISSING") for f in FINGERPRINT_FILES},
        next_action=dict(experiment=nxt[0], command=nxt[1]) if nxt else "all registered experiments complete",
        known_issues=manual.get("known_issues", []))
    (P5 / "checkpoint.json").write_text(json.dumps(ck, indent=1))
    # auto-generated status table (between markers in PHASE5_STATUS.md)
    st_md = P5 / "PHASE5_STATUS.md"
    table = ["<!-- AUTO-STATUS START (generated by resume5.py; do not edit by hand) -->",
             f"_Updated {ck['updated']}, git {ck['git_commit']}, {len(runs_done)} runs DONE._", "",
             "| Milestone | Status |", "|---|---|"] + [f"| {m} | **{s}** |" for m, s in ms.items()] + \
            ["", "| Experiment | Status | Runs done |", "|---|---|---|"] + \
            [f"| `{eid}` | {s} | {sum(1 for rs, _ in st.values() if rs == 'DONE')}/{len(st) or '–'} |" for eid, (s, st) in exps.items()] + \
            ["", f"**Next action:** `{nxt[1]}` ({nxt[0]})" if nxt else "**Next action:** none (all registered experiments complete)",
             "<!-- AUTO-STATUS END -->"]
    old = st_md.read_text() if st_md.exists() else "# Phase 5 status\n\n<!-- AUTO-STATUS START --><!-- AUTO-STATUS END -->\n"
    a, b = old.find("<!-- AUTO-STATUS START"), old.find("<!-- AUTO-STATUS END -->")
    new = old[:a] + "\n".join(table) + old[b + len("<!-- AUTO-STATUS END -->"):] if a >= 0 and b >= 0 else old + "\n" + "\n".join(table) + "\n"
    st_md.write_text(new)
    for m, s in ms.items():
        print(f"[{s:>11}] {m}")
    for eid, (s, st) in exps.items():
        if s in ("INTERRUPTED", "CORRUPT", "RUNNING"):
            print(f"  !! {eid}: {s} -> {EXPERIMENTS[eid]['command']}")
    print("\nnext:", f"{nxt[1]}   ({nxt[0]})" if nxt else "nothing pending")
    if "--run-next" in sys.argv and nxt:
        print("running:", nxt[1], flush=True)
        subprocess.run(nxt[1], shell=True, cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
