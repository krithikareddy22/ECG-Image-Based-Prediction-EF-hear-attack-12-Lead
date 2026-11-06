"""Phase-6 integrity: everything Phase 6 must NOT change.

Two independent checks, both required before every commit:
  1. git level (works in any clone): the working tree equals commit 27cbdbd (end of Phase 5) for every tracked path
     outside research_experiments/phase_6/ and the Phase 6A sub-study research_experiments/phase_6a_normal_mild_lstm/, and there is no untracked, non-ignored file outside phase_6/.
  2. content level (local): SHA-256 snapshot of the original project (LV_ECG_Dataset_*, data, results, models, ecg_ef,
     top-level files) and ALL of research_experiments except phase_6/ - i.e. Phases 1-5, including Phase 5 itself and
     the git-ignored private tables of Phase 4. Keys are hashed paths (dataset file names contain patient names).
     Files ADDED later (regenerated private inputs, downloaded weights) are reported but are not violations;
     a CHANGED or MISSING protected file is.

    python -m research_experiments.phase_6.integrity6 snapshot phase6_before
    python -m research_experiments.phase_6.integrity6 verify phase6_before

The snapshot stays local (phase_6/data/private/integrity/, git-ignored). Verification reports contain counts only and
are written to phase_6/results/integrity/.
"""
import hashlib
import json
import subprocess
import sys
import time

from research_experiments.common import ROOT

BASE_COMMIT = "27cbdbd"
SNAP = ROOT / "research_experiments" / "phase_6" / "data" / "private" / "integrity"
OUT = ROOT / "research_experiments" / "phase_6" / "results" / "integrity"
SKIP_PARTS = {"phase_6", "phase_6a_normal_mild_lstm", "phase_6b_threshold_sampling_lstm", "__pycache__", "logs", ".git"}
PHASE6_DIRS = (":!research_experiments/phase_6", ":!research_experiments/phase_6a_normal_mild_lstm",
               ":!research_experiments/phase_6b_threshold_sampling_lstm",   # additive Phase-6 work
               ":!research_experiments/phase_7")                            # additive Phase-7 work (PHASE7_PLAN.md)


def files():
    roots = [p for p in ROOT.iterdir() if p.name.startswith("LV_ECG_Dataset_") or p.name in ("data", "results", "models", "ecg_ef")]
    roots.append(ROOT / "research_experiments")
    for p in sorted(ROOT.iterdir()):
        if p.is_file():
            yield p
    for r in roots:
        for p in sorted(r.rglob("*")):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_PARTS):
                yield p


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def key(p):
    return hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:24]


def group(p):
    parts = p.relative_to(ROOT).parts
    if len(parts) == 1:
        return "<top-level>"
    return parts[0] + ("/" + parts[1] if parts[0] == "research_experiments" and len(parts) > 2 else "")


def git_check():
    """-> dict(ok, n_changed_tracked, n_untracked_outside_phase6)."""
    diff = subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", ".", *PHASE6_DIRS],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    diff = [d for d in diff if d]
    staged = subprocess.run(["git", "diff", "--cached", "--name-only", BASE_COMMIT, "--", ".", *PHASE6_DIRS],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    staged = [d for d in staged if d]
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", ".", *PHASE6_DIRS],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    untracked = [u for u in untracked if u]
    head_ok = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=ROOT).returncode == 0
    committed = subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "HEAD", "--", ".", *PHASE6_DIRS],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    committed = [c for c in committed if c]
    return dict(base_commit=BASE_COMMIT, base_is_ancestor_of_head=head_ok,
                n_tracked_changed_in_worktree=len(diff), n_staged_changed=len(staged),
                n_committed_changed_since_base=len(committed), n_untracked_outside_phase6=len(untracked),
                ok=head_ok and not diff and not staged and not committed and not untracked)


def main():
    cmd, tag = sys.argv[1], sys.argv[2]
    path = SNAP / f"integrity_{tag}.json"
    cur = {key(p): [sha(p), group(p)] for p in files()}
    if cmd == "snapshot":
        if path.exists():
            raise FileExistsError(f"{path} exists; snapshots are never overwritten")
        SNAP.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"created": time.strftime("%F %T"), "n": len(cur), "files": cur}))
        g = {}
        for v in cur.values():
            g[v[1]] = g.get(v[1], 0) + 1
        print(len(cur), "files;", g)
        return
    ref = json.loads(path.read_text())["files"]
    changed = [k for k in ref if k in cur and cur[k][0] != ref[k][0]]
    rep = dict(checked=time.strftime("%F %T"), snapshot=tag, n_ref=len(ref), n_cur=len(cur), changed=len(changed),
               changed_groups={}, missing=sum(k not in cur for k in ref), added_groups={}, git=git_check())
    for k in changed:
        rep["changed_groups"][ref[k][1]] = rep["changed_groups"].get(ref[k][1], 0) + 1
    for k in cur:
        if k not in ref:
            rep["added_groups"][cur[k][1]] = rep["added_groups"].get(cur[k][1], 0) + 1
    rep["ok_protected_unchanged"] = rep["changed"] == 0 and rep["missing"] == 0 and rep["git"]["ok"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"integrity_verify_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["ok_protected_unchanged"] else 1)


if __name__ == "__main__":
    main()
