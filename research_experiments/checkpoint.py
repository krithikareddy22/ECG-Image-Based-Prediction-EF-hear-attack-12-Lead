"""Audited checkpoint commit + push of COMPLETED research files to research/ecg-ef-model-improvements only.
(Authorised by the project owner for milestone checkpoints. Never force-pushes, never touches other branches.)

    python -m research_experiments.checkpoint "Checkpoint: <what>"
Excluded: datasets/PDFs/spreadsheets (outside research_experiments anyway), checkpoints, pretrained weights,
feature/activation caches (*/models/), logs, caches, files modified < 3 min ago, runs without final metrics,
integrity snapshots keyed by dataset file names (contain patient names). Aborts on any audit failure.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BRANCH = "research/ecg-ef-model-improvements"
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_0197pVhpv431s3tnRJePvqaD\n")


def sh(*a, check=True):
    return subprocess.run(a, cwd=ROOT, check=check, capture_output=True, text=True).stdout.strip()


INCLUDE_RECENT = set(a for a in sys.argv[2:])   # hand-written files (e.g. RESUME.md) that may be committed although just edited


def candidates():
    now = time.time(); out, skipped = [], {}
    for p in sorted((ROOT / "research_experiments").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT); s = str(rel); parts = rel.parts
        why = None
        if "__pycache__" in parts or "logs" in parts:
            why = "cache/log"
        elif len(parts) > 2 and parts[-2] != "results" and "models" in parts[:3]:
            why = "models/ (checkpoints, weights, caches)"
        elif p.name.startswith("integrity_") and "verify" not in p.name:
            why = "integrity snapshot (file names contain patient names)"
        elif now - p.stat().st_mtime < 180 and s not in INCLUDE_RECENT:
            why = "modified < 3 min ago"
        elif p.stat().st_size > 5 * 2 ** 20:
            why = "> 5 MB"
        elif "/results/" in s and p.parent.name != "results":
            d = p.parent
            complete = (d / "cv_metrics.json").exists() or (d / "metrics.json").exists() or \
                (d.name == "img_resnet18_baseline_recipe" and re.match(r"fold\d(\.npz|_info\.json)$", p.name)
                 and (d / (p.name[:5] + "_info.json")).exists() and (d / (p.name[:5] + ".npz")).exists()) or \
                (d.name == "img_resnet18_baseline_recipe" and (d / "oof.csv").exists())
            if not complete:
                why = "run without final metrics"
        elif "/locked_test/" in s:
            tag = p.stem
            if not any((ROOT / "research_experiments" / ph / "results" / tag / f).exists()
                       for ph in ("phase_2", "phase_3") for f in ("cv_metrics.json", "oof.csv")):
                why = "locked-test file of incomplete run"
        if why:
            skipped[why] = skipped.get(why, 0) + 1
        else:
            out.append(s)
    return out, skipped


def audit(files):
    src = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["source"]).source
    names = set()
    for s in src:
        m = re.search(r"\]_(.*?)_*\[", Path(s).name)
        if m:
            names |= {t for t in re.split(r"[\s_.]+", m.group(1).upper()) if len(t) >= 4 and t.isalpha()}
    allowed_words = {"MADE", "NORMAL", "SINUS", "RHYTHM", "LEFT", "RIGHT", "AXIS", "BLOCK", "WAVE", "ABNORMAL", "ATRIAL", "LEAD", "LEADS"}
    names -= allowed_words
    # patterns are written so that this source line does not match itself (e.g. "pass(word)")
    secret = re.compile(r"(api[_-]?key|secret[_-]?key|pass(word)|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH) PRIVATE|aws_access_(key))", re.I)
    bad_names, bad_secret = [], []
    for f in files:
        if Path(f).suffix.lower() in (".png", ".npz", ".npy", ".pt"):
            continue
        t = (ROOT / f).read_text(errors="ignore")
        if set(re.findall(r"[A-Z]{4,}", t.upper())) & names:
            bad_names.append(f)
        if secret.search(t):
            bad_secret.append(f)
    return bad_names, bad_secret


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "Checkpoint: research progress"
    assert sh("git", "branch", "--show-current") == BRANCH, "not on the research branch"
    files, skipped = candidates()
    tracked = set(sh("git", "ls-files", "research_experiments").splitlines())
    new_or_changed = [f for f in files if f not in tracked or sh("git", "diff", "--name-only", "--", f, check=False)]
    if not new_or_changed:
        print("nothing new to checkpoint"); return
    bad_names, bad_secret = audit(new_or_changed)
    if bad_names or bad_secret:
        print("AUDIT FAILED - nothing committed", {"name_hits": bad_names, "secret_hits": bad_secret}); sys.exit(1)
    lst = ROOT / ".git" / "checkpoint_files.txt"; lst.write_text("\n".join(new_or_changed) + "\n")
    sh("git", "add", f"--pathspec-from-file={lst}")
    staged = sh("git", "diff", "--cached", "--name-only").splitlines()
    outside = [f for f in staged if not f.startswith("research_experiments/")]
    mod_outside = sh("git", "diff", "--cached", "--name-only", "--diff-filter=MDR", "--", ".", ":(exclude)research_experiments").splitlines()
    big = [f for f in staged if (ROOT / f).exists() and os.path.getsize(ROOT / f) > 5 * 2 ** 20]
    if outside or mod_outside or big:
        sh("git", "reset", "-q"); print("STAGING CHECK FAILED - unstaged", outside, mod_outside, big); sys.exit(1)
    sh("git", "commit", "-q", "-m", msg + TRAILER)
    head = sh("git", "rev-parse", "HEAD")
    remote_before = sh("git", "ls-remote", "origin", f"refs/heads/{BRANCH}").split("\t")[0]
    local_parent = sh("git", "rev-parse", "HEAD~1")
    if remote_before and remote_before != local_parent:
        print(f"REMOTE DIVERGED (remote {remote_before[:8]} != local parent {local_parent[:8]}) - commit kept locally, NOT pushed"); sys.exit(1)
    for i in range(5):
        r = subprocess.run(["git", "push", "-u", "origin", BRANCH], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            break
        time.sleep(2 ** (i + 1))
    remote = sh("git", "ls-remote", "origin", f"refs/heads/{BRANCH}").split("\t")[0]
    orig = sh("git", "ls-remote", "origin", "refs/heads/claude/funny-gauss-udlmv0").split("\t")[0]
    print(dict(commit=head, pushed=remote == head, files=len(staged), skipped=skipped, original_branch_remote=orig))


if __name__ == "__main__":
    main()
