"""Phase-5 integrity: SHA-256 snapshot / verify of everything Phase 5 must NOT change - the original project
(LV_ECG_Dataset_*, data, results, models, ecg_ef) and ALL of research_experiments except phase_5/ (so phases 1-4,
including phase_4/ and its private tables, are protected). Keys are hashed paths (dataset file names contain names).
    python -m research_experiments.phase_5.integrity5 snapshot phase5_before
    python -m research_experiments.phase_5.integrity5 verify phase5_before
The snapshot stays local (git-excluded); verification reports (counts only) go to phase_5/results/.
"""
import hashlib
import json
import sys
import time

from research_experiments.common import ROOT

OUT = ROOT / "research_experiments" / "phase_5" / "results"
SKIP_PARTS = {"phase_5", "__pycache__", "logs", ".git"}


def files():
    roots = [p for p in ROOT.iterdir() if p.name.startswith("LV_ECG_Dataset_") or p.name in ("data", "results", "models", "ecg_ef")]
    roots.append(ROOT / "research_experiments")
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
    return parts[0] + ("/" + parts[1] if parts[0] == "research_experiments" and len(parts) > 2 else "")


def main():
    cmd, tag = sys.argv[1], sys.argv[2]
    path = OUT / f"integrity_{tag}.json"
    cur = {key(p): [sha(p), group(p)] for p in files()}
    if cmd == "snapshot":
        if path.exists():
            raise FileExistsError(f"{path} exists; snapshots are never overwritten")
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"created": time.strftime("%F %T"), "n": len(cur), "files": cur}))
        g = {}
        for v in cur.values():
            g[v[1]] = g.get(v[1], 0) + 1
        print(len(cur), "files;", g)
        return
    ref = json.loads(path.read_text())["files"]
    changed = [k for k in ref if k in cur and cur[k][0] != ref[k][0]]
    rep = dict(checked=time.strftime("%F %T"), n_ref=len(ref), n_cur=len(cur), changed=len(changed),
               changed_groups={}, missing=sum(k not in cur for k in ref), added_groups={})
    for k in changed:
        rep["changed_groups"][ref[k][1]] = rep["changed_groups"].get(ref[k][1], 0) + 1
    for k in cur:
        if k not in ref:
            rep["added_groups"][cur[k][1]] = rep["added_groups"].get(cur[k][1], 0) + 1
    rep["ok_protected_unchanged"] = rep["changed"] == 0 and rep["missing"] == 0
    (OUT / f"integrity_verify_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    sys.exit(0 if rep["ok_protected_unchanged"] else 1)


if __name__ == "__main__":
    main()
