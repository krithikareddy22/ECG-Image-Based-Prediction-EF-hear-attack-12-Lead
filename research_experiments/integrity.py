"""Dataset / baseline integrity check (SHA-256 of every protected file).

    python -m research_experiments.integrity snapshot before   # -> results/integrity_before.json
    python -m research_experiments.integrity verify before     # compare current files with a snapshot

Protected = the original ECG folders (PDFs + the matching spreadsheets: hashed as raw bytes
only, never opened or parsed), the derived data cache, the original results (incl. split.csv)
and the original checkpoints. Nothing here writes outside research_experiments/.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
PROTECTED = ["LV_ECG_Dataset_2020", "LV_ECG_Dataset_2021", "LV_ECG_Dataset_2022",
             "LV_ECG_Dataset_2024", "data", "results", "models"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot() -> dict:
    files = {}
    for top in PROTECTED:
        for p in sorted((ROOT / top).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files[str(p.relative_to(ROOT))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return files


def main():
    cmd, tag = sys.argv[1], sys.argv[2]
    path = OUT / f"integrity_{tag}.json"
    if cmd == "snapshot":
        files = snapshot()
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_files": len(files),
                                    "files": files}, indent=1))
        per_top = {t: sum(k.startswith(t + "/") for k in files) for t in PROTECTED}
        print(f"snapshot '{tag}': {len(files)} files {per_top}")
    elif cmd == "verify":
        ref = json.loads(path.read_text())["files"]
        cur = snapshot()
        changed = [k for k in ref if k in cur and cur[k]["sha256"] != ref[k]["sha256"]]
        missing = [k for k in ref if k not in cur]
        added = [k for k in cur if k not in ref]
        report = {"checked": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": path.name,
                  "n_reference": len(ref), "n_current": len(cur),
                  "changed": changed, "missing": missing, "added": added,
                  "ok": not (changed or missing or added)}
        (OUT / f"integrity_verify_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}))
        for k in changed + missing + added:
            print("  !!", k)
        sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
