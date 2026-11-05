"""Phase-2 integrity check: SHA-256 of the original data/project files AND all phase-1 outputs
(research_experiments/* except phase_2/ and caches), so nothing earlier can be overwritten unnoticed.
    python -m research_experiments.phase_2.integrity2 snapshot|verify <tag>
"""
import hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
ORIGINAL = ["LV_ECG_Dataset_2020", "LV_ECG_Dataset_2021", "LV_ECG_Dataset_2022", "LV_ECG_Dataset_2024",
            "data", "results", "models", "ecg_ef"]


def files():
    for top in ORIGINAL:
        for p in sorted((ROOT / top).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                yield p
    rex = ROOT / "research_experiments"
    for p in sorted(rex.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and "phase_2" not in p.relative_to(rex).parts[:1]:
            yield p


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snap():
    return {str(p.relative_to(ROOT)): sha(p) for p in files()}


cmd, tag = sys.argv[1], sys.argv[2]
path = OUT / f"integrity_{tag}.json"
if cmd == "snapshot":
    s = snap(); OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": time.strftime("%F %T"), "n_files": len(s), "files": s}, indent=0))
    groups = {}
    for k in s:
        g = k.split("/")[0] if not k.startswith("research_experiments") else "research_experiments(phase1)"
        groups[g] = groups.get(g, 0) + 1
    print(len(s), groups)
else:
    ref = json.loads(path.read_text())["files"]; cur = snap()
    rep = dict(checked=time.strftime("%F %T"), n_ref=len(ref), n_cur=len(cur),
               changed=[k for k in ref if k in cur and cur[k] != ref[k]], missing=[k for k in ref if k not in cur],
               added=[k for k in cur if k not in ref])
    rep["ok"] = not (rep["changed"] or rep["missing"] or rep["added"])
    (OUT / f"integrity_verify_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rep, indent=1))
    print({k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()})
    sys.exit(0 if rep["ok"] else 1)
