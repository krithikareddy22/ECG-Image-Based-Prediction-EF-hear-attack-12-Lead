"""Phase-4 integrity: SHA-256 of ORIGINAL data/project files (incl. the 4 spreadsheets) and of ALL phase 1-3 research
outputs (research_experiments/* except phase_4/, caches and logs). Nothing is ever written outside phase_4/.
    python -m research_experiments.phase_4.integrity4 snapshot|verify <tag>
"""
import hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
ORIGINAL = ["LV_ECG_Dataset_2020", "LV_ECG_Dataset_2021", "LV_ECG_Dataset_2022", "LV_ECG_Dataset_2024", "data", "results", "models", "ecg_ef"]
def files():
    for top in ORIGINAL:
        for p in sorted((ROOT / top).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                yield p
    rex = ROOT / "research_experiments"
    for p in sorted(rex.rglob("*")):
        rel = p.relative_to(rex).parts
        if p.is_file() and rel[0] != "phase_4" and "__pycache__" not in rel and "logs" not in rel:
            yield p
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
cmd, tag = sys.argv[1], sys.argv[2]
OUT.mkdir(parents=True, exist_ok=True)
# keys are hashed paths (dataset file names contain patient names) -> the snapshot itself contains no names
key = lambda p: hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:24]
group = lambda p: str(p.relative_to(ROOT)).split("/")[0] + ("/" + str(p.relative_to(ROOT)).split("/")[1] if str(p.relative_to(ROOT)).startswith("research_experiments") else "")
snap = lambda: {key(p): [sha(p), group(p)] for p in files()}
path = OUT / f"integrity_{tag}.json"
if cmd == "snapshot":
    s = snap(); path.write_text(json.dumps({"created": time.strftime("%F %T"), "n": len(s), "files": s}))
    g = {}
    for v in s.values():
        g[v[1]] = g.get(v[1], 0) + 1
    print(len(s), g)
else:
    ref = json.loads(path.read_text())["files"]; cur = snap()
    rep = dict(checked=time.strftime("%F %T"), n_ref=len(ref), n_cur=len(cur), changed=sum(cur[k][0] != ref[k][0] for k in ref if k in cur),
               missing=sum(k not in cur for k in ref), added_groups={})
    for k in cur:
        if k not in ref:
            rep["added_groups"][cur[k][1]] = rep["added_groups"].get(cur[k][1], 0) + 1
    rep["ok_original_and_previous_unchanged"] = rep["changed"] == 0 and rep["missing"] == 0
    (OUT / f"integrity_verify_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rep, indent=1))
    print(rep); sys.exit(0 if rep["ok_original_and_previous_unchanged"] else 1)
