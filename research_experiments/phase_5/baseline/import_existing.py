"""5A: re-evaluate EXISTING per-ECG development-CV predictions (phases 2-4) with the Phase-5 metric suite.
No training. Copies each source OOF into the git-ignored Phase-5 store and writes an aggregate per run.
Also stages the Phase-3 clinical feature table (development rows only) as feature family 'clin3'.
    python -m research_experiments.phase_5.baseline.import_existing
"""
import numpy as np
import pandas as pd

from research_experiments.phase_5.p5common import (FEAT5, IDS, P3, ROOT, done, guard5, load_oof_csv, mark_running, save_run)
from research_experiments.phase_5.registry import IMPORTS, IMPORTS_R0_ONLY

RE = ROOT / "research_experiments"


def src_path(pattern, s, r):
    seed = "" if s in (0, 42) else f"_seed{s}"
    sfx = "" if r == 0 else f"_r{r}"
    return RE / (pattern.format(seed=seed, sfx=sfx, s=s, r=r) + "/oof.csv")


def stage_clin3():
    out = FEAT5 / "clin3.csv"
    if out.exists():
        return
    C = pd.read_csv(P3 / "models" / "clinical_features.csv").set_index("ecg_id").loc[IDS]
    C = C.replace([np.inf, -np.inf], np.nan)
    guard5(FEAT5).mkdir(parents=True, exist_ok=True)
    C.to_csv(out)


def main():
    stage_clin3()
    jobs = [(eid, pat, s, r) for eid, (pat, seeds) in IMPORTS.items() for r in (0, 1, 2) for s in seeds]
    jobs += [(eid, pat, 0, 0) for eid, pat in IMPORTS_R0_ONLY.items()]
    for eid, pat, s, r in jobs:
        rid = f"{eid}__s{s}_r{r}"
        if done(rid):
            continue
        src = src_path(pat, s, r)
        mark_running(rid, "python -m research_experiments.phase_5.baseline.import_existing")
        P, pred = load_oof_csv(src)
        agg = save_run(rid, P, pred, r, dict(kind="import", source=str(src.relative_to(ROOT)), seed=s, repeat=r,
                                             decision="stored pred column" if "pred" in pd.read_csv(src, nrows=1) else "argmax"))
        m = agg["metrics"]
        print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f} "
              f"Severe R/P {m['recall_Severe']:.2f}/{m['precision_Severe']:.2f} min recall {m['min_recall']:.2f}", flush=True)


if __name__ == "__main__":
    main()
