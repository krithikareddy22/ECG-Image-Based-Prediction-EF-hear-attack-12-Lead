"""Phase-4 pipeline status (read-only): DONE / TODO per step and the next command. Complements
research_experiments/status.py (phases 2-3), which is a protected file and is left unchanged.
    python -m research_experiments.phase_4.status4
A run counts as done only if BOTH the committed aggregate (results/runs/<tag>.json) and the git-ignored per-ECG
predictions (data/private/runs/<tag>/oof.csv) exist; in a fresh clone the run scripts regenerate the latter
deterministically.
"""
from pathlib import Path

P4 = Path(__file__).resolve().parent


def done(p):
    return p.exists()


def n4(prefix):
    return sum(1 for f in (P4 / "results" / "runs").glob(prefix + "*.json") if (P4 / "data" / "private" / "runs" / f.stem / "oof.csv").exists())


M = "python -m research_experiments.phase_4."
STEPS = [
    ("P4 private ECG_ECHO data recovered (git-ignored)", done(P4 / "data" / "private" / "cohorts.csv"),
     "follow research_experiments/DATA_ACCESS.md (git show 9f2a883^ ... ; phase_4.audit_matching ; phase_4.cohorts)"),
    ("P4 benchmark predictions imported (18)", n4("bench_") >= 18, M + "label_quality --bench"),
    ("P4D EF regression (54 runs x 4 decoders = 216)", n4("reg_") >= 216,
     M + "regression --family ridge && " + M + "regression --family mlp"),
    ("P4E multitask (63 + 9 exploratory + 9 selected = 81)", n4("mt_") >= 81,
     M + "multitask && " + M + "multitask --variants lam1_A_dist && " + M + "multitask --select"),
    ("P4F label-quality training (36)", n4("lq_") >= 36, M + "label_quality"),
    ("P4H benchmark + fusions + spreadsheet", done(P4 / "results" / "phase4_results.xlsx") and n4("fus_") >= 9, M + "benchmark4"),
    ("P4 controls, analyses, tables, matched pool",
     all(done(P4 / "results" / f) for f in ("phase4_vs_controls.md", "mild_analysis.json", "g_cohort_analysis.json",
                                            "phase4_tables.md", "matched_pool_summary.json")),
     M + "controls4 && " + M + "analysis4 && " + M + "make_tables4 && " + M + "matched_pool"),
    ("P4 reports (8) + claims register",
     all(done(P4 / "reports" / f"PHASE4_{n}.md") for n in ("DATA_AUDIT", "LABEL_QUALITY_REPORT", "REGRESSION_REPORT",
         "MULTITASK_REPORT", "MILD_CLASS_ANALYSIS", "MODEL_BENCHMARK", "FINAL_REPORT", "CLAIMS_REGISTER")),
     "(written; see phase_4/reports)"),
]

if __name__ == "__main__":
    nxt = None
    for name, ok, cmd in STEPS:
        print(f"[{'DONE' if ok else 'TODO'}] {name}")
        if not ok and nxt is None:
            nxt = cmd
    print("\nnext command:", nxt or "nothing left in the phase-4 pipeline")
