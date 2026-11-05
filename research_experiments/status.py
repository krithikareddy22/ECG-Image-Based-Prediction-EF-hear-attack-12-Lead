"""Where is the research pipeline? Prints DONE / TODO for every step and the next command to run.
    python -m research_experiments.status
Safe to run at any time (read-only)."""
from pathlib import Path

R = Path(__file__).resolve().parent
P2, P3 = R / "phase_2" / "results", R / "phase_3" / "results"


def done(path):
    return path.exists()


def n_runs(base, prefix):
    return sum(1 for d in base.glob(prefix + "*") if (d / "cv_metrics.json").exists())


STEPS = [
    ("P2 image-CNN CV, 5 folds (oof.csv)", done(P2 / "img_resnet18_baseline_recipe" / "oof.csv"),
     "python -m research_experiments.phase_2.cv_image_cnn --threads 3"),
    ("P2 MLP-fusion HP seeds (mlp_hp_seeds, 20 runs)", n_runs(P2, "ff_mlp_") >= 30,
     "python -m research_experiments.phase_2.cv_features --group mlp_hp_seeds"),
    ("P2 robust seeds x repeats (fusion + ECGFounder MLP)", done(P2 / "ecgf_f6_mlp_seed44_r2" / "cv_metrics.json"),
     "python -m research_experiments.phase_2.cv_features --group robust_r0_extra && "
     "python -m research_experiments.phase_2.cv_features --group robust_seeds --repeat 1 && "
     "python -m research_experiments.phase_2.cv_features --group robust_seeds --repeat 2"),
    ("P3 clinical features file", done(R / "phase_3" / "models" / "clinical_features.csv"),
     "python -m research_experiments.phase_3.clinical_features"),
    ("P3 clinical + gated runs (seed 42, repeat 0)", done(P3 / "gate_scalar_hdr+clin+ecgf_dist" / "cv_metrics.json"),
     "python -m research_experiments.phase_3.exp_p3 --group clinical gated"),
    ("P3 matched 3-way vs 2-way runs (seeds x repeats)", done(P3 / "clin_mlpfusion_hdr+clin+ecgf_seed44_r2" / "cv_metrics.json")
     and done(P3 / "clin_hdrnum+clin_skLR_r2" / "cv_metrics.json"), "bash research_experiments/phase_3/run_matched.sh"),
    ("P3 gated-dist seeds x repeats", done(P3 / "gate_scalar_hdr+clin+ecgf_dist_seed44_r2" / "cv_metrics.json"),
     "E='python -m research_experiments.phase_3.exp_p3 --group gated --only gate_scalar_hdr+clin+ecgf_dist'; "
     "$E --seeds 43 44; $E --repeat 1 --seeds 42 43 44; $E --repeat 2 --seeds 42 43 44"),
    ("P3 masking audit features", done(P3 / "masking_audit_features.json"), "python -m research_experiments.phase_3.masking_audit"),
    ("P3 masking dose-response CV", done(P3 / "masking_dose_response.json"), "python -m research_experiments.phase_3.masking_cv"),
    ("P2 header+image fusions incl. reference R (needs CNN)", done(P2 / "R_hdrall+img_wavg" / "cv_metrics.json"),
     "python -m research_experiments.phase_2.run_fusion --stage combos"),
    ("P3 robust comparison tables (must include the header + ResNet-18 baseline)",
     done(P3 / "p3_robust_tables.md") and "Header + ResNet-18" in (P3 / "p3_robust_tables.md").read_text(),
     "python -m research_experiments.phase_3.p3_robust"),
    ("P2 nested calibration (fusion MLP, 3 repeats)", all(done(P2 / f"nested_calibration_fusion_mlp_r{r}.json") for r in (0, 1, 2)),
     "for r in 0 1 2; do python -m research_experiments.phase_2.nested_calibration --model fusion_mlp --repeat $r; done"),
    ("P3 complementary ensembles (vote / wavg / stack, incl. image combos)", done(P3 / "ens_all+image__vote" / "cv_metrics.json"),
     "python -m research_experiments.phase_3.p3_ensembles"),
    ("P3 nested calibration of the vote ensemble (3 repeats)", all(done(P3 / f"nested_calibration_vote_r{r}.json") for r in (0, 1, 2)),
     "for r in 0 1 2; do python -m research_experiments.phase_3.nested_calib_vote --repeat $r; done"),
    ("P3 calibration summary + confusion collection", done(P3 / "calibration_nested_tables.md") and done(P3 / "confusion_matrices_leading.md"),
     "python -m research_experiments.phase_3.calibration_summary && python -m research_experiments.phase_3.confusion_collection"),
    ("P3 report draft", done(R / "phase_3" / "reports" / "PHASE3_REPORT.md"), "(see RESUME.md)"),
    ("P3 report placeholders filled (no 'PLACEHOLDER' left)",
     done(R / "phase_3" / "reports" / "PHASE3_REPORT.md") and "PLACEHOLDER" not in (R / "phase_3" / "reports" / "PHASE3_REPORT.md").read_text(),
     "fill the *_PLACEHOLDER markers in phase_3/reports/PHASE3_REPORT.md from the result files (see RESUME.md step list)"),
]

if __name__ == "__main__":
    nxt = None
    for name, ok, cmd in STEPS:
        print(f"[{'DONE' if ok else 'TODO'}] {name}")
        if not ok and nxt is None:
            nxt = cmd
    print("\nnext command:", nxt or "nothing left in the automated pipeline")
