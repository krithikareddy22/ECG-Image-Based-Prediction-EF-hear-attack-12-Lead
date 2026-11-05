"""Audit of the phase-1 experiments (read-only on phase-1 outputs).
    python -m research_experiments.phase_2.audit_previous
"""
import json

import numpy as np
import pandas as pd

from research_experiments.common import RES as P1RES, ROOT, load_meta
from research_experiments.make_tables import holm as holm_p1
from research_experiments.phase_2.p2common import RES, holm, nadeau_bengio


def main():
    meta = load_meta()
    out = {}
    # 1-2. phase-1 CV folds: patient grouping
    oof = pd.read_csv(P1RES / "cv5_oof_predictions.csv")[["ecg_id", "fold"]].merge(meta[["ecg_id", "patient", "split", "label"]], on="ecg_id")
    per_pat = oof.groupby("patient").fold.nunique()
    out["phase1_cv"] = {
        "n_ecgs": len(oof), "n_patients": int(oof.patient.nunique()),
        "patients_in_more_than_one_fold": int((per_pat > 1).sum()),
        "patients_with_multiple_ecgs": int((oof.groupby("patient").size() > 1).sum()),
        "fold_class_counts": oof.groupby(["fold", "label"]).size().unstack().to_dict(orient="index"),
        "test_split_ecgs_inside_cv": int((oof.split == "test").sum()),
        "inner_validation_leak_check": "asserted in cv_grouped.py (no patient shared between inner-train/inner-val/outer fold)",
    }
    # 5a. corrected resampled t-test for the phase-1 CV comparisons
    pf = pd.read_csv(P1RES / "cv5_per_fold.csv")
    nb = {}
    for a, b in (("header_LR", "FU_header+ecgfA1"), ("header_LR", "ecgfA1_multinomial"),
                 ("headerNumbersOnly_LR", "FU_headerNumbersOnly+ecgfA1"), ("ecgfA1_multinomial", "ecgfA1_cumlink"),
                 ("header_LR", "header_cumlink")):
        for metric in ("acc", "bal_acc", "macro_f1"):
            d = pf[pf.model == b].set_index("fold")[metric] - pf[pf.model == a].set_index("fold")[metric]
            nb[f"{b} vs {a} [{metric}]"] = nadeau_bengio(d.values, n_train=int(0.8 * 0.8 * 1047), n_test=int(0.2 * 1047))
    boot = json.loads((P1RES / "cv5_tests.json").read_text())["paired_tests_pooled_oof"]
    out["phase1_cv_significance_recheck"] = {
        "nadeau_bengio_corrected_t": nb,
        "bootstrap_reported": {k: {m: v[m]["p_two_sided"] for m in ("accuracy", "balanced_accuracy")} for k, v in boot.items()},
        "note": "bootstrap p=0.0 means p < 1/2000 (= 0.0005), not 0; the pooled-OOF bootstrap ignores training-set "
                "variability, the corrected t-test (5 folds, 4 df) is far more conservative"}
    # 5b. bootstrap mechanics on the fixed test split
    t = meta[meta.split == "test"]
    rng = np.random.default_rng(42); pats = t.patient.unique(); groups = {p: np.flatnonzero(t.patient.values == p) for p in pats}
    skipped = 0
    for _ in range(2000):
        ii = np.concatenate([groups[p] for p in rng.choice(pats, len(pats))])
        skipped += len(np.unique(t.label_idx.values[ii])) < 4
    out["phase1_bootstrap"] = {"resamples": 2000, "resamples_skipped_missing_a_class": int(skipped),
                               "cluster_unit": "patient (210 test ECGs / 199 patients)", "ci_method": "percentile"}
    # 5c. Holm implementation check
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    expected = np.array([0.04, 0.09, 0.09, 0.025, 0.5])     # Holm by hand: 0.005*5, 0.01*4, 0.03*3, max(0.04*2, 0.09), 0.5*1
    out["holm_check"] = {"phase1_impl": holm_p1(p).round(4).tolist(), "phase2_impl": holm(p).round(4).tolist(),
                         "by_hand": expected.tolist(), "ok": bool(np.allclose(holm_p1(p), expected) and np.allclose(holm(p), expected))}
    # 6. population overlap
    out["evaluation_populations"] = {
        "fixed_split_test": "210 ECGs / 199 patients (original test split)",
        "phase1_cv": "all 1,047 ECGs incl. the same 210 test ECGs -> NOT independent of the fixed-split test",
        "phase2_cv": "837 development ECGs only (test split excluded) -> independent of the locked test set"}
    # 3. test influence (documented from the code and the phase-1 log of decisions)
    out["test_set_influence"] = [
        "Model/threshold/fusion-weight selection in phase-1 code used validation only (verified in heads.fit_head, train_deep, exp_fusion.fuse, header_ablation, ecg_ef/compare.py).",
        "ECGFounder input construction chosen with a label-free check on all ECGs (no EF labels) -> no label leakage.",
        "Researcher degrees of freedom: test results of ~40 phase-1 models (and ~30 earlier project models) were inspected; the decision to ADD the accuracy-selected fusion variants was taken after a preliminary look at test results (weights themselves chosen on validation). The fixed test set is therefore EXPOSED and the best test score across models is optimistically biased.",
        "Phase-1 'primary model' recommendations referred to test results -> must not be used to select phase-2 models; phase 2 selects on development-set CV only."]
    # 7. identical evaluation conditions
    out["evaluation_conditions"] = [
        "Fixed split: all phase-1 models scored on the same 210 ECGs with the same metric code; YES/NO threshold tuned on validation per model.",
        "NOT identical: the A2 'fine-tuned ECGFounder' row is a 3-seed ensemble, compared with single-seed image/header models (ensembling advantage); CORAL uses rank decoding (others argmax); fusion weights were tuned on only 168 validation ECGs.",
        "Phase-1 CV did not include the image CNN, the default fusion or fine-tuned ECGFounder -> the default could not be compared with ECGFounder under the same CV (fixed in phase 2)."]
    (RES / "audit_previous.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "phase1_cv"}, indent=1, default=float)[:6000])
    print({k: v for k, v in out["phase1_cv"].items() if k != "fold_class_counts"})


if __name__ == "__main__":
    main()
