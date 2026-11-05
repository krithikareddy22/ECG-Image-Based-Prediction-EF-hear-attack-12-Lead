# Claims register (living document)

Every result that may be cited carries one of four statuses:
- **VALID**: measured, with the statistics stated.
- **CORRECTED**: the number or significance changed.
- **RETRACTED**: must not be used.
- **HYPOTHESIS**: an estimate or expectation, not measured.

| # | Claim | Status | Evidence / correct wording |
|---|---|---|---|
| C1 | Saved checkpoints reproduce the default model exactly (test accuracy 0.552, balanced 0.454, macro-F1 0.433) | VALID | phase 1 §1.3; max \|Δprob\| < 1e-6 |
| C2 | Header + ECGFounder beats header in 5-fold CV with p < 0.001 | **RETRACTED** | corrected resampled t-test p = 0.12 (accuracy), 0.33 (balanced); see `reports/ERRATUM_phase1.md` E1 |
| C3 | Header + ECGFounder has higher accuracy than header in CV (+6.6 points, phase 1) | CORRECTED | point estimate valid; **not statistically significant** (p = 0.12) |
| C4 | Fine-tuned ECGFounder beats the image CNN on the fixed test split (balanced accuracy +10.8 points) | VALID, qualified | nominal p = 0.022, Holm 0.24 (n.s.); 3-seed ensemble vs single-seed CNN; test set exposed |
| C5 | Ordinal heads do not improve accuracy or balanced accuracy | VALID | phases 1 and 2 (development CV): CORAL / cumulative link lower S3 in every input family |
| C6 | The printed header's predictive value is in the numeric measurements, not the machine text | VALID | phase-2 development CV: numeric S3 0.432 vs text 0.368 vs all 0.405 |
| C7 | The ECGFounder input construction is physiologically valid (calibration, lead identity, timing, derived leads) | VALID | `phase_2/results/audit_waveform.json` |
| C8 | 58 % of the ECGFounder input is masked | VALID (definition) | 70 of 120 lead-seconds are not recorded on the printout; see the phase-3 masking audit for its effect |
| C9 | Header + ECGFounder MLP fusion reaches S3 0.505 / accuracy 0.559 | **CORRECTED** | that was the best single run; 3 seeds on repeat 0 give S3 0.479–0.505, and repeat 1 gives 0.476; robust mean ± SD in phase 2/3 |
| C10 | 60 % 4-class accuracy is achievable with this dataset | HYPOTHESIS | not demonstrated; current development-CV accuracy ≈ 0.53–0.56 |
| C11 | The dataset is 669 labelled ECGs | **RETRACTED wording** | 1,047 ECGs / 993 patients; 669 = original training split |
