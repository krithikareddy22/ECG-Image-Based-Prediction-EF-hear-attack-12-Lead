# Phase 6B exploratory amendment: ECHONEXT (external pre-trained EF model as an extra input)

**Date:** 2026-11-05. Written before any run. The owner asked for further ways to improve accuracy without new data
and approved the use of the EchoNext-Mini weights for this non-commercial project, with citation.

## External model
- **Source:** EchoNext-Mini, `PierreElias/IntroECG`, folder `7-EchoNext Minimodel`, upstream commit
  `15233e9392e9dc10c136a5624abf10199207bce9`. The repository has no licence file; the authors state that the weights
  are publicly released, and ask for citation. **Cite:**
  - Hughes JW, Jing L, Finer J, et al. EchoNext-Mini: A Dataset and Baseline AI Model for Detecting Structural Heart
    Disease from Electrocardiograms. NEJM AI, 2026. doi:10.1056/AIdbp2500516.
  - Poterucha TJ, et al. Detecting structural heart disease from electrocardiograms using AI. Nature 2025;
    644(8075). doi:10.1038/s41586-025-09227-0.
- **Files:** `weights.pt`, `tabular_transformer.joblib`, `waveform_normalization_params.json` and the model definition
  `resnet1d_tabular.py`. They are copied to the git-ignored `data/private/echonext_mini/` with their SHA-256 and are
  never committed. No EchoNext data is downloaded or used.
- **Model:** ResNet-1D (filter size 16) with a 7-value tabular input. It returns 12 sigmoid outputs: LVEF ≤ 45, LV wall
  thickness ≥ 13 mm, five valve lesions, RV dysfunction, pericardial effusion, PASP ≥ 45, TR velocity ≥ 3.2, and the
  composite structural heart disease (SHD) label.
- **Inference only:** the weights are frozen and never trained on these ECGs.

## Inputs (header-free)
- **Waveform (primary):** the P1 beat-synchronous 10-s 12-lead reconstruction of `AMENDMENT_ENH.md`, built from the
  digitised printout, with EchoNext's own pre-processing in place of the ECGFounder band-pass:
  1. two-stage median baseline removal (0.2 s, then 0.6 s);
  2. 500 → 250 Hz by taking every other sample, as for MUSE 500-Hz files;
  3. mV → MUSE units (× 1000 / 4.88);
  4. lead order I, II, III, aVR, aVL, aVF, V1–V6;
  5. per-lead truncation and z-scoring with the published parameters.
- **Waveform (diagnostic):** the same reconstruction with the ECGFounder band-pass.
- **Tabular:**
  - sex = 0.5 (unknown, between female 0 and male 1);
  - ventricular rate = heart rate measured on the rhythm strip of the image;
  - age, atrial rate, PR, QRS and QTc = the pipeline's training medians (unknown);
  - **diagnostic:** ventricular rate also set to the median.

  No header value is used.

## Descriptive transfer check (no training)
On the 837 development ECGs:
- AUROC of P(LVEF ≤ 45) for echo EF ≤ 45;
- AUROC for Normal vs Mild and Mild vs Moderate;
- Spearman ρ with echo EF.

These are compared with the in-house header-free EF ridge (M6 inputs; nested outer OOF).

## Members and combiners
- **New members** (nested inner OOF, same splits as ULT):
  - **M13:** class-balanced LR on the 12 EchoNext-Mini output logits;
  - **M14:** LR on [aligned f6 ‖ waveform features ‖ EchoNext logits], block-standardised;
  - C grid as ENH.
- **ULT2 combiners:** ULT2-select, ULT2-stack and ULT2-vote, i.e. the ULT combiners with M13 and M14 added. The
  ULT2-stack meta-features also include the EchoNext LVEF ≤ 45 logit.
- **Decision rule:** as ULT (maximise accuracy; every inner recall ≥ 0.18, fallback 0.15).
- **Primary: ULT2-stack** (stacking was the best ULT combiner; this choice is declared here, before ULT2 is run).

## Evaluation
- **Data:** all 837 ECGs, original folder labels, Phase-2 folds (5 × 3).
- **Reported:** accuracy (corrected 95 % CI), per-class recall and precision, admissibility.
- **Key test:** ULT2-stack vs ULT-stack. It is identical except for the EchoNext members, so it isolates the added
  value of the external model.
- **Also tested:** ULT2-stack vs C3 (with header) and vs IMG-C3; Holm over 3.
- **Status:** EXPLORATORY. The locked test set is not used.
