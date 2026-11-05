# Phase 6B exploratory amendment: ULT (large diverse header-free ensemble, owner objective)

**Date:** 2026-11-05. Written before any run, at the owner's request: "multi-stage or any complex model is fine; it
should give the highest accuracy with Mild above 15 %; same dataset".

## Objective (the owner's, stated before running)
Maximise four-class accuracy on the original folder labels (837 ECGs), with Mild recall ≥ 0.15 and every other class
recall ≥ 0.15 (admissibility). The macro-F1 guard used in ENH / HIER is dropped, because it is not part of the
owner's objective.

## Members
All members are header-free. Each gives four-class probabilities: nested inner OOF on the outer-training portion
(patient-grouped 4-fold, random_state 1000 + 10 r + k) plus outer OOF on the Phase-2 folds (5 × 3).

| Code | Member | Status |
|---|---|---|
| M1 | IMG-vote (ECGFounder MLP + linear distance-CE) | existing, cached |
| M2 | LR on synth ECGFounder f6 ‖ logits | existing, cached |
| M4 | MiniRocket LR | existing, cached |
| M5 | LR on aligned f6 ‖ waveform features | existing, cached |
| M6 | EF-ordinal ridge | existing, cached |
| H | HIER-ext two-stage probabilities | existing, cached |
| M8 | LR on multi-scale ECGFounder: aligned f4 ‖ f5 ‖ f6max | new; same C grid as ENH |
| M9 | kNN (distance-weighted) on a 64-component PCA of [aligned f6 ‖ synth f6], PCA fitted on training rows | new; k ∈ {15, 30, 60} by inner log-loss |
| M10 | Histogram gradient boosting (class-balanced, depth 3, learning rate 0.05, L2 1.0) on [waveform features ‖ 32-component PCA of aligned f6] | new; iterations ∈ {100, 300} by inner log-loss |
| M12 | Confusion-zone specialist: LR on the M5 inputs, trained only on training ECGs that are Mild, Moderate, or Normal with EF < 62. Outputs Normal / Mild / Moderate; Severe is set to 10⁻³ and the row renormalised. | new; C as ENH |

## Combiners (one complete four-class model each)
- **ULT-select (primary):** greedy ensemble selection with replacement (Caruana) on the inner OOF, minimising
  log-loss over 25 steps. Members are weighted by how many times they are selected.
- **ULT-stack:**
  - **Meta-learner:** multinomial LR on the members' log-probabilities plus the predicted EF μ̂/50 and (μ̂/50)².
  - **Meta penalty:** C ∈ {0.01, 0.03, 0.1, 0.3, 1}, chosen by patient-grouped 4-fold meta-CV log-loss on the
    training portion.
  - **Decision tuning:** on the meta-OOF.
- **ULT-vote:** equal-weight mean of all members.
- **Decision rule (all three):**
  - class weights w_Mild, w_Moderate, w_Severe ∈ {0.4, 0.5, …, 2.5}, chosen on the training-portion OOF to maximise
    accuracy;
  - subject to every class recall ≥ 0.18 (a margin of 0.03 so that the outer recall stays ≥ 0.15);
  - fallback: ≥ 0.15; then the untuned rule.
- **Diagnostic:** the same combiners with an inner floor of exactly 0.15 (no margin).

## Evaluation
- Accuracy (corrected 95 % CI), per-class recall ("% detected") and precision, macro-F1, neighbour confusion.
- **Admissibility:** outer recall ≥ 0.15 for every class.
- **Tests:** paired corrected t-tests (15 cells) of ULT-select, ULT-stack and ULT-vote vs C3 (with header), IMG-C3,
  ENH-vote and HIER-ext; Holm over 12.
- **Status:** EXPLORATORY. The primary is declared here. The best of the three combiners is a maximum and is
  reported as such. The locked test set is not used.
