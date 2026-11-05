# Phase 6B exploratory amendment: IMG-ONLY (no header information)

**Date:** 2026-11-05. Written before any run, at the owner's request ("only image based, without the header").

- **Header-free** means no printed machine measurements, no age and no sex, and no clinical features derived from them.
  Inputs are only the ECG drawing: the waveform digitised from the PDF (ECGFounder "aligned" embedding) or the page
  image (ResNet-18).
- **IMG-vote:** equal-weight vote of the header-free ECGFounder heads, the C3 recipe with the header-using members
  replaced:
  - MLP on ECGFounder f6 (`p2heads` MLP 64, dropout 0.5, class-balanced CE, 3 seeds);
  - ECGFounder f6 linear head with distance-penalised CE.
- **IMG-C3:** IMG-vote + the nested class-prior rule (protected objective), with nested inner OOF of both members
  (Phase 5 nested protocol).
- **Reported alongside** (existing results, no retraining): ECGFounder linear, ResNet-18 image (repeat 0 only),
  LSTM / CNN-LSTM (laptop runs).
- Original labels, all 837 ECGs, Phase 2 folds; compared with C3 (with header) to quantify the header's contribution.
  Exploratory; reported separately.
