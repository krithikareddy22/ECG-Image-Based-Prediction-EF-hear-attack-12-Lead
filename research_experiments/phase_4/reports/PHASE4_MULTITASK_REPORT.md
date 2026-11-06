# Phase 4E: regression–classification multitask learning

- **Pre-registration:** `configs/PREREG_PHASE4.md`.
- **Code:** `phase_4/multitask.py`.
- **Results:** `results/runs/mt_*.json`, `results/phase4_tables.md` (4E), `results/phase4_vs_controls.md`.

## 1. Design
- **Question:** does continuous EF supervision improve the shared representation's ability to separate Mild,
  Moderate and Severe?
- **Data and validation:**
  - 837 development ECGs / 794 patients;
  - folds, inner-train / inner-val, seeds (42 / 43 / 44) and repeats (0–2) identical to Phases 2–3;
  - all metrics are on all 837 ECGs against folder labels unless stated;
  - the locked test was neither predicted nor scored.
- **Network:**
  - input: header numeric ‖ ECGFounder f6, block-standardised on inner-train;
  - shared body: Dropout 0.5 → Linear(1,040 → 64) → GELU → Dropout 0.5, i.e. the Phase 2 MLP fusion body;
  - class head: 4-way, class-balanced CE on the folder label;
  - EF head: Huber(δ = 1) on z-scored nearest-echo EF, masked to the EF cohort.
- **Training:**
  - L2 ∈ {1e-4, 1e-3, 1e-2} and the epoch are selected on inner-val S3 (classification);
  - AdamW, batch 64 and patience 25, as in Phase 2.
- **Variants** (9 runs each = 3 seeds × 3 repeats):

  | Variant | Loss | EF cohort | Role |
  |---|---|---|---|
  | `mt_lam0_A` | CE only | – | **control** (classification-only) |
  | `mt_lam0.3_A`, `mt_lam1_A`, `mt_lam3_A` | CE + λ·Huber | A (837) | loss-weight dose |
  | `mt_sel` | λ ∈ {0, 0.3, 1, 3} chosen **per fold on inner-val S3** | A | pre-registered selection |
  | `mt_lam1_C`, `mt_lam1_G` | CE + 1·Huber | C (550) / G (357) | cleaner EF targets |
  | `mt_uw_A` | uncertainty weighting Σ exp(−sᵢ)Lᵢ + sᵢ | A | learned loss weights |
  | `mt_lam1_A_dist` | (CE + E_p[(k − y)²]/9) + 1·Huber | A | **exploratory** ordinal-aware head |

- **Regression-only training** is the 4D MLP regression (same body); classification-only is the control.
- **Implementation check [VALID]:** `mt_lam0_A` reproduces the stored Phase 2 MLP fusion predictions, max |ΔP| =
  5 × 10⁻⁶ (CSV rounding) with identical argmax. The EF head is initialised from a saved RNG state, so the class
  path is bit-identical. Every difference below is therefore caused by the EF loss, not by implementation changes.

## 2. Results [VALID]
Seed-combined within repeat, mean over 3 repeats.

| Variant | Accuracy | Balanced acc. | Macro-F1 | F1 Mild | F1 Severe | QWK | S3 | ΔS3 vs control (p) | ΔF1 Mild vs control (p) | EF head MAE A / G | R² G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mt_lam0_A` (control) | 0.544 | 0.466 | 0.440 | 0.186 | 0.373 | 0.583 | 0.483 | – | – | (untrained) | – |
| `mt_lam0.3_A` | 0.546 | 0.467 | 0.447 | 0.204 | 0.372 | 0.587 | 0.487 | +0.004 (0.80) | +0.018 (0.62) | 10.52 / 9.80 | 0.514 |
| `mt_lam1_A` | 0.546 | 0.467 | 0.447 | 0.211 | 0.366 | 0.587 | 0.487 | +0.004 (0.81) | +0.025 (0.62) | 10.43 / 9.63 | 0.528 |
| `mt_lam3_A` | 0.547 | 0.469 | 0.441 | 0.189 | 0.376 | 0.596 | 0.486 | +0.002 (0.87) | +0.004 (0.92) | 10.36 / 9.57 | 0.534 |
| `mt_sel` | 0.546 | 0.464 | 0.444 | 0.211 | 0.355 | 0.583 | 0.484 | +0.001 (0.93) | +0.024 (0.60) | 10.91 / 10.26 | 0.485 |
| `mt_lam1_C` | 0.548 | 0.469 | 0.447 | 0.207 | 0.374 | 0.586 | 0.488 | +0.005 (0.54) | +0.021 (0.39) | 10.26 / 9.44 | 0.537 |
| `mt_lam1_G` | 0.551 | 0.471 | 0.449 | 0.209 | 0.373 | 0.591 | **0.490** | +0.007 (0.39) | +0.020 (0.61) | 10.27 / 9.52 | 0.531 |
| `mt_uw_A` | 0.548 | 0.469 | 0.446 | 0.197 | 0.374 | 0.590 | 0.488 | +0.005 (0.74) | +0.013 (0.78) | 10.43 / 9.63 | 0.528 |
| `mt_lam1_A_dist` (exploratory) | 0.545 | 0.464 | 0.447 | 0.214 | 0.367 | **0.596** | 0.485 | +0.002 (0.87) | +0.029 (0.55) | 10.41 / 9.66 | 0.526 |
| Vote ensemble (benchmark) | 0.547 | 0.467 | 0.455 | 0.255 | 0.362 | 0.574 | 0.490 | – | – | – | – |

- **p-values** come from the corrected repeated-CV t-test on 15 matched (repeat, fold) cells. Holm-adjusted
  p = 1.00 for every variant and metric against the control.
- **Against the vote ensemble**, every variant has ΔS3 between −0.006 and +0.000, all p ≥ 0.70.
- **Bootstrap:** in the patient-cluster paired bootstrap (repeat 0), every Δ accuracy CI vs the vote ensemble
  includes 0 (`phase4_tables.md`).

## 3. Interpretation
1. **EF supervision does not harm the classifier and gives a uniformly small, non-significant positive shift.**
   - All 8 EF-supervised variants have ΔS3 ≥ 0 vs the control (+0.001 to +0.007) and higher Mild F1
     (+0.004 to +0.029).
   - The variants share seeds and folds, so these 8 signs are **not** independent evidence. The effect is
     smaller than the seed × split noise: S3 SD across the 3 repeats is 0.005–0.015 for these models, and SD over the 9
     single runs of `mt_lam1_A` is 0.008.
   - [HYPOTHESIS] continuous EF supervision regularises the representation slightly toward Mild; detecting an
     effect of +0.005 S3 would need far more ECGs.
2. **The joint model learns EF as well as the dedicated regressors:** EF-head MAE 10.3–10.5 on A and R² 0.53 on
   G, matching 4D. A single network can serve both tasks without loss.
3. **The loss weight does not matter** (λ 0.3–3 and learned weights are equivalent). Selecting λ on inner-val
   (`mt_sel`) is no better than fixing it. It chooses λ 0–3 inconsistently across folds, which is itself evidence
   that inner-val cannot distinguish the weights.
4. **Cleaner EF targets (C, G) give the highest point estimates** (S3 0.488 / 0.490) but are not significantly
   different from cohort A.
5. **The ordinal-aware loss** (exploratory) gives the highest QWK (0.596) and Mild F1 among the multitask variants,
   but not higher S3. In Phase 3 the ordinal heads also did not improve S3.
6. **None reaches the vote ensemble's Mild F1 (0.255).** Its ECGF-linear member keeps Mild predictions more
   frequent.

## 4. Limitations
- **Representation:** only the frozen ECGFounder embedding is trained on top of. The multitask signal cannot
  reshape the waveform representation itself; fine-tuning was equal to frozen in Phase 3.
- **Target:** the EF target has unquantified measurement noise and link uncertainty (`PHASE4_DATA_AUDIT.md`).
- **Power:** with 837 ECGs the corrected repeated-CV test has low power for differences < 0.02 S3.
