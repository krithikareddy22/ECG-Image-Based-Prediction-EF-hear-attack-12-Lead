# Phase 6B exploratory report: ECHONEXT (frozen EchoNext-Mini as an extra input)

Declared in `AMENDMENT_ECHONEXT.md` (commit 9db9d43). Code `echonext_mini.py`. Results:
`results/echonext_transfer.json` and `results/echonext_ult2.json`.
- **Data:** 837 development ECGs, original labels, patient-grouped 5-fold CV × 3 repeats.
- **Inputs:** header-free (sex unknown; heart rate measured on the image; other tabular inputs set to the pipeline
  medians). No EchoNext data were used, and the weights were not retrained.
- **Status:** EXPLORATORY.

**External model:** EchoNext-Mini, `PierreElias/IntroECG` @ 15233e93, used with the owner's approval. **Cite:**
- Hughes JW, Jing L, Finer J, et al. EchoNext-Mini: A Dataset and Baseline AI Model for Detecting Structural Heart
  Disease from Electrocardiograms. NEJM AI, 2026. doi:10.1056/AIdbp2500516.
- Poterucha TJ, et al. Detecting structural heart disease from electrocardiograms using AI. Nature 2025; 644(8075).
  doi:10.1038/s41586-025-09227-0.

## Transfer without training (descriptive)
| Separation | EchoNext-Mini P(LVEF ≤ 45), primary input | Band-pass input | In-house EF ridge (trained on these ECGs, nested OOF) |
|---|---|---|---|
| Echo EF ≤ 45 vs > 45 (AUROC) | 0.792 | 0.794 | **0.857** |
| Normal vs Mild | 0.707 | 0.713 | **0.768** |
| Mild vs Moderate | 0.606 | 0.614 | **0.650** |
| Spearman ρ with echo EF | −0.53 | −0.54 | 0.63 (+) |

Mean P(LVEF ≤ 45) by class: Normal 0.31, Mild 0.48, Moderate 0.57, Severe 0.65.
- **It transfers:** the external model carries real EF information into the printouts it never saw.
- **But it is weaker** than the in-house representation on every contrast.

## Ensemble (ULT2 = ULT + EchoNext members)
| Model | Accuracy [95 % CI] | Recall N / Mi / Mo / S | Admissible |
|---|---|---|---|
| ULT-stack (reproduced exactly) | 0.571 [0.542, 0.600] | 0.905 / 0.164 / 0.420 / 0.286 | yes |
| **ULT2-stack (declared primary)** | 0.569 [0.538, 0.599] | 0.898 / 0.151 / 0.439 / 0.278 | yes (Mild 0.151) |
| ULT2-vote | 0.561 [0.536, 0.586] | 0.879 / 0.180 / 0.401 / 0.304 | yes |
| ULT2-select | 0.550 [0.512, 0.588] | 0.889 / 0.144 / 0.392 / 0.253 | no |
| M13, EchoNext-Mini outputs + LR, alone | 0.486 [0.436, 0.536] | 0.718 / 0.105 / 0.360 / 0.531 | no |

- **ULT2-stack details:** precision 0.701 / 0.264 / 0.459 / 0.367; per-repeat accuracy 0.576 / 0.558 / 0.572.
- **Key test, ULT2-stack vs ULT-stack:** −0.002 (p 0.73; Holm 1.00).
- **Other tests:** vs C3 +0.008 (Holm 1.00); vs IMG-C3 +0.010 (Holm 1.00).
- **Greedy selection weights:** the EchoNext members receive 0.11 (M14 0.10, M13 0.01). The EF-ordinal member keeps
  0.47.

## Interpretation
- **No added value:** EchoNext-Mini adds no accuracy. Its EF information on these printouts is a weaker version of
  what the in-house ECGFounder-based models already extract from the same 837 ECGs.
- **Likely causes:**
  1. domain shift: digital simultaneous 10-s ECGs vs a reconstruction from 2.5-s printed columns;
  2. a binary ≤ 45 % target that does not resolve the 41–52 % Mild band;
  3. no age or sex input.
- **What remains:** the plateau of about 0.57 holds with outside knowledge too. What is left is fine-tuning such a model
  on many more labelled ECGs (`DATA_REQUEST.md`).
