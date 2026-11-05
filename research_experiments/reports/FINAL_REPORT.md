# ECG-based 4-class LVEF prediction: controlled experimental improvement — final report

Branch `research/ecg-ef-model-improvements`, created from `c2487308` (baseline branch
`claude/funny-gauss-udlmv0`). Everything was run on CPU (4 cores, no GPU). Nothing has been
committed or pushed.

**Evaluation protocol for every number below:**
- The **original patient-level split** is used: 669 train / 168 validation / 210 test ECGs from
  635 / 159 / 199 patients. No patient appears in more than one split.
- Model selection, thresholds and fusion weights use **validation data only**.
- The test set is used once per model.
- 95 % CIs come from a **patient-cluster bootstrap** (2,000 resamples).
- Differences are tested with a paired patient-cluster bootstrap plus an exact McNemar test.
  Holm correction is applied within each metric family over the 11 pre-specified comparisons.

> **Correction to the brief.** The project has **1,047 labelled ECGs** (993 patients). The figure
> 669 is the *training* set. The test set is 210 ECGs (96 Normal / 47 Mild / 44 Moderate /
> 23 Severe).

---

## 0. Executive answers

| Question | Answer |
|---|---|
| **Did ECGFounder improve performance?** | **Yes, against the image CNN; no, against the current default model on the fixed test set.** <br>• Fine-tuned ECGFounder (3-seed ensemble) vs the ResNet-18 image CNN: balanced accuracy **+10.8 points** (95 % CI +1 to +21), macro-F1 **+0.125** (+0.03 to +0.23), Mild F1 **+0.17**, QWK +0.23. These are nominal p = 0.022 / 0.012 and not significant after Holm correction (0.24 / 0.13). <br>• In 5-fold patient-grouped CV over all 1,047 ECGs, frozen ECGFounder (accuracy 0.530 ± 0.030, balanced accuracy 0.449 ± 0.038) outperforms the header model (0.465 ± 0.044 / 0.401 ± 0.051): +6.6 accuracy points (95 % CI +3.5 to +9.8, p < 0.001) for the fusion, which is ~90 % ECGFounder. <br>• Against the default fusion (header + ResNet-18, accuracy 0.552), no ECGFounder model or fusion has higher test accuracy (best 0.524–0.529). They are better on Mild F1, QWK, HF AUC and error severity, but none of these differences is significant. |
| **Did ordinal classification improve the classification of adjacent EF categories?** | **Partly, and at a cost.** <br>• Ordinal heads raise the *share* of errors that are adjacent: frozen ECGFounder 65 % → 71 % (cumulative link) → 82 % (CORAL). They reduce ≥ 2-category errors (37 → 32 → 23). <br>• But exact accuracy falls in every input family (−1.4 to −8.6 points), and balanced accuracy does not improve (−4.8 to +0.7 points). <br>• The ordinal image CNN is significantly *worse* in accuracy (−8.6 points, p = 0.009, Holm 0.099). <br>• CV confirms: no gain in accuracy or balanced accuracy. |
| **Did Mild EF performance improve?** | **Yes in point estimates; not statistically convincing.** <br>• Mild F1: 0.154 (default) → 0.268–0.282 (ECGFounder models). <br>• Mild recall: 0.106 → 0.23–0.26. <br>• Share of Mild ECGs called Normal: 47 % → 21–28 %. <br>• Mild one-vs-rest AUROC: 0.531 → 0.598 (A2). <br>• All Mild-F1 differences have p ≥ 0.07; Mild remains the weakest class (AUROC ≤ 0.61). |
| **Did overall balanced accuracy improve?** | **Marginally at best.** <br>• Best balanced accuracy on test: A2 3-seed ensemble, 0.464 [0.39–0.54], vs default 0.454 [0.38–0.53]: +1.0 point. <br>• Against the image CNN alone: +10.8 points. |
| **Are the improvements statistically convincing?** | **No, not on the fixed 210-ECG test set:** no comparison survives Holm correction. <br>The one convincing result is in **cross-validation**: ECGFounder-based models beat the header model (p < 0.001 for accuracy, p = 0.017 for balanced accuracy). The default model could not be cross-validated on CPU, so the decisive comparison is still missing. |
| **Which model should become the primary model?** | **Keep the current default (header + ResNet-18) as the primary model for 4-class accuracy for now.** No candidate beats its 55.2 % with statistical support. <br>Adopt the **fine-tuned ECGFounder (3-seed ensemble, waveform from the PDF)** as the **primary research candidate**. It has the best balanced accuracy, macro-F1, QWK, HF AUC, within-one accuracy and fewest ≥ 2-category errors, and needs neither the image nor the machine text. Switch only if the cross-validation in §12 confirms it. |
| **What experiments are still necessary before publication?** | See §12. Most important: 5-fold patient-grouped CV of **all** models, including the image CNN and fine-tuned ECGFounder (GPU); temporal validation; label cleaning / EF regression (needs your permission for the EF values); external validation. |

---

## 1. Original baseline and reproducibility (Phase 0)

### 1.1 Git and data integrity

| Check | Result |
|---|---|
| Original branch / HEAD | `claude/funny-gauss-udlmv0` @ `c2487308f6221c0a5132e2f9fd5a7073b30bb7e7`, working tree clean, nothing stashed |
| New branch | `research/ecg-ef-model-improvements`, created from the same commit |
| Protected files hashed (SHA-256) | **3,365**: `LV_ECG_Dataset_*` 1,051 (1,047 PDFs + 4 spreadsheets, hashed as raw bytes and never opened), `data/` 2,098, `results/` 203, `models/` 13. File: `results/integrity_before.json` |
| Split | `data/manifest.csv` equals `results/split.csv`; 0 patients in more than one split |
| Class counts (train / val / test) | Normal 307 / 77 / 96 · Mild 148 / 37 / 47 · Moderate 141 / 36 / 44 · Severe 73 / 18 / 23 |
| Privacy | The manifest's file-path column (it contains patient names) is dropped on load and never written. ECGs are identified only by `year_timestamp`. |

### 1.2 Pipeline identified

- Default model = **0.6 × header logistic regression + 0.4 × ResNet-18** on grid-filtered images
  (`results/best_model.json`).
- ResNet-18 recipe (`ecg_ef/train.py`):
  - 256×512 grayscale input, random init;
  - AdamW, lr 1e-3, wd 0.05, warm-up + cosine, 30 epochs, batch size 16;
  - EMA 0.99, inverse-frequency class weights, label smoothing 0.05;
  - in-memory augmentation;
  - selection score = mean(macro-F1, QWK, HF AUC) after epoch 8, patience 10.
- Header model (`ecg_ef/header.py`): 110 features (age, sex, 14 machine measurements, 4 verdicts,
  90 statements), standardised multinomial logistic regression, `class_weight=balanced`, C chosen
  on validation.

### 1.3 Reproduction

| Model | Stored test (acc / bal / mF1 / QWK) | Re-computed from saved checkpoints | Max \|Δprob\| |
|---|---|---|---|
| ResNet-18 image | 0.486 / 0.356 / 0.323 / 0.290 | identical | 6.2e-7 |
| Header logistic regression | 0.514 / 0.442 / 0.421 / 0.422 | identical; re-fitting from scratch gives the same C (0.01) and identical probabilities (Δ = 0) | 5.0e-7 |
| **Default fusion** | **0.552 / 0.454 / 0.433 / 0.447** | identical | 8.7e-7 |

The reported figures (55.2 % accuracy, 45.4 % balanced accuracy, macro-F1 0.43, weak Mild) are
**confirmed exactly**.

**Training reproducibility (Phase 0b).** The ResNet-18 was retrained with the **unchanged
original code** (`ecg_ef/train.py`), the original arguments (read from the checkpoint) and
seed 42. Only the output folders were redirected into `research_experiments/`.

| | Original | Retrain |
|---|---|---|
| Best epoch | 14 | 14 |
| Test accuracy / balanced acc. / macro-F1 / QWK | 0.486 / 0.356 / 0.323 / 0.290 | 0.481 / 0.347 / 0.310 / 0.301 |
| Same predicted class | – | 94.8 % of test ECGs |
| Max \|Δprob\| | – | 0.097 |

Training reproduces to within about 1 point, but is not bit-exact. This is expected from
multi-threaded CPU arithmetic and newer library versions. Training time was 52.9 min, inflated
by concurrent analysis jobs (original: 23.5 min). Single-run differences of about 1 point
between CNN trainings are therefore noise.

---

## 2. ECGFounder: investigation and implementation (Experiment A)

### 2.1 What ECGFounder is (verified from the paper, code and checkpoint)

| Item | Finding |
|---|---|
| Publication | Li J. et al., *An Electrocardiogram Foundation Model Built on over 10 Million Recordings*, **NEJM AI 2025;2(7):AIoa2401033** (arXiv 2410.04133) |
| Code | github.com/PKUDigitalHealth/ECGFounder (commit `04edac70`), **MIT licence** |
| Checkpoint | Hugging Face `PKUDigitalHealth/ECGFounder` (revision `d9b17939`), `12_lead_ECGFounder.pth`, 370 MB, SHA-256 `ee199f37…a7997` (matches the LFS hash), MIT. A 1-lead (lead I) model also exists; it is not usable here because only 2.5 s of lead I is printed. |
| Architecture | `Net1D`, a RegNet-style 1-D CNN: stem conv k16/s2 + 7 stages (filters 64-160-160-400-400-1024-1024, blocks 2-2-2-3-3-4-4), grouped conv k16 (group width 16), squeeze-excitation, Swish activations, **no BatchNorm / dropout** in the released configuration, global average pooling, 1,024-d embedding, 150-label head (machine-statement labels). 30.8 M parameters. |
| Input | **12 × 5,000 samples = 10 s at 500 Hz**, lead order I, II, III, aVR, aVL, aVF, V1–V6 |
| Pre-processing (required by the authors) | 50 Hz notch (Q 30), 4th-order Butterworth band-pass 0.67–40 Hz (`filtfilt`), 0.4-s median-filter baseline removal, z-score |
| Accepts images? | **No**, raw waveforms only |
| Safe loading | `torch.load(weights_only=True)`; only NumPy's scalar/dtype types are allow-listed (the file stores one NumPy scalar). No arbitrary unpickling. |

### 2.2 Compatibility with this project's waveforms

The PDFs are vector drawings, and `ecg_ef/signals.py` already recovers them exactly at 500 Hz.
**The printout layout, however, is 3×4 + rhythm**: each lead is shown for only 2.5 s, the four
columns are *successive* time windows, and only lead II spans 10 s. I verified this on all
1,047 ECGs:

| Check | Median r |
|---|---|
| lead II (column 1) vs rhythm strip 0–2.5 s | 0.9998 |
| Einthoven III vs II − I | 0.993 |
| 2(aVF − aVR)/3 (column 2) vs rhythm 2.5–5 s | **0.997** (vs 0–2.5 s: −0.04) |

So the rhythm strip is simultaneous with the grid, and the limb leads can be completed over
0–5 s with Einthoven/Goldberger identities. Three input constructions were built **in memory**
(`ecgfounder/inputs.py`):

- **aligned:** each lead at its true time position; lead II = full 10-s strip; I/III/aVR/aVL/aVF
  completed over 0–5 s; zeros where nothing was recorded.
- **tiled:** each 2.5-s segment repeated 4×.
- **short:** 12 × 1,250, as the existing 1-D CNN uses.

**Documented deviations** from the authors' preprocessing:
- filtering is applied per recorded segment;
- the z-score uses recorded samples only.

**Label-free domain check.** I compared ECGFounder's *pre-trained* statement outputs with the
statements the ECG machine printed on the same PDF (14 shared statements, ≥ 15 positives each):

| Construction | Median AUROC vs printed statements |
|---|---|
| **aligned** | **0.887** (LBBB 0.99, RBBB 0.99, LAFB 0.98, sinus tachycardia 0.97, AF 0.88) |
| tiled | 0.765 |
| short | 0.767 |

ECGFounder therefore works on the digitised printouts. **aligned** was pre-specified as the
primary input using this check alone, without EF labels. Validation agreed: it had the best
A1 validation score of the 9 construction × feature-set variants (0.612).

### 2.3 A1: frozen feature extraction

- ECGFounder frozen; 1,024-d embedding (14 ms/ECG on CPU).
- Standardised linear softmax head (4,100 parameters), class-weighted cross-entropy, L2 chosen on
  validation from {1e-3 … 3}, trained with L-BFGS in about 1 s.

### 2.4 A2: partial fine-tuning

- Start from the same checkpoint; **stages 5–6 unfrozen** (26.7 M of 30.7 M parameters), new
  dropout(0.3) + linear head.
- The baseline recipe otherwise: AdamW (backbone lr 1e-4, head lr 1e-3), wd 0.05, 1-epoch warm-up
  + cosine, ≤ 15 epochs, EMA 0.99, class-weighted cross-entropy with label smoothing 0.05,
  grad-clip 5, selection by the validation score after epoch 3, patience 5.
- In-memory augmentation: per-lead gain ±10 %, Gaussian noise 0.02 on recorded samples.
- Seeds 42, 43, 44; the pre-specified A2 estimate is the **3-seed probability average**.
- Ablation: stage 6 only (13.7 M trainable).

| Model (test, n = 210) | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | QWK | Macro AUROC | HF AUC |
|---|---|---|---|---|---|---|---|
| ResNet-18 image (existing baseline) | 0.486 [0.42–0.55] | 0.356 [0.30–0.42] | 0.323 | 0.109 | 0.290 | 0.652 | 0.735 |
| A1 frozen ECGFounder | 0.500 [0.44–0.57] | 0.421 [0.35–0.49] | 0.412 | 0.265 | 0.464 | 0.668 | 0.810 |
| A2 fine-tuned, seed 42 / 43 / 44 | 0.510 / 0.500 / 0.519 | 0.438 / 0.440 / 0.471 | 0.425 / 0.425 / 0.453 | 0.238 / 0.295 / 0.321 | 0.468 / 0.452 / 0.485 | 0.676 / 0.673 / 0.688 | 0.809 / 0.811 / 0.815 |
| A2 mean ± SD over seeds | 0.510 ± 0.010 | 0.450 ± 0.019 | 0.434 ± 0.016 | 0.285 ± 0.043 | 0.468 ± 0.017 | 0.679 | 0.812 |
| **A2 3-seed ensemble** | **0.524 [0.46–0.59]** | **0.464 [0.39–0.54]** | **0.448** | 0.282 | **0.519** | 0.680 | 0.812 |
| A2 stage 6 only (seed 42) | 0.510 | 0.441 | 0.427 | 0.225 | 0.455 | 0.677 | 0.812 |

**Feature extraction vs fine-tuning:**
- Fine-tuning adds +2.4 accuracy and +4.4 balanced-accuracy points over the frozen model (both
  n.s.).
- Unfreezing one stage or two makes no difference.
- Seed-to-seed spread (SD ≈ 1–2 points) is far smaller than the test CI (±7 points).

---

## 3. Ordinal classification (Experiment B)

### 3.1 Implementation

- **Cumulative link (proportional odds):** `ordinal_layer.py`, `heads.py`. One severity score s;
  P(y ≤ k) = σ(t_k − s), with t₁ < t₂ < t₃ enforced by construction (t_k = t₁ + Σ softplus).
  Class probabilities are differences of the cumulative probabilities. They are non-negative and
  monotone by construction, so **0 inconsistent predictions** are possible. Loss: class-weighted
  NLL. Decoding: argmax.
- **CORAL** (Cao et al. 2020): shared weights, 3 biases, sum of binary cross-entropies, rank
  decoding (count of P(y > k) > 0.5). Rank consistency was checked: **0 inconsistent samples** on
  validation and test.
- **Comparators** with identical inputs, optimiser, regularisation grid and selection rule:
  multinomial softmax (the baseline loss) and focal loss (γ = 2).
- On validation, the cumulative link beat CORAL on both feature sets (0.634 vs 0.596 ECGFounder;
  0.555 vs 0.491 header). It was therefore chosen as the ordinal head for the deep models (A2,
  ResNet-18).

### 3.2 Results (test)

| Input | Head | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | QWK | Adjacent share of errors | ≥ 2-category errors |
|---|---|---|---|---|---|---|---|---|
| Header | multinomial / focal | 0.524 / 0.519 | 0.458 / 0.461 | 0.425 / 0.436 | 0.147 / 0.184 | 0.453 / 0.476 | 0.56 (sklearn baseline) | 45 |
| Header | **cumulative link** / CORAL | 0.462 / 0.414 | 0.422 / 0.382 | 0.409 / 0.373 | **0.247** / 0.310 | **0.480** / 0.348 | 0.63 | 42 |
| ECGFounder frozen | multinomial / focal | 0.500 / 0.510 | 0.421 / 0.437 | 0.412 / 0.428 | 0.265 / 0.265 | 0.464 / 0.471 | 0.65 | 37 |
| ECGFounder frozen | **cumulative link** / CORAL | 0.467 / 0.390 | 0.427 / 0.376 | 0.421 / 0.345 | **0.308** / 0.329 | 0.463 / 0.394 | 0.71 / **0.82** | 32 / **23** |
| ECGFounder fine-tuned (3-seed ens.) | CE / **cumulative link** | 0.524 / 0.510 | 0.464 / 0.417 | 0.448 / 0.393 | 0.282 / 0.225 | 0.519 / **0.523** | 0.67 / 0.62 | 33 / 39 |
| ResNet-18 image | CE / **cumulative link** | 0.486 / **0.400** | 0.356 / 0.342 | 0.323 / 0.302 | 0.109 / 0.118 | 0.290 / 0.325 | 0.44 / 0.52 | 60 / 61 |

Paired tests (ordinal minus multinomial; Δ accuracy / Δ balanced accuracy, p):

| Input | Δ accuracy | Δ balanced acc. |
|---|---|---|
| Header | −6.2 points (p = 0.08) | −3.5 points (p = 0.34) |
| ECGFounder frozen | −3.3 (p = 0.42) | +0.7 (p = 0.88) |
| ECGFounder fine-tuned | −1.4 (p = 0.70) | −4.8 (p = 0.17) |
| Image CNN | **−8.6 (p = 0.009, Holm 0.099)** | −1.4 (p = 0.67) |

5-fold CV (§5.2): the cumulative link vs multinomial on ECGFounder gives −4.9 accuracy points
(p = 0.002) and −1.9 balanced-accuracy points (n.s.).

**Interpretation:**
- Ordinal losses pull predictions toward the middle classes. This raises Mild recall and makes
  errors more adjacent; CORAL is most extreme (49 % Mild recall, but only 44 % Normal F1).
- Exact accuracy drops because many Normal ECGs move into Mild.
- With class weights and so few Severe cases, the ordinal models never gained overall.
- On the image CNN, the ordinal head reached the same validation score as cross-entropy (0.542 vs
  0.541, best epoch 21 vs 14) but generalised worse to the test set. Validation could not tell the
  two heads apart.

---

## 4. Regression feasibility (Experiment C): **not executed**

`regression_feasibility.py` checked all authorised data:
- manifest, header features, waveform features, split file: no EF-like column;
- the text layer of all 1,047 PDFs: 0 files with EF / LVEF / ejection / fraction tokens;
- the bracketed file-name fields: all empty.

The labels are **folder categories only**. The continuous EF values exist only in the matching
spreadsheets (`LV_ECG_Dataset_*/*.xlsx`), which were **not opened** because you didn't give
permission.

In line with your instructions, **no regression was trained and no EF values were invented**.

**Data requirement:**
- permission to read the EF column (and ideally the echo date, for label cleaning) from those
  spreadsheets;
- they would need joining to the ECGs by timestamp only, without names.

The pipeline is ready to add an EF-regression head with sex-specific decoding.

---

## 5. Experimental comparison

### 5.1 Main table (original split, test n = 210; 95 % patient-cluster bootstrap CI)

| Model | Input | Accuracy | Balanced accuracy | Macro-F1 | Normal F1 | Mild F1 | Moderate F1 | Severe F1 | Macro AUROC | QWK | HF AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 image (B0) | image | 0.486 [0.42–0.55] | 0.356 [0.30–0.42] | 0.323 [0.26–0.39] | 0.689 | 0.109 | 0.250 | 0.246 | 0.652 | 0.290 | 0.735 |
| ResNet-18 image, exact retrain (R0) | image | 0.481 | 0.347 | 0.310 | – | 0.109 | – | – | 0.650 | 0.301 | 0.731 |
| Header LR (B0) | header | 0.514 [0.44–0.58] | 0.442 [0.37–0.51] | 0.421 [0.35–0.49] | 0.716 | 0.160 | 0.471 | 0.339 | 0.684 | 0.422 | 0.795 |
| **Default: header + ResNet-18 (B0)** | header + image | **0.552 [0.48–0.62]** | 0.454 [0.38–0.53] | 0.433 [0.36–0.50] | 0.741 | 0.154 | **0.468** | 0.370 | 0.688 | 0.447 | 0.791 |
| A1 frozen ECGFounder | waveform | 0.500 [0.44–0.57] | 0.421 [0.35–0.49] | 0.412 [0.35–0.48] | 0.728 | 0.265 | 0.315 | 0.340 | 0.668 | 0.464 | 0.810 |
| **A2 fine-tuned ECGFounder, 3-seed ens.** | waveform | 0.524 [0.46–0.59] | **0.464 [0.39–0.54]** | **0.448 [0.38–0.52]** | 0.743 | **0.282** | 0.345 | **0.421** | 0.680 | **0.519** | 0.812 |
| A2 ordinal (cum. link), 3-seed ens. | waveform | 0.510 [0.44–0.58] | 0.417 [0.35–0.49] | 0.393 [0.33–0.46] | **0.761** | 0.225 | 0.278 | 0.308 | 0.690 | 0.523 | 0.790 |
| ResNet-18 ordinal (cum. link) | image | 0.400 [0.33–0.46] | 0.342 [0.27–0.41] | 0.302 [0.24–0.36] | 0.636 | 0.118 | 0.197 | 0.256 | 0.618 | 0.325 | 0.715 |
| Header cum. link | header | 0.462 [0.39–0.52] | 0.422 [0.35–0.49] | 0.409 [0.34–0.47] | 0.656 | 0.247 | 0.353 | 0.379 | 0.642 | 0.480 | 0.790 |
| Header + A1 fusion (0.1 / 0.9) | header + waveform | 0.514 [0.45–0.58] | 0.437 [0.37–0.51] | 0.426 [0.36–0.49] | 0.737 | 0.268 | 0.322 | 0.377 | 0.674 | 0.483 | 0.815 |
| Header + A2 fusion (0.2 / 0.8) | header + waveform | 0.505 [0.44–0.57] | 0.445 [0.37–0.52] | 0.425 [0.35–0.49] | 0.741 | 0.238 | 0.322 | 0.400 | **0.695** | 0.497 | 0.819 |
| Header numbers only + A2 (0.4 / 0.6) | header (no text) + waveform | 0.510 [0.44–0.58] | 0.448 [0.37–0.52] | 0.426 [0.36–0.49] | 0.736 | 0.250 | 0.329 | 0.387 | **0.703** | 0.488 | **0.822** |
| Header + image + A1 (val-best, 0.1 / 0.2 / 0.7) | all three | 0.514 [0.45–0.58] | 0.428 [0.36–0.50] | 0.415 [0.35–0.48] | 0.738 | 0.234 | 0.310 | 0.377 | 0.683 | 0.481 | 0.805 |
| Header + image + A1, weights by val accuracy (0.2 / 0.5 / 0.3) | all three | 0.529 [0.46–0.60] | 0.420 [0.35–0.49] | 0.397 [0.32–0.46] | 0.735 | 0.125 | 0.400 | 0.327 | 0.689 | 0.432 | 0.786 |

**Notes on the table:**
- The full table, with every variant and ablation (41 runs), is in `results/comparison_main.md` /
  `.csv`.
- Per-class precision/recall: `results/per_class.csv`.
- Confusion matrices: `results/confusion_matrices.md` and `plots/cm_*.png`.
- Training curves: `plots/curves_*.png`.
- QWK CIs are in the CSV.

**Validation-selection caveat:**
- By the project's pre-specified selection score, the best model on validation is
  header + image + A1 (0.641 vs 0.612 for the default). On test, it is *worse* than the default
  in accuracy.
- Choosing weights by validation accuracy (secondary analysis, requested because accuracy is your
  main objective) also fails to beat 0.552.
- With 168 validation ECGs, validation differences of about 0.03 are within noise.

### 5.2 Patient-grouped 5-fold cross-validation (all 1,047 ECGs; secondary analysis)

**Leakage control:**
- StratifiedGroupKFold on the anonymous patient hash;
- inner patient-grouped validation for C/L2/fusion weights;
- statement vocabulary and imputation medians rebuilt per fold;
- ECGFounder features are label-free.

| Model (mean ± SD over 5 folds) | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | QWK | Macro AUROC | HF AUC |
|---|---|---|---|---|---|---|---|
| Header LR | 0.465 ± 0.044 | 0.401 ± 0.051 | 0.391 ± 0.047 | 0.232 ± 0.052 | 0.420 ± 0.030 | 0.645 ± 0.040 | 0.779 ± 0.007 |
| Header numbers only LR | 0.491 ± 0.007 | 0.422 ± 0.033 | 0.402 ± 0.021 | 0.197 ± 0.044 | 0.517 ± 0.044 | 0.694 ± 0.024 | 0.786 ± 0.023 |
| Header cum. link | 0.474 ± 0.012 | 0.418 ± 0.028 | 0.403 ± 0.022 | 0.268 ± 0.043 | 0.501 ± 0.065 | 0.678 ± 0.031 | 0.787 ± 0.037 |
| **ECGFounder A1 multinomial** | **0.530 ± 0.030** | **0.449 ± 0.038** | **0.442 ± 0.032** | 0.244 ± 0.028 | 0.521 ± 0.070 | 0.703 ± 0.039 | **0.826 ± 0.024** |
| ECGFounder A1 cum. link | 0.481 ± 0.051 | 0.430 ± 0.059 | 0.428 ± 0.055 | 0.272 ± 0.058 | 0.551 ± 0.109 | 0.705 ± 0.041 | 0.811 ± 0.041 |
| Header + A1 fusion | 0.531 ± 0.031 | 0.449 ± 0.038 | 0.442 ± 0.032 | 0.253 ± 0.020 | 0.527 ± 0.079 | 0.704 ± 0.039 | 0.825 ± 0.024 |
| **Header numbers only + A1 fusion** | **0.531 ± 0.027** | **0.452 ± 0.031** | 0.437 ± 0.025 | 0.231 ± 0.044 | 0.542 ± 0.084 | **0.720 ± 0.021** | **0.828 ± 0.023** |
| Ordinal header + A1 fusion | 0.499 ± 0.036 | 0.447 ± 0.038 | 0.437 ± 0.031 | **0.277 ± 0.057** | **0.558 ± 0.083** | 0.706 ± 0.032 | 0.816 ± 0.036 |

**Paired tests on the pooled out-of-fold predictions (n = 1,047):**
- Header + A1 vs header: accuracy **+6.6 points [+3.5, +9.8], p < 0.001**; balanced accuracy
  **+4.8 [+1.0, +8.6], p = 0.017**; better in 4 of 5 folds.
- Numbers-only header + A1 vs numbers-only header: accuracy +4.0 [+1.4, +6.6], p < 0.001;
  balanced accuracy +3.0 (p = 0.08).
- Ordinal vs multinomial (ECGFounder): accuracy −4.9 (p = 0.002).
- Ordinal vs multinomial (header): +0.9 (n.s.).

**Not cross-validated (CPU cost):** the image CNN, the default fusion and fine-tuned ECGFounder.

Note that the header model scores **lower in CV (0.465) than on the fixed test split (0.514)**.
The fixed test split seems comparatively favourable to the header model. This is one reason the
test-split comparison and the CV comparison disagree.

---

## 6. Class-wise performance (test)

| Model | Normal P / R | Mild P / R | Moderate P / R | Severe P / R | OvR AUROC N / Mi / Mo / S |
|---|---|---|---|---|---|
| Default (header + ResNet-18) | 0.648 / **0.865** | 0.278 / 0.106 | **0.545** / **0.409** | 0.323 / 0.435 | 0.791 / 0.531 / **0.725** / 0.704 |
| ResNet-18 image | 0.568 / 0.875 | 0.375 / 0.064 | 0.400 / 0.182 | 0.206 / 0.304 | 0.735 / 0.527 / 0.682 / 0.664 |
| Header LR | 0.686 / 0.750 | 0.214 / 0.128 | 0.488 / 0.455 | 0.278 / 0.435 | 0.795 / 0.532 / 0.689 / **0.720** |
| A1 frozen ECGFounder | 0.717 / 0.740 | 0.306 / 0.234 | 0.311 / 0.318 | 0.300 / 0.391 | 0.810 / 0.560 / 0.632 / 0.671 |
| A2 3-seed ens. | **0.747** / 0.740 | 0.316 / **0.255** | 0.349 / 0.341 | **0.353** / **0.522** | **0.812** / **0.598** / 0.626 / 0.687 |
| A2 ordinal 3-seed ens. | 0.716 / 0.812 | **0.333** / 0.170 | 0.314 / 0.250 | 0.238 / 0.435 | 0.790 / 0.551 / 0.722 / 0.696 |

**Pattern:**
- ECGFounder is better than the header/image models on **Normal precision, Mild and Severe**.
- It is **worse on Moderate**, where the header measurements are strongest (Moderate F1 0.47 vs
  0.32–0.35).
- This complementarity is what fusion should exploit. With validation-chosen weights it did not,
  on this test set.

Confusion matrices (rows = true N / Mi / Mo / S):

| Model | N | Mi | Mo | S |
|---|---|---|---|---|
| Default | [83 6 3 4] | [22 5 10 10] | [14 5 18 7] | [9 2 2 10] |
| A2 3-seed ens. | [71 13 8 4] | [12 12 16 7] | [7 11 15 11] | [5 2 4 12] |

---

## 7. Mild EF analysis (47 Mild test ECGs)

| Model | Mild predicted (n) | Mild recall | Mild precision | Mild OvR AUROC | Mild-vs-Normal AUROC | Mild → Normal | Mild → Moderate | Mild → Severe |
|---|---|---|---|---|---|---|---|---|
| Image CNN | 8 | 0.064 | 0.375 | 0.527 | 0.684 | **57 %** | 11 % | 26 % |
| Header LR | 28 | 0.128 | 0.214 | 0.532 | 0.759 | 36 % | 26 % | 26 % |
| **Default fusion** | 18 | 0.106 | 0.278 | 0.531 | 0.755 | **47 %** | 21 % | 21 % |
| A1 frozen ECGFounder | 36 | 0.234 | 0.306 | 0.560 | 0.775 | 26 % | 36 % | 15 % |
| A1 + cum. link | 57 | 0.340 | 0.281 | 0.494 | 0.762 | 19 % | 34 % | 13 % |
| A1 + CORAL | 93 | **0.489** | 0.247 | 0.491 | 0.756 | 9 % | 40 % | 2 % |
| **A2 3-seed ens.** | 38 | 0.255 | **0.316** | **0.598** | **0.788** | 26 % | 34 % | 15 % |
| Header + A2 fusion | 37 | 0.213 | 0.270 | **0.603** | **0.793** | 21 % | 36 % | 21 % |

**Findings:**
1. The default model hides Mild almost completely: 47 % of Mild ECGs are called Normal, and only
   18 ECGs in the whole test set are predicted Mild.
2. ECGFounder roughly **halves Mild → Normal misses** (47 % → 21–26 %) and **doubles Mild
   recall** while also improving Mild precision.
3. Mild is still not separable from its neighbours: OvR AUROC ≤ 0.61. The improvement is mostly
   better separation from Normal (Mild-vs-Normal AUROC 0.755 → 0.788–0.793); Mild ECGs now drift
   to Moderate instead.
4. Mild recall can be raised to 49 % with CORAL, but only by over-calling Mild (93 predictions
   for 47 cases) and losing Normal accuracy. That is a threshold trade-off, not better
   discrimination.
5. **By sex:** Mild recall in females is 0.00 for the image CNN and 0.17 for the default;
   ECGFounder models reach 0.25. The female Normal/Mild cut-off is 54 % vs 52 % for males, but
   there are only 12 female Mild test ECGs, so this is unreliable.

---

## 8. Error analysis

| Model | Exact | Off by 1 | Off by 2 | Off by 3 | Mean \|class error\| | Normal → Mod/Sev | Severe → Normal | Under- / over-grading |
|---|---|---|---|---|---|---|---|---|
| Image CNN | 102 | 48 | 42 | 18 | 0.886 | 10 | 12 | 70 / 38 |
| Header LR | 108 | 57 | 32 | 13 | 0.762 | 13 | 5 | 48 / 54 |
| Default fusion | **116** | 52 | 29 | 13 | 0.710 | **7** | 9 | 54 / 40 |
| A1 frozen | 105 | 68 | 27 | 10 | 0.724 | 12 | 6 | 46 / 59 |
| **A2 3-seed ens.** | 110 | 67 | **24** | **9** | **0.676** | 12 | **5** | 41 / 59 |
| A2 ordinal ens. | 107 | 64 | 31 | 8 | 0.714 | 10 | 5 | 45 / 58 |
| A1 + CORAL | 82 | 105 | **18** | **5** | 0.743 | 11 | 4 | 41 / 87 |

**Findings:**
- **Clinically dangerous misses:** Severe called Normal drops from 9 (default) and 12 (image) to
  5–6 with ECGFounder.
- **Over-calls:** Normal called Moderate/Severe stays around 7–13.
- **Grading direction:** the default under-grades (54 under vs 40 over); ECGFounder models
  over-grade (≈ 41 under vs 59 over). For a screening tool, over-grading is the safer direction.
- **Within-one-category accuracy:** 0.800 (default) vs 0.843 (A2 ensemble).
- **By year:** see `results/subgroup_analysis.csv`. The default ranges 52–63 %, A2 50–55 %; the
  2024 test year remains hardest for the frozen model (41 %).

---

## 9. Computational cost (CPU, 4 threads)

**End-to-end inference per ECG PDF** (batch 1, median of 28 test PDFs,
`results/timing_end_to_end.json`):

| Step | ms |
|---|---|
| Render + grid-filter image | 145 |
| ResNet-18 forward | 54 |
| Header text parse + model | 46 |
| Waveform extraction from PDF vectors | 60 |
| ECGFounder input build (filter, align, z-score) | 9 |
| ECGFounder forward | 51 |

| Pipeline | ms per ECG |
|---|---|
| Image CNN | 199 |
| **Default (header + image)** | **245** |
| ECGFounder only | **120** |
| Header + ECGFounder | 166 |

The ECGFounder pipelines are **faster** than the default, because image rendering is the
slowest step.

| Model | Total params | Trainable params | Training time | Checkpoint / weights |
|---|---|---|---|---|
| ResNet-18 image (baseline) | 11.17 M | 11.17 M | 23.5 min (original log) | 44.8 MB |
| ResNet-18 ordinal | 11.17 M | 11.17 M | 37.3 min | 42.7 MB |
| Header LR | 444 | 444 | < 1 s | < 1 MB |
| A1 frozen ECGFounder + linear head | 30.8 M (frozen) + 4.1 k | 4,100 | feature extraction 15 s for 1,047 ECGs + head ≈ 1 s | 117 MB fp32 (public checkpoint) + 16 KB head |
| A2 fine-tuned (stages 5–6) | 30.7 M | 26.7 M | 4.7–5.2 min per seed (seed 42: 10.0 min, slowed by a concurrent job) | 117 MB fp32 for deployment (3 seeds: 3 × 102 MB trainable part + shared frozen part) |
| A2 stage 6 only | 30.7 M | 13.7 M | 4.8 min | 117 MB fp32 for deployment (trainable part 52 MB) |

ECGFounder fine-tuning on these 669 ECGs is **cheaper than training the image CNN** (≈ 5 vs
≈ 24 min), because the frozen part has no gradient and the network converges in 5–9 epochs.

---

## 10. Limitations and failed experiments

**Limitations:**
1. **Small test set (n = 210).** Accuracy CIs are about ±7 points, so differences under about
   7 points cannot be confirmed on the fixed split. 41 runs were evaluated on the same test set;
   only 11 comparisons were pre-specified and tested, with Holm correction. Other differences in
   the tables are descriptive.
2. **Small validation set (n = 168).** Validation-optimal choices (fusion weights, the val-best
   three-way fusion) did not transfer to the test set. All weights chosen on validation should be
   treated as noisy.
3. **Domain shift for ECGFounder.**
   - It was pre-trained on simultaneous 10-s 12-lead ECGs; here 8 leads have ≤ 5 s of signal and
     zero-filled gaps.
   - It was pre-trained on US data; this cohort is Indian, single-centre, single device.
   - The label-free check shows good compatibility, but performance could improve with full
     simultaneous leads (other ECG export formats).
4. **Label noise:** name-matched ECG–echo pairs with long intervals (see Document 3) limit all
   models, especially at the Normal/Mild boundary.
5. **Regularisation grid edge:** the header heads' best L2 (3.0), and the original header model's
   C = 0.01, lie at the edge of their grids. All heads shared the same grid, so the comparisons
   are fair, but the header models may be slightly under-regularised.
6. **CPU-only:** cross-validation of the CNN and fine-tuned models, larger seed counts and full
   fine-tuning were not affordable.
7. **Timing contamination:** training times for A2 seed 42 and the baseline retrain were
   inflated by concurrent jobs. The other timings come from runs that had the CPU to themselves.

**Failed, negative or not completed:**

| Experiment | Status |
|---|---|
| C – EF regression | Not executed: no continuous EF in the authorised data (§4) |
| Ordinal image CNN | Ran; **negative**: −8.6 accuracy points vs cross-entropy (p = 0.009) |
| CORAL heads | Ran; negative for accuracy (−11 / −11 points); best adjacent-error profile |
| Focal loss | Ran; no gain (header +0.3 bal. points, ECGFounder +1.6, n.s.) |
| Fusion improving on the default's accuracy | Not achieved (best 0.529 vs 0.552) |
| CV of the image CNN / default / A2 | Not run (CPU cost, about 2 h per model); **essential next step** |
| Full fine-tuning of all ECGFounder stages | Not run; stage 6 vs stages 5–6 showed no depth effect |
| 1-lead ECGFounder | Not applicable (only 2.5 s of lead I is printed) |
| Augmentation on/off ablation for A2 | Not run |
| Calibration / decision-curve analysis | Not run |

---

## 11. Research contribution of each investigated change

| Change | Contribution | Evidence strength |
|---|---|---|
| **ECGFounder transfer from exact PDF-vector waveforms with Einthoven-completed, time-aligned input** | Shows that a public foundation model can be applied to **3×4 PDF printouts**; a label-free agreement check validates it (median AUROC 0.887 vs printed statements). Replaces the image CNN with a stronger waveform branch: balanced accuracy +6.5 to +10.8 points, Mild F1 ×2.5, QWK +0.17 to +0.23, fewer Severe → Normal misses. Cheaper to train than the image CNN. | Consistent across seeds and in CV vs the header model (p < 0.001); vs the image CNN nominal p = 0.01–0.02, not Holm-significant |
| Partial vs frozen fine-tuning | Small, non-significant gain (+2.4 accuracy / +4.4 balanced-accuracy points); depth of unfreezing irrelevant | Weak |
| Ordinal classification (cumulative link, CORAL) | **Negative result:** consistent ordinal outputs and more adjacent errors, but no gain, and often a loss, in accuracy / balanced accuracy at n ≈ 700 | Moderate (all input families, CV) |
| Header ablation | **The header's value is in the numeric measurements and demographics, not the machine's text.** Numbers only: bal. 0.449 vs 0.442 full; text only: 0.379. CV: numbers-only ≥ full header. Answers the "the model reads the machine diagnosis" objection. | Moderate |
| Accuracy-targeted fusion weights | Did not beat the default; shows validation-selection noise | – |
| EF regression | Data requirement documented; no result | – |

---

## 12. Recommendations for the next research stage

In priority order:

1. **5-fold patient-grouped CV of every headline model, including the image CNN, the default
   fusion and fine-tuned ECGFounder.**
   - Use the folds in `results/cv5_oof_predictions.csv` (stored by ECG id, so they can be reused).
   - This is the only way to settle "default vs ECGFounder" fairly. Needs a GPU: on this CPU it
     costs about 25 min per fold for the CNN, and Colab GPU is much faster.
2. **Pre-specify the primary endpoint** before the next experiments: accuracy, or balanced
   accuracy / macro-F1. The ranking of the candidates depends on it.
3. **Temporal validation:** train 2020–2022, test 2024.
4. **With your permission, use the spreadsheet EF values and echo dates** for:
   - an ECG–echo interval filter / label-cleaning sensitivity analysis;
   - EF regression with sex-specific decoding (Experiment C).
5. **Fusion fitted on out-of-fold predictions** (from step 1) instead of the 168-ECG validation
   set. It should exploit ECGFounder's Mild/Severe strength together with the header's Moderate
   strength.
6. **Calibration** (reliability curves, Brier score) and a **3-class ≤ 40 % secondary analysis**,
   clearly labelled.
7. **External validation**, e.g. the EchoNext LVEF ≤ 45 % binary task.

---

## 13. Final safety verification

Verified 2026-11-04 14:59 (`results/integrity_verify_before_20261104_145943.json`).

| Check | Result |
|---|---|
| 1. Original branch HEAD unchanged | `claude/funny-gauss-udlmv0` = `origin/claude/funny-gauss-udlmv0` = `c2487308f6221c0a5132e2f9fd5a7073b30bb7e7` (unchanged; no commits, no pushes, no stash) |
| 2. Original dataset files unchanged | SHA-256 of all 1,051 files in `LV_ECG_Dataset_*`: **0 changed, 0 missing, 0 added** |
| 3. Split / label / derived files unchanged | `data/` (manifest, signals, header/waveform features, 2,094 cached images), `results/` (incl. `split.csv`, 203 files), `models/` (13 checkpoints): **all 3,365 protected files identical** |
| 4. Code changes only on the new branch | Current branch `research/ecg-ef-model-improvements` (= `c2487308` + untracked files). `git diff HEAD` is **empty**: no tracked file was modified. The only change is the new untracked folder `research_experiments/`. No file outside it was created or modified during the session (checked by modification time, excluding `.git`). |
| 5. Outputs stored separately | All outputs are in `research_experiments/` (1.1 GB, mostly checkpoints and the ECGFounder download, git-ignored through `research_experiments/.gitignore`). `research_experiments/logs/` is also ignored by the repository's root `.gitignore` pattern `logs/`. |
| Spreadsheets | Hashed as raw bytes for the integrity check only; never opened or parsed |
| External downloads | ECGFounder code (cloned to the session scratchpad, outside the repository) and checkpoint (into `research_experiments/models/pretrained/`). No hospital data downloaded. No packages installed. |

**6. Files added, all under `research_experiments/`:**
- **Code (28):** `__init__.py`, `README.md`, `.gitignore`, `common.py`, `integrity.py`,
  `phase0_reproduce.py`, `reproduce_resnet18.py`, `collect_repro.py`, `heads.py`,
  `ordinal_layer.py`, `exp_feature_heads.py`, `train_deep.py`, `run_queue.sh`, `exp_fusion.py`,
  `header_ablation.py`, `cv_grouped.py`, `analysis_mild_errors.py`, `regression_feasibility.py`,
  `timing.py`, `make_tables.py`, and `ecgfounder/{__init__.py, net1d.py (verbatim MIT copy),
  LICENSE_ECGFounder, tasks_150_labels.txt, inputs.py, model.py, extract_features.py,
  sanity_check.py}`.
- **Generated:**
  - `configs/`: 9 JSON (8 run configs + environment/seeds/provenance);
  - `models/`: 15 files (7 A2 checkpoints, the ordinal ResNet-18, the retrain checkpoint,
    3 ECGFounder feature files, the pre-trained checkpoint + its HF README/config);
  - `results/`: 55 run folders + 24 summary tables/JSON, 200 files in total;
  - `plots/`: 64 PNG (confusion matrices, training curves, summary figures);
  - `logs/`: 20 run logs;
  - `reports/FINAL_REPORT.md`.
- **Modified files: none.**
- **Deleted files:** only my own smoke-test outputs (`smoke_test` run) inside
  `research_experiments/`.

**7. Experiments not completed:**
- Experiment C (no authorised EF values);
- cross-validation of the image CNN, default fusion and fine-tuned ECGFounder (CPU cost);
- full fine-tuning of all ECGFounder stages;
- augmentation ablation;
- calibration analysis.

See §10.
