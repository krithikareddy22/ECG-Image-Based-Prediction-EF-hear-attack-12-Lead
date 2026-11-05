# Phase 4H: fair evaluation and model benchmark

- **Code:** `phase_4/benchmark4.py`, `controls4.py`, `make_tables4.py`.
- **Full tables:** `results/phase4_tables.md`, `results/phase4_vs_controls.md`.
- **Filterable spreadsheet:** `results/phase4_results.xlsx`, with two sheets:
  - the 68 models, seed-combined;
  - all 360 evaluated runs: 162 trained runs (54 regression × 4 decoders, 72 multitask, 36 label-quality), 9 λ
    selections, 9 fusions and 18 benchmark re-evaluations.
- **Figures:** `plots/model_comparison_vs_vote.png` (forest plot), `plots/confusion_matrices_key_models.png`.

## 1. Protocol (pre-registered, `configs/PREREG_PHASE4.md`, committed before any run)
- **Data:** development set D, 837 ECGs / 794 patients.
- **Validation:**
  - patient-grouped StratifiedGroupKFold(5), repeats 0–2, **identical folds for every model** (Phases 2–4);
  - all preprocessing, feature standardisation, hyper-parameters, loss weights, decoding cut-points,
    recalibration and early stopping are chosen on inner-val inside each outer training portion;
  - the outer fold is used only for scoring.
- **Primary score:**
  - all 837 ECGs against **folder labels**, mean over the 3 repeats;
  - seeds are combined within a repeat: mean probabilities, or the ordinal median for hard EF decoders;
  - selection metric S3 = (accuracy + balanced accuracy + macro-F1) / 3.
- **Statistics:**
  - corrected repeated-CV t-test (Bouckaert–Frank; n_train / n_test = 535 / 167) on 15 matched (repeat, fold)
    cells;
  - Holm over the **26-model confirmatory family**: primary-decoder regression models, multitask models,
    label-quality models and fusions;
  - patient-cluster paired bootstrap (2,000 resamples, repeat 0) with exact McNemar;
  - corrected 95 % CIs.
- **"Genuine improvement":** better than the vote ensemble, Holm-adjusted p < 0.05 **and** the same sign in all 3
  repeats.
- **Locked test set:**
  - the 210-ECG test set was **not predicted and not scored** in Phase 4;
  - it was exposed in Phase 1 (about 40 models inspected), so it is no longer a clean hold-out;
  - an **external or temporally independent validation cohort is required** before any performance claim.

## 2. Headline results [VALID]
Means over repeats; all 837 ECGs; folder labels.

| Model | Runs | Accuracy | Balanced acc. | Macro-F1 | F1 Mild | F1 Severe | QWK | Macro AUROC | S3 [corrected 95 % CI] | ΔS3 vs vote (p; Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Vote ensemble** (Phase 3 benchmark) | 3 | 0.547 | 0.467 | **0.455** | 0.255 | 0.362 | 0.574 | 0.726 | 0.490 [0.449, 0.530] | – |
| MLP fusion (Phase 2) | 9 | 0.544 | 0.466 | 0.440 | 0.186 | 0.373 | 0.583 | – | 0.483 [0.442, 0.523] | −0.007 (0.65; –) |
| `lq_dropdisagree` (train without folder ≠ echo ECGs) | 9 | 0.554 | **0.471** | 0.449 | 0.197 | **0.383** | 0.586 | – | **0.491** [0.451, 0.530] | +0.001 (0.94; 1.00) |
| `mt_lam1_G` (multitask, EF loss on G) | 9 | 0.551 | **0.471** | 0.449 | 0.209 | 0.373 | 0.591 | – | 0.490 [0.459, 0.520] | +0.000 (0.97; 1.00) |
| `mt_sel` (multitask, λ chosen on inner-val) | 9 | 0.546 | 0.464 | 0.444 | 0.211 | 0.355 | 0.583 | – | 0.484 | −0.006 (0.70; 1.00) |
| `fus_vote+mlpregA` (vote + MLP-regression member) | 3 | **0.560** | 0.458 | 0.446 | 0.226 | 0.360 | 0.578 | – | 0.488 | −0.002 (0.93; 1.00) |
| `fus_vote+ridgeA` (vote + ridge-regression member) | 3 | 0.557 | 0.455 | 0.443 | 0.222 | 0.359 | 0.573 | – | 0.485 | −0.004 (0.80; 1.00) |
| Best EF-regression `tuned` (MLP, C) | 9 | 0.546 | 0.435 | 0.432 | 0.248 | 0.267 | 0.580 | – | 0.471 | −0.022 (0.29; 1.00) |
| EF regression, clinical thresholds (MLP, C) | 9 | 0.558 | 0.416 | 0.418 | 0.287 | 0.207 | 0.552 | – | 0.464 | −0.026 (0.22; n.c.) |
| Ridge ECGF, clinical thresholds | 3 | 0.547 | 0.397 | 0.394 | **0.309** | 0.149 | 0.527 | – | 0.446 | −0.044 (0.05; n.c.) |
| Ordinal-aware multitask (exploratory) | 9 | 0.545 | 0.464 | 0.447 | 0.214 | 0.367 | **0.596** | – | 0.485 | −0.004 (0.72; n.c.) |
| Reference: calibrated MLP fusion (Phase 3; predictions not stored) | 3 | 0.568 | 0.463 | 0.442 | 0.159 | – | 0.591 | – | – | not paired |

- Macro AUROC is shown for the vote ensemble only; the regression models reach 0.744 (§3 for paired tests).
- n.c. = not in the confirmatory family (secondary decoder or exploratory), so no Holm adjustment.

**Confirmatory family (26 models):**

| Metric | Models with Δ > 0 vs vote | Smallest Holm p | Genuine improvements |
|---|---|---|---|
| S3 | 2 | 0.25 | **0** |
| Accuracy | 6 | 0.57 | **0** |
| Balanced accuracy | 7 | 0.39 | **0** |
| Macro-F1 | 0 | 0.35 | **0** |

**Paired bootstrap (repeat 0):** the largest accuracy gain is `fus_vote+mlpregA`, Δ accuracy +0.014
[−0.006, +0.034], Δ balanced −0.012 [−0.034, +0.010], McNemar 45 : 33, p = 0.21. No Phase 4 model has a bootstrap
CI for Δ accuracy that excludes 0.

## 3. Against each experiment's own control (`results/phase4_vs_controls.md`)

| Experiment | Control | ΔS3 range | Best (p; Holm) | Direction |
|---|---|---|---|---|
| 4E multitask (8 variants) | λ = 0 (identical net, no EF loss) | +0.001 to +0.007 | `mt_lam1_G` +0.007 (0.39; 1.00) | all ≥ 0, none significant |
| 4F label quality (4) | MLP fusion | −0.007 to +0.008 | `lq_dropdisagree` +0.008 (0.56; 1.00) | none significant |
| 4D EF regression, tuned (12) | MLP fusion (direct 4-class) | −0.068 to −0.016 | – | all negative; Severe F1 −0.10 to −0.15 |

**Ranking (AUROC), candidate − vote, corrected t-test:**

| Candidate | Δ macro AUROC (p; Holm) | Δ AUROC EF < 41 | Δ AUROC EF < 31 |
|---|---|---|---|
| Vote + MLP-regression member | +0.014 (0.007; 0.07) | +0.006 (0.23) | +0.005 (0.61) |
| Vote + ridge-regression member | +0.013 (0.008; 0.07) | +0.007 (0.20) | +0.005 (0.51) |
| MLP EF regression (A) | +0.014 (0.33; 1.00) | +0.002 (0.89) | −0.017 (0.62) |

- Adding an EF-regression member gives a **consistent but small** ranking gain: macro AUROC 0.726 → about 0.74,
  borderline after Holm. It does not change the 4-class decisions.
- **Clinically standard binary endpoints** (all ECGs, folder labels):
  - EF < 41 (Moderate + Severe): AUROC 0.81–0.82 for every strong model;
  - EF < 31 (Severe): 0.78–0.80;
  - against the EF-derived class on G these rise to 0.85–0.87 and 0.83–0.87, an evaluation-set effect.

## 4. Best model per metric (development CV; none significantly better than the vote ensemble)

| Metric | Target | Best model | Value | Vote ensemble | Target met? |
|---|---|---|---|---|---|
| Accuracy | ≥ 0.60 | vote + MLP-regression member | 0.560 | 0.547 | **no** (calibrated MLP reference 0.568 also no) |
| Balanced accuracy | ≥ 0.50 | `mt_lam1_G` / `lq_dropdisagree` | 0.471 | 0.467 | **no** |
| Macro-F1 | ≥ 0.48 | vote ensemble | 0.455 | 0.455 | **no** |
| Mild F1 | ≥ 0.35 | ridge ECGF, clinical thresholds | 0.309 (Severe recall 0.10) | 0.255 | **no** |
| QWK | ≥ 0.60 | ordinal-aware multitask / `mt_lam3_A` | 0.596 | 0.574 | **no** (0.004 short) |
| S3 (selection) | – | `lq_dropdisagree` | 0.491 | 0.490 | – (tie) |
| Macro AUROC | – | EF regression (MLP, A) | 0.744 | 0.726 | – |

- **Best overall balanced model:** **unchanged, the Phase 3 vote ensemble.** No Phase 4 model is statistically
  distinguishable from it on any 4-class metric.
- Among Phase 4 models, `lq_dropdisagree` and `mt_lam1_G` have the same S3 with slightly higher accuracy and
  balanced accuracy, but lower Mild F1.
- **Best accuracy** is still the Phase 3 calibrated MLP fusion (0.568). It sacrifices Mild (F1 0.159).

## 5. Genuine improvement vs cohort selection vs label quality

| Observation | Classification |
|---|---|
| Every model is 5–8 points more accurate on G than on non-G (vote 0.588 vs 0.517) | **evaluation-set effect** (`PHASE4_LABEL_QUALITY_REPORT.md` §2.4); not a model property |
| Accuracy against EF-derived labels on G is 0.6 points higher than against folder labels | **label-definition effect** |
| Training on G / C / cleaned / EF-derived labels | no change vs control (|ΔS3| ≤ 0.008), i.e. **no label-quality gain** |
| EF supervision (multitask) | +0.001 to +0.007 S3 vs control, n.s.: **no demonstrable gain** |
| EF regression → class | lower S3 than direct classification: **worse** |
| Regression member in the vote | +0.010 to +0.013 accuracy, −0.009 to −0.012 balanced, +0.013 macro AUROC (Holm 0.07): **trade-off / borderline ranking gain** |

## 6. Limitations
- **Optimism:** development-CV estimates of the best of 68 models are optimistically biased. The corrected
  t-test addresses split dependence, not selection over many models; Holm addresses the latter only within the
  confirmatory family.
- **Locked test:** exposed in Phase 1 and not scored. External validation is required.
- **Power:** with 837 ECGs, differences of about 0.01–0.02 in S3 are below detectability.
- **Labels:** folder labels come from name-matched echoes; label noise affects every model equally.
