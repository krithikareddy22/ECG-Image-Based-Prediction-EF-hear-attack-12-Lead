# Phase 5 claims register

**Status labels:**
- **VALID**: directly measured;
- **CORRECTED**: a previously stated value revised;
- **RETRACTED**: unsupported or incorrect;
- **HYPOTHESIS**: plausible but unverified;
- **ESTIMATE**: projected, not measured.

Earlier phases keep their registers (`phase_4/reports/PHASE4_CLAIMS_REGISTER.md`, `phase_3/reports/CLAIMS_REGISTER.md`).

| # | Claim | Status | Evidence |
|---|---|---|---|
| P5-1 | The 56.8 % benchmark is a four-class development-CV accuracy on the same labels and the same 837 ECGs (not a test-set or three-class result) | VALID | `REPOSITORY_AUDIT.md` §4 |
| P5-2 | The 56.8 % model has Mild recall 0.08–0.16 and is not admissible under the Phase 5 recall-floor rule | VALID | `REPOSITORY_AUDIT.md` §4; `PHASE5_PLAN.md` §3 |
| P5-3 | Validation is grouped by a name-derived key; 29 near-identical name pairs exist under different keys (15 within development, 11 straddling development / locked test) | VALID (counts); same-person status unverifiable | `data_audit/grouping_audit.json` |
| P5-4 | There are no duplicate recordings (0 exact, 0 with r ≥ 0.98 rhythm-strip correlation) | VALID | `data_audit/grouping_audit.json` |
| P5-5 | The calibrated MLP (56.8 %) is reproduced exactly with stored predictions (0.5711 / 0.5687 / 0.5651; identical per-fold α) | VALID | `reports/PHASE5_BASELINE_REPORT.md` |
| P5-6 | Best admissible baseline accuracy is 0.560 (vote + EF-regression member). No baseline is significantly more accurate than the vote ensemble (smallest p 0.06) | VALID | `results/phase5_baseline_results.csv` |
| P5-7 | The accuracy gain of calibration (+0.021 to +0.025) comes with Mild recall 0.09–0.11 (below the floor); under the recall floor it vanishes (0.549) | VALID | 5A report |
| P5-8 | Mild errors go both ways (35 % → Normal, 27 % → Moderate) and are only slightly more boundary-concentrated than correct Mild predictions | VALID | `reports/PHASE5_ERROR_ANALYSIS.md` §1 |
| P5-9 | Missed Severe ECGs cluster near EF 31 (58 % within 5 points vs 33 % of detected) and have narrower QRS | VALID | 5B §8 |
| P5-10 | Patient keys with ≥ 2 ECGs are classified about 13–15 points less accurately; the possible split identities show no accuracy advantage (no evidence of leakage) | VALID | 5B §5 |
| P5-11 | "HF power" in digitised printouts reflects QRS sharpness, not noise (confounded quality indicator) | HYPOTHESIS | 5B §4 |
| P5-12 | Accuracy on the 50 % most confident ECGs is 0.66–0.68 (selective prediction; not the four-class target) | VALID | 5B §7 |
| P5-13 | QRS morphology is the only informative engineered family on top of the header numbers (LR 0.538 vs 0.489; p 0.043, Holm 0.38); all families together dilute it (0.503) | VALID (nominal; not Holm-significant) | `reports/PHASE5_FEATURE_REPORT.md` §3 |
| P5-14 | No engineered feature family improves the MLP fusion (E1: \|Δacc\| ≤ 0.010, Holm p = 1.00) | VALID | 5C report §4 |
| P5-15 | Gradient boosting (E2) does not beat the MLP fusion (best 0.544; Δ ≤ 0) | VALID | `results/phase5_family_comparisons.csv` |
| P5-16 | Nested stacking (E3) does not beat the equal-weight vote; its accuracy-oriented variant (0.567) makes Mild undetectable (recall 0.04) | VALID | same |
| P5-17 | The hierarchical two-stage model (E4) reaches 0.567 but with Mild recall 0.07 (not admissible) | VALID | same |
| P5-18 | Nested decision rules on the vote (E5) raise accuracy to 0.561 while keeping every class recall ≥ 0.18 (prior / protected) or ≥ 0.20 (bias / floor). The gain of +0.014 over C1 is not significant (p ≥ 0.16, Holm 1.00) | VALID | same |
| P5-19 | Signal-quality-aware training (E6) and training-strategy changes (E7: focal, smoothing, mixup, cosine LR, accuracy-based selection, waveform augmentation) change accuracy by −0.014 to +0.009 vs the MLP fusion, none significant | VALID | same |
| P5-20 | Class-unweighted CE gives 0.577 on repeat 0 (Phase 2 runs) but 0.572 with Mild recall 0.144 over all three repeats: an accuracy-for-Mild trade, not admissible | VALID | same |
| C5-1 | First 5F selection output nominated the repeat-0-only unweighted model (0.577). **Corrected**: the pre-registered estimate is the mean over repeats 0–2, so repeat-0-only models are reported but not eligible; the 3-repeat version is not admissible | CORRECTED | `validation/final_stats.py` (eligibility rule), `results/phase5_selection.json` |
| P5-21 | No Phase 5 model is a genuine improvement over C1 or C2 (0 of 43 candidates with Holm p < 0.05 and consistent sign); the 65 % target was not reached (best admissible about 0.56; best of any model 0.572) | VALID | `results/phase5_selection.json` |
