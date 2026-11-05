# Phase 6A status (Normal/Mild boundary + LSTM study)

| Date | Step | Outcome |
|---|---|---|
| 2026-11-05 | Label rule and cohort facts | Exact rule recovered from the matching notebook's code; class, patient and fold distributions documented (`PHASE6A_PLAN.md` §1–2, `results/label_configurations.json`). |
| 2026-11-05 | Experiment A design | 270 is below the original training split's 307 Normal ECGs but above the 245–247 Normal ECGs each CV model is fitted on; A270 raises inner-train Normal to exactly 270 in every fold by moving whole Normal patients from inner-val (23–25 ECGs per fold). |
| 2026-11-05 | Label configurations | B2 (EF ≥ 50 % → Normal, consistent links only): 27 Mild ECGs / 27 patients relabelled. B3: 45 uncertain Normal/Mild ECGs (44 patients) excluded from inner-train only. |
| 2026-11-05 | Matching / data check (owner request) | Uploaded sheets identical to the ones used; linking, EF values and labels verified on development ECGs; found and quantified a day/month swap in the 2021 sheet's ECG-date column (C6A-1): it changes interval-based counts and cohort G, not labels, EF targets or the 6A configurations. |
| 2026-11-05 | Existing model under B1 / A270 / B2 / B3 | Done; B1 reproduces the stored C1 / C3 exactly. |
| 2026-11-05 | Existing model: A270 / B2 / B3 results | No configuration improves accuracy on the original labels (all Holm p 1.00); B2's apparent gain is a definition effect; B2 / B3 reduce Mild recall for C3 (interim comparison committed; final comparison after the LSTM runs). |
| 2026-11-05 | 80 % feasibility (owner request) | Four-class 80 % needs r ≈ 0.975 (have 0.66): not attainable with these data; selective prediction tops out at ≈ 0.70. Binary Normal-vs-reduced EF reaches 0.775 (AUROC 0.83) from existing models. |
| 2026-11-05 | Handover | Cloud LSTM grid stopped (L4 29/45, L1 1/45 folds; not committed). Open work (LSTM L0-L4 grid, nested selection, LSTM-fixed under A270 / B2 / B3, conditional CNN-LSTM / fusion, final comparison and report) continues on the owner's GPU laptop. |
| 2026-11-05 | Pre-registration | `PHASE6A_PLAN.md` + SHA-256 written before any Phase 6A model run. |
| 2026-11-05 | GPU laptop: restore | Fresh clone on the owner's laptop (RTX 3060). All private inputs regenerated; the 84 existing-model runs (B1/B2/B3/A270) reproduce their committed aggregates within 0.005 (max 0.0047, header + clinical LR B1 r1); `existing --check` True. |
| 2026-11-05 | LSTM grid L0-L4 (B1) | 225 folds / 45 runs on the GPU (≈ 22 min). Accuracy 0.43-0.48; no variant admissible (Mild recall ≤ 0.08). |
| 2026-11-05 | Nested selection | L3 chosen in 7/15 folds (L4 4, L2 3, L1 1) -> LSTM-fixed = L3. LSTM-sel 0.470 [0.424, 0.515], not admissible. |
| 2026-11-05 | LSTM-fixed under A270 / B2 / B3 | 0.461 / 0.466 / 0.462 vs 0.467 (B1); Holm p 1.00. |
| 2026-11-05 | Conditional models | CNN-LSTM condition (LSTM-sel admissible, macro-F1 ≥ 0.389) and complementarity criterion **not met**; both were run unconditionally in Phase 6B (L5, F3/F4) and are reported descriptively here. |
| 2026-11-05 | Final comparison and report | `compare.py` re-run; `PHASE6A_FINAL_REPORT.md`, `PHASE6A_MODEL_COMPARISON.csv`, `PHASE6A_EXPERIMENT_INVENTORY.md`. No genuine improvement; only the LSTM inferiority is significant. **Phase 6A complete.** |

## Expected compute
| Stage | Estimate |
|---|---|
| Existing best model under B1 / B2 / B3 / A270 (4 parallel single-thread processes; nested inner OOF re-fitted for B2 / B3) | ≈ 20–30 min |
| LSTM L0–L4, 3 seeds × 3 repeats (45 fold fits each; 4 parallel jobs) | ≈ 3–4 h in total (timed: 2–12 s per epoch) |
| LSTM-fixed under A270 / B2 / B3 | ≈ 0.5–1.5 h each |
| CNN-LSTM, fusion (conditional) | ≈ 1–2 h |
