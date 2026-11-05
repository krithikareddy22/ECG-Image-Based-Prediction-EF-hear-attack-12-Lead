# Phase 6B status

| Date | Step | Outcome |
|---|---|---|
| 2026-11-05 | Feasibility | All Normal sizes feasible: outer-training Normal 307-308 -> 300 / 270 / 240 / 200 by removing 7-8 / 37-38 / 67-68 / 107-108 Normal ECGs (whole patients) from inner-train; T50 moves 27 Mild -> Normal, T55 moves 10 Normal -> Mild, XUNC removes 45 ECGs from training. |
| 2026-11-05 | Pre-registration | PHASE6B_PLAN.md + SHA-256 committed (08d3126) before any 6B run. |
| 2026-11-05 | Handover | No 6B model run yet. All 6B runs (audit report, existing model under T55 / N-sizes, LSTM family, fusions, comparison, report) continue on the owner's GPU laptop session. |
| 2026-11-05 | Amendment 1 | Operational details (test families, eligibility, 60 % criterion, GPU compute rule) committed (f88bcff) before any 6B LSTM-family result. Compute rule: 0.53 h estimated -> 3 seeds for all configurations. |
| 2026-11-05 | Audit | `PHASE6B_AUDIT.md`: 30 borderline Mild ECGs = 15 matching uncertainty / 10 unreliable EF / 5 high-confidence / < 5 genuinely difficult. |
| 2026-11-05 | Existing recipe T55, N300-N200 | `existing6b.py` (CPU): C3 0.569 (T55), 0.559 / 0.558 / 0.558 / 0.553 (N300-N200) on the original labels; Mild recall rises as Normal shrinks (0.178 -> 0.220). |
| 2026-11-05 | LSTM-fixed (L3) under T55, N300-N200 | 45 runs (GPU, 18 min): 0.458-0.474; never admissible. |
| 2026-11-05 | L5, L6, F1, F2 | 36 runs (GPU, 12 min): 0.484 / 0.497 / 0.540 / 0.551. |
| 2026-11-05 | F3, F4 | late fusion: 0.553 / 0.559 (C3 0.561). |
| 2026-11-05 | Comparison, statistics, report | `compare6b.py`, `PHASE6B_FINAL_REPORT.md`: no genuine improvement in any family; conclusion 1 = C3 0.561 [0.520, 0.603]; conclusion 2 = C3 recipe trained on T55, 0.567 [0.535, 0.598] on T55 labels; 60 % not reached. **Phase 6B complete.** |
| 2026-11-05 | MMCLEAN (exploratory, owner / supervisor request) | `AMENDMENT_MMCLEAN.md`, declared before running; `mmclean.py`, `MMCLEAN_REPORT.md`: MMC / MMNC / MN240 / MN200 / ECHO with MLP fusion, C1, C3 (CPU, about 30 min per config pair). All 837 ECGs: 0.525-0.553 vs C3 0.562 (Holm p 1.00); cleaner subsets: max 0.677 (not admissible); selective prediction: 0.713 [0.637, 0.820] at 22 % coverage (C3 MMC; Normal / Severe only). 70 % four-class accuracy not reached. |
| 2026-11-05 | NF / NFS and MILDFIX (exploratory, owner request) | Noise-free training (`AMENDMENT_MMCLEAN.md`), then one nested fix per reason Mild is missed (`AMENDMENT_MILDFIX.md`, `mildfix.py`, `MILDFIX_REPORT.md`). All 837 ECGs: nothing beats C3 0.562 (fixes 0.505-0.560, Holm p >= 0.06). Best Mild: Mild-aware rule (recall 0.218). Abstention: 0.708 at 34 % (Normal / Severe only). Clean clear-cut subset: 0.707-0.720. |
| 2026-11-05 | MILDFIX2 (exploratory, owner request) | `AMENDMENT_MILDFIX2.md`, `mildfix2.py`, `MILDFIX2_REPORT.md`: 12 noise-aware EF regressors, nested selection. EF error unchanged (best SD 13.22 vs 13.4); four-class unchanged (C3 0.562 best); agreement abstention 0.698 [0.652, 0.744] answering 57 % (Mild never correct). |
| 2026-11-05 | MILDFEAT (exploratory, owner request: image-only Mild features) | `AMENDMENT_MILDFEAT.md` declared before any model (42dc57f); `mildfeat.py` 121 features (validated vs printed machine values), `mildmodels.py`; `MILDFEAT_REPORT.md`: no gain over ECGFounder (Holm p 1.00); best image-only 0.558; Mild vs Normal AUROC <= 0.757, Mild vs Moderate <= 0.63. |

## Already available from Phase 6 / 6A (re-used, not repeated)
- Existing model under ORIG (= 6A B1, reproduces C1 / C3 exactly), T50 (= B2), XUNC (= B3), A270X (= A270): no accuracy gain on the original labels (Holm p 1.00); B2 / B3 reduce C3 Mild recall.
- Matching / data check (6A `results/matching_data_check.json`), 80 % feasibility (6A `results/feasibility_80.json`).
- Phase 6: fine-tuning (A1 / A2), ECG-FM and ECGFounder + ECG-FM, input-fidelity D1 / D2 and the name-pair sensitivity all failed to improve; see `phase_6/PHASE6_CLAIMS_REGISTER.md`.
