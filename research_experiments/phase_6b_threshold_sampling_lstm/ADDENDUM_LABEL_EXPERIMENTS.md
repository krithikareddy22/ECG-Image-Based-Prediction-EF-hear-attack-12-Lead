# Phase 6B addendum: exploratory label experiments (cloud session, 2026-11-05)

Definitions declared before running (`AMENDMENT_BRES.md`, `AMENDMENT_VERIFIED.md`). Existing best recipe (MLP fusion,
C1 vote, C3 rule; nested inner models re-fitted), development CV, mean over 3 repeats, all 837 ECGs. Each model is
scored on the original labels (primary) and on its own labels. "B1 model on new labels" is the unchanged original
model, so the difference to it is the training effect; the difference between B1 on new and on original labels is
the definition effect. Results: `results/bres_exploratory.json`, `results/verified_labels_exploratory.json`.

| Configuration (labels changed) | C3 on original labels | C3 on own labels | B1 model on own labels | Mild recall (own) | Admissible |
|---|---|---|---|---|---|
| Original (reference) | 0.561 | – | – | 0.177 | yes |
| VER: 15 labels corrected to a high-confidence echo | 0.565 | 0.568 | 0.565 | 0.187 | **yes** |
| VERB: VER + 26 high-confidence borderline Mild resolved | 0.558 | 0.571 | 0.567 | 0.136 | no |
| BRES: 45 consistent borderline Mild resolved (16 → Normal, 29 → Moderate) | 0.566 | 0.592 | 0.573 | 0.064 | no |

- **VER** (model-independent label correction) is the only admissible configuration: +0.4 points on the original
  labels and +0.3 on its own labels versus the unchanged model. Label noise explains under half a point of accuracy.
- **VERB / BRES** raise accuracy on their own labels (+0.4 / +1.9 points of training effect on top of a
  +0.6 / +1.2-point definition effect) only by giving up Mild detection (C3 Mild recall 0.136 / 0.064). They are not
  admissible and do not improve the original task.
- **Circularity check (not a result):** relabelling the 255 border ECGs (within 3 points of a boundary or disagreeing
  with the nearest echo) to the model's own prediction would give C3 0.764, but random guessing would also jump from
  0.244 to 0.470. This is why model-dependent relabelling is not used. Descriptive subgroup: C3 is 0.660 accurate on
  the 582 non-border ECGs (0.561 overall).
