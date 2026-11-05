| Group | Model | Input | Accuracy [95% CI] | Balanced acc. [95% CI] | Macro-F1 [95% CI] | Normal F1 | Mild F1 | Moderate F1 | Severe F1 | Macro AUROC | QWK | HF AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (existing) | B0_default_fusion_header0.6_resnet18 | header + image | 0.552 [0.48-0.62] | 0.454 [0.38-0.53] | 0.433 [0.36-0.50] | 0.741 | 0.154 | 0.468 | 0.370 | 0.688 | 0.447 | 0.791 |
| Baseline (existing) | B0_header_logreg_multinomial | printed header (measurements + statements + age/sex) | 0.514 [0.44-0.58] | 0.442 [0.37-0.51] | 0.421 [0.35-0.49] | 0.716 | 0.160 | 0.471 | 0.339 | 0.684 | 0.422 | 0.795 |
| Baseline (existing) | B0_resnet18_image_ce | ECG image (filtered, 256x512) | 0.486 [0.42-0.55] | 0.356 [0.30-0.42] | 0.323 [0.26-0.39] | 0.689 | 0.109 | 0.250 | 0.246 | 0.652 | 0.290 | 0.735 |
| Baseline retrain | R0_resnet18_filtered_retrain_seed42 | ECG image (filtered 256x512) | 0.481 [0.41-0.55] | 0.347 [0.29-0.41] | 0.310 [0.25-0.38] | 0.685 | 0.109 | 0.200 | 0.246 | 0.650 | 0.301 | 0.731 |
| A1 frozen ECGFounder | A1_ecgf_aligned_deep_multinomial | ECGFounder frozen (aligned waveform, deep) | 0.500 [0.44-0.57] | 0.421 [0.35-0.49] | 0.412 [0.35-0.48] | 0.728 | 0.265 | 0.315 | 0.340 | 0.668 | 0.464 | 0.810 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_ce_3seed_ens | mean of seeds | 0.524 [0.46-0.59] | 0.464 [0.39-0.54] | 0.448 [0.38-0.52] | 0.743 | 0.282 | 0.345 | 0.421 | 0.680 | 0.519 | 0.812 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_ce_seed42 | ECGFounder 'aligned' waveform 12x5000 | 0.510 [0.44-0.58] | 0.438 [0.36-0.52] | 0.425 [0.35-0.50] | 0.740 | 0.238 | 0.360 | 0.364 | 0.676 | 0.468 | 0.809 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_ce_seed43 | ECGFounder 'aligned' waveform 12x5000 | 0.500 [0.43-0.56] | 0.440 [0.37-0.51] | 0.425 [0.36-0.49] | 0.731 | 0.295 | 0.295 | 0.379 | 0.673 | 0.452 | 0.811 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_ce_seed44 | ECGFounder 'aligned' waveform 12x5000 | 0.519 [0.45-0.59] | 0.471 [0.39-0.55] | 0.453 [0.38-0.52] | 0.717 | 0.321 | 0.362 | 0.414 | 0.688 | 0.485 | 0.815 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_cumlink_3seed_ens | mean of seeds | 0.510 [0.44-0.58] | 0.417 [0.35-0.49] | 0.393 [0.33-0.46] | 0.761 | 0.225 | 0.278 | 0.308 | 0.690 | 0.523 | 0.790 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_cumlink_seed42 | ECGFounder 'aligned' waveform 12x5000 | 0.510 [0.44-0.58] | 0.420 [0.35-0.49] | 0.406 [0.34-0.47] | 0.754 | 0.286 | 0.293 | 0.290 | 0.685 | 0.515 | 0.785 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_cumlink_seed43 | ECGFounder 'aligned' waveform 12x5000 | 0.495 [0.43-0.56] | 0.407 [0.33-0.48] | 0.383 [0.31-0.45] | 0.720 | 0.164 | 0.320 | 0.328 | 0.688 | 0.490 | 0.786 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft56_cumlink_seed44 | ECGFounder 'aligned' waveform 12x5000 | 0.505 [0.44-0.57] | 0.430 [0.36-0.50] | 0.400 [0.34-0.46] | 0.754 | 0.243 | 0.263 | 0.338 | 0.687 | 0.520 | 0.793 |
| A2 fine-tuned ECGFounder | A2_ecgf_ft6_ce_seed42 | ECGFounder 'aligned' waveform 12x5000 | 0.510 [0.44-0.58] | 0.441 [0.37-0.52] | 0.427 [0.36-0.50] | 0.729 | 0.225 | 0.383 | 0.370 | 0.677 | 0.455 | 0.812 |
| B ordinal / loss (ECGFounder frozen) | AB_ecgf_aligned_deep_coral | ECGFounder frozen (aligned waveform, deep) | 0.390 [0.33-0.46] | 0.376 [0.31-0.44] | 0.345 [0.28-0.41] | 0.435 | 0.329 | 0.474 | 0.143 | 0.673 | 0.394 | 0.785 |
| B ordinal / loss (ECGFounder frozen) | AB_ecgf_aligned_deep_cumlink | ECGFounder frozen (aligned waveform, deep) | 0.467 [0.40-0.53] | 0.427 [0.35-0.51] | 0.421 [0.35-0.49] | 0.615 | 0.308 | 0.451 | 0.311 | 0.679 | 0.463 | 0.792 |
| B ordinal / loss (ECGFounder frozen) | AB_ecgf_aligned_deep_focal | ECGFounder frozen (aligned waveform, deep) | 0.510 [0.44-0.58] | 0.437 [0.37-0.51] | 0.428 [0.36-0.49] | 0.732 | 0.265 | 0.330 | 0.385 | 0.678 | 0.471 | 0.812 |
| B ordinal / loss (header) | B_header_coral | printed header (all features) | 0.414 [0.35-0.48] | 0.382 [0.31-0.45] | 0.373 [0.30-0.44] | 0.529 | 0.310 | 0.417 | 0.235 | 0.655 | 0.348 | 0.730 |
| B ordinal / loss (header) | B_header_cumlink | printed header (all features) | 0.462 [0.39-0.52] | 0.422 [0.35-0.49] | 0.409 [0.34-0.47] | 0.656 | 0.247 | 0.353 | 0.379 | 0.642 | 0.480 | 0.790 |
| B ordinal / loss (header) | B_header_focal | printed header (all features) | 0.519 [0.45-0.59] | 0.461 [0.39-0.53] | 0.436 [0.37-0.50] | 0.721 | 0.184 | 0.427 | 0.414 | 0.684 | 0.476 | 0.806 |
| B ordinal / loss (header) | B_header_multinomial | printed header (all features) | 0.524 [0.45-0.59] | 0.458 [0.38-0.53] | 0.425 [0.36-0.49] | 0.729 | 0.147 | 0.447 | 0.375 | 0.686 | 0.453 | 0.807 |
| B ordinal (image CNN) | B_resnet18_image_cumlink_seed42 | ECG image (filtered 256x512) | 0.400 [0.33-0.46] | 0.342 [0.27-0.41] | 0.302 [0.24-0.36] | 0.636 | 0.118 | 0.197 | 0.256 | 0.618 | 0.325 | 0.715 |
| Header ablation | HA_H1_demographics | header subset: H1_demographics | 0.390 [0.32-0.46] | 0.341 [0.27-0.42] | 0.331 [0.26-0.40] | 0.568 | 0.214 | 0.342 | 0.200 | 0.614 | 0.211 | 0.667 |
| Header ablation | HA_H2_measurements | header subset: H2_measurements | 0.495 [0.43-0.56] | 0.408 [0.34-0.48] | 0.385 [0.32-0.45] | 0.733 | 0.174 | 0.349 | 0.286 | 0.686 | 0.431 | 0.793 |
| Header ablation | HA_H3_demographics+measurements | header subset: H3_demographics+measurements (no machine text) | 0.510 [0.44-0.58] | 0.449 [0.38-0.52] | 0.414 [0.35-0.48] | 0.734 | 0.137 | 0.386 | 0.400 | 0.698 | 0.488 | 0.802 |
| Header ablation | HA_H4_verdict_only | header subset: H4_verdict_only | 0.352 [0.29-0.42] | 0.341 [0.29-0.39] | 0.218 [0.18-0.25] | 0.636 | 0.000 | 0.000 | 0.235 | 0.607 | 0.254 | 0.693 |
| Header ablation | HA_H5_statements_only | header subset: H5_statements_only | 0.414 [0.35-0.48] | 0.359 [0.29-0.43] | 0.337 [0.27-0.40] | 0.600 | 0.080 | 0.427 | 0.242 | 0.628 | 0.242 | 0.729 |
| Header ablation | HA_H6_verdict+statements | header subset: H6_verdict+statements (machine text only) | 0.424 [0.36-0.50] | 0.379 [0.31-0.45] | 0.355 [0.29-0.42] | 0.617 | 0.137 | 0.454 | 0.213 | 0.642 | 0.258 | 0.755 |
| Header ablation | HA_H7_all_minus_statements | header subset: H7_all_minus_statements | 0.490 [0.42-0.56] | 0.450 [0.38-0.52] | 0.415 [0.35-0.48] | 0.699 | 0.189 | 0.396 | 0.377 | 0.702 | 0.426 | 0.801 |
| Header ablation | HA_H8_all | header subset: H8_all (= baseline header model) | 0.514 [0.44-0.58] | 0.442 [0.37-0.51] | 0.421 [0.35-0.49] | 0.716 | 0.160 | 0.471 | 0.339 | 0.684 | 0.422 | 0.795 |
| Fusion | FU_header+ecgfA1 | late fusion | 0.514 [0.45-0.58] | 0.437 [0.37-0.51] | 0.426 [0.36-0.49] | 0.737 | 0.268 | 0.322 | 0.377 | 0.674 | 0.483 | 0.815 |
| Fusion | FU_header+ecgfA2 | late fusion | 0.505 [0.44-0.57] | 0.445 [0.37-0.52] | 0.425 [0.35-0.49] | 0.741 | 0.238 | 0.322 | 0.400 | 0.695 | 0.497 | 0.819 |
| Fusion | FU_header+image+ecgfA1 | late fusion | 0.514 [0.45-0.58] | 0.428 [0.36-0.50] | 0.415 [0.35-0.48] | 0.738 | 0.234 | 0.310 | 0.377 | 0.683 | 0.481 | 0.805 |
| Fusion | FU_header+image+ecgfA2 | late fusion | 0.524 [0.46-0.59] | 0.426 [0.36-0.50] | 0.402 [0.33-0.47] | 0.750 | 0.147 | 0.346 | 0.364 | 0.693 | 0.479 | 0.793 |
| Fusion | FU_header+image_re-tuned | late fusion | 0.552 [0.48-0.62] | 0.454 [0.38-0.53] | 0.433 [0.36-0.50] | 0.741 | 0.154 | 0.468 | 0.370 | 0.688 | 0.447 | 0.791 |
| Fusion | FU_headerNumbersOnly+ecgfA1 | late fusion | 0.500 [0.43-0.57] | 0.423 [0.35-0.49] | 0.410 [0.34-0.47] | 0.727 | 0.250 | 0.299 | 0.364 | 0.674 | 0.480 | 0.814 |
| Fusion | FU_headerNumbersOnly+ecgfA2 | late fusion | 0.510 [0.44-0.58] | 0.448 [0.37-0.52] | 0.426 [0.36-0.49] | 0.736 | 0.250 | 0.329 | 0.387 | 0.703 | 0.488 | 0.822 |
| Fusion | FU_ordinal_header+ecgfA1 | late fusion | 0.490 [0.42-0.56] | 0.451 [0.37-0.53] | 0.444 [0.37-0.51] | 0.651 | 0.283 | 0.474 | 0.367 | 0.677 | 0.493 | 0.802 |
| Fusion | FU_ordinal_header+ecgfA2 | late fusion | 0.510 [0.44-0.58] | 0.417 [0.35-0.49] | 0.393 [0.33-0.46] | 0.761 | 0.225 | 0.278 | 0.308 | 0.690 | 0.523 | 0.790 |
| Fusion | FU_ordinal_header+image | late fusion | 0.424 [0.36-0.49] | 0.361 [0.29-0.43] | 0.338 [0.27-0.40] | 0.663 | 0.145 | 0.274 | 0.270 | 0.637 | 0.389 | 0.774 |
| Fusion, weights by val accuracy (secondary) | FUacc_header+ecgfA1 | late fusion | 0.514 [0.45-0.58] | 0.437 [0.37-0.51] | 0.426 [0.36-0.49] | 0.737 | 0.268 | 0.322 | 0.377 | 0.674 | 0.483 | 0.815 |
| Fusion, weights by val accuracy (secondary) | FUacc_header+ecgfA2 | late fusion | 0.524 [0.46-0.59] | 0.464 [0.39-0.54] | 0.448 [0.38-0.52] | 0.743 | 0.282 | 0.345 | 0.421 | 0.680 | 0.519 | 0.812 |
| Fusion, weights by val accuracy (secondary) | FUacc_header+image+ecgfA1 | late fusion | 0.529 [0.46-0.60] | 0.420 [0.35-0.49] | 0.397 [0.32-0.46] | 0.735 | 0.125 | 0.400 | 0.327 | 0.689 | 0.432 | 0.786 |
| Fusion, weights by val accuracy (secondary) | FUacc_header+image+ecgfA2 | late fusion | 0.524 [0.46-0.59] | 0.426 [0.36-0.50] | 0.402 [0.33-0.47] | 0.750 | 0.147 | 0.346 | 0.364 | 0.693 | 0.479 | 0.793 |
| Fusion, weights by val accuracy (secondary) | FUacc_header+image_re-tuned | late fusion | 0.552 [0.48-0.62] | 0.454 [0.38-0.53] | 0.433 [0.36-0.50] | 0.741 | 0.154 | 0.468 | 0.370 | 0.688 | 0.447 | 0.791 |
| Fusion, weights by val accuracy (secondary) | FUacc_ordinal_header+ecgfA2 | late fusion | 0.510 [0.44-0.58] | 0.417 [0.35-0.49] | 0.393 [0.33-0.46] | 0.761 | 0.225 | 0.278 | 0.308 | 0.690 | 0.523 | 0.790 |
| A1 input/feature ablation | A1_ecgf_aligned_deep_logits_multinomial | ECGFounder frozen (aligned waveform, deep+logits) | 0.514 [0.45-0.58] | 0.451 [0.38-0.52] | 0.429 [0.36-0.50] | 0.727 | 0.219 | 0.348 | 0.421 | 0.688 | 0.499 | 0.811 |
| A1 input/feature ablation | A1_ecgf_aligned_logits_multinomial | ECGFounder frozen (aligned waveform, logits) | 0.514 [0.44-0.58] | 0.462 [0.38-0.54] | 0.442 [0.37-0.51] | 0.727 | 0.286 | 0.368 | 0.387 | 0.703 | 0.519 | 0.801 |
| A1 input/feature ablation | A1_ecgf_short_deep_logits_multinomial | ECGFounder frozen (short waveform, deep+logits) | 0.481 [0.41-0.55] | 0.395 [0.32-0.47] | 0.392 [0.31-0.46] | 0.677 | 0.262 | 0.378 | 0.250 | 0.652 | 0.420 | 0.760 |
| A1 input/feature ablation | A1_ecgf_short_deep_multinomial | ECGFounder frozen (short waveform, deep) | 0.467 [0.40-0.54] | 0.373 [0.30-0.44] | 0.370 [0.30-0.44] | 0.673 | 0.214 | 0.378 | 0.213 | 0.649 | 0.440 | 0.765 |
| A1 input/feature ablation | A1_ecgf_short_logits_multinomial | ECGFounder frozen (short waveform, logits) | 0.471 [0.41-0.54] | 0.377 [0.31-0.44] | 0.344 [0.28-0.40] | 0.704 | 0.143 | 0.275 | 0.254 | 0.633 | 0.403 | 0.736 |
| A1 input/feature ablation | A1_ecgf_tiled_deep_logits_multinomial | ECGFounder frozen (tiled waveform, deep+logits) | 0.452 [0.38-0.52] | 0.357 [0.29-0.42] | 0.349 [0.28-0.41] | 0.667 | 0.187 | 0.337 | 0.204 | 0.641 | 0.382 | 0.745 |
| A1 input/feature ablation | A1_ecgf_tiled_deep_multinomial | ECGFounder frozen (tiled waveform, deep) | 0.486 [0.42-0.55] | 0.398 [0.33-0.47] | 0.392 [0.32-0.46] | 0.683 | 0.250 | 0.400 | 0.235 | 0.633 | 0.402 | 0.734 |
| A1 input/feature ablation | A1_ecgf_tiled_logits_multinomial | ECGFounder frozen (tiled waveform, logits) | 0.429 [0.36-0.50] | 0.343 [0.28-0.40] | 0.314 [0.26-0.37] | 0.670 | 0.061 | 0.282 | 0.242 | 0.631 | 0.360 | 0.732 |
