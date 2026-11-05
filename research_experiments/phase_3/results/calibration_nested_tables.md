| Model | Rule | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | Mild recall | QWK | Normal recall | Predicted Normal |
|---|---|---|---|---|---|---|---|---|---|
| fusion_mlp | original | 0.544 | 0.466 | 0.440 | 0.186 | 0.141 | 0.583 | 0.800 | 50.3% |
| fusion_mlp | calibrated: prior, accuracy | 0.572 (+0.029) | 0.443 (-0.023) | 0.423 (-0.016) | 0.140 (-0.046) | 0.092 (-0.049) | 0.571 (-0.012) | 0.900 | 60.4% |
| fusion_mlp | calibrated: prior, protected | 0.568 (+0.025) | 0.463 (-0.003) | 0.442 (+0.002) | 0.159 (-0.027) | 0.112 (-0.029) | 0.591 (+0.008) | 0.865 | 55.8% |
| fusion_mlp | calibrated: bias, accuracy | 0.571 (+0.028) | 0.443 (-0.023) | 0.399 (-0.041) | 0.031 (-0.155) | 0.016 (-0.124) | 0.577 (-0.006) | 0.914 | 61.1% |
| fusion_mlp | calibrated: bias, protected | 0.566 (+0.022) | 0.450 (-0.016) | 0.415 (-0.025) | 0.090 (-0.096) | 0.054 (-0.086) | 0.585 (+0.002) | 0.891 | 58.4% |
| ecgf_linear | original | 0.528 | 0.442 | 0.434 | 0.244 | 0.214 | 0.543 | 0.766 | 48.4% |
| ecgf_linear | calibrated: prior, accuracy | 0.556 (+0.028) | 0.433 (-0.010) | 0.426 (-0.007) | 0.202 (-0.042) | 0.157 (-0.058) | 0.545 (+0.002) | 0.865 | 57.9% |
| ecgf_linear | calibrated: prior, protected | 0.544 (+0.016) | 0.434 (-0.008) | 0.427 (-0.006) | 0.207 (-0.037) | 0.168 (-0.047) | 0.546 (+0.003) | 0.831 | 54.7% |
| ecgf_linear | calibrated: bias, accuracy | 0.563 (+0.035) | 0.438 (-0.004) | 0.408 (-0.025) | 0.088 (-0.156) | 0.056 (-0.159) | 0.565 (+0.022) | 0.892 | 59.7% |
| ecgf_linear | calibrated: bias, protected | 0.542 (+0.014) | 0.431 (-0.011) | 0.412 (-0.021) | 0.138 (-0.106) | 0.101 (-0.114) | 0.545 (+0.002) | 0.847 | 56.2% |
| header_numeric_lr | original | 0.489 | 0.414 | 0.393 | 0.198 | 0.162 | 0.493 | 0.738 | 49.3% |
| header_numeric_lr | calibrated: prior, accuracy | 0.505 (+0.016) | 0.383 (-0.031) | 0.370 (-0.023) | 0.156 (-0.042) | 0.114 (-0.049) | 0.447 (-0.046) | 0.839 | 62.2% |
| header_numeric_lr | calibrated: prior, protected | 0.498 (+0.009) | 0.408 (-0.006) | 0.390 (-0.004) | 0.184 (-0.014) | 0.142 (-0.020) | 0.480 (-0.013) | 0.780 | 54.2% |
| header_numeric_lr | calibrated: bias, accuracy | 0.516 (+0.027) | 0.380 (-0.034) | 0.339 (-0.054) | 0.036 (-0.161) | 0.020 (-0.142) | 0.436 (-0.057) | 0.844 | 62.4% |
| header_numeric_lr | calibrated: bias, protected | 0.504 (+0.015) | 0.406 (-0.008) | 0.382 (-0.011) | 0.143 (-0.055) | 0.095 (-0.067) | 0.462 (-0.031) | 0.789 | 55.4% |

Confusion matrices summed over the 3 CV repeats (rows = true N/Mi/Mo/S; 3 x 837 ECGs)

**fusion_mlp - original**
```
  922   125    54    51
  214    78   158   105
   95    55   232   149
   33    20    87   133
```

**fusion_mlp - calibrated: prior, accuracy**
```
 1037    40    62    13
  285    51   178    41
  135    47   279    70
   59    24   120    70
```

**fusion_mlp - calibrated: prior, protected**
```
  997    73    57    25
  247    62   179    67
  111    49   266   105
   46    27    98   102
```

**fusion_mlp - calibrated: bias, accuracy**
```
 1053     4    77    18
  282     9   209    55
  138     9   294    90
   60     5   129    79
```

**fusion_mlp - calibrated: bias, protected**
```
 1027    32    69    24
  265    30   189    71
  123    28   267   113
   52    12   113    96
```
