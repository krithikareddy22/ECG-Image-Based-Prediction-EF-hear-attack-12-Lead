# Confusion matrices (development CV; rows = true, columns = predicted; summed over CV repeats)

**Header + ResNet-18 (current default, R)** (1 repeat(s) x 837 ECGs)

| true \ pred | Normal | Mild | Moderate | Severe | recall |
|---|---|---|---|---|---|
| Normal | 290 (76%) | 33 (9%) | 17 (4%) | 44 (11%) | 0.755 |
| Mild | 69 (37%) | 26 (14%) | 28 (15%) | 62 (34%) | 0.141 |
| Moderate | 36 (20%) | 18 (10%) | 37 (21%) | 86 (49%) | 0.209 |
| Severe | 13 (14%) | 7 (8%) | 10 (11%) | 61 (67%) | 0.670 |

**ECGFounder frozen (linear + distance penalty)** (3 repeat(s) x 837 ECGs)

| true \ pred | Normal | Mild | Moderate | Severe | recall |
|---|---|---|---|---|---|
| Normal | 855 (74%) | 195 (17%) | 66 (6%) | 36 (3%) | 0.742 |
| Mild | 180 (32%) | 137 (25%) | 167 (30%) | 71 (13%) | 0.247 |
| Moderate | 94 (18%) | 95 (18%) | 243 (46%) | 99 (19%) | 0.458 |
| Severe | 34 (12%) | 49 (18%) | 95 (35%) | 95 (35%) | 0.348 |

**Header numeric + clinical (LR)** (3 repeat(s) x 837 ECGs)

| true \ pred | Normal | Mild | Moderate | Severe | recall |
|---|---|---|---|---|---|
| Normal | 814 (71%) | 186 (16%) | 84 (7%) | 68 (6%) | 0.707 |
| Mild | 195 (35%) | 135 (24%) | 119 (21%) | 106 (19%) | 0.243 |
| Moderate | 106 (20%) | 98 (18%) | 183 (34%) | 144 (27%) | 0.345 |
| Severe | 36 (13%) | 32 (12%) | 78 (29%) | 127 (47%) | 0.465 |

**2-way MLP fusion (3-seed ens.)** (3 repeat(s) x 837 ECGs)

| true \ pred | Normal | Mild | Moderate | Severe | recall |
|---|---|---|---|---|---|
| Normal | 922 (80%) | 125 (11%) | 54 (5%) | 51 (4%) | 0.800 |
| Mild | 214 (39%) | 78 (14%) | 158 (28%) | 105 (19%) | 0.141 |
| Moderate | 95 (18%) | 55 (10%) | 232 (44%) | 149 (28%) | 0.437 |
| Severe | 33 (12%) | 20 (7%) | 87 (32%) | 133 (49%) | 0.487 |

**3-way MLP fusion (3-seed ens.)** (3 repeat(s) x 837 ECGs)

| true \ pred | Normal | Mild | Moderate | Severe | recall |
|---|---|---|---|---|---|
| Normal | 909 (79%) | 138 (12%) | 53 (5%) | 52 (5%) | 0.789 |
| Mild | 209 (38%) | 92 (17%) | 150 (27%) | 104 (19%) | 0.166 |
| Moderate | 101 (19%) | 62 (12%) | 227 (43%) | 141 (27%) | 0.427 |
| Severe | 32 (12%) | 28 (10%) | 81 (30%) | 132 (48%) | 0.484 |

**Vote ensemble (2-way MLP + ECGFounder lin + header+clin LR)** (3 repeat(s) x 837 ECGs)

| true \ pred | Normal | Mild | Moderate | Severe | recall |
|---|---|---|---|---|---|
| Normal | 903 (78%) | 159 (14%) | 46 (4%) | 44 (4%) | 0.784 |
| Mild | 196 (35%) | 122 (22%) | 151 (27%) | 86 (15%) | 0.220 |
| Moderate | 95 (18%) | 80 (15%) | 233 (44%) | 123 (23%) | 0.439 |
| Severe | 33 (12%) | 39 (14%) | 85 (31%) | 116 (42%) | 0.425 |
