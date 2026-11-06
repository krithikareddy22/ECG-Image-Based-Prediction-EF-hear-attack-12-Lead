# Phase 6B exploratory amendment: SUB591 (random class-capped subset, owner request)

**Date:** 2026-11-05. Written before any run, at the owner's request ("take 150 Mild and Moderate, 200 Normal, all Severe
and train").

- **Subset:** a random, seeded (`numpy default_rng(591)`), patient-level selection from the development set:
  - Normal 200, Mild 150, Moderate 150, Severe all 91 (target 591 ECGs);
  - whole patients are added in random order while the class stays at or below its target.

  Selection uses only the folder label and patient key: never EF difficulty, link quality or model output.
- **Training:** in every fold (Phase 2 folds, 3 repeats), only subset ECGs may be used for fitting (outer members,
  nested inner models, rule and prior fitting). Inner-val keeps all its ECGs; early stopping is unchanged.
- **Evaluation:**
  1. all 837 development ECGs, original labels (comparable with every other result);
  2. the subset ECGs only (the subset's own class mix);
  3. compared with the ORIG models on the same ECGs.
- **Status:** exploratory class-distribution sensitivity analysis; reported separately.
