# Prompt for the laptop Claude Code session: image-only route (EchoNext drawn as hospital printouts)

Paste everything below the line into the laptop Claude Code chat that is already working with EchoNext.

---

New task on branch `research/public-ef`: the **image-only** version of the EchoNext study. The owner needs the model
to classify LVEF from the ECG **image only**. Read `research_experiments/public_ef/IMG_PLAN.md` (pre-registration)
first and follow it exactly. The code is written and smoke-tested; your job is to run it on the real data, on the GPU.

## Hard rules (same as before)
- **EchoNext data:** stays in `C:\krithika_major\public_data\echonext\`. The drawn EchoNext images
  (`img_echonext.npy`) and the trained weights (`img_en_weights.pt`) are EchoNext-derived. They stay in the
  git-ignored `research_experiments/public_ef/data/private/`: never commit, push, upload or paste them anywhere.
- **Hospital locked test set:** do not load or score it. `img_prep.py` reads the 837 development PDFs only.
- **What may be committed:** only code and aggregate JSON (`results/img_*.json`, `results/img_prep_summary.json`)
  plus a short report.
- **Before pushing:**
  - run the privacy scan (`research_experiments.phase_5.p5common.privacy_scan`) on the staged files;
  - check that the staged diff is only `research_experiments/public_ef/`;
  - push only to `research/public-ef`.
- **If anything below fails or looks wrong, stop and report it.** Do not change the plan's settings (image size,
  model, epochs, decision rule) to get a better number.
- **If the signal-route job (`en_prep` / `en_train`) is still running, let it finish.** This is a separate,
  additional study.

## Steps (PowerShell, from the repository root, project Python active, e.g. `C:\krithika_major\p6.ps1`)
1. **Update the branch:**
   ```
   git fetch origin research/public-ef
   git checkout research/public-ef
   git pull origin research/public-ef
   ```
   Check that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
2. **Set the paths** (same as for `en_prep`):
   ```
   $env:ECHONEXT_DIR = "C:\krithika_major\public_data\echonext"
   $env:ECHONEXT_MINI_DIR = "C:\krithika_major\public_data\IntroECG\7-EchoNext Minimodel"
   ```
3. **Hospital images** (837 development PDFs → text removed → waveform area → grid filter → 320×640; about 1 min):
   ```
   python -m research_experiments.public_ef.img_prep --hospital
   ```
4. **Fidelity check before drawing EchoNext.** Redraw 30 development ECGs from `data/signals.npz` with
   `img_render.drawn_image` and compare each with `img_render.hospital_image` of the same PDF.
   - The cloud reference is median pixel correlation 0.993 and median trace-mask IoU 0.967.
   - Report your median and minimum (aggregates only).
   - If the median correlation is below 0.97, stop and report.
5. **Draw EchoNext as printout images** (val + test splits; parallel; roughly 0.4 s per image per core):
   ```
   python -m research_experiments.public_ef.img_prep --echonext --splits val,test
   ```
   Then check `results/img_prep_summary.json`: the image count and the class counts.
   - Open ONE drawn image locally to confirm it shows four trace rows, column ticks and calibration pulses.
   - Do not upload or paste it anywhere.
6. **Pre-train on the EchoNext images** (GPU; ImageNet weights download once from download.pytorch.org):
   ```
   python -m research_experiments.public_ef.img_train --pretrain
   ```
   It writes `results/img_pretrain.json` (hold-out accuracy, per-class recall, EF r).
   - If GPU memory runs out, rerun with `--workers 0`.
   - If memory still runs out, stop and report. Do not shrink the images.
7. **Evaluate on the 837 hospital images** (zero-shot, fine-tuned from EchoNext, fine-tuned from ImageNet only;
   Phase-2 folds 5 × 3; it resumes if interrupted):
   ```
   python -m research_experiments.public_ef.img_train --evaluate
   ```
   It writes `results/img_results.json`.
8. **Report to the owner** in a short `IMG_REPORT.md`, with aggregates only:
   - the fidelity numbers;
   - the EchoNext image and class counts;
   - the hold-out metrics;
   - for ZS / FT-EN / FT-IN and C3: accuracy with 95 % CI, per-class recall and precision, macro-F1, admissibility;
   - the two tests (FT-EN vs FT-IN, FT-EN vs C3) with Holm p.

   **Context to include:**
   - the best image-derived results so far are ULT-stack 0.571 (traces read from the image) and the triage model
     (0.694 on 56 % answered; locked test 0.747 on 50 %);
   - the original pixel ResNet-18 scored 0.471 in CV.

   Say plainly whether EchoNext image pre-training helped.
9. **Commit and push** `research_experiments/public_ef/IMG_REPORT.md` and `research_experiments/public_ef/results/*.json`
   only, after the privacy checks, to `research/public-ef`.

## Optional (only if the owner asks afterwards)
- **More data:** add the `train` split (16 GB) with `--splits train,val,test`. Re-run `--pretrain` and `--evaluate`
  under a new dated amendment in `IMG_PLAN.md`.
- **Three seeds** (42, 43, 44) for eligibility, under an amendment written before running.
