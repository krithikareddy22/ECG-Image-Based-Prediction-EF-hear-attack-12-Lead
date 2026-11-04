"""Shared constants for the ECG -> LVEF category project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # generated cache (git-ignored)
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# Folder names in LV_ECG_Dataset_<year>/ are the labels.
# Order = increasing severity of LV systolic dysfunction, so index is ordinal.
CLASSES = ["Normal", "Mild", "Moderate", "Severe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# How the folders were labelled from the echo report (ECG-echo matching notebook, since removed;
# EF by Teichholz). Kept here for documentation and for the README.
EF_RANGES = {
    "Normal": "EF > 52% (male) / EF >= 54% (female)",
    "Mild": "EF 41-52% (male) / 41-53% (female)",
    "Moderate": "EF 31-40%",
    "Severe": "EF < 31%",
}

# Screening interpretation of each category.
# "Heart-failure likelihood" = P(Mild) + P(Moderate) + P(Severe) = 1 - P(Normal),
# i.e. the model's probability that LV systolic function is reduced.
FURTHER_EVALUATION = {
    "Normal": "NO  - LV function predicted normal; routine care",
    "Mild": "YES - mildly reduced EF; refer for echo / cardiology follow-up",
    "Moderate": "YES - moderately reduced EF; cardiology evaluation and treatment advised",
    "Severe": "YES - severely reduced EF; urgent cardiology evaluation advised",
}
HF_THRESHOLD = 0.5            # fallback; each checkpoint stores its own tuned threshold
HF_TARGET_SENSITIVITY = 0.80  # threshold is tuned on validation to reach this sensitivity

# All 1,047 PDFs come from the same ECG machine/layout: one page of 2296 x 1728 pt.
# The pink grid (= the waveform area) is at y = 547..1697 pt on every file.
# We crop to the waveform area only, which removes the header (patient name, age,
# sex, and the machine's own text interpretation) and the footer.
PAGE_SIZE_PT = (2296, 1728)
WAVEFORM_CLIP_PT = (60, 547, 2200, 1600)  # x0, y0, x1, y1 in PDF points

# Image size fed to the networks (height, width). Aspect matches the crop (~2:1).
IMG_H, IMG_W = 256, 512
# Size stored in the on-disk cache (2x the training size, so a larger input can be
# tried without re-rendering).
CACHE_H, CACHE_W = 512, 1024
