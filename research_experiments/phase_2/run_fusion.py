"""Experiment 1 / 5: multimodal late fusion, stacking and decision rules on the phase-2 OOF predictions.
    python -m research_experiments.phase_2.run_fusion --stage combos|decisions
"""
import argparse

from research_experiments.phase_2.fusion import run

H, HN, I = "hdr_all_skLR", "hdr_numeric_skLR", "img_resnet18_baseline_recipe"


def combos(F, extra=()):
    plan = [
        # A. header + image (R = the current default architecture: original header model + ResNet-18, wavg)
        ("wavg", [H, I], "R_hdrall+img_wavg"), ("wavg", [HN, I], None), ("vote", [H, I], None),
        ("stack_lr", [H, I], None), ("stack_lr_hdr", [I], "stack_lr_hdrnum+img"),
        # C. header + ECGFounder (late)
        ("wavg", [HN, F], None), ("wavg", [H, F], None), ("vote", [HN, F], None),
        ("stack_lr", [HN, F], None), ("stack_lr_hdr", [F], "stack_lr_hdrnum+ecgf"),
        ("stack_xgb_hdr", [F], "stack_xgb_hdrnum+ecgf"), ("stack_mlp_hdr", [F], "stack_mlp_hdrnum+ecgf"),
        # D. image + ECGFounder
        ("wavg", [I, F], None), ("vote", [I, F], None), ("stack_lr", [I, F], None),
        # E. header + image + ECGFounder
        ("wavg", [HN, I, F], None), ("wavg", [H, I, F], None), ("vote", [HN, I, F], None),
        ("stack_lr", [HN, I, F], None), ("stack_lr_hdr", [I, F], "stack_lr_hdrnum+img+ecgf"),
        ("stack_xgb_hdr", [I, F], "stack_xgb_hdrnum+img+ecgf"), ("stack_mlp_hdr", [I, F], "stack_mlp_hdrnum+img+ecgf"),
    ]
    for m, tags, name in list(plan) + list(extra):
        run(m, tags, name=name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True)
    ap.add_argument("--ecgf", default="ecgf_f6_lin_cedist1"); ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    if a.stage == "combos":
        combos(a.ecgf)


if __name__ == "__main__":
    main()
