"""Assemble the robust comparison (development-set CV only; locked test untouched).
    python -m research_experiments.phase_2.robust_run [--fusion-prefix ff_hdrnum+ecgf_mlp]
"""
import argparse

from research_experiments.phase_2.p2common import RES
from research_experiments.phase_2.robust_eval import fmt, make_seed_ensembles, summarise


def runs(prefix, seeds=(42, 43, 44), repeats=(0, 1, 2), seed42_name=None):
    """Tag naming of cv_features: seed 42 -> '<prefix>' (or seed42_name), other seeds '<prefix>_seed<s>';
    repeat r>0 appends '_r<r>'."""
    out = []
    for r in repeats:
        for s in seeds:
            base = (seed42_name or prefix) if s == 42 else f"{prefix}_seed{s}"
            out.append(base + (f"_r{r}" if r else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-prefix", default="ff_hdrnum+ecgf_mlp")
    ap.add_argument("--out", default="robust")
    a = ap.parse_args()
    fus = runs(a.fusion_prefix)
    emlp = ["ecgf_aligned_f6_mlp", "ecgf_f6_mlp_seed43", "ecgf_f6_mlp_seed44"] + \
        [f"ecgf_f6_mlp_seed{s}_r{r}" for r in (1, 2) for s in (42, 43, 44)]
    ens_f = make_seed_ensembles("fusion_mlp", fus)
    ens_e = make_seed_ensembles("ecgf_mlp", emlp)
    det = lambda t: [t, t + "_r1", t + "_r2"]
    models = {
        "Header all (orig. LR)": det("hdr_all_skLR"),
        "Header numeric (LR)": det("hdr_numeric_skLR"),
        "ResNet-18 image": ["img_resnet18_baseline_recipe"],
        "Header + ResNet-18 (R, current default)": ["R_hdrall+img_wavg"],
        "ECGFounder frozen, linear": det("ecgf_aligned_f6_lin"),
        "ECGFounder frozen, linear + dist. penalty": det("ecgf_f6_lin_cedist1"),
        "ECGFounder frozen, MLP (single seed)": emlp,
        "ECGFounder frozen, MLP (3-seed ens.)": ens_e,
        "Header num. + ECGFounder, linear fusion": det("ff_hdrnum+ecgf_lin_w0.5"),
        "Header num. + ECGFounder, MLP fusion (single seed)": fus,
        "Header num. + ECGFounder, MLP fusion (3-seed ens.)": ens_f,
    }
    M = "Header num. + ECGFounder, MLP fusion (3-seed ens.)"
    comparisons = [("Header + ResNet-18 (R, current default)", M), ("ECGFounder frozen, MLP (3-seed ens.)", M),
                   ("ECGFounder frozen, linear", M), ("Header numeric (LR)", M), ("Header all (orig. LR)", M),
                   ("Header num. + ECGFounder, linear fusion", M),
                   ("Header num. + ECGFounder, MLP fusion (single seed)", M)]
    S, C = summarise(models, comparisons, a.out)
    txt = fmt(S, C)
    (RES / f"{a.out}_tables.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
