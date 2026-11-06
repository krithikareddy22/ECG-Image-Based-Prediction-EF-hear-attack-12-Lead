"""IMG-ONLY exploratory recipe (AMENDMENT_IMGONLY.md): header-free ECGFounder vote and vote + nested prior rule.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.imgonly
"""
import json

import numpy as np
import torch

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.experiments.nested import BASES, LIN, MLPF
from research_experiments.phase_6.p6common import inner_oof
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm.existing6b import fit_member

SPEC = {"ecgfmlp": (["ecgf"], "p2heads", MLPF, B.SEEDS), "ecgflindist": BASES["ecgflindist"]}
P2 = B.ROOT / "research_experiments" / "phase_2" / "results"


def sfx(r):
    return "" if r == 0 else f"_r{r}"


def outer_ecgfmlp(r):
    """Committed Phase 2 OOF of the MLP on ECGFounder f6 (seeds 42/43/44) if present; same recipe as SPEC."""
    paths = [P2 / f"ecgf_aligned_f6_mlp{sfx(r)}" / "oof.csv" if s == 42 and r == 0 else P2 / (f"ecgf_f6_mlp_seed{s}{sfx(r)}") / "oof.csv"
             for s in B.SEEDS]
    if all(p.exists() for p in paths):
        return np.mean([B.load_oof_csv(p)[0] for p in paths], 0), "phase_2 committed OOF"
    P = np.zeros((B.nD, 4))
    for k in range(5):
        itr, iva, ote = B.split_indices(r, k)
        fm = np.zeros(B.nD, bool); fm[itr] = True
        X, pre = B.make_X(["ecgf"], fm)
        P[ote] = np.mean([fit_member("mlpfusion", X[itr], B.Y[itr], X[iva], B.Y[iva], s, pre)(X[ote]) for s in B.SEEDS], 0)
    return P, "re-trained"


def main():
    torch.set_num_threads(1)
    out = dict(models={})
    for r in (0, 1, 2):
        Pm, src = outer_ecgfmlp(r)
        Pl = B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]
        Pv = (Pm + Pl) / 2
        fo = B.fold_vec(r); pred = np.zeros(B.nD, int); al = []
        for k in range(5):
            inn = [inner_oof(b, r, k, SPEC[b]) for b in ("ecgfmlp", "ecgflindist")]
            TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
            prior = np.bincount(B.Y[TR], minlength=4) / len(TR)
            a = choose_alpha(Pin, B.Y[TR], prior, "protected"); al.append(a)
            pred[fo == k] = B.decide(Pv[fo == k], "prior", a, prior)
        for name, P, p in (("ECGF-MLP (no header)", Pm, Pm.argmax(1)), ("IMG-vote", Pv, Pv.argmax(1)), ("IMG-C3", Pv, pred)):
            out["models"].setdefault(name, []).append(dict(B.metrics(B.Y, P, p), source=src if name.startswith("ECGF") else "", alpha=al if name == "IMG-C3" else None))
    summ = {n: {k: float(np.mean([x[k] for x in v])) for k in ("accuracy", "balanced_accuracy", "macro_f1", "recall_Normal", "recall_Mild",
                                                              "recall_Moderate", "recall_Severe", "f1_Mild", "min_recall")}
            for n, v in out["models"].items()}
    acc_cells = {}
    for n in summ:
        summ[n]["accuracy_per_repeat"] = [round(x["accuracy"], 4) for x in out["models"][n]]
    (B.guard(B.RES) / "imgonly_exploratory.json").write_text(json.dumps(dict(summary=summ), indent=1))
    for n, v in summ.items():
        print(n, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})


if __name__ == "__main__":
    main()
