"""Paired comparisons against each experiment's OWN control (same architecture, same folds, same seeds):
  4E multitask variants  vs  mt_lam0_A          (identical network without EF supervision = phase-2 MLP fusion)
  4F label-quality runs   vs  bench_mlpfusion    (same MLP fusion trained on all ECGs / folder labels)
  4D regression decoders  vs  bench_mlpfusion    (direct 4-class model on the same features)
Corrected repeated-CV t-test on 15 (repeat, fold) cells, all 837 development ECGs, folder labels; Holm within each
experiment family and metric.  -> results/phase4_vs_controls.{json,md}
    python -m research_experiments.phase_4.controls4
"""
from __future__ import annotations

import json

import numpy as np

from research_experiments.phase_2.p2common import holm
from research_experiments.phase_4.benchmark4 import cells, combined, model_groups
from research_experiments.phase_4.p4common import RES4, REPEATS, corr_ci, corr_t

METS = ("s3", "accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "f1_Severe")


def main():
    G = model_groups()
    fams = {"4E multitask vs lambda-0 control": ("mt_lam0_A", [m for m in G if m.startswith("mt_") and m != "mt_lam0_A"]),
            "4F label quality vs MLP fusion": ("bench_mlpfusion", [m for m in G if m.startswith("lq_")]),
            "4D regression (tuned) vs MLP fusion": ("bench_mlpfusion", [m for m in G if m.startswith("reg_") and m.endswith("__tuned")])}
    out, L = {}, ["# Phase 4: comparisons against each experiment's own control", "",
                  "837 development ECGs, folder labels, corrected repeated-CV t-test on 15 (repeat, fold) cells; Holm within family "
                  "and metric. Differences are candidate − control.", ""]
    for fam, (ctl, cands) in fams.items():
        cc = {r: combined(ctl, G[ctl][r]) for r in REPEATS}
        cell_c = [c for r in REPEATS for c in cells(*cc[r][:2], r)]
        res = {}
        for m in cands:
            cm = {r: combined(m, G[m][r]) for r in REPEATS}
            cell_m = [c for r in REPEATS for c in cells(*cm[r][:2], r)]
            res[m] = {}
            for k in METS:
                dl = [b[k] - a[k] for a, b in zip(cell_c, cell_m)]
                d, p = corr_t(dl); res[m][k] = dict(diff=d, p=p, ci95=corr_ci(dl))
        for k in METS:
            ps = holm([res[m][k]["p"] for m in cands]) if cands else []
            for m, ph in zip(cands, ps):
                res[m][k]["p_holm"] = float(ph)
        out[fam] = dict(control=ctl, results=res)
        L += [f"## {fam} (control `{ctl}`)", "", "| Candidate | " + " | ".join(f"Δ {k} (p; Holm)" for k in METS) + " |",
              "|---|" + "---|" * len(METS)]
        for m in sorted(cands, key=lambda m: -res[m]["s3"]["diff"]):
            L.append(f"| `{m}` | " + " | ".join(f"{res[m][k]['diff']:+.3f} ({res[m][k]['p']:.2f}; {res[m][k]['p_holm']:.2f})" for k in METS) + " |")
        L.append("")
    # AUROC comparisons vs the vote ensemble (ranking quality; Holm within metric over the listed candidates)
    from sklearn.metrics import roc_auc_score
    from research_experiments.phase_4.p4common import Y, fold_vec
    def aucs(P, mk):
        y = Y[mk]; q = P[mk]
        return dict(macro_auroc=float(np.mean([roc_auc_score(y == k, q[:, k]) for k in range(4)])),
                    ef_lt41=float(roc_auc_score(np.isin(y, [2, 3]), q[:, 2] + q[:, 3])),
                    ef_lt31=float(roc_auc_score(y == 3, q[:, 3])))
    cands = [m for m in ("reg_mlp_hdr+ecgf_A__tuned", "reg_ridge_hdr+ecgf_A__tuned", "reg_mlp_hdr+ecgf_C__tuned", "mt_sel",
                         "mt_lam1_A", "lq_dropdisagree", "fus_vote+ridgeA", "fus_vote+mlpregA", "fus_vote+mtsel", "bench_mlpfusion")
             if m in G]
    vv = {r: combined("bench_vote", G["bench_vote"][r]) for r in REPEATS}
    res = {}
    for m in cands:
        cm = {r: combined(m, G[m][r]) for r in REPEATS}
        dl = {k: [] for k in ("macro_auroc", "ef_lt41", "ef_lt31")}
        for r in REPEATS:
            fo = fold_vec(r)
            for f in range(5):
                a, b = aucs(vv[r][0], fo == f), aucs(cm[r][0], fo == f)
                for k in dl:
                    dl[k].append(b[k] - a[k])
        res[m] = {k: dict(zip(("diff", "p"), corr_t(v))) | {"ci95": corr_ci(v)} for k, v in dl.items()}
    for k in ("macro_auroc", "ef_lt41", "ef_lt31"):
        for m, ph in zip(cands, holm([res[m][k]["p"] for m in cands])):
            res[m][k]["p_holm"] = float(ph)
    out["AUROC vs vote ensemble"] = dict(control="bench_vote", results=res)
    L += ["## AUROC vs the vote ensemble (ranking; candidate − vote)", "",
          "| Candidate | Δ macro AUROC (p; Holm) | Δ AUROC EF < 41 (p; Holm) | Δ AUROC EF < 31 (p; Holm) |", "|---|---|---|---|"]
    for m in cands:
        L.append(f"| `{m}` | " + " | ".join(f"{res[m][k]['diff']:+.3f} ({res[m][k]['p']:.3f}; {res[m][k]['p_holm']:.2f})"
                                          for k in ("macro_auroc", "ef_lt41", "ef_lt31")) + " |")
    L.append("")
    (RES4 / "phase4_vs_controls.json").write_text(json.dumps(out, indent=1, default=float))
    (RES4 / "phase4_vs_controls.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
