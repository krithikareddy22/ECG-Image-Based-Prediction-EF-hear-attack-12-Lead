"""Is the header + ECGFounder MLP fusion learning COMPLEMENTARY information? (development-set OOF only)

1. error overlap of the single-modality models (header numeric LR vs ECGFounder): joint correctness table,
   Cohen's kappa between their predictions, correlation of their P(reduced EF), oracle accuracy
2. what the fusion fixes: fusion accuracy within each joint-correctness cell
3. block-permutation importance inside the refitted fusion MLP (repeat 0, 3 seeds, 20 permutations):
   permute the header block (or the ECGFounder block) across ECGs of the outer fold -> metric drop
4. how much of each printed measurement is linearly decodable from the frozen ECGFounder embedding (CV R^2)
    python -m research_experiments.phase_2.complementarity --fusion-cfg base
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.analyses import q1_measurements_from_embedding
from research_experiments.phase_2.cv_features import ALL, D, H_NUM, F6, LIN, make_X, nD
from research_experiments.phase_2.p2common import PCOLS, RES, s3
from research_experiments.phase_2.protocol import split_indices
from ecg_ef.metrics import evaluate

Y = D.label_idx.values


def oof(tag):
    o = pd.read_csv(RES / tag / "oof.csv").set_index("ecg_id").loc[D.ecg_id]
    P = o[PCOLS].values
    return P, (o.pred.values if "pred" in o else P.argmax(1))


def overlap(tag_h, tag_e, tag_f):
    Ph, ph = oof(tag_h); Pe, pe = oof(tag_e); Pf, pf = oof(tag_f)
    ch, ce, cf = ph == Y, pe == Y, pf == Y
    cells = {"both right": ch & ce, "header only right": ch & ~ce, "ECGFounder only right": ~ch & ce, "both wrong": ~ch & ~ce}
    tab = {k: dict(n=int(m.sum()), share=float(m.mean()), fusion_accuracy_in_cell=float(cf[m].mean()) if m.any() else None)
           for k, m in cells.items()}
    by_class = {}
    for k, c in enumerate(["Normal", "Mild", "Moderate", "Severe"]):
        m = Y == k
        by_class[c] = dict(header_recall=float(ch[m].mean()), ecgf_recall=float(ce[m].mean()), fusion_recall=float(cf[m].mean()),
                           oracle_recall=float((ch | ce)[m].mean()))
    return dict(joint_correctness=tab, kappa_between_predictions=float(cohen_kappa_score(ph, pe)),
                corr_P_reduced=float(np.corrcoef(1 - Ph[:, 0], 1 - Pe[:, 0])[0, 1]),
                corr_expected_severity=float(np.corrcoef(Ph @ np.arange(4), Pe @ np.arange(4))[0, 1]),
                accuracy=dict(header=float(ch.mean()), ecgf=float(ce.mean()), fusion=float(cf.mean()), oracle=float((ch | ce).mean())),
                recall_by_class=by_class)


def permutation_importance(cfg, seeds=(42, 43, 44), n_perm=20):
    spec = dict(blocks=[H_NUM, F6]); rng = np.random.default_rng(0)
    rows = []
    for sd in seeds:
        P_base = np.zeros((nD, 4)); P_perm = {b: np.zeros((n_perm, nD, 4)) for b in ("header", "ecgfounder")}
        for k in range(5):
            itr, iva, ote = split_indices(D, 0, k)
            fm = np.zeros(len(ALL), bool); fm[itr] = True
            X, _ = make_X(spec, fm)
            f = p2heads.fit(X[itr], Y[itr], X[iva], Y[iva], dict(cfg, no_std=True), seed=sd)
            P_base[ote] = f["predict"](X[ote])[0]
            dh = 16  # header numeric block width
            for b, sl in (("header", slice(0, dh)), ("ecgfounder", slice(dh, X.shape[1]))):
                for j in range(n_perm):
                    Xp = X[ote].copy(); Xp[:, sl] = Xp[rng.permutation(len(ote))][:, sl]
                    P_perm[b][j, ote] = f["predict"](Xp)[0]
        mb = evaluate(Y, P_base)
        for b in P_perm:
            ms = [evaluate(Y, P_perm[b][j]) for j in range(n_perm)]
            rows.append(dict(seed=sd, block=b, base_accuracy=mb["accuracy"], base_balanced_accuracy=mb["balanced_accuracy"],
                             base_macro_f1=mb["macro_f1"],
                             drop_accuracy=float(mb["accuracy"] - np.mean([m["accuracy"] for m in ms])),
                             drop_balanced_accuracy=float(mb["balanced_accuracy"] - np.mean([m["balanced_accuracy"] for m in ms])),
                             drop_macro_f1=float(mb["macro_f1"] - np.mean([m["macro_f1"] for m in ms])),
                             drop_f1_mild=float(mb["per_class"]["Mild"]["f1"] - np.mean([m["per_class"]["Mild"]["f1"] for m in ms])),
                             drop_s3=float(s3(mb) - np.mean([s3(m) for m in ms]))))
        print(rows[-2:], flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default="hdr_numeric_skLR"); ap.add_argument("--ecgf", required=True)
    ap.add_argument("--fusion", required=True); ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    cfg = dict(LIN, head="mlp", hidden=64, dropout=0.5, **json.loads(a.cfg))
    out = dict(overlap=overlap(a.header, a.ecgf, a.fusion), ecgfounder_decodes_measurements_cv_r2=q1_measurements_from_embedding(),
               fusion_cfg=cfg)
    print(json.dumps(out, indent=1), flush=True)
    pi = permutation_importance(cfg)
    pi.to_csv(RES / "complementarity_permutation_importance.csv", index=False)
    out["permutation_importance_mean_over_seeds"] = pi.groupby("block")[[c for c in pi.columns if c.startswith("drop")]].mean().to_dict(orient="index")
    (RES / "complementarity.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["permutation_importance_mean_over_seeds"], indent=1))


if __name__ == "__main__":
    main()
