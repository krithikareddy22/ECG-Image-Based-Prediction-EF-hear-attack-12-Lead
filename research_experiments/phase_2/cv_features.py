"""CV runner for models on fixed (label-free) features: header blocks, ECGFounder representations,
and feature-level fusion of both. Identical folds (protocol.py), inner-train fit, inner-val selection.

    python -m research_experiments.phase_2.cv_features --group audit_inputs
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.common import ROOT
from research_experiments.header_ablation import DEMO, MEAS
from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.p2common import MOD, PCOLS, RES, cv_report, s3, save_cv, short
from research_experiments.phase_2.protocol import dev_and_test, split_indices
from ecg_ef.header import design_matrix
from ecg_ef.metrics import evaluate

META, D, T = dev_and_test()
ALL = pd.concat([D, T], ignore_index=True)          # row order: D then T
nD = len(D)
_H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[ALL.ecg_id].reset_index()
_ECGF = {}


def ecgf(mode="aligned", kind="f6"):
    if mode not in _ECGF:
        z = np.load(MOD / f"ecgf_{mode}_features.npz")
        pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[ALL.ecg_id].values
        _ECGF[mode] = {k: z[k][pos] for k in z.files if k != "ecg_id"}
    z = _ECGF[mode]
    return np.concatenate([z[k] for k in kind.split("+")], 1)


def header_block(fit_mask, part="numeric"):
    """Header design matrix for ALL rows; vocabulary + medians from the fitting rows only."""
    tr = _H[fit_mask]
    cnt = pd.Series([s for L in tr.statements.fillna("").str.split("|") for s in L if s]).value_counts()
    vocab = sorted(cnt[cnt >= 5].index)
    X = design_matrix(_H, vocab)
    verdict = [c for c in X.columns if c.startswith("summary=")]; stm = [c for c in X.columns if c.startswith("stmt=")]
    cols = {"numeric": DEMO + MEAS, "measurements": MEAS, "demographics": DEMO, "verdict": verdict, "statements": stm,
            "text": verdict + stm, "all": DEMO + MEAS + verdict + stm, "numeric+verdict": DEMO + MEAS + verdict}[part]
    X = X[cols]
    return X.fillna(X[fit_mask].median()).fillna(0).values.astype(float)


def blockstd(blocks, fit_mask, weights=None):
    """Standardise each block on the fitting rows, scale block b by sqrt(1/d_b) * w_b (equal total variance)."""
    out = []
    for i, B in enumerate(blocks):
        mu, sd = B[fit_mask].mean(0), B[fit_mask].std(0) + 1e-6
        w = (weights[i] if weights else 1.0) / np.sqrt(B.shape[1])
        out.append((B - mu) / sd * w)
    return np.concatenate(out, 1)


def make_X(spec, fit_mask):
    """spec: list of ('hdr', part) / ('ecgf', mode, kind) blocks; >1 block -> block-standardised concat."""
    blocks = [header_block(fit_mask, s[1]) if s[0] == "hdr" else ecgf(s[1], s[2]) for s in spec["blocks"]]
    if len(blocks) == 1:
        return blocks[0], False
    return blockstd(blocks, fit_mask, spec.get("block_weights")), True


def run(tag, spec, cfg, repeat=0, seed=42, sk_lr=False, overwrite=False):
    if (RES / tag / "cv_metrics.json").exists() and not overwrite:
        return json.loads((RES / tag / "cv_metrics.json").read_text())
    y = D.label_idx.values
    P = np.zeros((nD, 4)); pred = np.zeros(nD, int); fold = np.full(nD, -1); PT = []; info = []
    t0 = time.time()
    for k in range(5):
        itr, iva, ote = split_indices(D, repeat, k)
        fit_mask = np.zeros(len(ALL), bool); fit_mask[itr] = True
        X, prestd = make_X(spec, fit_mask)
        XT = X[nD:]
        if sk_lr:     # the project's original header recipe (sklearn LR, C grid, S3 on inner-val)
            best = None
            for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
                clf.fit(X[itr], y[itr]); sc = s3(evaluate(y[iva], clf.predict_proba(X[iva])))
                if best is None or sc > best[0]:
                    best = (sc, C, clf)
            sc, C, clf = best
            P[ote] = clf.predict_proba(X[ote]); pred[ote] = P[ote].argmax(1); PT.append(clf.predict_proba(XT))
            info.append(dict(fold=k, C=C, val_s3=sc))
        else:
            c = dict(cfg, no_std=prestd or cfg.get("no_std", False))
            f = p2heads.fit(X[itr], y[itr], X[iva], y[iva], c, seed=seed)
            P[ote], pred[ote] = f["predict"](X[ote]); PT.append(f["predict"](XT)[0])
            info.append(dict(fold=k, l2=f["l2"], val_s3=f["val_s3"], n_params=f["n_params"], best_epoch=f["best_epoch"]))
        fold[ote] = k
    oof = pd.DataFrame(P, columns=PCOLS); oof.insert(0, "ecg_id", D.ecg_id.values); oof.insert(1, "fold", fold)
    oof["pred"] = pred
    locked = pd.DataFrame(np.mean(PT, 0), columns=PCOLS); locked.insert(0, "ecg_id", T.ecg_id.values)
    save_cv(tag, oof, locked, dict(spec=spec, cfg=cfg, repeat=repeat, seed=seed, sk_lr=sk_lr, folds=info,
                                   minutes=(time.time() - t0) / 60), overwrite=True)
    m = cv_report(tag, D, P, pred if cfg.get("loss") == "coral" else None, repeat=repeat)
    print(f"[{tag}] {short(m)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return m


H_NUM, H_ALL = ("hdr", "numeric"), ("hdr", "all")
E = lambda mode="aligned", kind="f6": ("ecgf", mode, kind)
LIN = dict(head="linear", loss="ce", weight="balanced")

GROUPS = {
    "reference": [
        ("hdr_all_skLR", dict(blocks=[H_ALL]), {}, dict(sk_lr=True)),
        ("hdr_numeric_skLR", dict(blocks=[H_NUM]), {}, dict(sk_lr=True)),
        ("hdr_text_skLR", dict(blocks=[("hdr", "text")]), {}, dict(sk_lr=True)),
        ("hdr_numeric+verdict_skLR", dict(blocks=[("hdr", "numeric+verdict")]), {}, dict(sk_lr=True)),
    ],
    "audit_inputs": [(f"ecgf_{m}_f6_lin", dict(blocks=[E(m)]), LIN, {}) for m in
                     ("aligned", "aligned_meas", "leadII12", "onelead_II", "tiled")],
    "ecgf_features": [
        ("ecgf_aligned_f6+f6max_lin", dict(blocks=[E("aligned", "f6+f6max")]), LIN, {}),
        ("ecgf_aligned_f5_lin", dict(blocks=[E("aligned", "f5")]), LIN, {}),
        ("ecgf_aligned_f4+f5+f6_lin", dict(blocks=[E("aligned", "f4+f5+f6")]), LIN, {}),
        ("ecgf_aligned_logits_lin", dict(blocks=[E("aligned", "logits")]), LIN, {}),
        ("ecgf_aligned_f6+logits_lin", dict(blocks=[E("aligned", "f6+logits")]), LIN, {}),
        ("ecgf_aligned_f6_mlp", dict(blocks=[E("aligned")]), dict(LIN, head="mlp", hidden=64, dropout=0.5), {}),
    ],
}

F6 = E("aligned", "f6")
GROUPS["imbalance"] = [
    ("ecgf_f6_lin_ce_unweighted", dict(blocks=[F6]), dict(LIN, weight="none"), {}),
    ("ecgf_f6_lin_ce_sqrtw", dict(blocks=[F6]), dict(LIN, weight="sqrt"), {}),
    ("ecgf_f6_lin_ce_effnum", dict(blocks=[F6]), dict(LIN, weight="effnum"), {}),
    ("ecgf_f6_lin_focal1", dict(blocks=[F6]), dict(LIN, loss="focal", gamma=1.0), {}),
    ("ecgf_f6_lin_focal2", dict(blocks=[F6]), dict(LIN, loss="focal", gamma=2.0), {}),
    ("ecgf_f6_lin_ce_smooth0.1", dict(blocks=[F6]), dict(LIN, smooth=0.1), {}),
    ("ecgf_f6_mlp_balsampler", dict(blocks=[F6]), dict(LIN, head="mlp", hidden=64, dropout=0.5, sampler="balanced"), {}),
    ("ecgf_f6_mlp_balsampler_noise", dict(blocks=[F6]), dict(LIN, head="mlp", hidden=64, dropout=0.5, sampler="balanced", feat_noise=0.3), {}),
]
GROUPS["ordinal"] = [
    ("ecgf_f6_lin_coral", dict(blocks=[F6]), dict(LIN, loss="coral"), {}),
    ("ecgf_f6_lin_cumlink", dict(blocks=[F6]), dict(LIN, loss="cumlink"), {}),
    ("ecgf_f6_lin_cedist1", dict(blocks=[F6]), dict(LIN, loss="ce_dist", lam=1.0), {}),
    ("ecgf_f6_lin_cedist3", dict(blocks=[F6]), dict(LIN, loss="ce_dist", lam=3.0), {}),
    ("hdrnum_lin_ce", dict(blocks=[H_NUM]), LIN, {}),
    ("hdrnum_lin_cumlink", dict(blocks=[H_NUM]), dict(LIN, loss="cumlink"), {}),
    ("hdrnum_lin_cedist1", dict(blocks=[H_NUM]), dict(LIN, loss="ce_dist", lam=1.0), {}),
]
GROUPS["featfusion"] = [
    ("ff_hdrnum+ecgf_lin", dict(blocks=[H_NUM, F6]), LIN, {}),
    ("ff_hdrnum+ecgf_lin_w2", dict(blocks=[H_NUM, F6], block_weights=[2.0, 1.0]), LIN, {}),
    ("ff_hdrnum+ecgf_lin_w0.5", dict(blocks=[H_NUM, F6], block_weights=[0.5, 1.0]), LIN, {}),
    ("ff_hdrall+ecgf_lin", dict(blocks=[H_ALL, F6]), LIN, {}),
    ("ff_hdrnumverdict+ecgf_lin", dict(blocks=[("hdr", "numeric+verdict"), F6]), LIN, {}),
    ("ff_hdrnum+ecgf_mlp", dict(blocks=[H_NUM, F6]), dict(LIN, head="mlp", hidden=64, dropout=0.5), {}),
    ("ff_hdrnum+ecgf_lin_cedist1", dict(blocks=[H_NUM, F6]), dict(LIN, loss="ce_dist", lam=1.0), {}),
    ("ff_hdrnum+ecgf_lin_cumlink", dict(blocks=[H_NUM, F6]), dict(LIN, loss="cumlink"), {}),
]

MLPF = dict(LIN, head="mlp", hidden=64, dropout=0.5)
FUS = dict(blocks=[H_NUM, F6])
GROUPS["mlp_robust"] = [            # seed variance of the best fusion + its linear counterpart
    ("ff_hdrnum+ecgf_mlp_seed43", FUS, MLPF, dict(seed=43)),
    ("ff_hdrnum+ecgf_mlp_seed44", FUS, MLPF, dict(seed=44)),
    ("ecgf_f6_mlp_seed43", dict(blocks=[F6]), MLPF, dict(seed=43)),
    ("hdrnum_mlp", dict(blocks=[H_NUM]), MLPF, {}),
]
GROUPS["mlp_hp"] = [                 # staged HP search on the MLP fusion (Experiment 6), one factor at a time
    ("ff_mlp_h32", FUS, dict(MLPF, hidden=32), {}),
    ("ff_mlp_h128", FUS, dict(MLPF, hidden=128), {}),
    ("ff_mlp_do0.3", FUS, dict(MLPF, dropout=0.3), {}),
    ("ff_mlp_do0.7", FUS, dict(MLPF, dropout=0.7), {}),
    ("ff_mlp_cedist1", FUS, dict(MLPF, loss="ce_dist", lam=1.0), {}),
    ("ff_mlp_balsampler", FUS, dict(MLPF, sampler="balanced"), {}),
    ("ff_mlp_unweighted", FUS, dict(MLPF, weight="none"), {}),
    ("ff_mlp_hdrw2", dict(FUS, block_weights=[2.0, 1.0]), MLPF, {}),
    ("ff_mlp_f6max", dict(blocks=[H_NUM, E("aligned", "f6+f6max")]), MLPF, {}),
    ("ff_mlp_hdrnumverdict", dict(blocks=[("hdr", "numeric+verdict"), F6]), MLPF, {}),
]
GROUPS["repeats"] = [                # split variance (run with --repeat 1 / 2)
    ("ff_hdrnum+ecgf_mlp", FUS, MLPF, {}),
    ("ff_hdrnum+ecgf_lin_w0.5", dict(FUS, block_weights=[0.5, 1.0]), LIN, {}),
    ("ecgf_f6_lin_cedist1", dict(blocks=[F6]), dict(LIN, loss="ce_dist", lam=1.0), {}),
    ("ecgf_aligned_f6_lin", dict(blocks=[F6]), LIN, {}),
    ("hdr_numeric_skLR", dict(blocks=[H_NUM]), {}, dict(sk_lr=True)),
    ("hdr_all_skLR", dict(blocks=[H_ALL]), {}, dict(sk_lr=True)),
]

# Experiment 6 / robust validation: every MLP-fusion variant (and the base config) on 3 seeds
GROUPS["mlp_hp_seeds"] = [(f"{t}_seed{sd}", sp, c, dict(kw, seed=sd))
                          for t, sp, c, kw in GROUPS["mlp_hp"] for sd in (43, 44)]

# robust validation runs (use with --repeat 1 / --repeat 2; seed 42 of the fusion comes from "repeats")
GROUPS["robust_seeds"] = [(f"ff_hdrnum+ecgf_mlp_seed{sd}", FUS, MLPF, dict(seed=sd)) for sd in (43, 44)] + \
                         [(f"ecgf_f6_mlp_seed{sd}", dict(blocks=[F6]), MLPF, dict(seed=sd)) for sd in (42, 43, 44)]
GROUPS["robust_r0_extra"] = [("ecgf_f6_mlp_seed44", dict(blocks=[F6]), MLPF, dict(seed=44))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--group", nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=0)
    a = ap.parse_args()
    for g in a.group:
        for tag, spec, cfg, kw in GROUPS[g]:
            run(tag + (f"_r{a.repeat}" if a.repeat else ""), spec, cfg, repeat=a.repeat, **kw)


if __name__ == "__main__":
    main()
