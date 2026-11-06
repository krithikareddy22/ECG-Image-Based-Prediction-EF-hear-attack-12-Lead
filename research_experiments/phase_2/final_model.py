"""Final model: header numeric + frozen ECGFounder -> MLP fusion, trained on the FULL development set.

1. hyper-parameters fixed from the development CV (no test data): L2 = most frequently selected value,
   epochs = median best epoch of the CV fits of this configuration
2. train on all 837 development ECGs (3 seeds -> probability ensemble)
3. locked-test probabilities written to locked_test/FINAL_<name>.csv  (NOT scored here)
4. calibration parameters (prior alpha / per-class bias) selected on the development OOF predictions of the
   same configuration (seed ensembles of all CV repeats pooled), prior from development labels only
5. checkpoint + preprocessing statistics saved for best_model/predict_pdf.py

    python -m research_experiments.phase_2.final_model --name fusion_mlp --cv-prefix ff_hdrnum+ecgf_mlp
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from research_experiments.common import seed_all
from research_experiments.phase_2.accuracy_tradeoff import ALPHAS, BIASES, decide, mets
from research_experiments.phase_2.cv_features import ALL, D, T, MLPF, ecgf, header_block, nD
from research_experiments.phase_2.p2common import LOCK, PCOLS, RES, guard
from research_experiments.phase_2.p2heads import Head, class_w

P2 = RES.parent
Y = D.label_idx.values


def cv_hparams(prefix):
    l2s, eps = [], []
    for d in sorted(RES.glob(prefix + "*")):
        if d.name.startswith(prefix) and (d / "info.json").exists() and ("seed" in d.name[len(prefix):] or d.name[len(prefix):] in ("", "_r1", "_r2")):
            for fo in json.loads((d / "info.json").read_text()).get("folds", []):
                l2s.append(fo["l2"])
                if fo.get("best_epoch") is not None:
                    eps.append(fo["best_epoch"] + 1)            # best_epoch is 0-based in p2heads
    return Counter(l2s).most_common(1)[0][0], int(np.median(eps)) if eps else None, len(l2s), len(eps)


def blocks_full():
    fm = np.zeros(len(ALL), bool); fm[:nD] = True
    H = header_block(fm, "numeric"); F = ecgf("aligned", "f6")
    stats = {}
    out = []
    for name, B in (("header_numeric", H), ("ecgfounder_f6", F)):
        mu, sd = B[:nD].mean(0), B[:nD].std(0) + 1e-6
        out.append((B - mu) / sd / np.sqrt(B.shape[1])); stats[name] = dict(mu=mu.tolist(), sd=sd.tolist(), dim=B.shape[1])
    hdr_median = pd.DataFrame(header_block(fm, "numeric")[:nD]).median().tolist()
    return np.concatenate(out, 1), stats


def train_fixed(X, y, cfg, l2, epochs, seed):
    seed_all(seed)
    m = Head(X.shape[1], cfg.get("loss", "ce"), "mlp", cfg.get("hidden", 64), cfg.get("dropout", 0.5))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=l2 * 100)
    cw = class_w(y, cfg.get("weight", "balanced")); xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train(); idx = torch.randperm(len(y), generator=g)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; lo = m.loss(xt[b], yt[b], cw, cfg); opt.zero_grad(); lo.backward(); opt.step()
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fusion_mlp"); ap.add_argument("--cv-prefix", default="ff_hdrnum+ecgf_mlp")
    ap.add_argument("--cfg", default="{}"); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--ens-tags", nargs="+", default=["ens3__fusion_mlp_r0", "ens3__fusion_mlp_r1", "ens3__fusion_mlp_r2"])
    a = ap.parse_args()
    cfg = dict(MLPF, **json.loads(a.cfg))
    l2, epochs, n_l2, n_ep = cv_hparams(a.cv_prefix)
    X, stats = blocks_full()
    models = [train_fixed(X[:nD], Y, cfg, l2, epochs, s) for s in a.seeds]
    with torch.no_grad():
        PT = np.mean([m.probs(torch.tensor(X[nD:], dtype=torch.float32)).numpy() for m in models], 0)
        PD_train = np.mean([m.probs(torch.tensor(X[:nD], dtype=torch.float32)).numpy() for m in models], 0)
    # calibration parameters from development OOF predictions only (all repeats pooled), prior = development labels
    P_oof = np.concatenate([pd.read_csv(RES / t / "oof.csv").set_index("ecg_id").loc[D.ecg_id][PCOLS].values for t in a.ens_tags])
    y_oof = np.tile(Y, len(a.ens_tags)); prior = np.bincount(Y, minlength=4) / len(Y)
    calib = {}
    for method, cands in (("prior", ALPHAS), ("bias", BIASES)):
        base = mets(y_oof, P_oof.argmax(1))
        for obj in ("accuracy", "protected"):
            best = None
            for c in cands:
                mm = mets(y_oof, decide(P_oof, method, c, prior))
                if obj == "protected" and (mm["balanced_accuracy"] < base["balanced_accuracy"] - 0.01 or mm["macro_f1"] < base["macro_f1"] - 0.01):
                    continue
                if best is None or mm["accuracy"] > best[1] + 1e-12:
                    best = (c, mm["accuracy"])
            calib[f"{method}, {obj}"] = float(best[0]) if method == "prior" else np.asarray(best[0]).tolist()
    out = guard(P2 / "best_model"); out.mkdir(exist_ok=True)
    torch.save(dict(state_dicts=[m.state_dict() for m in models], cfg=cfg, l2=l2, epochs=epochs, seeds=a.seeds,
                    block_stats=stats, prior=prior.tolist(), calibration=calib, in_dim=X.shape[1],
                    header_numeric_columns=["age", "male", "Rate", "RR", "PR", "QRSD", "QT", "QTcB", "QTcF", "PR_missing",
                                            "P_axis_sin", "P_axis_cos", "QRS_axis_sin", "QRS_axis_cos", "T_axis_sin", "T_axis_cos"],
                    header_numeric_train_medians=pd.DataFrame(header_block(np.r_[np.ones(nD, bool), np.zeros(len(T), bool)], "numeric")[:nD]).median().tolist()),
               out / f"FINAL_{a.name}.pt")
    locked = pd.DataFrame(PT, columns=PCOLS); locked.insert(0, "ecg_id", T.ecg_id.values)
    for k, c in calib.items():
        method = k.split(",")[0]
        locked[f"pred_calibrated[{k}]"] = decide(PT, method, np.asarray(c) if method == "bias" else c, prior)
    locked["pred_original"] = PT.argmax(1)
    guard(LOCK).mkdir(exist_ok=True); locked.to_csv(LOCK / f"FINAL_{a.name}.csv", index=False, float_format="%.6f")
    info = dict(name=a.name, cfg=cfg, l2=l2, epochs=epochs, n_cv_fits_for_l2=n_l2, n_cv_fits_for_epochs=n_ep, seeds=a.seeds,
                trained_on="all 837 development ECGs", calibration_selected_on=a.ens_tags, calibration=calib,
                dev_resubstitution_accuracy_not_a_performance_estimate=float(np.mean(PD_train.argmax(1) == Y)),
                locked_test="probabilities written, NOT scored")
    (out / f"FINAL_{a.name}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
