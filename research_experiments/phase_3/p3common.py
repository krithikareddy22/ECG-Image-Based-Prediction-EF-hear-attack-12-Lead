"""Phase-3 storage + CV runner. Same dataset, labels, patient-grouped folds (phase_2/configs/folds_D.csv),
inner-train / inner-val protocol, metrics and statistics as phase 2. Writes ONLY to research_experiments/phase_3/;
phase-1/2 outputs are read-only. The locked test set is predicted (stored) but never scored here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.common import ROOT, bootstrap_ci, full_metrics
from research_experiments.phase_2 import p2heads
from research_experiments.phase_2.cv_features import ALL, D, T, header_block, nD
from research_experiments.phase_2.cv_features import ecgf as ecgf_p2
from research_experiments.phase_2.p2common import PCOLS, s3
from research_experiments.phase_2.protocol import folds, split_indices
from ecg_ef.metrics import evaluate

P3 = ROOT / "research_experiments" / "phase_3"
RES3, LOCK3, MOD3, PLOT3, CONF3 = (P3 / d for d in ("results", "locked_test", "models", "plots", "configs"))
P2RES = ROOT / "research_experiments" / "phase_2" / "results"
Y = D.label_idx.values
_CACHE: dict = {}
EXTRA_BLOCKS: dict = {}        # name -> function(fit_mask) -> matrix over ALL rows (registered by experiment modules)


def guard3(path) -> Path:
    path = Path(path).resolve()
    if P3.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_3/: {path}")
    return path


def ecgf3(mode, kind="f6"):
    key = ("p3", mode)
    if key not in _CACHE:
        z = np.load(MOD3 / f"ecgf_{mode}_features.npz")
        pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[ALL.ecg_id].values
        _CACHE[key] = {k: z[k][pos] for k in z.files if k != "ecg_id"}
    return np.concatenate([_CACHE[key][k] for k in kind.split("+")], 1)


def block(item, fit_mask):
    kind = item[0]
    if kind == "hdr":
        return header_block(fit_mask, item[1])
    if kind == "ecgf":
        return ecgf_p2(item[1], item[2])
    if kind == "ecgf3":
        return ecgf3(item[1], item[2])
    return EXTRA_BLOCKS[kind](fit_mask, *item[1:])


def make_X(spec, fit_mask):
    blocks = [block(b, fit_mask) for b in spec["blocks"]]
    if len(blocks) == 1:
        return blocks[0], False
    out = []
    for i, B in enumerate(blocks):
        mu, sd = B[fit_mask].mean(0), B[fit_mask].std(0) + 1e-6
        # raw_std: plain per-block standardisation (gated/attention models have their own per-modality projections)
        w = (spec.get("block_weights") or [1.0] * len(blocks))[i] / (1.0 if spec.get("raw_std") else np.sqrt(B.shape[1]))
        out.append((B - mu) / sd * w)
    return np.concatenate(out, 1), True


def save_cv3(tag, P, pred, PT, info, repeat):
    d = guard3(RES3 / tag); d.mkdir(parents=True, exist_ok=True)
    f = folds(repeat); fo = f[f.role == "outer_test"].set_index("ecg_id").loc[D.ecg_id].fold.values
    oof = pd.DataFrame(P, columns=PCOLS); oof.insert(0, "ecg_id", D.ecg_id.values); oof.insert(1, "fold", fo); oof["pred"] = pred
    oof.to_csv(d / "oof.csv", index=False, float_format="%.6f")
    if PT is not None:
        guard3(LOCK3).mkdir(exist_ok=True)
        lk = pd.DataFrame(PT, columns=PCOLS); lk.insert(0, "ecg_id", T.ecg_id.values); lk.to_csv(LOCK3 / f"{tag}.csv", index=False, float_format="%.6f")
    (d / "info.json").write_text(json.dumps(dict(info, repeat=repeat), indent=1, default=float))


def cv_report3(tag, P, pred, repeat=0):
    pooled = full_metrics(Y, P, 0.5, pred)
    pooled["ci95"] = bootstrap_ci(Y, P, D.patient.values, pred)
    pooled["s3"] = s3(pooled)
    (RES3 / tag / "cv_metrics.json").write_text(json.dumps(dict(tag=tag, repeat=repeat, pooled=pooled), indent=1, default=float))
    p = pooled
    print(f"[{tag}] acc {p['accuracy']:.3f} bal {p['balanced_accuracy']:.3f} mF1 {p['macro_f1']:.3f} "
          f"Mild {p['per_class']['Mild']['f1']:.3f} QWK {p['qwk']:.3f} AUC {p['auroc_ovr']['macro']:.3f} S3 {p['s3']:.3f}", flush=True)
    return pooled


def run3(tag, spec, cfg=None, repeat=0, seed=42, sk_lr=False, fitter=None, overwrite=False):
    """Standard phase-2 protocol. `fitter(Xtr, ytr, Xva, yva, seed) -> predict_fn(X) -> (probs, pred)` overrides
    the default heads (used for gated fusion / boosting)."""
    tag = tag + (f"_r{repeat}" if repeat else "")
    if (RES3 / tag / "cv_metrics.json").exists() and not overwrite:
        return json.loads((RES3 / tag / "cv_metrics.json").read_text())["pooled"]
    P = np.zeros((nD, 4)); pred = np.zeros(nD, int); PT = []; info = []; t0 = time.time()
    for k in range(5):
        itr, iva, ote = split_indices(D, repeat, k)
        fm = np.zeros(len(ALL), bool); fm[itr] = True
        X, pre = make_X(spec, fm)
        if fitter is not None:
            pf, extra = fitter(X[itr], Y[itr], X[iva], Y[iva], seed)
        elif sk_lr:
            best = None
            for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[itr], Y[itr])
                sc = s3(evaluate(Y[iva], clf.predict_proba(X[iva])))
                if best is None or sc > best[0]:
                    best = (sc, C, clf)
            clf = best[2]; pf = lambda Z, c=clf: (c.predict_proba(Z), c.predict_proba(Z).argmax(1)); extra = dict(C=best[1])
        else:
            fobj = p2heads.fit(X[itr], Y[itr], X[iva], Y[iva], dict(cfg, no_std=pre), seed=seed)
            pf = fobj["predict"]; extra = dict(l2=fobj["l2"], best_epoch=fobj["best_epoch"], n_params=fobj["n_params"])
        P[ote], pred[ote] = pf(X[ote]); PT.append(pf(X[nD:])[0]); info.append(dict(fold=k, **extra))
    save_cv3(tag, P, pred, np.mean(PT, 0), dict(spec=spec, cfg=cfg, seed=seed, sk_lr=sk_lr, folds=info,
                                                minutes=(time.time() - t0) / 60), repeat)
    return cv_report3(tag, P, pred, repeat)
