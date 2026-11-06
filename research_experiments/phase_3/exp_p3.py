"""Phase-3 experiments: clinical feature engineering, gated / attention multimodal fusion, gradient boosting.
Same folds / inner-val protocol as phase 2 (p3common.run3). Locked test stored, never scored.

Gated fusion (small, for n = 837):
    x_m (standardised block m) -> dropout -> Linear(d_m, h) -> GELU            modality-specific projection z_m
    g = sigmoid(Linear([z_1..z_M], M or M*h))                                   per-modality (scalar) or per-unit (vector) gate
    fused = sum_m g_m * z_m  -> dropout -> Linear(h, 4)
Attention fusion: tokens z_m + modality embedding; one learned query; single-head scaled dot-product weights
    alpha_m = softmax(q . W_k z_m / sqrt(h)); fused = sum_m alpha_m W_v z_m.
Training: AdamW, class-weighted CE (optionally + distance penalty), early stopping on inner-val S3.

    python -m research_experiments.phase_3.exp_p3 --group clinical gated [--repeat 1] [--seeds 42 43 44]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_experiments.common import ROOT, seed_all
from research_experiments.phase_2.cv_features import ALL
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_3.p3common import EXTRA_BLOCKS, MOD3, run3
from ecg_ef.metrics import evaluate

# ----------------------------------------------------------------------------- feature blocks
_CLIN = pd.read_csv(MOD3 / "clinical_features.csv").set_index("ecg_id").loc[ALL.ecg_id]
_NING = pd.read_csv(ROOT / "data" / "ecg_features.csv").set_index("ecg_id").reindex(ALL.ecg_id).select_dtypes("number")


def _impute(df, fit_mask):
    v = df.values.astype(float); med = np.nanmedian(v[fit_mask], 0); med = np.where(np.isfinite(med), med, 0)
    return np.where(np.isfinite(v), v, med[None])


EXTRA_BLOCKS["clin"] = lambda fm: _impute(_CLIN, fm)
EXTRA_BLOCKS["ning"] = lambda fm: _impute(_NING, fm)

H_NUM, F6, CLIN, NING = ("hdr", "numeric"), ("ecgf", "aligned", "f6"), ("clin",), ("ning",)
LIN = dict(head="linear", loss="ce", weight="balanced")
MLPF = dict(LIN, head="mlp", hidden=64, dropout=0.5)


# ----------------------------------------------------------------------------- gated / attention fusion
class GatedFusion(nn.Module):
    def __init__(self, dims, h=32, dropout=0.5, gate="scalar", attention=False):
        super().__init__()
        self.dims, self.gate_type, self.attention = dims, gate, attention
        self.proj = nn.ModuleList([nn.Sequential(nn.Dropout(dropout if d > 64 else dropout / 2), nn.Linear(d, h), nn.GELU()) for d in dims])
        M = len(dims)
        if attention:
            self.mod_emb = nn.Parameter(torch.zeros(M, h)); self.q = nn.Parameter(torch.randn(h) / h ** 0.5)
            self.k, self.v = nn.Linear(h, h, bias=False), nn.Linear(h, h, bias=False)
        else:
            self.g = nn.Linear(M * h, M if gate == "scalar" else M * h)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(h, 4); self.h = h

    def split(self, x):
        return torch.split(x, self.dims, dim=1)

    def forward(self, x, return_weights=False):
        z = [p(xm) for p, xm in zip(self.proj, self.split(x))]
        Z = torch.stack(z, 1)                                                   # [B, M, h]
        if self.attention:
            Zt = Z + self.mod_emb[None]
            a = torch.softmax((self.k(Zt) @ self.q) / self.h ** 0.5, 1)         # [B, M]
            fused = (a[..., None] * self.v(Zt)).sum(1); w = a
        else:
            g = torch.sigmoid(self.g(Z.flatten(1)))
            g = g[..., None] if self.gate_type == "scalar" else g.view(-1, len(self.dims), self.h)
            fused = (g * Z).sum(1); w = g.mean(-1) if self.gate_type != "scalar" else g[..., 0]
        logits = self.out(self.drop(fused))
        return (logits, w) if return_weights else logits


def gated_fitter(dims, h=32, dropout=0.5, gate="scalar", attention=False, wd_grid=(0.01, 0.1), lr=1e-3,
                 epochs=200, patience=25, lam=0.0):
    def fit(Xtr, ytr, Xva, yva, seed):
        xtr, xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
        yt = torch.tensor(ytr); cw = class_w(ytr, "balanced")
        best = None
        for wd in wd_grid:
            seed_all(seed)
            m = GatedFusion(dims, h, dropout, gate, attention)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            g = torch.Generator().manual_seed(seed); sc_best, state, bad, best_ep = -1, None, 0, 0
            for ep in range(epochs):
                m.train(); idx = torch.randperm(len(ytr), generator=g)
                for i in range(0, len(idx), 64):
                    b = idx[i:i + 64]; lo = m(xtr[b])
                    li = F.cross_entropy(lo, yt[b], reduction="none")
                    if lam:
                        p = F.softmax(lo, 1); k = torch.arange(4, dtype=torch.float32)
                        li = li + lam * (p * (k[None] - yt[b][:, None].float()) ** 2).sum(1) / 9
                    loss = (li * cw[yt[b]]).sum() / cw[yt[b]].sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    pv = F.softmax(m(xva), 1).numpy()
                sc = s3(evaluate(yva, pv))
                if sc > sc_best + 1e-6:
                    sc_best, bad, best_ep = sc, 0, ep; state = {k_: v.clone() for k_, v in m.state_dict().items()}
                else:
                    bad += 1
                if bad >= patience:
                    break
            m.load_state_dict(state)
            if best is None or sc_best > best[0]:
                best = (sc_best, wd, m, best_ep)
        sc, wd, m, best_ep = best; m.eval()

        def predict(X):
            with torch.no_grad():
                p = F.softmax(m(torch.tensor(X, dtype=torch.float32)), 1).numpy()
            return p, p.argmax(1)
        predict.model = m
        return predict, dict(wd=wd, val_s3=sc, best_epoch=best_ep, n_params=sum(p.numel() for p in m.parameters()))
    return fit


def xgb_fitter(depths=(2, 3), n_est=(100, 300)):
    import xgboost as xgb

    def fit(Xtr, ytr, Xva, yva, seed):
        sw = (1.0 / np.bincount(ytr, minlength=4))[ytr] * len(ytr) / 4
        best = None
        for d in depths:
            for n in n_est:
                clf = xgb.XGBClassifier(max_depth=d, n_estimators=n, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                                        min_child_weight=5, reg_lambda=5.0, tree_method="hist", n_jobs=1, random_state=seed,
                                        eval_metric="mlogloss")
                clf.fit(Xtr, ytr, sample_weight=sw)
                sc = s3(evaluate(yva, clf.predict_proba(Xva)))
                if best is None or sc > best[0]:
                    best = (sc, d, n, clf)
        sc, d, n, clf = best
        return (lambda X: (clf.predict_proba(X), clf.predict_proba(X).argmax(1))), dict(depth=d, n_estimators=n, val_s3=sc)
    return fit


NCLIN = len(_CLIN.columns)
GROUPS = {
    "clinical": [
        ("clin_hdrnum+clin_skLR", dict(blocks=[H_NUM, CLIN]), None, dict(sk_lr=True)),
        ("clin_clinonly_skLR", dict(blocks=[CLIN]), None, dict(sk_lr=True)),
        ("clin_hdrnum+clin_xgb", dict(blocks=[H_NUM, CLIN]), None, dict(fitter=xgb_fitter())),
        ("clin_hdrnum+clin+ning_xgb", dict(blocks=[H_NUM, CLIN, NING]), None, dict(fitter=xgb_fitter())),
        ("clin_mlpfusion_hdr+clin+ecgf", dict(blocks=[H_NUM, CLIN, F6]), MLPF, {}),
    ],
    "gated": [
        ("gate_scalar_hdr+ecgf", dict(blocks=[H_NUM, F6], raw_std=True), None, dict(fitter=gated_fitter([16, 1024]))),
        ("gate_vector_hdr+ecgf", dict(blocks=[H_NUM, F6], raw_std=True), None, dict(fitter=gated_fitter([16, 1024], gate="vector"))),
        ("gate_scalar_hdr+clin+ecgf", dict(blocks=[H_NUM, CLIN, F6], raw_std=True), None, dict(fitter=gated_fitter([16, NCLIN, 1024]))),
        ("attn_hdr+clin+ecgf", dict(blocks=[H_NUM, CLIN, F6], raw_std=True), None, dict(fitter=gated_fitter([16, NCLIN, 1024], attention=True))),
        ("gate_scalar_hdr+clin+ecgf_h64", dict(blocks=[H_NUM, CLIN, F6], raw_std=True), None, dict(fitter=gated_fitter([16, NCLIN, 1024], h=64))),
        ("gate_scalar_hdr+clin+ecgf_dist", dict(blocks=[H_NUM, CLIN, F6], raw_std=True), None, dict(fitter=gated_fitter([16, NCLIN, 1024], lam=1.0))),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", required=True); ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    for g in a.group:
        for tag, spec, cfg, kw in GROUPS[g]:
            if a.only and tag not in a.only:
                continue
            for sd in a.seeds:
                run3(tag + ("" if sd == 42 else f"_seed{sd}"), spec, cfg, repeat=a.repeat, seed=sd, **kw)


if __name__ == "__main__":
    main()
