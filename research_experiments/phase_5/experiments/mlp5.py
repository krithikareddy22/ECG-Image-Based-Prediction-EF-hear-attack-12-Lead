"""Phase-5 MLP trainer: identical to the phase-2 MLP head (phase_2/p2heads.py, head='mlp': Dropout -> Linear(d, 64) -> GELU ->
Dropout -> Linear(64, K); AdamW lr 1e-3, weight_decay = 100 * l2, l2 in {1e-4, 1e-3, 1e-2}; batch 64; <= 200 epochs;
patience 25 on the inner-val selection score; same RNG consumption) so that the default reproduces phase 2 exactly
(verified in tests). One option at a time can be switched on for controlled ablations (E4 / E6 / E7):
  n_classes  K (E4 stage models)          loss      'ce' | 'focal' (gamma)          smooth   label smoothing
  mixup      alpha (Beta(alpha, alpha))   sched     'none' | 'cosine'               select   's3' | 'accuracy'
  sample_w   per-sample training weights (E6)      weight   'balanced' | 'none'
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def quick_s3(y, p_or_pred, K):
    pred = p_or_pred.argmax(1) if p_or_pred.ndim == 2 else p_or_pred
    rc, f1 = [], []
    for k in range(K):
        m = y == k
        rc.append(np.mean(pred[m] == k) if m.any() else 0.0)
        tp = np.sum((pred == k) & m); d = np.sum(pred == k) + m.sum(); f1.append(2 * tp / d if d else 0.0)
    return (np.mean(pred == y) + np.mean(rc) + np.mean(f1)) / 3


def class_w(y, K, mode):
    c = np.bincount(y, minlength=K).astype(float)
    w = c.sum() / (K * np.maximum(c, 1)) if mode == "balanced" else np.ones(K)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class Head(nn.Module):
    def __init__(self, d, K, hidden=64, dropout=0.5):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, K)

    def forward(self, x):
        return self.out(self.body(x))


def fit(Xtr, ytr, Xva, yva, cfg: dict | None = None, seed: int = 42):
    cfg = dict(cfg or {})
    K = cfg.get("n_classes", 4); select = cfg.get("select", "s3")
    if cfg.get("no_std"):
        mu, sd = np.zeros(Xtr.shape[1]), np.ones(Xtr.shape[1])
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    T = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    xtr, xva = T(Xtr), T(Xva); yt = torch.tensor(ytr)
    cw = class_w(ytr, K, cfg.get("weight", "balanced"))
    sw = torch.tensor(cfg["sample_w"], dtype=torch.float32) if cfg.get("sample_w") is not None else None
    best = None
    for l2 in cfg.get("l2_grid", (1e-4, 1e-3, 1e-2)):
        torch.manual_seed(seed)
        m = Head(Xtr.shape[1], K, cfg.get("hidden", 64), cfg.get("dropout", 0.5))
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=l2 * 100)
        epochs = cfg.get("epochs", 200)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cfg.get("sched") == "cosine" else None
        g = torch.Generator().manual_seed(seed)
        mg = np.random.default_rng(seed)
        best_ep, sc_best, state, bad = 0, -1, None, 0
        for ep in range(epochs):
            m.train(); idx = torch.randperm(len(ytr), generator=g)
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                xb = xtr[b] + 0.0 * torch.randn_like(xtr[b])          # same RNG draw as p2heads (feat_noise = 0)
                logits = m(xb)
                if cfg.get("mixup"):
                    lam = float(mg.beta(cfg["mixup"], cfg["mixup"])); perm = torch.tensor(mg.permutation(len(b)))
                    logits = m(lam * xb + (1 - lam) * xb[perm])
                    li = lam * F.cross_entropy(logits, yt[b], reduction="none") + (1 - lam) * F.cross_entropy(logits, yt[b][perm], reduction="none")
                    w = lam * cw[yt[b]] + (1 - lam) * cw[yt[b][perm]]
                elif cfg.get("loss") == "focal":
                    logp = F.log_softmax(logits, 1).gather(1, yt[b][:, None])[:, 0]
                    li = -(1 - logp.exp()) ** cfg.get("gamma", 2.0) * logp; w = cw[yt[b]]
                else:
                    li = F.cross_entropy(logits, yt[b], label_smoothing=cfg.get("smooth", 0.0), reduction="none"); w = cw[yt[b]]
                if sw is not None:
                    w = w * sw[b]
                lo = (li * w).sum() / w.sum()
                opt.zero_grad(); lo.backward(); opt.step()
            if sched is not None:
                sched.step()
            m.eval()
            with torch.no_grad():
                p = F.softmax(m(xva), 1).numpy()
            if select == "accuracy":
                sc = float(np.mean(p.argmax(1) == yva))
            elif K == 4 and select == "s3":
                from ecg_ef.metrics import evaluate
                from research_experiments.phase_2.p2common import s3
                sc = s3(evaluate(yva, p))
            else:
                sc = quick_s3(yva, p, K)
            if sc > sc_best + 1e-6:
                sc_best, best_ep, bad = sc, ep, 0; state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
            if bad >= cfg.get("patience", 25):
                break
        m.load_state_dict(state)
        if best is None or sc_best > best[0] + 1e-9:
            best = (sc_best, l2, m, best_ep)
    sc, l2, m, ep = best; m.eval()

    def predict(X):
        with torch.no_grad():
            return F.softmax(m(T(X)), 1).numpy()
    return predict, dict(l2=l2, val_score=float(sc), best_epoch=ep)


def fitter(cfg):
    """Adapter for runner.cv_run(kind='fitter')."""
    def f(Xtr, ytr, Xva, yva, seed):
        return fit(Xtr, ytr, Xva, yva, cfg, seed)
    return f
