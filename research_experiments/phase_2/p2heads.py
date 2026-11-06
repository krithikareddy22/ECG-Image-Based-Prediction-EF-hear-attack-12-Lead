"""Phase-2 classifiers on fixed feature vectors, with configurable loss / imbalance handling.

cfg keys (all optional):
  head      'linear' | 'mlp'                 (mlp: 1 hidden layer, `hidden` units, GELU, dropout)
  loss      'ce' | 'focal' | 'coral' | 'cumlink' | 'ce_dist'
            ce_dist = CE + lam * E_p[(k - y)^2] / 9  (distance-sensitive penalty)
  weight    'none' | 'balanced' | 'sqrt' | 'effnum'  (effective number of samples, beta = 0.999)
  sampler   'none' | 'balanced'                    (class-balanced mini-batches; mlp only)
  gamma     focal gamma (2.0);  smooth  label smoothing (ce/ce_dist);  lam  distance weight (1.0)
  l2_grid   L2 strengths searched on inner-val (S3);  hidden, dropout, epochs (mlp)
Linear heads are fitted full-batch with L-BFGS; MLP heads with AdamW + early stopping on inner-val S3.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3


def class_w(y, mode, n=4, beta=0.999):
    c = np.bincount(y, minlength=n).astype(float)
    if mode == "balanced":
        w = c.sum() / (n * c)
    elif mode == "sqrt":
        w = np.sqrt(c.sum() / (n * c))
    elif mode == "effnum":
        w = (1 - beta) / (1 - beta ** c)
    else:
        w = np.ones(n)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class Head(nn.Module):
    def __init__(self, d, loss, head="linear", hidden=64, dropout=0.3):
        super().__init__()
        self.loss_name = loss
        k_out = 1 if loss == "cumlink" else (3 if loss == "coral" else 4)
        if head == "mlp":
            self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
            d = hidden
        else:
            self.body = nn.Identity()
        if loss == "coral":
            self.w = nn.Linear(d, 1, bias=False); self.b = nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))
        elif loss == "cumlink":
            self.w = nn.Linear(d, 1, bias=False)
            self.t1 = nn.Parameter(torch.tensor([-1.0])); self.dt = nn.Parameter(torch.tensor([1.0, 1.0]))
        else:
            self.out = nn.Linear(d, k_out)

    def weight_params(self):
        return [p for n, p in self.named_parameters() if n.endswith("weight") and p.dim() == 2]

    def cum_gt(self, h):
        s = self.w(h)
        if self.loss_name == "coral":
            return torch.sigmoid(s + self.b)
        t = torch.cat([self.t1, self.t1 + torch.cumsum(F.softplus(self.dt), 0)])
        return 1 - torch.sigmoid(t[None] - s)

    def probs(self, x):
        h = self.body(x)
        if self.loss_name in ("coral", "cumlink"):
            g = self.cum_gt(h)
            p = torch.cat([torch.ones_like(g[:, :1]), g], 1) - torch.cat([g, torch.zeros_like(g[:, :1])], 1)
            p = p.clamp_min(1e-7); return p / p.sum(1, keepdim=True)
        return F.softmax(self.out(h), 1)

    def decode(self, x):
        if self.loss_name == "coral":
            return (self.cum_gt(self.body(x)) > 0.5).sum(1)
        return self.probs(x).argmax(1)

    def loss(self, x, y, cw, cfg):
        w = cw[y]; h = self.body(x)
        if self.loss_name in ("ce", "focal", "ce_dist"):
            logits = self.out(h)
            if self.loss_name == "focal":
                logp = F.log_softmax(logits, 1).gather(1, y[:, None])[:, 0]
                li = -(1 - logp.exp()) ** cfg.get("gamma", 2.0) * logp
            else:
                li = F.cross_entropy(logits, y, label_smoothing=cfg.get("smooth", 0.0), reduction="none")
                if self.loss_name == "ce_dist":
                    p = F.softmax(logits, 1); k = torch.arange(4, dtype=torch.float32)
                    li = li + cfg.get("lam", 1.0) * (p * (k[None] - y[:, None].float()) ** 2).sum(1) / 9
            return (li * w).sum() / w.sum()
        if self.loss_name == "coral":
            s = self.w(h) + self.b
            tgt = (y[:, None] > torch.arange(3)[None]).float()
            return (F.binary_cross_entropy_with_logits(s, tgt, reduction="none").sum(1) * w).sum() / w.sum()
        p = self.probs(x)
        return (-torch.log(p.gather(1, y[:, None])[:, 0]) * w).sum() / w.sum()


def _eval(m, x, y):
    with torch.no_grad():
        p = m.probs(x).numpy(); d = m.decode(x).numpy()
    mm = evaluate(y, p)
    if m.loss_name == "coral":
        m2 = evaluate(y, np.eye(4)[d] * 0.97 + 0.01)
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk"):
            mm[k] = m2[k]
    return s3(mm), p, d


def fit(Xtr, ytr, Xva, yva, cfg: dict, seed: int = 42):
    """Returns dict(predict=fn(X)->(probs, pred), l2=..., val_s3=..., n_params=...)."""
    if cfg.get("no_std"):
        mu, sd = np.zeros(Xtr.shape[1]), np.ones(Xtr.shape[1])
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    T = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    xtr, xva = T(Xtr), T(Xva); yt = torch.tensor(ytr)
    cw = class_w(ytr, cfg.get("weight", "balanced"))
    loss = cfg.get("loss", "ce"); head = cfg.get("head", "linear")
    grid = cfg.get("l2_grid", (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0) if head == "linear" else (1e-4, 1e-3, 1e-2))
    best = None
    for l2 in grid:
        torch.manual_seed(seed)
        m = Head(Xtr.shape[1], loss, head, cfg.get("hidden", 64), cfg.get("dropout", 0.3))
        if head == "linear":
            opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=cfg.get("steps", 300), history_size=20,
                                    line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                lo = m.loss(xtr, yt, cw, cfg) + l2 * sum((p ** 2).sum() for p in m.weight_params())
                lo.backward(); return lo
            opt.step(closure); m.eval()
            sc, _, _ = _eval(m, xva, yva); state = None; best_ep = None
        else:
            opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=l2 * 100)
            g = torch.Generator().manual_seed(seed)
            if cfg.get("sampler") == "balanced":
                sw = (1.0 / np.bincount(ytr, minlength=4))[ytr]
                samp = lambda: torch.multinomial(torch.tensor(sw), len(ytr), replacement=True, generator=g)
                cw_use = torch.ones(4)
            else:
                samp = lambda: torch.randperm(len(ytr), generator=g); cw_use = cw
            best_ep, sc_best, state, bad = 0, -1, None, 0
            for ep in range(cfg.get("epochs", 200)):
                m.train(); idx = samp()
                for i in range(0, len(idx), 64):
                    b = idx[i:i + 64]
                    xb = xtr[b] + cfg.get("feat_noise", 0.0) * torch.randn_like(xtr[b])
                    lo = m.loss(xb, yt[b], cw_use, cfg); opt.zero_grad(); lo.backward(); opt.step()
                m.eval(); sc, _, _ = _eval(m, xva, yva)
                if sc > sc_best + 1e-6:
                    sc_best, best_ep, bad = sc, ep, 0; state = {k: v.clone() for k, v in m.state_dict().items()}
                else:
                    bad += 1
                if bad >= cfg.get("patience", 25):
                    break
            m.load_state_dict(state); sc = sc_best
        if best is None or sc > best[0] + 1e-9:
            best = (sc, l2, m, best_ep)
    sc, l2, m, best_ep = best
    m.eval()

    def predict(X):
        with torch.no_grad():
            x = T(X); return m.probs(x).numpy(), m.decode(x).numpy()
    return dict(predict=predict, l2=l2, val_s3=sc, best_epoch=best_ep, model=m, mu=mu, sd=sd,
                n_params=sum(p.numel() for p in m.parameters()))
