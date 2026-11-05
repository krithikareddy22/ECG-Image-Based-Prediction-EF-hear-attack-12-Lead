"""Lightweight classifiers on fixed feature vectors (ECGFounder embeddings, header features).

One trainer, four output heads (identical optimiser, regularisation grid and selection rule,
so differences come from the head/loss only):

  multinomial : softmax, (class-weighted) cross-entropy            [the project's baseline loss]
  focal       : softmax, class-weighted focal loss (gamma = 2)
  coral       : CORAL (Cao et al. 2020) - one shared weight vector, 3 thresholds,
                P(y > k) = sigmoid(w.x + b_k), loss = sum_k weighted BCE; decoded by counting
                P(y > k) > 0.5 (rank decoding)
  cumlink     : cumulative-link / proportional-odds logit model: P(y <= k) = sigmoid(t_k - w.x)
                with t_1 < t_2 < t_3 enforced by construction (t_k = t_1 + sum softplus), loss =
                multinomial likelihood of the implied class probabilities; argmax decoding

Class probabilities for the ordinal heads: p_k = P(y > k-1) - P(y > k) (clipped at 0 and
renormalised for CORAL, whose thresholds are not guaranteed ordered; the number of rank-
inconsistent samples is reported). Model selection (L2 strength) uses the VALIDATION
selection score of the project, mean(macro-F1, QWK, HF AUC).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ecg_ef.metrics import evaluate
from ecg_ef.train import selection_score

HEADS = ("multinomial", "focal", "coral", "cumlink")


def class_weights(y, n=4, mode="balanced"):
    c = np.bincount(y, minlength=n).astype(float)
    w = c.sum() / (n * c) if mode == "balanced" else np.ones(n)
    return w / w.mean()


class LinearHead(torch.nn.Module):
    def __init__(self, d, head):
        super().__init__()
        self.head = head
        if head in ("multinomial", "focal"):
            self.lin = torch.nn.Linear(d, 4)
        else:
            self.w = torch.nn.Linear(d, 1, bias=False)
            if head == "coral":
                self.b = torch.nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))
            else:
                self.t1 = torch.nn.Parameter(torch.tensor([-1.0]))
                self.dt = torch.nn.Parameter(torch.tensor([0.5, 0.5]))

    def cum_gt(self, x):
        """P(y > k) for k = 0, 1, 2  -> [N, 3]."""
        s = self.w(x)                                       # [N,1]
        if self.head == "coral":
            return torch.sigmoid(s + self.b)
        t = torch.cat([self.t1, self.t1 + torch.cumsum(F.softplus(self.dt), 0)])
        return 1 - torch.sigmoid(t - s)                     # P(y > k) = 1 - P(y <= k)

    def probs(self, x):
        if self.head in ("multinomial", "focal"):
            return F.softmax(self.lin(x), 1)
        g = self.cum_gt(x)
        one, zero = torch.ones_like(g[:, :1]), torch.zeros_like(g[:, :1])
        upper = torch.cat([one, g], 1); lower = torch.cat([g, zero], 1)
        p = (upper - lower).clamp_min(0) + 1e-8
        return p / p.sum(1, keepdim=True)

    def loss(self, x, y, cw, gamma=2.0):
        w = cw[y]
        if self.head == "multinomial":
            return (F.cross_entropy(self.lin(x), y, reduction="none") * w).sum() / w.sum()
        if self.head == "focal":
            logp = F.log_softmax(self.lin(x), 1).gather(1, y[:, None])[:, 0]
            return (-(1 - logp.exp()) ** gamma * logp * w).sum() / w.sum()
        if self.head == "coral":
            s = self.w(x) + self.b
            target = (y[:, None] > torch.arange(3)[None]).float()
            bce = F.binary_cross_entropy_with_logits(s, target, reduction="none").sum(1)
            return (bce * w).sum() / w.sum()
        p = self.probs(x)
        return (-torch.log(p.gather(1, y[:, None])[:, 0]) * w).sum() / w.sum()

    def decode(self, x):
        if self.head == "coral":
            return (self.cum_gt(x) > 0.5).sum(1)
        return self.probs(x).argmax(1)


def fit_head(Xtr, ytr, Xva, yva, head, l2_grid=(1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0), weighting="balanced",
             gamma=2.0, steps=400, seed=42):
    """Standardise with TRAIN statistics, fit for each L2 value, keep the best on VALIDATION."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    T = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    xtr, xva = T(Xtr), T(Xva)
    ytr_t = torch.tensor(ytr)
    cw = torch.tensor(class_weights(ytr, mode=weighting), dtype=torch.float32)
    best = None
    for l2 in l2_grid:
        torch.manual_seed(seed)
        m = LinearHead(Xtr.shape[1], head)
        opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=steps, history_size=20,
                                line_search_fn="strong_wolfe")
        wparams = [p for n, p in m.named_parameters() if n in ("lin.weight", "w.weight")]

        def closure():
            opt.zero_grad()
            loss = m.loss(xtr, ytr_t, cw, gamma) + l2 * sum((p ** 2).sum() for p in wparams)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pv = m.probs(xva).numpy(); dv = m.decode(xva).numpy()
        mv = evaluate(yva, pv)
        if head == "coral":   # class metrics with rank decoding
            mv2 = evaluate(yva, np.eye(4)[dv] * 0.97 + 0.01)
            for k in ("macro_f1", "qwk"):
                mv[k] = mv2[k]
        sc = selection_score(mv)
        if best is None or sc > best[0]:
            best = (sc, l2, m)
    sc, l2, m = best

    def predict(X):
        with torch.no_grad():
            x = T(X)
            out = dict(probs=m.probs(x).numpy(), pred=m.decode(x).numpy())
            if head in ("coral", "cumlink"):
                g = m.cum_gt(x).numpy()
                out["cum_gt"] = g
                out["n_rank_inconsistent"] = int(np.sum(np.any(np.diff(g, axis=1) > 1e-9, axis=1)))
            return out
    return dict(model=m, l2=l2, val_score=sc, predict=predict, n_params=sum(p.numel() for p in m.parameters()))
