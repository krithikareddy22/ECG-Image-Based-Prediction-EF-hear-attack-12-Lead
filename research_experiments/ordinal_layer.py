"""Cumulative-link (proportional-odds) output layer for deep networks.
The network emits one severity score s; P(y <= k) = sigmoid(t_k - s) with ordered cut-points
t_1 < t_2 < t_3 (t_k = t_1 + cumulative softplus), so cumulative probabilities are monotone and
class probabilities p_k = P(y<=k) - P(y<=k-1) are non-negative by construction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CumulativeLink(nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = nn.Parameter(torch.tensor([-1.0]))
        self.dt = nn.Parameter(torch.tensor([1.0, 1.0]))

    def cutpoints(self):
        return torch.cat([self.t1, self.t1 + torch.cumsum(F.softplus(self.dt), 0)])

    def cum_le(self, s):                     # s [N] or [N,1] -> P(y <= k) [N,3]
        return torch.sigmoid(self.cutpoints()[None] - s.reshape(-1, 1))

    def probs(self, s):
        c = self.cum_le(s)
        one, zero = torch.ones_like(c[:, :1]), torch.zeros_like(c[:, :1])
        p = torch.cat([c, one], 1) - torch.cat([zero, c], 1)
        return p.clamp_min(1e-7) / p.clamp_min(1e-7).sum(1, keepdim=True)

    def nll(self, s, y, w=None):
        lp = torch.log(self.probs(s).gather(1, y[:, None])[:, 0])
        w = torch.ones_like(lp) if w is None else w
        return -(lp * w).sum() / w.sum()
