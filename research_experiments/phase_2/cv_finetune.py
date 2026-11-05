"""ECGFounder partial fine-tuning on CACHED frozen activations (exactly equivalent to fine-tuning with the
earlier stages frozen, because Net1D has no BatchNorm/dropout and the frozen part is deterministic).

  depth 'block3' : last block of stage 6 trainable  (input = cached stage-6 block-2 output, 1024 x 20)
  depth 'stage6' : all 4 blocks of stage 6 trainable (input = cached stage-5 output, 1024 x 40)
Head: dropout + linear (CE family) or cumulative-link.  Discriminative learning rates (backbone lr_b,
head lr_h), AdamW, warm-up + cosine, EMA, early stopping on inner-val S3, feature-space augmentation
(Gaussian noise on cached activations, random temporal crop), class weighting / balanced sampling.

    python -m research_experiments.phase_2.cv_finetune --tag ft_block3_base --depth block3
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_experiments.common import seed_all
from research_experiments.ordinal_layer import CumulativeLink
from research_experiments.phase_2.ecgf_features import load_model
from research_experiments.phase_2.p2common import CONF, MOD, PCOLS, RES, cv_report, guard, s3, save_cv, short
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_2.protocol import dev_and_test, split_indices
from ecg_ef.metrics import evaluate
from ecg_ef.train import EMA

META, D, T = dev_and_test()


def load_cache(depth):
    ids = np.load(MOD / "ecgf_aligned_features.npz")["ecg_id"]
    arr = np.load(MOD / ("ecgf_aligned_cache_stage6block2out.npy" if depth == "block3" else "ecgf_aligned_cache_stage5out.npy"),
                  mmap_mode="r")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return torch.from_numpy(np.ascontiguousarray(arr[pos.loc[D.ecg_id].values])), \
        torch.from_numpy(np.ascontiguousarray(arr[pos.loc[T.ecg_id].values]))


class FT(nn.Module):
    def __init__(self, depth, head, dropout):
        super().__init__()
        base = load_model(False)
        st = base.stage_list[6]
        self.blocks = nn.ModuleList([copy.deepcopy(st.block_list[3])] if depth == "block3" else
                                    [copy.deepcopy(b) for b in st.block_list])
        del base
        self.drop = nn.Dropout(dropout)
        self.head = head
        self.fc = nn.Linear(1024, 1 if head == "cumlink" else 4)
        self.link = CumulativeLink() if head == "cumlink" else None

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.fc(self.drop(x.mean(-1)))

    def probs(self, out):
        return F.softmax(out, 1) if self.head != "cumlink" else self.link.probs(out[:, 0])


def augment(x, noise, crop):
    if noise > 0:
        x = x + noise * x.std() * torch.randn_like(x)
    if crop < 1.0:
        L = x.shape[-1]; n = max(4, int(round(L * crop))); s = int(torch.randint(0, L - n + 1, (1,)))
        x = x[..., s:s + n]
    return x


@torch.no_grad()
def predict(m, X, bs=64):
    m.eval(); return torch.cat([m.probs(m(X[i:i + bs])) for i in range(0, len(X), bs)]).numpy()


def train(Xtr, ytr, Xva, yva, a, seed):
    seed_all(seed)
    m = FT(a.depth, a.head, a.dropout)
    cw = class_w(ytr.numpy(), a.weight)
    bb = [p for p in m.blocks.parameters()]
    hd = list(m.fc.parameters()) + (list(m.link.parameters()) if m.link else [])
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_b}, {"params": hd, "lr": a.lr_h}], weight_decay=a.wd)
    spe = max(1, len(Xtr) // a.bs); total, warm = a.epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ema = EMA(m, a.ema) if a.ema > 0 else None
    em = ema.model if ema else m
    best, state, bad, best_ep = -1, None, 0, 0
    sw = (1.0 / np.bincount(ytr.numpy(), minlength=4))[ytr.numpy()]
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, a.epochs + 1):
        m.train()
        perm = (torch.multinomial(torch.tensor(sw), len(ytr), replacement=True, generator=g) if a.sampler == "balanced"
                else torch.randperm(len(ytr), generator=g))
        w_use = torch.ones(4) if a.sampler == "balanced" else cw
        for i in range(spe):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            out = m(augment(Xtr[idx], a.noise, a.crop)); y = ytr[idx]
            if a.head == "cumlink":
                loss = m.link.nll(out[:, 0], y, w_use[y])
            elif a.loss == "focal":
                logp = F.log_softmax(out, 1).gather(1, y[:, None])[:, 0]
                loss = (-(1 - logp.exp()) ** a.gamma * logp * w_use[y]).sum() / w_use[y].sum()
            else:
                ce = F.cross_entropy(out, y, label_smoothing=a.smooth, reduction="none")
                loss = (ce * w_use[y]).sum() / w_use[y].sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
            if ema:
                ema.update(m)
        pv = predict(em, Xva); sc = s3(evaluate(yva.numpy(), pv))
        if ep >= a.min_epochs:
            if sc > best + 1e-6:
                best, bad, best_ep = sc, 0, ep; state = {k: v.clone() for k, v in em.state_dict().items()}
            else:
                bad += 1
            if bad >= a.patience:
                break
    m.load_state_dict(state)
    return m, dict(best_epoch=best_ep, val_s3=best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depth", default="block3", choices=["block3", "stage6"])
    ap.add_argument("--head", default="ce", choices=["ce", "cumlink"])
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"])
    ap.add_argument("--weight", default="balanced", choices=["balanced", "none", "sqrt", "effnum"])
    ap.add_argument("--sampler", default="none", choices=["none", "balanced"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--lr-b", dest="lr_b", type=float, default=1e-4)
    ap.add_argument("--lr-h", dest="lr_h", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--crop", type=float, default=0.9)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min-epochs", dest="min_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if (RES / a.tag / "cv_metrics.json").exists():
        print("exists", a.tag); return
    guard(CONF).mkdir(parents=True, exist_ok=True)
    (CONF / f"{a.tag}.json").write_text(json.dumps(vars(a), indent=1))
    XD, XT = load_cache(a.depth); y = torch.tensor(D.label_idx.values)
    P = np.zeros((len(D), 4)); fold = np.full(len(D), -1); PT = []; infos = []; t0 = time.time()
    for k in range(5):
        itr, iva, ote = split_indices(D, 0, k)
        ps, pts = [], []
        for sd in a.seeds:
            m, inf = train(XD[itr], y[itr], XD[iva], y[iva], a, sd)
            t1 = time.time(); ps.append(predict(m, XD[ote])); inf["infer_s_per_ecg_cached"] = (time.time() - t1) / len(ote)
            pts.append(predict(m, XT)); infos.append(dict(fold=k, seed=sd, **inf))
        P[ote] = np.mean(ps, 0); PT.append(np.mean(pts, 0)); fold[ote] = k
    n_tr = sum(p.numel() for p in FT(a.depth, a.head, a.dropout).parameters())
    oof = pd.DataFrame(P, columns=PCOLS); oof.insert(0, "ecg_id", D.ecg_id.values); oof.insert(1, "fold", fold)
    locked = pd.DataFrame(np.mean(PT, 0), columns=PCOLS); locked.insert(0, "ecg_id", T.ecg_id.values)
    save_cv(a.tag, oof, locked, dict(cfg=vars(a), runs=infos, trainable_params=n_tr, minutes=(time.time() - t0) / 60))
    mt = cv_report(a.tag, D, P)
    print(f"[{a.tag}] {short(mt)} ({(time.time() - t0) / 60:.1f} min, trainable {n_tr:,})", flush=True)


if __name__ == "__main__":
    main()
