"""Experiment A (pre-registered, PHASE6_PLAN.md §4.1): joint class + EF fine-tuning of ECGFounder on CPU (FP32).

  R0  backbone frozen (input: cached f6 = mean-pooled stage-6 output)              reference, same code
  A1  stage 6 trainable (input: cached stage-5 activations), progressive unfreezing
  A2  stages 5-6 trainable (input: cached stage-4 activations), progressive unfreezing
Head = the Phase-4 multitask body on pooled stage-6 output || header numeric (16): Dropout 0.5 -> Linear(1040, 64) ->
GELU -> Dropout 0.5 -> class head (4-way, class-balanced CE, folder label) + EF head (Huber delta 1 on z-scored
nearest-echo EF; z-scoring fitted on inner-train). Loss = CE + 1.0 * Huber. Echo EF is a TARGET only.
Unfreezing: epochs 0-4 head only; 5-9 + stage-6 block 3; >= 10 all of stage 6; A2: >= 15 + stage 5.
Learning rates: head 1e-3 (weight decay 0.1), stage 6 1e-4, stage 5 5e-5 (weight decay 1e-2); AdamW; batch 32;
<= 40 epochs. Early stopping on the fold's inner-val only: criterion = mean(inner-val S3, inner-val EF Pearson r),
patience 8, no stop before epoch 10; the best inner-val epoch is restored. The outer fold is predicted once.
Jobs are (experiment, seed, repeat, fold); each finished fold is cached privately, so an interrupted run resumes at
fold granularity. A run (5 folds) is assembled and saved when all folds exist.
    python -m research_experiments.phase_6.experiments.finetune --exp R0 [--seeds 42 43 44] [--repeats 0 1 2] [--jobs 4]
"""
from __future__ import annotations

import argparse
import copy
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_6.p6common import (PRIV6, Y, done, ef_target, guard6, header_numeric, mark_running, nD,
                                                   save_run, split_indices)

EXPS = {"R0": dict(exp_id="P6R0-joint-frozen-v1", start=None),
        "A1": dict(exp_id="P6A1-ft-s6-joint-v1", start=6),
        "A2": dict(exp_id="P6A2-ft-s56-joint-v1", start=5)}
LAM, BS, MAX_EP, MIN_EP, PATIENCE = 1.0, 32, 40, 10, 8
LR = dict(head=1e-3, s6=1e-4, s5=5e-5)
WD = dict(head=0.1, backbone=1e-2)
FOLDS6 = PRIV6 / "fold_cache"
_CACHE = {}


def cache():
    if not _CACHE:
        from research_experiments.phase_6.experiments.embed_ecgf_cache import load_cache
        s4, s5, f6 = load_cache()
        _CACHE.update(stage4=torch.from_numpy(s4), stage5=torch.from_numpy(s5), f6=torch.from_numpy(f6))
    return _CACHE


class Joint(nn.Module):
    def __init__(self, start, hidden=64, dropout=0.5):
        super().__init__()
        self.start = start
        if start is not None:
            from research_experiments.phase_2.ecgf_features import load_model
            base = load_model(False)
            self.stages = nn.ModuleList([copy.deepcopy(base.stage_list[s]) for s in range(start, 7)])
            del base
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024 + 16, hidden), nn.GELU(), nn.Dropout(dropout))
        self.cls, self.ef = nn.Linear(hidden, 4), nn.Linear(hidden, 1)

    def pooled(self, x):
        if self.start is None:
            return x
        for s in self.stages:
            x = s(x)
        return x.mean(-1)

    def forward(self, x, h):
        z = self.body(torch.cat([self.pooled(x), h], 1))
        return self.cls(z), self.ef(z)[:, 0]

    def stage(self, s):
        return self.stages[s - self.start]

    def set_trainable(self, epoch):
        if self.start is None:
            return
        for p in self.stages.parameters():
            p.requires_grad_(False)
        if epoch >= 5:
            for p in self.stage(6).block_list[3].parameters():
                p.requires_grad_(True)
        if epoch >= 10:
            for p in self.stage(6).parameters():
                p.requires_grad_(True)
        if self.start == 5 and epoch >= 15:
            for p in self.stage(5).parameters():
                p.requires_grad_(True)


def inputs_for(start):
    c = cache()
    return c["f6"] if start is None else (c["stage5"] if start == 6 else c["stage4"])


@torch.no_grad()
def predict(m, X, H, idx, bs=64):
    m.eval(); P, E = [], []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        lo, ef = m(X[j], H[j]); P.append(F.softmax(lo, 1)); E.append(ef)
    return torch.cat(P).numpy(), torch.cat(E).numpy()


def fit_fold(exp, seed, r, k):
    f = guard6(FOLDS6 / EXPS[exp]["exp_id"] / f"s{seed}_r{r}_k{k}.npz")
    if f.exists():
        return str(f)
    torch.set_num_threads(1)
    t0 = time.time()
    itr, iva, ote = split_indices(r, k)
    fm = np.zeros(nD, bool); fm[itr] = True
    Hn = header_numeric(fm); mu, sd = Hn[itr].mean(0), Hn[itr].std(0) + 1e-6
    H = torch.tensor((Hn - mu) / sd, dtype=torch.float32)
    ef = ef_target()["ef"]; emask = np.isfinite(ef)
    em, es = ef[itr][emask[itr]].mean(), ef[itr][emask[itr]].std()
    ez = torch.tensor(np.where(emask, (ef - em) / es, 0.0), dtype=torch.float32); emt = torch.tensor(emask)
    X = inputs_for(EXPS[exp]["start"])
    yt = torch.tensor(Y)
    torch.manual_seed(seed * 100 + 10 * r + k)
    m = Joint(EXPS[exp]["start"])
    groups = [dict(params=list(m.body.parameters()) + list(m.cls.parameters()) + list(m.ef.parameters()),
                   lr=LR["head"], weight_decay=WD["head"])]
    if m.start is not None:
        groups.append(dict(params=list(m.stage(6).parameters()), lr=LR["s6"], weight_decay=WD["backbone"]))
        if m.start == 5:
            groups.append(dict(params=list(m.stage(5).parameters()), lr=LR["s5"], weight_decay=WD["backbone"]))
    opt = torch.optim.AdamW(groups)
    cw = class_w(Y[itr], "balanced")
    g = torch.Generator().manual_seed(seed * 100 + 10 * r + k)
    best, bad, hist = None, 0, []
    for ep in range(MAX_EP):
        m.set_trainable(ep); m.train()
        perm = itr[torch.randperm(len(itr), generator=g).numpy()]
        for i in range(0, len(perm), BS):
            j = perm[i:i + BS]
            lo, efh = m(X[j], H[j])
            loss = F.cross_entropy(lo, yt[j], weight=cw)
            if emt[j].any():
                loss = loss + LAM * F.huber_loss(efh[emt[j]], ez[j][emt[j]], delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        Pv, Ev = predict(m, X, H, iva)
        vm = iva[emask[iva]]
        rv = float(np.corrcoef(Ev[emask[iva]], ef[vm])[0, 1])
        sv = s3(evaluate(Y[iva], Pv)); crit = (sv + rv) / 2
        hist.append([ep, round(sv, 4), round(rv, 4)])
        if best is None or crit > best[0] + 1e-9:
            best = (crit, ep, copy.deepcopy(m.state_dict()), sv, rv); bad = 0
        else:
            bad += 1
        if ep + 1 >= MIN_EP and bad >= PATIENCE:
            break
    m.load_state_dict(best[2])
    Po, Eo = predict(m, X, H, ote)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, ote=ote, P=Po, efhat=Eo * es + em, best_epoch=best[1], val_s3=best[3], val_r=best[4],
             epochs_run=len(hist), minutes=(time.time() - t0) / 60, hist=np.array(hist))
    print(f"[{exp} s{seed} r{r} k{k}] best ep {best[1]} val S3 {best[3]:.3f} r {best[4]:.3f} "
          f"({len(hist)} ep, {(time.time() - t0) / 60:.1f} min)", flush=True)
    return str(f)


def _job(a):
    return fit_fold(*a)


def assemble(exp, seed, r, cmd):
    eid = EXPS[exp]["exp_id"]; rid = f"{eid}__s{seed}_r{r}"
    if done(rid):
        return None
    P = np.zeros((nD, 4)); E = np.full(nD, np.nan); info = []
    for k in range(5):
        z = np.load(FOLDS6 / eid / f"s{seed}_r{r}_k{k}.npz")
        P[z["ote"]] = z["P"]; E[z["ote"]] = z["efhat"]
        info.append(dict(fold=k, best_epoch=int(z["best_epoch"]), val_s3=float(z["val_s3"]), val_r=float(z["val_r"]),
                         epochs_run=int(z["epochs_run"]), minutes=float(z["minutes"])))
    cfg = dict(exp_id=eid, start_stage=EXPS[exp]["start"], lam=LAM, batch=BS, max_epochs=MAX_EP, min_epochs=MIN_EP,
               patience=PATIENCE, lr=LR, weight_decay=WD, unfreeze="head 0-4; +s6 block3 5-9; +s6 >=10; (+s5 >=15 for A2)",
               select="mean(inner-val S3, inner-val EF Pearson r)", seed=seed, repeat=r, ef_target="nearest echo (Phase 4, cohort A)")
    agg = save_run(rid, P, P.argmax(1), r, cfg, efhat=E, extra=dict(folds=info, minutes=sum(i["minutes"] for i in info)))
    m, e = agg["metrics"], agg["ef_regression"]["A"]
    print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f} minR {m['min_recall']:.2f} "
          f"| EF r {e['pearson']:.3f} MAE {e['mae']:.2f}", flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cmd = f"python -m research_experiments.phase_6.experiments.finetune --exp {a.exp} --seeds {' '.join(map(str, a.seeds))}"
    eid = EXPS[a.exp]["exp_id"]
    todo = [(s, r) for s in a.seeds for r in a.repeats if not done(f"{eid}__s{s}_r{r}")]
    for s, r in todo:
        mark_running(f"{eid}__s{s}_r{r}", cmd)
    cache()                                                        # load once before forking
    jobs = [(a.exp, s, r, k) for s, r in todo for k in range(5)]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            list(pool.imap_unordered(_job, jobs))
    else:
        for j in jobs:
            _job(j)
    for s, r in todo:
        assemble(a.exp, s, r, cmd)


if __name__ == "__main__":
    main()
