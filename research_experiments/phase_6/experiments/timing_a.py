"""M1: timing run for experiment A (ECGFounder fine-tuning on CPU, FP32). Measures compute cost only - no label, no
EF value and no metric is read or recorded. Development ECGs only (a fixed subset of 64); the locked test is never used.

Measured (per ECG, batch 32, with 4 and 1 torch threads):
  * frozen forward to the stage-4 and stage-5 outputs (one-off cost of caching the frozen part for all 837 ECGs);
  * one optimisation step (forward + backward + AdamW) with stage 6 trainable from cached stage-5 activations (A1);
  * the same with stages 5-6 trainable from cached stage-4 activations (A2);
  * the same with the full backbone trainable from the raw 12 x 5000 input (for the GPU question only; not run in A);
  * evaluation forward of the trainable part (inner-val / outer-test prediction).
Projection (pre-registered recipe, PHASE6_PLAN.md §4.1): <= 40 epochs per outer fold, expected ~25 with early
stopping; per epoch one pass over inner-train (~535 ECGs) + one evaluation pass over inner-val (~134); 5 folds per
repeat. Output: phase_6/results/timing_A.json
    python -m research_experiments.phase_6.experiments.timing_a
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research_experiments.common import ROOT
from research_experiments.ecgfounder.model import verify_checkpoint
from research_experiments.phase_2.audit_waveform import geometry
from research_experiments.phase_2.ecgf_features import build_p2, load_model
from research_experiments.phase_6.p6common import IDS, RES6, guard6

N_SUB, BS, REPS = 64, 32, 3


def dev_inputs(n):
    ids = IDS[:n]
    src = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id").loc[ids].source.values
    G = [geometry(s) for s in src]                       # PDF layout gaps; file paths stay in memory only
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[ids].values
    return np.stack([build_p2(l, r, "aligned", g) for l, r, g in zip(z["leads"][pos], z["rhythm"][pos], G)])


class Part(nn.Module):
    """Trainable top of Net1D (stages from `start`), mean pooling, the Phase-4 multitask head (1,040 -> 64 -> 4 + 1)."""
    def __init__(self, base, start):
        super().__init__()
        self.first = None
        if start == "raw":
            self.first = nn.Sequential(copy.deepcopy(base.first_conv), copy.deepcopy(base.first_activation))
            self.stages = nn.ModuleList([copy.deepcopy(s) for s in base.stage_list])
        else:
            self.stages = nn.ModuleList([copy.deepcopy(base.stage_list[s]) for s in range(start, 7)])
        self.body = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024 + 16, 64), nn.GELU(), nn.Dropout(0.5))
        self.cls, self.ef = nn.Linear(64, 4), nn.Linear(64, 1)

    def forward(self, x, h):
        if self.first is not None:
            x = self.first(x)
        for s in self.stages:
            x = s(x)
        z = self.body(torch.cat([x.mean(-1), h], 1))
        return self.cls(z), self.ef(z)[:, 0]


@torch.no_grad()
def frozen_to(base, X, upto):
    out = base.first_activation(base.first_conv(torch.from_numpy(X)))
    for s in range(upto + 1):
        out = base.stage_list[s](out)
    return out


def time_step(part, x, reps=REPS):
    h = torch.zeros(len(x), 16); y = torch.zeros(len(x), dtype=torch.long); e = torch.zeros(len(x))
    opt = torch.optim.AdamW(part.parameters(), lr=1e-4)
    part.train(); ts = []
    for i in range(reps + 1):
        t0 = time.perf_counter()
        lo, ef = part(x, h)
        loss = nn.functional.cross_entropy(lo, y) + nn.functional.huber_loss(ef, e)
        opt.zero_grad(); loss.backward(); opt.step()
        if i:
            ts.append(time.perf_counter() - t0)
    part.eval()
    with torch.no_grad():
        t0 = time.perf_counter(); part(x, h); te = time.perf_counter() - t0
    return float(np.median(ts)) / len(x), te / len(x)


def main():
    verify_checkpoint()
    t0 = time.time(); X = dev_inputs(N_SUB); t_build = (time.time() - t0) / N_SUB
    base = load_model(False)
    for p in base.parameters():
        p.requires_grad_(False)
    out = dict(n_ecgs_timed=N_SUB, batch=BS, input_build_s_per_ecg=t_build, threads={})
    for th in (4, 1):
        torch.set_num_threads(th); R = {}
        t0 = time.perf_counter(); c4 = torch.cat([frozen_to(base, X[i:i + BS], 4) for i in range(0, N_SUB, BS)])
        R["frozen_forward_to_stage4_s_per_ecg"] = (time.perf_counter() - t0) / N_SUB
        t0 = time.perf_counter(); c5 = torch.cat([frozen_to(base, X[i:i + BS], 5) for i in range(0, N_SUB, BS)])
        R["frozen_forward_to_stage5_s_per_ecg"] = (time.perf_counter() - t0) / N_SUB
        R["cache_shapes"] = dict(stage4=list(c4.shape[1:]), stage5=list(c5.shape[1:]))
        for name, start, x in (("A1_stage6", 6, c5[:BS]), ("A2_stages5-6", 5, c4[:BS]), ("full_backbone_raw", "raw", torch.from_numpy(X[:BS]))):
            part = Part(base, start)
            for p in part.parameters():
                p.requires_grad_(True)
            tr, ev = time_step(part, x)
            R[name] = dict(train_step_s_per_ecg=tr, eval_s_per_ecg=ev,
                           trainable_params=int(sum(p.numel() for p in part.parameters())))
        out["threads"][str(th)] = R
    # projection (4 threads, one run at a time; and 4 parallel single-thread runs)
    n_tr, n_va, n_te, n_all = 535, 134, 167, 837
    proj = {}
    for name in ("A1_stage6", "A2_stages5-6", "full_backbone_raw"):
        for th, par in (("4", 1), ("1", 4)):
            r = out["threads"][th][name]
            per_epoch = n_tr * r["train_step_s_per_ecg"] + n_va * r["eval_s_per_ecg"]
            for ep in (25, 40):
                fold = ep * per_epoch + n_te * r["eval_s_per_ecg"]
                proj[f"{name}__threads{th}x{par}__{ep}ep"] = dict(
                    hours_per_outer_fold=fold / 3600,
                    hours_stage1_15_folds=15 * fold / 3600 / par,
                    hours_stage2_extra_30_folds=30 * fold / 3600 / par,
                    hours_E_inner_oof_seed42_60_fits=60 * fold * (0.75 * n_tr / n_tr) / 3600 / par)
    c = out["threads"]["4"]
    proj["one_off_cache_all_837_dev_hours"] = dict(stage5=n_all * c["frozen_forward_to_stage5_s_per_ecg"] / 3600,
                                                   stage4=n_all * c["frozen_forward_to_stage4_s_per_ecg"] / 3600,
                                                   input_build=n_all * t_build / 3600)
    out["projection"] = proj
    out["note"] = ("compute only; random targets of zero, no label / EF / metric read. Projections assume the "
                   "pre-registered recipe and no parallel contention beyond the stated thread layout.")
    guard6(RES6).mkdir(parents=True, exist_ok=True)
    (RES6 / "timing_A.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
