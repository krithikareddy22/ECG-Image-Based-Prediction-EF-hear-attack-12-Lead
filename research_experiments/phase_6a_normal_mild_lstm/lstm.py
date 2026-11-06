"""Phase 6A experiment C/D: standalone LSTM classifiers on the digitised waveform (no ECGFounder, no ResNet).

Variants (PHASE6A_PLAN.md section 5):  L0 rhythm 1x uni | L1 aligned 1x uni | L2 aligned 2x uni | L3 aligned 1x bi |
L4 aligned 2x bi, inter-layer dropout 0.3 | L5 CNN-LSTM (conditional) - all hidden 64.
Head: [last hidden state (all directions) || mean of outputs over recorded steps] -> Dropout 0.3 -> Linear(-> 4).
Training: class-balanced CE (Phase 2 weights), AdamW lr 1e-3 wd 1e-2, batch 32, grad clip 1.0, <= 60 epochs, early
stopping on inner-val S3 (patience 12, not before epoch 10), best epoch restored. Inputs: signals.py (100 Hz, 10 s).
Labels / training rows per configuration (B1, B2, B3, A270) via p6acommon.split / y_train.
Jobs = (variant, config, seed, repeat, fold); each finished fold is cached (predictions + inner-val score + timing) and
its best weights saved to the git-ignored checkpoint store, so interrupted work resumes at fold granularity.
    python -m research_experiments.phase_6a_normal_mild_lstm.lstm --variants L0 L1 L2 L3 L4 --config B1 --jobs 4
    python -m research_experiments.phase_6a_normal_mild_lstm.lstm --select          # nested selection -> LSTM-sel, LSTM-fixed
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_6a_normal_mild_lstm import signals
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import (CKPT, CONF, FOLDC, RES, SEEDS, done, guard, load_run,
                                                                     mark_running, metrics, nD, save_run, split, y_train, Y)

VARIANTS = {
    "L0": dict(input="rhythm", layers=1, bi=False, inter_dropout=0.0, cnn=False),
    "L1": dict(input="aligned", layers=1, bi=False, inter_dropout=0.0, cnn=False),
    "L2": dict(input="aligned", layers=2, bi=False, inter_dropout=0.0, cnn=False),
    "L3": dict(input="aligned", layers=1, bi=True, inter_dropout=0.0, cnn=False),
    "L4": dict(input="aligned", layers=2, bi=True, inter_dropout=0.3, cnn=False),
    "L5": dict(input="aligned", layers=1, bi=True, inter_dropout=0.0, cnn=True),       # CNN-LSTM (conditional)
}
HID, HEAD_DO, LR, WD, BS, CLIP, MAX_EP, MIN_EP, PATIENCE = 64, 0.3, 1e-3, 1e-2, 32, 1.0, 60, 10, 12
CMD = "python -m research_experiments.phase_6a_normal_mild_lstm.lstm"
_X = {}
import os  # noqa: E402
DEVICE = os.environ.get("P6A_DEVICE", "cpu")   # "--device cuda" on a GPU machine (use --jobs 1); env var so that spawned
                                               # worker processes (Windows / macOS) inherit it; CPU results unaffected


def X_of(kind):
    if kind not in _X:
        _X[kind] = torch.from_numpy(signals.load(kind))
    return _X[kind]


class LSTMNet(nn.Module):
    def __init__(self, cin, layers, bi, inter_dropout, cnn):
        super().__init__()
        self.cnn = None
        if cnn:
            self.cnn = nn.Sequential(nn.Conv1d(cin, 32, 7, 2, 3), nn.GELU(), nn.Conv1d(32, 64, 7, 2, 3), nn.GELU(),
                                     nn.Conv1d(64, 64, 7, 2, 3), nn.GELU())
            cin = 64
        self.lstm = nn.LSTM(cin, HID, layers, batch_first=True, bidirectional=bi, dropout=inter_dropout if layers > 1 else 0.0)
        d = HID * (2 if bi else 1)
        self.drop = nn.Dropout(HEAD_DO)
        self.out = nn.Linear(2 * d, 4)
        self.bi = bi

    def forward(self, x):
        w = x[:, :, -1] if x.shape[2] == 26 else torch.ones(x.shape[:2], device=x.device)   # rhythm mask = recorded steps
        if self.cnn is not None:
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            w = F.max_pool1d(w[:, None], 8, 8, ceil_mode=True)[:, 0][:, :x.shape[1]]
        o, (h, _) = self.lstm(x)
        last = torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1]
        mean = (o * w[:, :, None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1.0)
        return self.out(self.drop(torch.cat([last, mean], 1)))


def n_params(v):
    spec = VARIANTS[v]
    return sum(p.numel() for p in LSTMNet(26 if spec["input"] == "aligned" else 1, spec["layers"], spec["bi"],
                                          spec["inter_dropout"], spec["cnn"]).parameters())


@torch.no_grad()
def predict(m, X, idx, bs=128):
    m.eval()
    return torch.cat([F.softmax(m(X[idx[i:i + bs]]), 1) for i in range(0, len(idx), bs)]).cpu().numpy()


def exp_id(v, cfg):
    return f"P6A-lstm{v}-{cfg}-v1"


def fit_fold(v, cfg, seed, r, k):
    f = guard(FOLDC / exp_id(v, cfg) / f"s{seed}_r{r}_k{k}.npz")
    if f.exists():
        return str(f)
    torch.set_num_threads(1)
    spec = VARIANTS[v]; X = X_of(spec["input"]).to(DEVICE); yt = torch.tensor(y_train(cfg)).to(DEVICE)
    itr, iva, ote = split(cfg, r, k)
    torch.manual_seed(seed * 1000 + 10 * r + k)
    m = LSTMNet(X.shape[2], spec["layers"], spec["bi"], spec["inter_dropout"], spec["cnn"]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    cw = class_w(yt[itr].cpu().numpy(), "balanced").to(DEVICE)
    g = torch.Generator().manual_seed(seed * 1000 + 10 * r + k)
    best, bad, hist, t0 = None, 0, [], time.time()
    for ep in range(MAX_EP):
        m.train()
        perm = itr[torch.randperm(len(itr), generator=g).numpy()]
        for i in range(0, len(perm), BS):
            j = perm[i:i + BS]
            loss = F.cross_entropy(m(X[j]), yt[j], weight=cw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
        sv = s3(evaluate(yt[iva].cpu().numpy(), predict(m, X, iva))); hist.append(round(sv, 4))
        if best is None or sv > best[0] + 1e-9:
            best = (sv, ep, {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}); bad = 0
        else:
            bad += 1
        if ep + 1 >= MIN_EP and bad >= PATIENCE:
            break
    train_min = (time.time() - t0) / 60
    m.load_state_dict(best[2])
    t1 = time.perf_counter(); Po = predict(m, X, ote); infer_ms = (time.perf_counter() - t1) / len(ote) * 1000
    ck = guard(CKPT / exp_id(v, cfg) / f"s{seed}_r{r}_k{k}.pt"); ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best[2], ck)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, ote=ote, P=Po, val_s3=best[0], best_epoch=best[1], epochs_run=len(hist), train_minutes=train_min,
             infer_ms_per_ecg=infer_ms, hist=np.array(hist))
    print(f"[{v} {cfg} s{seed} r{r} k{k}] best ep {best[1]} val S3 {best[0]:.3f} ({len(hist)} ep, {train_min:.1f} min)", flush=True)
    return str(f)


def _job(a):
    return fit_fold(*a)


def fold(v, cfg, seed, r, k):
    return np.load(FOLDC / exp_id(v, cfg) / f"s{seed}_r{r}_k{k}.npz")


def assemble(v, cfg, seed, r):
    rid = f"{exp_id(v, cfg)}__s{seed}_r{r}"
    if done(rid):
        return
    P = np.zeros((nD, 4)); info = []
    for k in range(5):
        z = fold(v, cfg, seed, r, k); P[z["ote"]] = z["P"]
        info.append(dict(fold=k, val_s3=float(z["val_s3"]), best_epoch=int(z["best_epoch"]), epochs_run=int(z["epochs_run"]),
                         train_minutes=float(z["train_minutes"]), infer_ms_per_ecg=float(z["infer_ms_per_ecg"])))
    spec = VARIANTS[v]
    cfgd = dict(model=f"LSTM {v}", **spec, hidden=HID, head_dropout=HEAD_DO, optimiser="AdamW", lr=LR, weight_decay=WD, batch=BS,
                grad_clip=CLIP, max_epochs=MAX_EP, min_epochs=MIN_EP, patience=PATIENCE, loss="class-balanced CE",
                selection="inner-val S3", input_hz=100, steps=1000, channels=26 if spec["input"] == "aligned" else 1,
                label_config=cfg, seed=seed, repeat=r)
    save_run(rid, P, P.argmax(1), r, cfg, cfgd, extra=dict(folds=info, n_params=n_params(v),
                                                          train_minutes=sum(i["train_minutes"] for i in info),
                                                          infer_ms_per_ecg=float(np.mean([i["infer_ms_per_ecg"] for i in info]))))


def select(cfg="B1", variants=("L0", "L1", "L2", "L3", "L4")):
    """Nested selection: per (repeat, fold) the variant with the highest mean inner-val S3 over seeds (outer fold unseen)."""
    choice = {}
    for r in (0, 1, 2):
        for k in range(5):
            sc = {v: float(np.mean([fold(v, cfg, s, r, k)["val_s3"] for s in SEEDS])) for v in variants}
            choice[f"r{r}_k{k}"] = max(sc, key=lambda v: (sc[v], -n_params(v)))
    for s in SEEDS:
        for r in (0, 1, 2):
            rid = f"P6A-lstmsel-{cfg}-v1__s{s}_r{r}"
            if done(rid):
                continue
            P = np.zeros((nD, 4))
            for k in range(5):
                z = fold(choice[f"r{r}_k{k}"], cfg, s, r, k); P[z["ote"]] = z["P"]
            save_run(rid, P, P.argmax(1), r, cfg, dict(model="LSTM-sel (nested inner-val selection over L0-L4)", label_config=cfg,
                                                       seed=s, repeat=r, variants=list(variants)),
                     extra=dict(choice_per_fold=[choice[f"r{r}_k{k}"] for k in range(5)]))
    cnt = Counter(choice.values())
    fixed = sorted(cnt, key=lambda v: (-cnt[v], n_params(v)))[0]
    out = dict(choice_per_fold=choice, counts=dict(cnt), lstm_fixed=fixed, rule="most often selected; ties -> fewer parameters",
               params={v: n_params(v) for v in variants})
    (guard(RES) / "lstm_selection.json").write_text(json.dumps(out, indent=1))
    (guard(CONF) / "lstm_fixed.json").write_text(json.dumps(dict(lstm_fixed=fixed, fixed_on="B1 nested selection counts"), indent=1))
    print(json.dumps(out, indent=1))
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["L0", "L1", "L2", "L3", "L4"])
    ap.add_argument("--config", default="B1")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device; os.environ["P6A_DEVICE"] = a.device
    if a.select:
        select(a.config); return
    cost = {"L4": 5, "L3": 3, "L2": 3, "L5": 2, "L1": 1, "L0": 1}
    jobs = [(v, a.config, s, r, k) for v in sorted(a.variants, key=lambda v: -cost[v]) for s in a.seeds for r in (0, 1, 2) for k in range(5)]
    for v in a.variants:
        for s in a.seeds:
            for r in (0, 1, 2):
                if not done(f"{exp_id(v, a.config)}__s{s}_r{r}"):
                    mark_running(f"{exp_id(v, a.config)}__s{s}_r{r}", CMD)
    X_of("aligned"); X_of("rhythm")
    with Pool(a.jobs) as pool:
        list(pool.imap_unordered(_job, jobs))
    for v in a.variants:
        for s in a.seeds:
            for r in (0, 1, 2):
                assemble(v, a.config, s, r)
                m = load_run(f"{exp_id(v, a.config)}__s{s}_r{r}")
                mm = metrics(Y, *m)
                print(f"[{exp_id(v, a.config)} s{s} r{r}] acc {mm['accuracy']:.3f} bal {mm['balanced_accuracy']:.3f} "
                      f"mF1 {mm['macro_f1']:.3f} Mild F1 {mm['f1_Mild']:.3f} minR {mm['min_recall']:.2f}", flush=True)


if __name__ == "__main__":
    main()
