"""Image CNN (ResNet-18, the project's existing baseline recipe, unchanged) under the phase-2
5-fold patient-grouped CV on the development set. Per fold: train on inner-train, early-stop /
select on inner-val with the original rule (mean(macro-F1, QWK, HF AUC) after epoch 8,
patience 10), predict the outer fold (OOF) and the locked test set (stored, not scored).

    python -m research_experiments.phase_2.cv_image_cnn [--threads 3] [--folds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from research_experiments.common import CLASSES, seed_all
from research_experiments.phase_2.p2common import LOCK, LOGS, MOD, PCOLS, RES, guard
from research_experiments.phase_2.protocol import dev_and_test, split_indices
from ecg_ef.data import Normalizer, augment, load_split
from ecg_ef.metrics import evaluate
from ecg_ef.models import build_model
from ecg_ef.train import EMA, predict_probs, selection_score

TAG = "img_resnet18_baseline_recipe"


def load_images(ids):
    pool = {}
    for s in ("train", "val", "test"):
        x, _, df = load_split(s, "filtered", (256, 512))
        for i, e in enumerate(df.ecg_id):
            pool[e] = x[i]
    return torch.stack([pool[e] for e in ids])


def train_fold(xtr, ytr, xva, yva, seed=42, epochs=30, patience=10, min_epochs=8, bs=16, lr=1e-3, wd=0.05, log=None):
    seed_all(seed)
    norm = Normalizer.fit(xtr)
    counts = torch.bincount(ytr, minlength=4).float(); cw = counts.sum() / (4 * counts); cw = cw / cw.mean()
    model = build_model("resnet18", 4).to(memory_format=torch.channels_last)
    ema = EMA(model, 0.99)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    spe = len(xtr) // bs; total, warm = epochs * spe, 2 * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best, best_state, best_ep, bad, hist = -1, None, 0, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(len(xtr))
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            x = norm(augment(xtr[idx].float() / 255)).contiguous(memory_format=torch.channels_last)
            y = ytr[idx]
            ce = F.cross_entropy(model(x), y, weight=cw, label_smoothing=0.05, reduction="none")
            loss = ce.sum() / cw[y].sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); ema.update(model)
        pva = predict_probs(ema.model, xva, norm); sc = selection_score(evaluate(yva.numpy(), pva))
        hist.append(dict(epoch=ep, val_score=sc, minutes=(time.time() - t0) / 60))
        flag = ""
        if ep >= min_epochs:
            if sc > best:
                best, best_ep, bad, flag = sc, ep, 0, " *"
                best_state = {k: v.detach().clone() for k, v in ema.model.state_dict().items()}
            else:
                bad += 1
        if log:
            print(f"  ep {ep:2d} val_score {sc:.3f} {hist[-1]['minutes']:.1f}m{flag}", file=log, flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, norm, dict(best_epoch=best_ep, best_val_score=best, epochs_run=len(hist),
                             train_minutes=(time.time() - t0) / 60, history=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    meta, D, T = dev_and_test()
    xD, xT = load_images(D.ecg_id), load_images(T.ecg_id)
    yD = torch.tensor(D.label_idx.values)
    out_dir = guard(RES / TAG); out_dir.mkdir(parents=True, exist_ok=True)
    log = open(guard(LOGS / f"{TAG}.log"), "a")
    for k in a.folds:
        part = out_dir / f"fold{k}.npz"
        if part.exists():
            continue
        itr, iva, ote = split_indices(D, 0, k)
        print(f"fold {k}: inner-train {len(itr)} inner-val {len(iva)} outer {len(ote)}", file=log, flush=True)
        model, norm, info = train_fold(xD[itr], yD[itr], xD[iva], yD[iva], log=log)
        t1 = time.time(); p_out = predict_probs(model, xD[ote], norm); inf = (time.time() - t1) / len(ote)
        p_val = predict_probs(model, xD[iva], norm); p_T = predict_probs(model, xT, norm)
        np.savez(part, ote=ote, iva=iva, p_out=p_out, p_val=p_val, p_T=p_T)
        torch.save(dict(state_dict=model.state_dict(), normalizer=norm.state(), fold=k, info=info), guard(MOD / f"{TAG}_fold{k}.pt"))
        info["inference_s_per_ecg_model_only"] = inf
        (out_dir / f"fold{k}_info.json").write_text(json.dumps(info, indent=1))
        print(f"fold {k} done: best epoch {info['best_epoch']} ({info['train_minutes']:.1f} min)", file=log, flush=True)
    if all((out_dir / f"fold{k}.npz").exists() for k in range(5)):
        P = np.zeros((len(D), 4)); fold = np.full(len(D), -1); PT = []
        for k in range(5):
            z = np.load(out_dir / f"fold{k}.npz"); P[z["ote"]] = z["p_out"]; fold[z["ote"]] = k; PT.append(z["p_T"])
        oof = pd.DataFrame(P, columns=PCOLS); oof.insert(0, "ecg_id", D.ecg_id.values); oof.insert(1, "fold", fold)
        locked = pd.DataFrame(np.mean(PT, 0), columns=PCOLS); locked.insert(0, "ecg_id", T.ecg_id.values)
        infos = [json.loads((out_dir / f"fold{k}_info.json").read_text()) for k in range(5)]
        oof.to_csv(out_dir / "oof.csv", index=False, float_format="%.6f")
        guard(LOCK).mkdir(parents=True, exist_ok=True); locked.to_csv(LOCK / f"{TAG}.csv", index=False, float_format="%.6f")
        (out_dir / "info.json").write_text(json.dumps(dict(
            input="filtered ECG image 256x512", recipe="ecg_ef/train.py defaults (random init)", params=11172292,
            best_epochs=[i["best_epoch"] for i in infos], train_minutes=[i["train_minutes"] for i in infos],
            inference_s_per_ecg_model_only=float(np.mean([i["inference_s_per_ecg_model_only"] for i in infos]))), indent=1))
        print("all folds done", file=log, flush=True)


if __name__ == "__main__":
    main()
