"""1-D CNN on the waveforms digitised from the PDF vector paths (ecg_ef/signals.py).

Same split, loss, EMA, selection rule and metrics as the image CNNs (train.py), so the
results are directly comparable.

    python -m ecg_ef.signals                       # once: extract waveforms
    python -m ecg_ef.train1d                       # 12 leads (12 x 2.5 s)
    python -m ecg_ef.train1d --rhythm              # + 10-s lead-II rhythm strip branch
    python -m ecg_ef.train1d --rhythm --seeds 5    # 5 seeds, averaged (seed ensemble)
    python -m ecg_ef.train1d --sampling undersample # equal images per class (drop extras)
    python -m ecg_ef.train1d --sampling balanced    # equal-class batches, all images
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

from .config import CLASSES, HF_TARGET_SENSITIVITY, MODELS_DIR, RESULTS_DIR
from .metrics import evaluate, pick_hf_threshold
from .preprocess import MANIFEST
from .signals import SIGNALS
from .train import EMA, plot_confusion, plot_history, save_predictions, seed_all, selection_score


# ----------------------------------------------------------------------------- model
class SE1d(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(ch, ch // r, 1), nn.ReLU(True),
                                nn.Conv1d(ch // r, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class Block1d(nn.Module):
    def __init__(self, cin, cout, stride, k=9, drop=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(True),
            nn.Dropout(drop),
            nn.Conv1d(cout, cout, k, 1, k // 2, bias=False), nn.BatchNorm1d(cout), SE1d(cout))
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


def _encoder(cin, widths, drop):
    layers = [nn.Conv1d(cin, widths[0], 15, 2, 7, bias=False), nn.BatchNorm1d(widths[0]), nn.ReLU(True)]
    for a, b in zip(widths[:-1], widths[1:]):
        layers += [Block1d(a, b, 2, drop=drop), Block1d(b, b, 1, drop=drop)]
    return nn.Sequential(*layers)


class ResNet1d(nn.Module):
    """12-lead 1-D SE-ResNet (+ optional rhythm-strip branch). [GAP || GMP] head."""

    def __init__(self, num_classes=4, rhythm=False, widths=(64, 96, 128, 192, 256), drop=0.1):
        super().__init__()
        self.leads = _encoder(12, widths, drop)
        feat = 2 * widths[-1]
        self.rhythm = _encoder(1, (32, 48, 64, 96, 128), drop) if rhythm else None
        if rhythm:
            feat += 2 * 128
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, num_classes))

    @staticmethod
    def _pool(f):
        return torch.cat([f.mean(-1), f.amax(-1)], 1)

    def forward(self, leads, rhythm=None):
        f = self._pool(self.leads(leads))
        if self.rhythm is not None:
            f = torch.cat([f, self._pool(self.rhythm(rhythm[:, None]))], 1)
        return self.head(f)


# ----------------------------------------------------------------------------- data
def load_signal_split(split):
    m = pd.read_csv(MANIFEST)
    z = np.load(SIGNALS, allow_pickle=True)
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    df = m[m.split == split].reset_index(drop=True)
    ii = [idx[e] for e in df.ecg_id]
    leads = torch.from_numpy(z["leads"][ii]); rhythm = torch.from_numpy(z["rhythm"][ii])
    # remove each lead's offset (isoelectric level); keep absolute mV amplitude, which
    # carries information (e.g. low QRS voltage)
    leads = leads - leads.median(-1, keepdim=True).values
    rhythm = rhythm - rhythm.median(-1, keepdim=True).values
    return leads, rhythm, torch.tensor(df.label_idx.values), df


def augment1d(leads, rhythm):
    """Gain +-10 %, random 90 % time crop, small noise, slow baseline wander.
    No time reversal / polarity flip (both change the diagnosis)."""
    b = leads.shape[0]
    g = 1 + (torch.rand(b, 1, 1) * 2 - 1) * 0.10
    n1, n2 = int(leads.shape[-1] * 0.9), int(rhythm.shape[-1] * 0.9)
    s1 = int(torch.randint(0, leads.shape[-1] - n1 + 1, (1,)))
    s2 = int(torch.randint(0, rhythm.shape[-1] - n2 + 1, (1,)))
    L = leads[..., s1:s1 + n1] * g
    R = rhythm[..., s2:s2 + n2] * g[:, 0]
    t = torch.linspace(0, 1, n1)
    wander = (torch.rand(b, 1, 1) * 0.05) * torch.sin(2 * math.pi * (torch.rand(b, 1, 1) * 0.5 * t
                                                                       + torch.rand(b, 1, 1)))
    L = L + wander + 0.01 * torch.randn_like(L)
    R = R + 0.01 * torch.randn_like(R)
    return L, R


@torch.no_grad()
def predict1d(model, leads, rhythm, bs=64):
    model.eval()
    return torch.cat([F.softmax(model(leads[i:i + bs], rhythm[i:i + bs]), 1)
                      for i in range(0, len(leads), bs)]).numpy()


# ----------------------------------------------------------------------------- train
def train_one(args, seed, data):
    (ltr, rtr, ytr, _), (lva, rva, yva, _) = data["train"], data["val"]
    seed_all(seed)
    counts = torch.bincount(ytr, minlength=4).float()
    class_w = counts.sum() / (4 * counts); class_w = class_w / class_w.mean()
    if args.sampling == "undersample":
        # keep the same number of ECGs from every class (= the smallest class)
        n = int(counts.min())
        keep = torch.cat([torch.nonzero(ytr == c).flatten()[torch.randperm(int(counts[c]))[:n]]
                          for c in range(4)])
        ltr, rtr, ytr = ltr[keep], rtr[keep], ytr[keep]
    if args.sampling != "weights":
        class_w = torch.ones(4)                       # balance comes from the data instead
    sample_p = (1 / counts)[ytr]                      # for --sampling balanced
    model = ResNet1d(4, rhythm=args.rhythm)
    ema = EMA(model, 0.99)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    spe = len(ltr) // args.batch_size
    total, warm = args.epochs * spe, 3 * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best, best_state, best_epoch, bad, hist = -1, None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = (torch.multinomial(sample_p, len(ltr), replacement=True)
                if args.sampling == "balanced" else torch.randperm(len(ltr)))
        tl = tc = tn = 0
        for i in range(spe):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            L, R = augment1d(ltr[idx], rtr[idx])
            logits = model(L, R)
            loss = F.cross_entropy(logits, ytr[idx], weight=class_w, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); ema.update(model)
            tl += loss.item() * len(idx); tc += (logits.argmax(1) == ytr[idx]).sum().item(); tn += len(idx)
        pva = predict1d(ema.model, lva, rva)
        vm = evaluate(yva.numpy(), pva)
        score = selection_score(vm)
        hist.append(dict(epoch=epoch, train_loss=tl / tn, train_acc=tc / tn,
                         val_loss=F.nll_loss(torch.log(torch.from_numpy(pva).clamp_min(1e-8)), yva).item(),
                         val_acc=vm["accuracy"], val_macro_f1=vm["macro_f1"], val_qwk=vm["qwk"],
                         val_hf_auc=vm["hf_screen"]["auc"], val_score=score))
        if epoch >= args.min_epochs:
            if score > best:
                best, best_state, best_epoch, bad = score, copy.deepcopy(ema.model.state_dict()), epoch, 0
            else:
                bad += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"  seed {seed} ep {epoch:3d} loss {tl / tn:.3f} acc {tc / tn:.3f} | val F1 "
                  f"{vm['macro_f1']:.3f} QWK {vm['qwk']:.3f} HF-AUC {vm['hf_screen']['auc']:.3f} "
                  f"score {score:.3f}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, pd.DataFrame(hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhythm", action="store_true", help="add the 10-s rhythm-strip branch")
    ap.add_argument("--seeds", type=int, default=1, help=">1 = average several seeds")
    ap.add_argument("--sampling", default="weights", choices=["weights", "undersample", "balanced"],
                    help="class imbalance: loss weights (default), equal images per class by "
                         "dropping extras, or equal-class batches drawn from all images")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tag = args.tag or ("resnet1d" + ("_rhythm" if args.rhythm else "") +
                       (f"_x{args.seeds}" if args.seeds > 1 else ""))
    out = RESULTS_DIR / tag; out.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)
    data = {s: load_signal_split(s) for s in ("train", "val", "test")}
    (lva, rva, yva, dva), (lte, rte, yte, dte) = data["val"], data["test"]

    t0 = time.time(); states, pvs, pts, epochs, per_seed = [], [], [], [], []
    for k in range(args.seeds):
        model, be, hist = train_one(args, 42 + k, data)
        states.append(model.state_dict()); epochs.append(be)
        pvs.append(predict1d(model, lva, rva)); pts.append(predict1d(model, lte, rte))
        per_seed.append(evaluate(yte.numpy(), pts[-1]))
        if k == 0:
            plot_history(hist, out / "training_curves.png"); hist.to_csv(out / "history.csv", index=False)
    pva, pte = np.mean(pvs, 0), np.mean(pts, 0)
    hf_thr = pick_hf_threshold(yva.numpy(), pva, HF_TARGET_SENSITIVITY)
    val_m, test_m = evaluate(yva.numpy(), pva, hf_thr), evaluate(yte.numpy(), pte, hf_thr)
    n_params = sum(p.numel() for p in model.parameters())
    torch.save(dict(model_name="resnet1d", variant="signal", rhythm=args.rhythm, classes=CLASSES,
                    state_dicts=states, hf_threshold=hf_thr, best_epochs=epochs, args=vars(args)),
               MODELS_DIR / f"{tag}.pt")
    summary = dict(tag=tag, model="resnet1d", variant="signal", pretrained=False, img_hw=[12, 1250],
                   rhythm=args.rhythm, seeds=args.seeds, params=n_params, best_epoch=epochs,
                   train_minutes=round((time.time() - t0) / 60, 1), lr=args.lr,
                   hf_threshold=hf_thr, sampling=args.sampling, val=val_m, test=test_m,
                   test_per_seed=[{k: v for k, v in m.items() if k in
                                   ("accuracy", "balanced_accuracy", "macro_f1", "qwk")} |
                                  {"hf_auc": m["hf_screen"]["auc"],
                                   "recall": [m["per_class"][c]["recall"] for c in CLASSES]}
                                  for m in per_seed])
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_confusion(test_m["confusion_matrix"], f"{tag} - test (acc {test_m['accuracy']:.1%})",
                   out / "confusion_matrix.png")
    save_predictions(pte, dte, out / "test_predictions.csv")
    save_predictions(pva, dva, out / "val_predictions.csv")
    hf = test_m["hf_screen"]
    print(f"\n[{tag}] {args.seeds} seed(s), best epochs {epochs}, {summary['train_minutes']} min")
    print(f"  VAL score {selection_score(val_m):.3f}")
    print(f"  TEST acc {test_m['accuracy']:.3f}  bal-acc {test_m['balanced_accuracy']:.3f}  "
          f"macro-F1 {test_m['macro_f1']:.3f}  QWK {test_m['qwk']:.3f}")
    print(f"  HF AUC {hf['auc']:.3f}  sens {hf['sensitivity']:.3f}  spec {hf['specificity']:.3f}  "
          f"| EF<=40% AUC {test_m['ef_le_40']['auc']:.3f}")


if __name__ == "__main__":
    main()
