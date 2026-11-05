"""Train one CNN on the ECG images and evaluate it on the held-out test patients.

Examples
--------
    python -m ecg_ef.train --model ecgnet
    python -m ecg_ef.train --model resnet18 --pretrained      # needs internet for weights
    python -m ecg_ef.train --model ecgnet --variant raw        # no grid filtering
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import CLASSES, HF_TARGET_SENSITIVITY, IMG_H, IMG_W, MODELS_DIR, RESULTS_DIR
from .data import Normalizer, augment, load_split
from .metrics import evaluate, pick_hf_threshold
from .models import MODEL_NAMES, build_model
from .preprocess import VARIANTS


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class EMA:
    """Exponential moving average of the weights (incl. BN statistics). With ~670
    training images the raw weights swing a lot between epochs; the averaged model is
    much more stable and is the one that is evaluated and saved."""

    def __init__(self, model, decay: float = 0.99):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))   # warm-up
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_probs(model, imgs_u8, norm, bs=32):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(imgs_u8), bs):
        x = norm(imgs_u8[i:i + bs].to(dev).float() / 255).contiguous(memory_format=torch.channels_last)
        out.append(F.softmax(model(x)[:, :len(CLASSES)], 1).cpu())   # [:, :4]: ignore aux heads
    return torch.cat(out).numpy()


def selection_score(m: dict) -> float:
    """Validation score used for early stopping / checkpoint and model selection."""
    return (m["macro_f1"] + m["qwk"] + m["hf_screen"]["auc"]) / 3


def save_predictions(probs, df, path):
    """Per-ECG probabilities (ids are year_timestamp; no patient names)."""
    pred = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
    pred.insert(0, "ecg_id", df.ecg_id.values); pred.insert(1, "true", df.label.values)
    pred.insert(2, "pred", [CLASSES[i] for i in probs.argmax(1)])
    pred["hf_likelihood"] = 1 - probs[:, 0]
    pred.to_csv(path, index=False, float_format="%.6f")


def plot_confusion(cm, title, path):
    cm = np.asarray(cm)
    frac = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (echo)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if frac[i, j] > 0.55 else "black")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_history(hist: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(hist.epoch, hist.train_loss, label="train")
    axes[0].plot(hist.epoch, hist.val_loss, label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(hist.epoch, hist.train_acc, label="train acc")
    axes[1].plot(hist.epoch, hist.val_acc, label="val acc")
    axes[1].plot(hist.epoch, hist.val_macro_f1, label="val macro-F1")
    axes[1].set_title("Validation"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ecgnet", choices=MODEL_NAMES)
    ap.add_argument("--variant", default="filtered", choices=list(VARIANTS),
                    help="image preprocessing (see preprocess.VARIANTS)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-epochs", type=int, default=8, help="no model selection before this")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "sqrt", "none"])
    ap.add_argument("--ema", type=float, default=0.99, help="EMA decay (0 disables)")
    ap.add_argument("--multitask", action="store_true",
                    help="extra binary heads 'EF reduced?' and 'EF <= 40%%?' sharing the backbone "
                         "(multi-task learning, after Boribalburephan et al., Sci Rep 2024)")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of the auxiliary losses")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="focal loss gamma (0 = plain cross-entropy; Swamy et al. 2025 used 0.2)")
    ap.add_argument("--img-size", type=int, nargs=2, default=[IMG_H, IMG_W], metavar=("H", "W"))
    ap.add_argument("--pretrained", action="store_true", help="ImageNet init (downloads weights)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    tag = args.tag or (f"{args.model}_{args.variant}" + ("_pretrained" if args.pretrained else "")
                       + ("_mtl" if args.multitask else "")
                       + (f"_focal{args.focal_gamma:g}" if args.focal_gamma else ""))
    lr = args.lr or (3e-4 if args.pretrained else 1e-3)
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    img_hw = tuple(args.img_size)

    xtr, ytr, _ = load_split("train", args.variant, img_hw)
    xva, yva, dva = load_split("val", args.variant, img_hw)
    xte, yte, dte = load_split("test", args.variant, img_hw)
    norm = Normalizer.fit(xtr)
    print(f"[{tag}] train {len(xtr)}  val {len(xva)}  test {len(xte)}  img {img_hw}  lr {lr}  "
          f"device {DEVICE}")
    torch.backends.cudnn.benchmark = True

    # Class imbalance (Normal 46% ... Severe 11%): inverse-frequency loss weights.
    counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
    class_w = counts.sum() / (len(CLASSES) * counts)
    if args.class_weight == "sqrt":
        class_w = class_w.sqrt()
    elif args.class_weight == "none":
        class_w = torch.ones_like(class_w)
    class_w = class_w / class_w.mean()
    print("class weights", dict(zip(CLASSES, class_w.round(decimals=2).tolist())))

    n_outputs = len(CLASSES) + (2 if args.multitask else 0)
    model = build_model(args.model, n_outputs, pretrained=args.pretrained)
    model = model.to(DEVICE, memory_format=torch.channels_last)
    class_w = class_w.to(DEVICE)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    eval_model = ema.model if ema else model
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    steps_per_epoch = len(xtr) // args.batch_size
    total, warm = args.epochs * steps_per_epoch, 2 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    best, best_state, best_epoch, hist, bad = -1.0, None, 0, [], 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        tl, tc, tn = 0.0, 0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            x = augment(xtr[idx].to(DEVICE).float() / 255)
            x = norm(x).contiguous(memory_format=torch.channels_last)
            y = ytr[idx].to(DEVICE)
            out = model(x)
            logits = out[:, :len(CLASSES)]
            ce = F.cross_entropy(logits, y, weight=class_w, label_smoothing=args.label_smoothing,
                                 reduction="none")
            if args.focal_gamma:      # down-weight easy examples
                p_true = F.softmax(logits, 1).gather(1, y[:, None])[:, 0].detach()
                ce = (1 - p_true) ** args.focal_gamma * ce
            loss = ce.sum() / class_w[y].sum()          # = weighted mean, as F.cross_entropy
            if args.multitask:        # aux heads: EF reduced (class > 0), EF <= 40% (class >= 2)
                aux_t = torch.stack([(y > 0), (y >= 2)], 1).float()
                loss = loss + args.aux_weight * F.binary_cross_entropy_with_logits(out[:, 4:6], aux_t)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            if ema:
                ema.update(model)
            tl += loss.item() * len(y); tc += (logits.argmax(1) == y).sum().item(); tn += len(y)

        pva = predict_probs(eval_model, xva, norm)
        vloss = F.nll_loss(torch.log(torch.from_numpy(pva).clamp_min(1e-8)), yva).item()
        vm = evaluate(yva.numpy(), pva)
        hist.append(dict(epoch=epoch, lr=sched.get_last_lr()[0], train_loss=tl / tn,
                         train_acc=tc / tn, val_loss=vloss, val_acc=vm["accuracy"],
                         val_bal_acc=vm["balanced_accuracy"], val_macro_f1=vm["macro_f1"],
                         val_qwk=vm["qwk"], val_hf_auc=vm["hf_screen"]["auc"],
                         elapsed_min=(time.time() - t0) / 60))
        # Model selection: mean of three validation metrics (less noisy than any one on
        # 168 ECGs), and only after a few epochs so a lucky early epoch can't win.
        score = selection_score(vm)
        hist[-1]["val_score"] = score
        flag = ""
        if epoch >= args.min_epochs:
            if score > best:
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
                best, best_epoch, bad, flag = score, epoch, 0, " *"
            else:
                bad += 1
        h = hist[-1]
        print(f"ep {epoch:2d}  loss {h['train_loss']:.3f}/{h['val_loss']:.3f}  "
              f"acc {h['train_acc']:.3f}/{h['val_acc']:.3f}  vF1 {h['val_macro_f1']:.3f}  "
              f"vQWK {h['val_qwk']:.3f}  vHF-AUC {h['val_hf_auc']:.3f}  score {score:.3f}  "
              f"{h['elapsed_min']:.1f}m{flag}", flush=True)
        if bad >= args.patience:
            print(f"early stop (no val improvement for {args.patience} epochs)")
            break

    train_min = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    pva = predict_probs(model, xva, norm)
    pte = predict_probs(model, xte, norm)
    hf_thr = pick_hf_threshold(yva.numpy(), pva, HF_TARGET_SENSITIVITY)   # chosen on val only
    val_m = evaluate(yva.numpy(), pva, hf_thr)
    test_m = evaluate(yte.numpy(), pte, hf_thr)
    test_m["per_year_accuracy"] = {
        str(y): float((pte.argmax(1)[dte.year.values == y] == yte.numpy()[dte.year.values == y]).mean())
        for y in sorted(dte.year.unique())}

    ckpt = dict(model_name=args.model, variant=args.variant, img_hw=img_hw, classes=CLASSES,
                n_outputs=n_outputs,
                normalizer=norm.state(), state_dict=best_state, best_epoch=best_epoch,
                hf_threshold=hf_thr, val_metrics=val_m, args=vars(args))
    torch.save(ckpt, MODELS_DIR / f"{tag}.pt")

    summary = dict(tag=tag, model=args.model, variant=args.variant, pretrained=args.pretrained,
                   img_hw=img_hw, params=n_params, epochs_run=len(hist), best_epoch=best_epoch,
                   train_minutes=round(train_min, 1), lr=lr, hf_threshold=hf_thr,
                   val=val_m, test=test_m)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    hist = pd.DataFrame(hist); hist.to_csv(out_dir / "history.csv", index=False)
    plot_history(hist, out_dir / "training_curves.png")
    plot_confusion(test_m["confusion_matrix"],
                   f"{tag} - test (acc {test_m['accuracy']:.1%})", out_dir / "confusion_matrix.png")
    save_predictions(pte, dte, out_dir / "test_predictions.csv")
    save_predictions(pva, dva, out_dir / "val_predictions.csv")

    print(f"\n[{tag}] best epoch {best_epoch}  ({train_min:.1f} min)")
    print(f"  TEST  acc {test_m['accuracy']:.3f}  bal-acc {test_m['balanced_accuracy']:.3f}  "
          f"macro-F1 {test_m['macro_f1']:.3f}  QWK {test_m['qwk']:.3f}")
    hf = test_m["hf_screen"]
    print(f"  HF screen (reduced EF vs normal): AUC {hf['auc']:.3f}  sens {hf['sensitivity']:.3f}  "
          f"spec {hf['specificity']:.3f}  (threshold {hf_thr:.3f}, tuned on val)")
    print(f"  EF<=40% detection: AUC {test_m['ef_le_40']['auc']:.3f}")


if __name__ == "__main__":
    main()
