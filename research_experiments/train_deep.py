"""Deep-model training for Experiments A2 and B, using the SAME recipe as the baseline
(ecg_ef/train.py): AdamW, 2-epoch warm-up + cosine, EMA 0.99, inverse-frequency class weights,
grad-clip 5, model selection on VALIDATION by mean(macro-F1, QWK, HF AUC) after --min-epochs,
early stopping. Only the architecture / output head / loss differ, as stated by the flags.

  --arch resnet18  : ResNet-18 on the filtered ECG images (256x512), random init, in-memory
                     augmentation of ecg_ef.data.augment (identical to the baseline)
  --arch ecgfounder: pre-trained ECGFounder on the 'aligned' 12 x 5000 waveforms (built in
                     memory), stages not in --unfreeze kept frozen (A2 partial fine-tuning);
                     in-memory augmentation: per-lead gain +-10 %, Gaussian noise 0.02 (z units)
                     on recorded samples only
  --head ce        : softmax + class-weighted cross-entropy with label smoothing 0.05 (baseline)
  --head cumlink   : cumulative-link ordinal head (research_experiments/ordinal_layer.py),
                     class-weighted NLL (label smoothing not defined for it -> not used)

    python -m research_experiments.train_deep --arch ecgfounder --head ce --tag A2_ecgf_ft56_ce
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from research_experiments.common import (CONFIGS, MOD, PLOTS, RES, ROOT, _guard, env_record, load_meta,  # noqa: E402
                                         save_run, seed_all)
from research_experiments.ordinal_layer import CumulativeLink  # noqa: E402
from ecg_ef.metrics import evaluate  # noqa: E402
from ecg_ef.train import EMA, selection_score  # noqa: E402


# ----------------------------------------------------------------------------- models
class ImageNet(nn.Module):
    def __init__(self, head):
        super().__init__()
        import timm
        self.head = head
        self.net = timm.create_model("resnet18", pretrained=False, in_chans=1,
                                     num_classes=4 if head == "ce" else 1, drop_rate=0.3)
        self.link = CumulativeLink() if head == "cumlink" else None

    def forward(self, x):
        return self.net(x)


class FounderFT(nn.Module):
    def __init__(self, head, unfreeze):
        super().__init__()
        from research_experiments.ecgfounder.model import load_ecgfounder
        self.head = head
        self.base = load_ecgfounder()                       # pre-trained weights, 150-label head
        self.base.dense = nn.Identity()
        self.unfreeze = set(unfreeze)
        for name, p in self.base.named_parameters():
            stage = int(name.split(".")[1]) if name.startswith("stage_list.") else -1
            p.requires_grad_(stage in self.unfreeze)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 4 if head == "ce" else 1)
        self.link = CumulativeLink() if head == "cumlink" else None

    def forward(self, x):
        b = self.base
        first = min(self.unfreeze)
        with torch.no_grad():                               # frozen part: no graph, faster
            out = b.first_activation(b.first_conv(x))
            for i in range(first):
                out = b.stage_list[i](out)
        for i in range(first, b.n_stages):
            out = b.stage_list[i](out)
        return self.fc(self.drop(out.mean(-1)))


def probs_from_out(model, out):
    if model.head == "ce":
        return F.softmax(out, 1)
    return model.link.probs(out[:, 0])


def loss_fn(model, out, y, cw, smoothing):
    if model.head == "ce":
        ce = F.cross_entropy(out, y, weight=cw, label_smoothing=smoothing, reduction="none")
        return ce.sum() / cw[y].sum()
    return model.link.nll(out[:, 0], y, cw[y])


# ----------------------------------------------------------------------------- data
def load_data(arch, meta):
    if arch == "resnet18":
        from ecg_ef.data import Normalizer, load_split
        d = {}
        for s in ("train", "val", "test"):
            x, y, df = load_split(s, "filtered", (256, 512))
            assert (df.ecg_id.values == meta[meta.split == s].ecg_id.values).all()
            d[s] = (x, y)
        norm = Normalizer.fit(d["train"][0])
        prep = lambda xb, train: _img_prep(xb, norm, train)
        return d, prep, dict(normalizer=norm.state())
    from research_experiments.ecgfounder.inputs import build_all
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)       # read-only
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    d = {}
    for s in ("train", "val", "test"):
        ms = meta[meta.split == s]
        ii = np.array([idx[e] for e in ms.ecg_id])
        d[s] = (torch.from_numpy(build_all(z["leads"][ii], z["rhythm"][ii], "aligned")),
                torch.tensor(ms.label_idx.values))
    return d, _wave_prep, {}


def _img_prep(xb, norm, train):
    from ecg_ef.data import augment
    x = xb.float() / 255
    if train:
        x = augment(x)
    return norm(x).contiguous(memory_format=torch.channels_last)


def _wave_prep(xb, train):
    if not train:
        return xb
    rec = xb != 0
    g = 1 + (torch.rand(xb.shape[0], xb.shape[1], 1) * 2 - 1) * 0.10
    return (xb * g + 0.02 * torch.randn_like(xb)) * rec


@torch.no_grad()
def predict(model, x, prep, bs=32):
    model.eval()
    return torch.cat([probs_from_out(model, model(prep(x[i:i + bs], False))) for i in range(0, len(x), bs)]).numpy()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "ecgfounder"], required=True)
    ap.add_argument("--head", choices=["ce", "cumlink"], required=True)
    ap.add_argument("--unfreeze", default="5,6", help="ECGFounder stages to fine-tune (0-6)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--min-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    img = a.arch == "resnet18"
    a.epochs = a.epochs or (30 if img else 15)
    a.patience = a.patience or (10 if img else 5)
    a.min_epochs = a.min_epochs or (8 if img else 3)
    a.lr = a.lr or (1e-3 if img else 1e-4)
    torch.set_num_threads(a.threads); seed_all(a.seed)
    _guard(CONFIGS).mkdir(parents=True, exist_ok=True)
    (CONFIGS / f"{a.tag}.json").write_text(json.dumps(dict(vars(a), env=env_record()), indent=2))

    meta = load_meta()
    data, prep, extra_ck = load_data(a.arch, meta)
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    counts = torch.bincount(ytr, minlength=4).float()
    cw = counts.sum() / (4 * counts); cw = cw / cw.mean()

    if img:
        model = ImageNet(a.head).to(memory_format=torch.channels_last)
        groups = [{"params": model.parameters(), "lr": a.lr}]
    else:
        model = FounderFT(a.head, [int(s) for s in a.unfreeze.split(",")])
        bb = [p for p in model.base.parameters() if p.requires_grad]
        hd = list(model.fc.parameters()) + (list(model.link.parameters()) if model.link else [])
        groups = [{"params": bb, "lr": a.lr}, {"params": hd, "lr": a.head_lr}]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.tag}] params total {n_total:,} trainable {n_train:,}", flush=True)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    eval_model = ema.model if ema else model
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    spe = len(xtr) // a.batch_size
    total, warm = a.epochs * spe, (2 if img else 1) * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    best, best_state, best_epoch, hist, bad = -1.0, None, 0, [], 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        if not img:
            model.base.eval()        # no BN/dropout inside Net1D (use_bn/use_do False); explicit anyway
        perm = torch.randperm(len(xtr)); tl, tn = 0.0, 0
        for i in range(spe):
            idx = perm[i * a.batch_size:(i + 1) * a.batch_size]
            out = model(prep(xtr[idx], True))
            loss = loss_fn(model, out, ytr[idx], cw, a.label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step(); sched.step()
            if ema:
                ema.update(model)
            tl += loss.item() * len(idx); tn += len(idx)
        pva = predict(eval_model, xva, prep)
        vm = evaluate(yva.numpy(), pva); sc = selection_score(vm)
        hist.append(dict(epoch=ep, train_loss=tl / tn,
                         val_loss=float(F.nll_loss(torch.log(torch.from_numpy(pva).clamp_min(1e-8)), yva)),
                         val_acc=vm["accuracy"], val_bal_acc=vm["balanced_accuracy"], val_macro_f1=vm["macro_f1"],
                         val_qwk=vm["qwk"], val_hf_auc=vm["hf_screen"]["auc"], val_score=sc,
                         elapsed_min=(time.time() - t0) / 60))
        flag = ""
        if ep >= a.min_epochs:
            if sc > best:
                best, best_epoch, bad, flag = sc, ep, 0, " *"
                best_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
            else:
                bad += 1
        h = hist[-1]
        print(f"ep {ep:2d} loss {h['train_loss']:.3f}/{h['val_loss']:.3f} vacc {h['val_acc']:.3f} "
              f"vbal {h['val_bal_acc']:.3f} vF1 {h['val_macro_f1']:.3f} vQWK {h['val_qwk']:.3f} "
              f"vHF {h['val_hf_auc']:.3f} score {sc:.3f} {h['elapsed_min']:.1f}m{flag}", flush=True)
        if bad >= a.patience:
            print("early stop"); break
    train_min = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    pva = predict(model, xva, prep)
    t1 = time.time(); pte = predict(model, xte, prep); inf = (time.time() - t1) / len(xte)

    _guard(MOD).mkdir(parents=True, exist_ok=True)
    ck = dict(arch=a.arch, head=a.head, args=vars(a), state_dict=best_state, best_epoch=best_epoch, **extra_ck)
    if not img:   # store only the trainable / changed part; frozen part = public checkpoint
        ck["state_dict"] = {k: v for k, v in best_state.items()
                            if not k.startswith("base.") or any(k.startswith(f"base.stage_list.{s}.") for s in model.unfreeze)}
    torch.save(ck, _guard(MOD / f"{a.tag}.pt"))
    size_mb = (MOD / f"{a.tag}.pt").stat().st_size / 2 ** 20
    if model.head == "cumlink":
        cut = model.link.cutpoints().detach().numpy().tolist()
    else:
        cut = None
    meta_v = meta[meta.split == "val"].reset_index(drop=True); meta_t = meta[meta.split == "test"].reset_index(drop=True)
    save_run(a.tag, meta_v, pva, meta_t, pte, dict(
        input="ECG image (filtered 256x512)" if img else "ECGFounder 'aligned' waveform 12x5000",
        arch=a.arch, head=a.head, unfreeze=a.unfreeze if not img else None, params_total=n_total,
        params_trainable=n_train, best_epoch=best_epoch, epochs_run=len(hist), train_minutes=train_min,
        inference_seconds_per_ecg_model_only=inf, checkpoint_mb=size_mb, cumlink_cutpoints=cut,
        val_selection_score=best))
    hist = pd.DataFrame(hist); _guard(RES / a.tag).mkdir(parents=True, exist_ok=True)
    hist.to_csv(RES / a.tag / "history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train"); ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set_title("loss"); ax[0].legend()
    for c in ("val_bal_acc", "val_macro_f1", "val_qwk", "val_score"):
        ax[1].plot(hist.epoch, hist[c], label=c)
    ax[1].axvline(best_epoch, ls=":", c="k"); ax[1].set_title("validation"); ax[1].legend(fontsize=7)
    fig.suptitle(a.tag, fontsize=9); fig.tight_layout(); fig.savefig(_guard(PLOTS / f"curves_{a.tag}.png"), dpi=110)
    print(f"[{a.tag}] best epoch {best_epoch}, {train_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
