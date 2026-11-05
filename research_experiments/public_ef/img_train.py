"""Image-only CNN: pre-train on EchoNext images drawn as hospital printouts, evaluate on the 837 hospital development
ECG images (IMG_PLAN.md). GPU laptop. Per-ECG outputs and weights stay in the git-ignored data/private/.
    python -m research_experiments.public_ef.img_train --pretrain      (EN on EchoNext images; hold-out metrics)
    python -m research_experiments.public_ef.img_train --evaluate      (ZS, FT-EN, FT-IN on the Phase-2 folds)
Re-running skips finished folds (cached in data/private/img_runs/).
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from research_experiments.phase_5.p5common import corr_ci, corr_t, holm
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
from research_experiments.public_ef.en_prep import PRIV, RES

warnings.filterwarnings("ignore")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS = PRIV / "img_runs"
MEAN, STD = 0.449, 0.226


class Imgs(Dataset):
    def __init__(self, path, idx, y, ef, train):
        self.path, self.idx, self.y, self.ef, self.train, self.arr = str(path), np.asarray(idx), y, ef, train, None
        self.aug = v2.Compose([v2.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                               v2.ColorJitter(brightness=0.2, contrast=0.2)])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(self.path, mmap_mode="r")                    # opened lazily in each worker
        j = self.idx[i]
        x = torch.from_numpy(np.array(self.arr[j], np.float32) / 255.0)[None]
        if self.train:
            x = self.aug(x)
        return (x - MEAN) / STD, int(self.y[j]), float((self.ef[j] - 50.0) / 15.0)


class Net(nn.Module):
    def __init__(self, imagenet=True):
        super().__init__()
        self.body = torchvision.models.resnet18(weights="IMAGENET1K_V1" if imagenet else None)
        self.body.fc = nn.Identity()
        self.cls, self.reg = nn.Linear(512, 4), nn.Linear(512, 1)

    def forward(self, x):
        h = self.body(x.repeat(1, 3, 1, 1))
        return self.cls(h), self.reg(h)[:, 0]


def macro_f1(y, p):
    f = []
    for c in range(4):
        tp = np.sum((p == c) & (y == c)); f.append(0.0 if tp == 0 else 2 * tp / ((p == c).sum() + (y == c).sum()))
    return float(np.mean(f))


@torch.no_grad()
def predict(model, path, idx, y, ef, bs=64):
    model.eval(); P, R = [], []
    for x, _, _ in DataLoader(Imgs(path, idx, y, ef, False), batch_size=bs, num_workers=0):
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            lo, r = model(x.to(DEV))
        P.append(torch.softmax(lo.float(), 1).cpu().numpy()); R.append(r.float().cpu().numpy())
    return np.concatenate(P), np.concatenate(R) * 15.0 + 50.0


def train(model, path, tr, va, y, ef, epochs, lr, patience, workers, bs=32):
    w = np.bincount(y[tr], minlength=4).astype(float); w = torch.tensor(w.sum() / (4 * np.maximum(w, 1)), dtype=torch.float32, device=DEV)
    ce, mse = nn.CrossEntropyLoss(weight=w), nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    dl = DataLoader(Imgs(path, tr, y, ef, True), batch_size=bs, shuffle=True, num_workers=workers, drop_last=len(tr) > bs,
                    persistent_workers=workers > 0)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, yy, ee in dl:
            x, yy, ee = x.to(DEV), yy.to(DEV), ee.float().to(DEV)
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                lo, r = model(x); loss = ce(lo.float(), yy) + 0.5 * mse(r.float(), ee)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        P, _ = predict(model, path, va, y, ef); f = macro_f1(y[va], P.argmax(1))
        if f > best + 1e-4:
            best, bad = f, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


# ------------------------------------------------------------------ data
def hospital_ef():
    from research_experiments.phase_6b_threshold_sampling_lstm.enh import EF
    return EF.astype(float)


def pretrain(a):
    m = np.load(PRIV / "img_echonext_meta.npz")
    y, ef, g = m["y"], m["ef"], m["patient_key"]
    path = PRIV / "img_echonext.npy"
    from research_experiments.public_ef.en_split import locked          # IMG_PLAN Amendment 2: locked test patients
    L = locked(g)
    idx = np.flatnonzero(~L) if not a.limit else np.flatnonzero(~L)[: a.limit]
    tr, va = next(GroupShuffleSplit(1, test_size=0.1, random_state=0).split(idx, y[idx], g[idx]))
    tr, va = idx[tr], idx[va]
    torch.manual_seed(42); np.random.seed(42)
    model = Net(imagenet=not a.no_imagenet).to(DEV)
    t0 = time.time()
    model, f1, ep = train(model, path, tr, va, y, ef, a.epochs or 15, 3e-4, 3, a.workers)
    P, R = predict(model, path, va, y, ef)
    torch.save(model.state_dict(), PRIV / "img_en_weights.pt")
    rec = [float(np.mean(P.argmax(1)[y[va] == c] == c)) for c in range(4)]
    out = dict(n_train=int(len(tr)), n_holdout=int(len(va)), epochs_run=ep, holdout_macro_f1=f1,
               holdout_accuracy=float(np.mean(P.argmax(1) == y[va])), holdout_recall=dict(zip(B.CLASSES, rec)),
               holdout_ef_r=float(pearsonr(R, ef[va])[0]), minutes=round((time.time() - t0) / 60, 1), device=str(DEV))
    lk = np.flatnonzero(L)
    if len(lk) and not a.limit:                                         # scored once, never used to choose anything
        Pl, Rl = predict(model, path, lk, y, ef)
        out["echonext_locked_test"] = dict(n=int(len(lk)), patients=int(len(set(g[lk]))), **metrics(y[lk], Pl.argmax(1)),
                                           ef_r=float(pearsonr(Rl, ef[lk])[0]))
    RES.mkdir(parents=True, exist_ok=True); (RES / "img_pretrain.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def metrics(y, p):
    rec = [float(np.mean(p[y == c] == c)) for c in range(4)]
    pre = [float(np.mean(y[p == c] == c)) if (p == c).any() else 0.0 for c in range(4)]
    return dict(accuracy=float(np.mean(p == y)), macro_f1=macro_f1(y, p), min_recall=min(rec),
                **{f"recall_{c}": rec[i] for i, c in enumerate(B.CLASSES)}, **{f"precision_{c}": pre[i] for i, c in enumerate(B.CLASSES)})


def evaluate(a):
    ids = np.load(PRIV / "img_hospital_ids.npy"); pos = {e: i for i, e in enumerate(ids)}
    path = PRIV / "img_hospital_ordered.npy"
    np.save(path, np.load(PRIV / "img_hospital.npy")[np.array([pos[e] for e in B.IDS])])
    Y, EFh = B.Y, hospital_ef()
    RUNS.mkdir(parents=True, exist_ok=True)
    en_state = torch.load(PRIV / "img_en_weights.pt", map_location=DEV, weights_only=True)
    zs = Net(imagenet=False).to(DEV); zs.load_state_dict(en_state)
    Pzs, _ = predict(zs, path, np.arange(B.nD), Y, EFh)
    preds, cells = {}, {}
    folds = [(r, k) for r in (0, 1, 2) for k in range(5)][: (a.folds or 15)]
    for r in sorted({f[0] for f in folds}):
        pr = {n: np.full(B.nD, -1) for n in ("ZS", "FT-EN", "FT-IN")}
        for k in [f[1] for f in folds if f[0] == r]:
            itr, iva, ote = B.split_indices(r, k); TR = np.r_[itr, iva]
            pr["ZS"][ote] = (Pzs[ote] * U.tune_floor(Pzs[TR], Y[TR], 0.18)).argmax(1)
            for name, init_en, lr in (("FT-EN", True, 1e-4), ("FT-IN", False, 3e-4)):
                f = RUNS / f"{name}_r{r}_k{k}.npz"
                if not f.exists():
                    torch.manual_seed(42); np.random.seed(42)
                    m = Net(imagenet=not init_en and not a.no_imagenet).to(DEV)
                    if init_en:
                        m.load_state_dict(en_state)
                    tr_idx = itr if not a.max_train else itr[: a.max_train]
                    m, f1, ep = train(m, path, tr_idx, iva, Y, EFh, a.epochs or 25, lr, 4, a.workers)
                    Pi, _ = predict(m, path, iva, Y, EFh); Po, _ = predict(m, path, ote, Y, EFh)
                    np.savez(f, Pi=Pi, Po=Po, ote=ote, epochs=ep)
                z = np.load(f)
                pr[name][ote] = (z["Po"] * U.tune_floor(z["Pi"], Y[iva], 0.18)).argmax(1)
            print(f"r{r} k{k} done", flush=True)
        pr["C3 (with header)"] = B.load_run(f"P6A-c3-B1-v1__s0_r{r}")[1]
        fo = B.fold_vec(r); done = pr["FT-EN"] >= 0
        for n, p in pr.items():
            preds.setdefault(n, []).append((p, done))
            cells.setdefault(n, []).extend(float(np.mean(p[(fo == k) & done] == Y[(fo == k) & done])) for k in range(5) if ((fo == k) & done).any())
    out = dict(models={}, comparisons=[], n_folds=len(folds))
    for n, lst in preds.items():
        ms = [metrics(Y[d], p[d]) for p, d in lst]
        s = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        s.update(accuracy_ci95_corrected=corr_ci(cells[n]) if len(cells[n]) > 1 else None, admissible=bool(s["min_recall"] >= 0.15))
        out["models"][n] = s
    if len(cells["FT-EN"]) > 1:
        pv = []
        for x, yv in (("FT-EN", "FT-IN"), ("FT-EN", "C3 (with header)")):
            d = np.array(cells[x]) - np.array(cells[yv]); md, p = corr_t(d); pv.append(p)
            out["comparisons"].append(dict(model=x, vs=yv, delta_accuracy=md, p=p))
        for c, h in zip(out["comparisons"], holm(pv)):
            c["p_holm"] = float(h)
    (RES / "img_results.json").write_text(json.dumps(out, indent=1, default=float))
    for n, s in out["models"].items():
        print(f"{n:18s} acc {s['accuracy']:.3f} {s['accuracy_ci95_corrected']} rec {[round(s[f'recall_{c}'], 3) for c in B.CLASSES]} adm {s['admissible']}")
    print(json.dumps(out["comparisons"], indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", action="store_true"); ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None, help="smoke tests only"); ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-train", type=int, default=None, help="smoke tests only"); ap.add_argument("--no-imagenet", action="store_true", help="smoke tests only")
    a = ap.parse_args()
    if a.pretrain:
        pretrain(a)
    if a.evaluate:
        evaluate(a)
