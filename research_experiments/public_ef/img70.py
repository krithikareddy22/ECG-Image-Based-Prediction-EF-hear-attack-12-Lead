"""IMG70 (IMG_PLAN Amendment 3): experiments towards 70 % four-class accuracy on the 837 hospital development ECG IMAGES,
pre-training on the EchoNext test file only (owner) with its locked 20 % kept out. GPU laptop.
Per-ECG outputs, images and weights stay in the git-ignored data/private/img70/; results are aggregates.

    python -m research_experiments.public_ef.img70 --prep 384x768            image caches at 384 x 768
    python -m research_experiments.public_ef.img70 --pretrain E2-convnext    (E2-effnet, E3, ...)
    python -m research_experiments.public_ef.img70 --evaluate E2-convnext [--seed 43]
    python -m research_experiments.public_ef.img70 --report
Re-running skips finished folds.
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from scipy.stats import norm, pearsonr
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from research_experiments.phase_5.p5common import corr_ci, corr_t, holm
from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
from research_experiments.public_ef import img_prep as IP
from research_experiments.public_ef import img_render as IR
from research_experiments.public_ef import img_train as IT
from research_experiments.public_ef.en_prep import CLS, PRIV, RES, norm_params, source_dir, to_mv
from research_experiments.public_ef.en_split import locked

warnings.filterwarnings("ignore")
DEV = IT.DEV
D70 = PRIV / "img70"
Y, nD = B.Y, B.nD
NCUT = Bd.normal_cut(52.0, 2.0)                               # per-ECG Normal limit (52 men / 54 women)

# ------------------------------------------------------------------------------------------------ configurations
# Declared in IMG_PLAN Amendment 3 before running. Loss = w_ce * CE(class-balanced) + w_ef * MSE((EF-50)/15).
CFG = {
    "E2-convnext": dict(backbone="convnext_tiny", hw=(384, 768), w_ce=1.0, w_ef=0.5, micro=8),
    "E2-effnet": dict(backbone="efficientnet_b2", hw=(384, 768), w_ce=1.0, w_ef=0.5, micro=8),
    "E3": dict(backbone="E2", hw=(384, 768), w_ce=0.5, w_ef=1.0, micro=8, ef_grading=True),
    "E4": dict(base="E3-or-E2", recipe="llrd-mixup", tta=True, seeds=(42, 43, 44)),
}
PRE = dict(lr=3e-4, wd=1e-4, epochs=15, patience=3, holdout=0.1, batch=32)       # as IMG_PLAN EN pre-training
FT = dict(lr=1e-4, epochs=25, patience=4, batch=32)                              # as IMG_PLAN FT-EN
LLRD = dict(head_lr=1e-3, backbone_lr=1e-4)                                      # E4 recipe
MIXUP = 0.2
SHIFTS = (0.01, -0.01)                                                           # E4 TTA: +-1 % shifts in x and y
DELTAS = (-4.0, -2.0, 0.0, 2.0, 4.0)                                             # E3 EF cut-point shift (inner-chosen)


# --------------------------------------------------------------------------------------------------------- images
def _init(template_pdf, hw):
    IR.set_template(template_pdf)
    IP.HW = hw


def _hosp(path):
    return IR.hospital_image(path, IP.HW)


def _drawn(x250):
    return IP._drawn(x250)


def img_paths(hw):
    tag = f"{hw[0]}x{hw[1]}"
    return D70 / f"hosp_{tag}.npy", D70 / f"en_{tag}.npy"


def prep(hw, jobs):
    """Hospital (837 development ECGs, B.IDS order) and EchoNext test-split images (img_echonext_meta order) at hw."""
    D70.mkdir(parents=True, exist_ok=True)
    hp, ep = img_paths(hw)
    man = IP.dev_manifest().set_index("ecg_id").loc[list(B.IDS)].reset_index()   # development ECGs only, B.IDS order
    tpl = IP.dev_manifest().source.iloc[0]
    t0 = time.time()
    with Pool(jobs, initializer=_init, initargs=(tpl, hw)) as p:
        if not hp.exists():
            np.save(hp, np.stack(p.map(_hosp, man.source.tolist(), chunksize=8)).astype(np.uint8))
        if not ep.exists():
            m = np.load(PRIV / "img_echonext_meta.npz")
            src, params = source_dir(), norm_params()
            meta = pd.read_csv(src / "echonext_metadata_100k.csv")
            M = meta[meta.split == "test"].reset_index(drop=True)
            keep = np.flatnonzero((M.lvef_value.notna() & M.sex.isin(["male", "female"])).values)
            assert len(keep) == len(m["y"]), "EchoNext rows differ from img_echonext_meta"
            W = np.load(src / "EchoNext_test_waveforms.npy", mmap_mode="r")
            normalised = float(np.asarray(W[:200], float).std()) < 5
            out = np.lib.format.open_memmap(ep, mode="w+", dtype=np.uint8, shape=(len(keep), *hw))
            for s in range(0, len(keep), 512):
                imgs = p.map(_drawn, [to_mv(W[i, 0], params, normalised) for i in keep[s:s + 512]], chunksize=16)
                out[s:s + len(imgs)] = np.stack(imgs)
            out.flush()
    return dict(hw=list(hw), seconds=round(time.time() - t0))


# ---------------------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, backbone="resnet18", imagenet=True):
        super().__init__()
        w = "IMAGENET1K_V1" if imagenet else None
        if backbone == "resnet18":
            self.body = torchvision.models.resnet18(weights=w); d = self.body.fc.in_features; self.body.fc = nn.Identity()
        elif backbone == "convnext_tiny":
            self.body = torchvision.models.convnext_tiny(weights=w); d = self.body.classifier[2].in_features
            self.body.classifier[2] = nn.Identity()
        elif backbone == "efficientnet_b2":
            self.body = torchvision.models.efficientnet_b2(weights=w); d = self.body.classifier[1].in_features
            self.body.classifier = nn.Identity()
        else:
            raise KeyError(backbone)
        self.cls, self.reg = nn.Linear(d, 4), nn.Linear(d, 1)

    def forward(self, x):
        h = self.body(x.repeat(1, 3, 1, 1))
        return self.cls(h), self.reg(h)[:, 0]


def backbone_of(name):
    c = CFG[name]
    if c.get("backbone") == "E2":
        return chosen_e2()
    return c["backbone"]


def chosen_e2():
    """E2 backbone = higher EchoNext hold-out macro-F1 of the two E2 pre-trainings (declared)."""
    r = {n: json.loads((RES / f"img70_pretrain_{n}.json").read_text())["holdout_macro_f1"] for n in ("E2-convnext", "E2-effnet")}
    return CFG[max(r, key=r.get)]["backbone"]


# -------------------------------------------------------------------------------------------------------- training
def _loader(path, idx, y, ef, train, bs, workers):
    return DataLoader(IT.Imgs(path, idx, y, ef, train), batch_size=bs, shuffle=train, num_workers=workers,
                      drop_last=train and len(idx) > bs, persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, path, idx, y, ef, tta=False, bs=32):
    model.eval()
    views = [(0.0, 0.0)] + ([(s, 0.0) for s in SHIFTS] + [(0.0, s) for s in SHIFTS] if tta else [])
    P, R = 0, 0
    for dx, dy in views:
        Pv, Rv = [], []
        for x, _, _ in _loader(path, idx, y, ef, False, bs, 0):
            if dx or dy:
                x = v2.functional.affine(x, angle=0.0, translate=[int(dx * x.shape[-1]), int(dy * x.shape[-2])], scale=1.0, shear=[0.0])
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                lo, r = model(x.to(DEV))
            Pv.append(torch.softmax(lo.float(), 1).cpu().numpy()); Rv.append(r.float().cpu().numpy())
        P = P + np.concatenate(Pv); R = R + np.concatenate(Rv)
    return P / len(views), R / len(views) * 15.0 + 50.0


def train(model, path, tr, va, y, ef, lr, epochs, patience, w_ce, w_ef, micro, workers, recipe="plain", wd=1e-4):
    w = np.bincount(y[tr], minlength=4).astype(float)
    w = torch.tensor(w.sum() / (4 * np.maximum(w, 1)), dtype=torch.float32, device=DEV)
    ce = nn.CrossEntropyLoss(weight=w)
    if recipe == "llrd-mixup":
        heads = list(model.cls.parameters()) + list(model.reg.parameters())
        opt = torch.optim.AdamW([{"params": model.body.parameters(), "lr": LLRD["backbone_lr"]},
                                 {"params": heads, "lr": LLRD["head_lr"]}], weight_decay=wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    acc = max(1, 32 // micro)
    dl = _loader(path, tr, y, ef, True, micro, workers)
    best, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        for i, (x, yy, ee) in enumerate(dl):
            x, yy, ee = x.to(DEV), yy.to(DEV), ee.float().to(DEV)
            if recipe == "llrd-mixup":
                lam = float(np.random.beta(MIXUP, MIXUP)); perm = torch.randperm(len(x), device=DEV)
                x = lam * x + (1 - lam) * x[perm]
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                lo, r = model(x)
                if recipe == "llrd-mixup":
                    l_ce = lam * ce(lo.float(), yy) + (1 - lam) * ce(lo.float(), yy[perm])
                    l_ef = nn.functional.mse_loss(r.float(), lam * ee + (1 - lam) * ee[perm])
                else:
                    l_ce, l_ef = ce(lo.float(), yy), nn.functional.mse_loss(r.float(), ee)
                loss = (w_ce * l_ce + w_ef * l_ef) / acc
            scaler.scale(loss).backward()
            if (i + 1) % acc == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        sched.step()
        P, _ = predict(model, path, va, y, ef)
        f = IT.macro_f1(y[va], P.argmax(1))
        if f > best + 1e-4:
            best, bad = f, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


# ----------------------------------------------------------------------------------------------------- pre-training
def pretrain(name, workers):
    c = CFG[name]
    bb = backbone_of(name)
    _, ep = img_paths(c["hw"])
    m = np.load(PRIV / "img_echonext_meta.npz")
    y, ef, g = m["y"], m["ef"], m["patient_key"]
    L = locked(g); idx = np.flatnonzero(~L)
    tr, va = next(GroupShuffleSplit(1, test_size=PRE["holdout"], random_state=0).split(idx, y[idx], g[idx]))
    tr, va = idx[tr], idx[va]
    torch.manual_seed(42); np.random.seed(42)
    model = Net(bb).to(DEV)
    t0 = time.time()
    model, f1, n_ep = train(model, ep, tr, va, y, ef, PRE["lr"], PRE["epochs"], PRE["patience"], c["w_ce"], c["w_ef"],
                            c["micro"], workers, wd=PRE["wd"])
    torch.save(model.state_dict(), D70 / f"{name}_en.pt")
    P, R = predict(model, ep, va, y, ef)
    lk = np.flatnonzero(L)
    Pl, Rl = predict(model, ep, lk, y, ef)
    out = dict(config=name, backbone=bb, hw=list(c["hw"]), n_train=int(len(tr)), n_holdout=int(len(va)), epochs_run=n_ep,
               holdout_macro_f1=f1, holdout=IT.metrics(y[va], P.argmax(1)), holdout_ef_r=float(pearsonr(R, ef[va])[0]),
               echonext_locked_test=dict(n=int(len(lk)), **IT.metrics(y[lk], Pl.argmax(1)), ef_r=float(pearsonr(Rl, ef[lk])[0])),
               minutes=round((time.time() - t0) / 60, 1), max_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    (RES / f"img70_pretrain_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float), flush=True)


# ------------------------------------------------------------------------------------------------ hospital evaluation
def ef_band(mu, sd, idx, delta):
    cN, cM, cS = (norm.cdf((c + delta - mu) / sd) for c in (NCUT[idx], 41.0, 31.0))
    return np.clip(np.c_[1 - cN, cN - cM, cM - cS, cS], 1e-6, 1)


def decide(Pi, Po, mi, mo, iva, ote, grading):
    """Inner-chosen decision. grading=False: tune_floor on the class head. True (E3): combine the class head with EF
    grading N(mu, sd_inner) over the class bands shifted by delta; delta and the class weights chosen on inner-val."""
    if not grading:
        w = U.tune_floor(Pi, Y[iva], 0.18)
        return (Po * w).argmax(1), float(np.mean((Pi * w).argmax(1) == Y[iva]))
    sd = float(np.std(mi - B_EF()[iva]))
    best = None
    for d in DELTAS:
        Ci = (Pi + ef_band(mi, sd, iva, d)) / 2
        w = U.tune_floor(Ci, Y[iva], 0.18); a = float(np.mean((Ci * w).argmax(1) == Y[iva]))
        if best is None or a > best[0]:
            best = (a, d, w)
    a, d, w = best
    return ((Po + ef_band(mo, sd, ote, d)) / 2 * w).argmax(1), a


_EF = None


def B_EF():
    global _EF
    if _EF is None:
        _EF = IT.hospital_ef()
    return _EF


def evaluate(name, seed, workers):
    """Fine-tune from the named EchoNext pre-training on each outer-training fold (Phase-2 inner split); outputs cached."""
    c = CFG[name]
    base = name if "base" not in c else e4_base()
    cb = CFG[base]
    bb, hw = backbone_of(base), cb["hw"]
    hp, _ = img_paths(hw)
    state = torch.load(D70 / f"{base}_en.pt", map_location=DEV, weights_only=True)
    runs = D70 / "runs"; runs.mkdir(parents=True, exist_ok=True)
    EFh = B_EF()
    for r in (0, 1, 2):
        for k in range(5):
            f = runs / f"{name}_s{seed}_r{r}_k{k}.npz"
            if f.exists():
                continue
            itr, iva, ote = B.split_indices(r, k)
            torch.manual_seed(seed); np.random.seed(seed)
            m = Net(bb, imagenet=False).to(DEV); m.load_state_dict(state)
            recipe = c.get("recipe", "plain")
            m, f1, n_ep = train(m, hp, itr, iva, Y, EFh, FT["lr"], FT["epochs"], FT["patience"], cb["w_ce"], cb["w_ef"],
                                cb["micro"], workers, recipe=recipe)
            res = {}
            for tta in ((False, True) if c.get("tta") else (False,)):
                Pi, Mi = predict(m, hp, iva, Y, EFh, tta); Po, Mo = predict(m, hp, ote, Y, EFh, tta)
                res.update({f"Pi{int(tta)}": Pi, f"Po{int(tta)}": Po, f"Mi{int(tta)}": Mi, f"Mo{int(tta)}": Mo})
            np.savez(f, ote=ote, iva=iva, epochs=n_ep, **res)
            print(f"{name} seed {seed} r{r} k{k}: epochs {n_ep}, inner-val macro-F1 {f1:.3f}", flush=True)


def e4_base():
    """E4 starts from E3 if E3's mean inner-val accuracy beats E2's, else from the chosen E2 (declared)."""
    e2 = [n for n in ("E2-convnext", "E2-effnet") if CFG[n]["backbone"] == chosen_e2()][0]
    return "E3" if inner_score("E3", 42) > inner_score(e2, 42) else e2


def fold_preds(name, seed):
    """Outer predictions (all 837, per repeat) and inner-val accuracies, decision chosen inside each fold."""
    c = CFG[name]
    base = name if "base" not in c else e4_base()
    grading = CFG[base].get("ef_grading", False)
    seeds = c.get("seeds", (seed,)) if name == "E4" else (seed,)
    runs = D70 / "runs"
    preds, inner = [], []
    for r in (0, 1, 2):
        p = np.full(nD, -1)
        for k in range(5):
            zs = [np.load(runs / f"{name}_s{s}_r{r}_k{k}.npz") for s in seeds]
            ote, iva = zs[0]["ote"], zs[0]["iva"]
            cand = []
            for t in ((0, 1) if c.get("tta") else (0,)):
                Pi = np.mean([z[f"Pi{t}"] for z in zs], 0); Po = np.mean([z[f"Po{t}"] for z in zs], 0)
                Mi = np.mean([z[f"Mi{t}"] for z in zs], 0); Mo = np.mean([z[f"Mo{t}"] for z in zs], 0)
                po, a = decide(Pi, Po, Mi, Mo, iva, ote, grading)
                cand.append((a, po))
            a, po = max(cand, key=lambda x: x[0])                    # TTA on / off chosen on inner-val (E4)
            p[ote] = po; inner.append(a)
        preds.append(p)
    return preds, inner


def inner_score(name, seed):
    return float(np.mean(fold_preds(name, seed)[1]))


def summary(preds):
    fo = [B.fold_vec(r) for r in range(3)]
    ms = [IT.metrics(Y, p) for p in preds]
    s = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    cells = [float(np.mean(p[fo[r] == k] == Y[fo[r] == k])) for r, p in enumerate(preds) for k in range(5)]
    s.update(accuracy_ci95_corrected=corr_ci(cells), accuracy_per_repeat=[round(m["accuracy"], 4) for m in ms],
             admissible=bool(s["min_recall"] >= 0.15))
    return s, cells


def baseline_preds(name):
    """IMG baseline (img_train): FT-EN / FT-IN outer predictions and inner-val accuracies; ZS outer predictions."""
    preds, inner = [], []
    for r in (0, 1, 2):
        p = np.full(nD, -1)
        for k in range(5):
            z = np.load(IT.RUNS / f"{name}_r{r}_k{k}.npz")
            itr, iva, ote = B.split_indices(r, k)
            po, a = decide(z["Pi"], z["Po"], None, None, iva, ote, False)
            p[ote] = po; inner.append(a)
        preds.append(p)
    return preds, inner


def triage(name, seed, target=0.70, min_n=20):
    """E6: answer only when the max probability reaches a threshold chosen on inner-val (lowest threshold whose inner-val
    answered accuracy >= target with >= min_n answered); applied unchanged to the outer fold."""
    c = CFG[name]; base = name if "base" not in c else e4_base()
    seeds = c.get("seeds", (seed,)) if name == "E4" else (seed,)
    grading = CFG[base].get("ef_grading", False)
    reps, curve = [], []
    for r in (0, 1, 2):
        ans = np.zeros(nD, bool); pred = np.full(nD, -1); conf = np.zeros(nD)
        for k in range(5):
            zs = [np.load(D70 / "runs" / f"{name}_s{s}_r{r}_k{k}.npz") for s in seeds]
            ote, iva = zs[0]["ote"], zs[0]["iva"]
            Pi = np.mean([z["Pi0"] for z in zs], 0); Po = np.mean([z["Po0"] for z in zs], 0)
            Mi = np.mean([z["Mi0"] for z in zs], 0); Mo = np.mean([z["Mo0"] for z in zs], 0)
            po, _ = decide(Pi, Po, Mi, Mo, iva, ote, grading)
            pi, _ = decide(Pi, Pi, Mi, Mi, iva, iva, grading)
            ci, co = Pi.max(1), Po.max(1)
            ok = pi == Y[iva]
            ts = [t for t in np.unique(np.round(ci, 3)) if (ci >= t).sum() >= min_n and ok[ci >= t].mean() >= target]
            thr = ts[0] if ts else 1.01
            ans[ote] = co >= thr; pred[ote] = po; conf[ote] = co
        reps.append(dict(share_answered=float(ans.mean()), accuracy_answered=float(np.mean(pred[ans] == Y[ans])) if ans.any() else None,
                         answered_by_class={cl: round(float(ans[Y == i].mean()), 3) for i, cl in enumerate(B.CLASSES)}))
        o = np.argsort(-conf); cum = np.cumsum(pred[o] == Y[o]) / np.arange(1, nD + 1)
        ok70 = np.flatnonzero(cum >= target)
        curve.append(dict(largest_share_with_answered_accuracy_ge_target=float((ok70.max() + 1) / nD) if len(ok70) else 0.0,
                          accuracy_at_share={f"{s:.1f}": float(cum[int(s * nD) - 1]) for s in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)}))
    return dict(inner_chosen_threshold=dict(share_answered=float(np.mean([x["share_answered"] for x in reps])),
                                            accuracy_answered=float(np.mean([x["accuracy_answered"] for x in reps if x["accuracy_answered"] is not None])),
                                            per_repeat=reps),
                descriptive_after_the_fact=curve)


def report():
    from research_experiments.public_ef import en_image as EI
    out = dict(variants={}, inner_selection={}, comparisons=[], note="EchoNext test file only; E1 (train split) deferred by the owner")
    cand = {}
    for n in ("FT-EN", "FT-IN"):
        if (IT.RUNS / f"{n}_r2_k4.npz").exists():
            p, i = baseline_preds(n); s, cells = summary(p); out["variants"][f"IMG {n} (ResNet-18 320x640)"] = s
            cand[f"IMG {n} (ResNet-18 320x640)"] = (p, i, cells, "baseline")
    for n in ("E2-convnext", "E2-effnet", "E3", "E4"):
        if (D70 / "runs" / f"{n}_s42_r2_k4.npz").exists():
            p, i = fold_preds(n, 42); s, cells = summary(p); out["variants"][n] = s; cand[n] = (p, i, cells, n)
    for f in sorted(RES.glob("img70_pretrain_*.json")):
        out.setdefault("pretraining", {})[f.stem.replace("img70_pretrain_", "")] = json.loads(f.read_text())
    sel = {n: float(np.mean(v[1])) for n, v in cand.items() if not n.startswith("IMG FT-IN")}
    out["inner_selection"] = dict(criterion="mean inner-val accuracy after the decision rule, over the 15 folds", scores=sel)
    if sel:
        final = max(sel, key=sel.get); out["final_model"] = final
        key = cand[final][3]
        seeds = [s for s in (42, 43, 44) if key not in ("baseline", "E4") and (D70 / "runs" / f"{key}_s{s}_r2_k4.npz").exists()]
        if key == "E4":
            single = [summary(fold_preds_single("E4", s))[1] for s in (42, 43, 44)]
            fin_cells = list(np.mean(single, 0)); out["final_seeds"] = "E4 ensemble of seeds 42/43/44; mean of the single seeds:"
        elif len(seeds) == 3:
            fin_cells = list(np.mean([summary(fold_preds(key, s)[0])[1] for s in seeds], 0)); out["final_seeds"] = seeds
        else:
            fin_cells = cand[final][2]; out["final_seeds"] = "seed 42 only (other seeds not run yet)"
        out["final_mean_over_seeds"] = dict(accuracy=float(np.mean(fin_cells)), accuracy_ci95_corrected=corr_ci(fin_cells))
        refs = {"C3 (with header; reference)": [], "IMG FT-EN": cand.get("IMG FT-EN (ResNet-18 320x640)", (None, None, None))[2]}
        for r in (0, 1, 2):
            p = B.load_run(f"P6A-c3-B1-v1__s0_r{r}")[1]; fo = B.fold_vec(r)
            refs["C3 (with header; reference)"].extend(float(np.mean(p[fo == k] == Y[fo == k])) for k in range(5))
        pv = []
        for rn, rc in refs.items():
            if rc is None:
                continue
            d = np.array(fin_cells) - np.array(rc); md, p_ = corr_t(d); pv.append(p_)
            out["comparisons"].append(dict(model=final, vs=rn, delta_accuracy=md, p=p_))
        for c_, h in zip(out["comparisons"], holm(pv)):
            c_["p_holm"] = float(h)
        if key != "baseline":
            out["E6_triage"] = triage(key, 42)
    out["reached_70"] = bool(any(v["accuracy"] >= 0.70 and v["admissible"] for v in out["variants"].values()))
    (RES / "img70_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in out if k != "variants"}, indent=1, default=float)[:4000])
    for n, s in out["variants"].items():
        print(f"{n:34s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} rec "
              f"{[round(s[f'recall_{c}'], 2) for c in B.CLASSES]} F1 {s['macro_f1']:.3f} adm {s['admissible']}")


def fold_preds_single(name, seed):
    """E4 with a single seed (for the per-seed mean)."""
    saved = CFG[name].get("seeds"); CFG[name]["seeds"] = (seed,)
    try:
        return fold_preds(name, seed)[0]
    finally:
        CFG[name]["seeds"] = saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep"); ap.add_argument("--pretrain"); ap.add_argument("--evaluate"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); D70.mkdir(parents=True, exist_ok=True)
    if a.prep:
        print(prep(tuple(int(v) for v in a.prep.split("x")), a.jobs), flush=True)
    if a.pretrain:
        pretrain(a.pretrain, a.workers)
    if a.evaluate:
        for s in (CFG[a.evaluate].get("seeds") or (a.seed,)):
            evaluate(a.evaluate, s, a.workers)
    if a.report:
        report()
