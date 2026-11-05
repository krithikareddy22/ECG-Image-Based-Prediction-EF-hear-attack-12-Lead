"""Phase 6B LSTM family (PHASE6B_PLAN.md section 3), re-using the Phase 6A code unchanged: LSTMNet / VARIANTS / the
training recipe and seeding of phase_6a_normal_mild_lstm.lstm, the waveform inputs of signals.py, and the 6B
configurations of p6bcommon (training labels y_def, training-row exclusions in split).

  FIX  LSTM-fixed (6A configs/lstm_fixed.json) under the 6B-only configurations T55, N300, N270, N240, N200.
       (ORIG / T50 / XUNC / A270X = the 6A runs of the same variant under B1 / B2 / B3 / A270, made by phase_6a lstm.py.)
  L5   CNN-LSTM = 6A VARIANTS["L5"]: 3 x [Conv1d k7 s2 -> GELU] (26 -> 32 -> 64 -> 64) -> BiLSTM-64 -> 6A head      ORIG
  L6   L5 encoder; pooled [last hidden || masked mean] || standardised header numeric (16) -> Dropout 0.3 -> Linear(4)  ORIG
  F1   LSTM-fixed encoder pooled || block-standardised [header (16) || ECGFounder f6] (fitted on inner-train) ->
       Phase-2 MLP-fusion body (Dropout 0.5 -> Linear 64 -> GELU -> Dropout 0.5) -> Linear(4), trained jointly      ORIG
  F2   F1 with the L5 encoder                                                                                        ORIG
  F3   equal-weight vote: C1 members (MLP fusion seed-mean, ECGF linear distance-CE, header+clinical LR) + LSTM-sel    ORIG
  F4   F3 + nested class-prior rule (C3 recipe, protected objective); LSTM inner OOF = LSTM-fixed, seed 42             ORIG
Training (6A section 5, identical code path): class-balanced CE, AdamW lr 1e-3 / wd 1e-2, batch 32, clip 1.0, <= 60 epochs,
early stopping on the fold's inner-val S3 (training labels of the configuration; patience 12, not before epoch 10), best
epoch restored; seeds 42/43/44 x repeats 0-2; the outer fold is predicted once. Every fold records training minutes,
inference ms / ECG and the parameter count. Fold results are cached (git-ignored), so interrupted work resumes per fold.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.lstm6b --models L5 L6 F1 F2 --configs ORIG --device cuda
    python -m research_experiments.phase_6b_threshold_sampling_lstm.lstm6b --models FIX --configs T55 N300 N270 N240 N200
    python -m research_experiments.phase_6b_threshold_sampling_lstm.lstm6b --late          # F3, F4 (needs LSTM-sel)
    python -m research_experiments.phase_6b_threshold_sampling_lstm.lstm6b --estimate      # compute rule (plan section 3)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_6a_normal_mild_lstm import lstm as L
from research_experiments.phase_6a_normal_mild_lstm.existing import MEMBERS
from research_experiments.phase_6a_normal_mild_lstm.existing import inner_oof as inner_member
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import CONF as CONF6A
from research_experiments.phase_6b_threshold_sampling_lstm.p6bcommon import (
    CKPT, FOLDC, INNER, PAT, RES, SEEDS, Y, decide, done, fold_vec, guard, load_run, make_X, mark_running, metrics, nD,
    save_run, split, y_def)
from research_experiments.phase_5.p5common import header_numeric

CMD = "python -m research_experiments.phase_6b_threshold_sampling_lstm.lstm6b"
DEVICE = os.environ.get("P6A_DEVICE", "cpu")
MODELS = ("FIX", "L5", "L6", "F1", "F2")
DESCR = {"FIX": "LSTM-fixed", "L5": "CNN-LSTM", "L6": "CNN-LSTM + header", "F1": "ECGFounder f6 + header + LSTM-fixed (early fusion)",
         "F2": "ECGFounder f6 + header + CNN-LSTM (early fusion)"}


def lstm_fixed():
    return json.loads((CONF6A / "lstm_fixed.json").read_text())["lstm_fixed"]


def spec(model):
    """-> (encoder variant, static block, head)"""
    return {"FIX": (lstm_fixed(), None, None), "L5": ("L5", None, None), "L6": ("L5", "hdr", "head"),
            "F1": (lstm_fixed(), "hdr+ecgf", "mlp"), "F2": ("L5", "hdr+ecgf", "mlp")}[model]


def exp_id(model, cfg):
    return f"P6B-{'lstm' + lstm_fixed() if model == 'FIX' else model}-{cfg}-v1"


class StaticFusion(nn.Module):
    """6A LSTM / CNN-LSTM encoder (head removed) || static block -> 6A head or Phase-2 MLP-fusion body."""

    def __init__(self, v, d_static, head):
        super().__init__()
        s = L.VARIANTS[v]
        self.enc = L.LSTMNet(26 if s["input"] == "aligned" else 1, s["layers"], s["bi"], s["inter_dropout"], s["cnn"])
        self.enc.drop, self.enc.out = nn.Identity(), nn.Identity()          # enc(x) = [last hidden || masked mean]
        d = 2 * L.HID * (2 if s["bi"] else 1) + d_static
        self.head = (nn.Sequential(nn.Dropout(L.HEAD_DO), nn.Linear(d, 4)) if head == "head" else
                     nn.Sequential(nn.Dropout(0.5), nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.5), nn.Linear(64, 4)))

    def forward(self, x, z):
        return self.head(torch.cat([self.enc(x), z], 1))


def static_block(kind, fit_rows):
    """Static inputs with all statistics from the fitting rows only."""
    if kind is None:
        return None
    fm = np.zeros(nD, bool); fm[fit_rows] = True
    if kind == "hdr":
        H = header_numeric(fm); mu, sd = H[fm].mean(0), H[fm].std(0) + 1e-6
        return ((H - mu) / sd).astype(np.float32)
    X, pre = make_X(["hdr", "ecgf"], fm)
    assert pre
    return X.astype(np.float32)


def build(model, d_static):
    v, kind, head = spec(model)
    s = L.VARIANTS[v]
    if kind is None:
        return L.LSTMNet(26 if s["input"] == "aligned" else 1, s["layers"], s["bi"], s["inter_dropout"], s["cnn"])
    return StaticFusion(v, d_static, head)


def n_params(model):
    v, kind, _ = spec(model)
    d = {None: 0, "hdr": 16, "hdr+ecgf": 16 + int(make_X(["ecgf"], np.ones(nD, bool))[0].shape[1])}[kind]
    return sum(p.numel() for p in build(model, d).parameters())


@torch.no_grad()
def predict(m, X, Z, idx, bs=128):
    m.eval(); out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        out.append(F.softmax(m(X[j]) if Z is None else m(X[j], Z[j]), 1))
    return torch.cat(out).cpu().numpy()


def train(model, tr, va, pred_rows, y, seed_key, device):
    """The 6A fit_fold recipe on arbitrary (train, val) rows. -> (P[pred_rows], best val S3, best epoch, epochs run,
    train minutes, inference ms / ECG, best state)."""
    torch.set_num_threads(1)
    v, kind, _ = spec(model)
    X = L.X_of(L.VARIANTS[v]["input"]).to(device); yt = torch.tensor(y).to(device)
    Zn = static_block(kind, tr); Z = None if Zn is None else torch.from_numpy(Zn).to(device)
    torch.manual_seed(seed_key)
    m = build(model, 0 if Z is None else Z.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=L.LR, weight_decay=L.WD)
    cw = class_w(y[tr], "balanced").to(device)
    g = torch.Generator().manual_seed(seed_key)
    best, bad, hist, t0 = None, 0, [], time.time()
    for ep in range(L.MAX_EP):
        m.train()
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), L.BS):
            j = perm[i:i + L.BS]
            loss = F.cross_entropy(m(X[j]) if Z is None else m(X[j], Z[j]), yt[j], weight=cw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), L.CLIP); opt.step()
        sv = s3(evaluate(y[va], predict(m, X, Z, va))); hist.append(round(sv, 4))
        if best is None or sv > best[0] + 1e-9:
            best = (sv, ep, {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}); bad = 0
        else:
            bad += 1
        if ep + 1 >= L.MIN_EP and bad >= L.PATIENCE:
            break
    train_min = (time.time() - t0) / 60
    m.load_state_dict(best[2])
    t1 = time.perf_counter(); P = predict(m, X, Z, pred_rows); infer_ms = (time.perf_counter() - t1) / len(pred_rows) * 1000
    return P, best[0], best[1], len(hist), train_min, infer_ms, best[2], hist


def fit_fold(model, cfg, seed, r, k):
    f = guard(FOLDC / exp_id(model, cfg) / f"s{seed}_r{r}_k{k}.npz")
    if f.exists():
        return str(f)
    itr, iva, ote = split(cfg, r, k)
    P, sv, be, ne, tmin, ims, state, hist = train(model, itr, iva, ote, y_def(cfg), seed * 1000 + 10 * r + k, DEVICE)
    ck = guard(CKPT / exp_id(model, cfg) / f"s{seed}_r{r}_k{k}.pt"); ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ck)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, ote=ote, P=P, val_s3=sv, best_epoch=be, epochs_run=ne, train_minutes=tmin, infer_ms_per_ecg=ims,
             hist=np.array(hist), n_inner_train=len(itr), normal_inner_train=int((y_def(cfg)[itr] == 0).sum()))
    print(f"[{exp_id(model, cfg)} s{seed} r{r} k{k}] best ep {be} val S3 {sv:.3f} ({ne} ep, {tmin:.1f} min)", flush=True)
    return str(f)


def _job(a):
    return fit_fold(*a)


def assemble(model, cfg, seed, r):
    rid = f"{exp_id(model, cfg)}__s{seed}_r{r}"
    if done(rid):
        return rid
    P = np.zeros((nD, 4)); info = []
    for k in range(5):
        z = np.load(FOLDC / exp_id(model, cfg) / f"s{seed}_r{r}_k{k}.npz"); P[z["ote"]] = z["P"]
        info.append(dict(fold=k, val_s3=float(z["val_s3"]), best_epoch=int(z["best_epoch"]), epochs_run=int(z["epochs_run"]),
                         train_minutes=float(z["train_minutes"]), infer_ms_per_ecg=float(z["infer_ms_per_ecg"]),
                         n_inner_train=int(z["n_inner_train"]), normal_inner_train=int(z["normal_inner_train"])))
    v, kind, head = spec(model)
    cfgd = dict(model=DESCR[model], model_id=model, encoder_variant=v, encoder=L.VARIANTS[v], static=kind, head=head or "6A head",
                hidden=L.HID, optimiser="AdamW", lr=L.LR, weight_decay=L.WD, batch=L.BS, grad_clip=L.CLIP, max_epochs=L.MAX_EP,
                min_epochs=L.MIN_EP, patience=L.PATIENCE, loss="class-balanced CE", selection="inner-val S3 (configuration labels)",
                input_hz=100, steps=1000, config=cfg, seed=seed, repeat=r, device=DEVICE)
    save_run(rid, P, P.argmax(1), r, cfg, cfgd,
             extra=dict(folds=info, n_params=n_params(model), train_minutes=sum(i["train_minutes"] for i in info),
                        infer_ms_per_ecg=float(np.mean([i["infer_ms_per_ecg"] for i in info]))))
    m = metrics(Y, P, P.argmax(1))
    print(f"[{rid}] acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} mF1 {m['macro_f1']:.3f} Mild F1 {m['f1_Mild']:.3f} "
          f"minR {m['min_recall']:.2f}", flush=True)
    return rid


# ------------------------------------------------------------------ F3 / F4 (late fusion)
def lstm_inner(v, r, k):
    """Nested inner OOF of LSTM-fixed (seed 42) on the outer-training portion: same protocol, splits and seed keys as
    phase_6a fusion.lstm_inner (4-fold patient-grouped inner CV, own 80/20 split per inner model), ORIG labels."""
    f = guard(INNER / f"lstm{v}_ORIG" / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    TR = np.flatnonzero(fold_vec(r) != k); P = np.zeros((len(TR), 4)); mins = 0.0
    for j, (ia, ib) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], PAT[TR])):
        fit = TR[ia]
        a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=2000 + 10 * k + j).split(fit, Y[fit], PAT[fit]))
        out = train("FIX", fit[a], fit[b], TR[ib], Y, 42 * 1000 + 100 * r + 10 * k + j, DEVICE)
        P[ib] = out[0]; mins += out[4]
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, TR=TR, P=P, train_minutes=mins)
    print(f"[inner LSTM {v} r{r} k{k}] done ({mins:.1f} min)", flush=True)
    return TR, P


def _inner_job(a):
    return lstm_inner(*a)


def late(jobs):
    v = lstm_fixed()
    with Pool(jobs) as pool:
        list(pool.imap_unordered(_inner_job, [(v, r, k) for r in (0, 1, 2) for k in range(5)]))
    for r in (0, 1, 2):
        Pm = np.mean([load_run(f"P6A-mlpfusion-B1-v1__s{s}_r{r}")[0] for s in SEEDS], 0)
        Pl = load_run(f"P6A-ecgflindist-B1-v1__s42_r{r}")[0]; Ph = load_run(f"P6A-hdrclinlr-B1-v1__s0_r{r}")[0]
        Ps = np.mean([load_run(f"P6A-lstmsel-B1-v1__s{s}_r{r}")[0] for s in SEEDS], 0)
        Pv = np.mean([Pm, Pl, Ph, Ps], 0)
        rid3, rid4 = f"P6B-F3-ORIG-v1__s0_r{r}", f"P6B-F4-ORIG-v1__s0_r{r}"
        if not done(rid3):
            save_run(rid3, Pv, Pv.argmax(1), r, "ORIG", dict(model="F3 vote: C1 members + LSTM-sel (equal weights)", repeat=r,
                                                             members=list(MEMBERS) + ["lstmsel"]))
        if done(rid4):
            continue
        fo = fold_vec(r); pred = np.zeros(nD, int); alphas = []; inner_min = 0.0
        for k in range(5):
            inn = [inner_member(b, "B1", r, k) for b in MEMBERS] + [lstm_inner(v, r, k)]
            TR = inn[0][0]; assert all((x[0] == TR).all() for x in inn)
            inner_min += float(np.load(INNER / f"lstm{v}_ORIG" / f"r{r}_k{k}.npz")["train_minutes"])
            Pin = np.mean([x[1] for x in inn], 0); prior = np.bincount(Y[TR], minlength=4) / len(TR)
            al = choose_alpha(Pin, Y[TR], prior, "protected"); alphas.append(al)
            pred[fo == k] = decide(Pv[fo == k], "prior", al, prior)
        save_run(rid4, Pv, pred, r, "ORIG", dict(model="F4 = F3 + nested class-prior rule (protected; C3 recipe)", repeat=r,
                                                 lstm_inner=f"{v} seed 42"),
                 extra=dict(alpha_per_fold=alphas, lstm_inner_train_minutes=inner_min))
        for rid in (rid3, rid4):
            m = metrics(Y, *load_run(rid))
            print(f"[{rid}] acc {m['accuracy']:.3f} mF1 {m['macro_f1']:.3f} Mild F1 {m['f1_Mild']:.3f} minR {m['min_recall']:.2f}", flush=True)


# ------------------------------------------------------------------ compute rule (plan section 3)
def estimate(jobs):
    """LSTM-fixed under the 8 sensitivity configurations: 3 seeds if the estimated wall clock (this machine, `jobs`
    parallel workers) is <= 10 h, else seed 42 only. Estimate = 8 x 45 fold fits x mean B1 fold time of LSTM-fixed."""
    v = lstm_fixed()
    mins = [float(np.load(L.FOLDC / L.exp_id(v, "B1") / f"s{s}_r{r}_k{k}.npz")["train_minutes"])
            for s in SEEDS for r in (0, 1, 2) for k in range(5)]
    total_h = 8 * 45 * float(np.mean(mins)) / 60 / jobs
    out = dict(lstm_fixed=v, mean_fold_minutes_B1=float(np.mean(mins)), parallel_jobs=jobs, device=DEVICE,
               estimated_wall_clock_hours_8_configs_3_seeds=total_h, rule="3 seeds if <= 10 h, else seed 42 only",
               decision="3 seeds" if total_h <= 10 else "seed 42 only")
    guard(RES).mkdir(parents=True, exist_ok=True)
    (RES / "compute_rule.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["L5", "L6", "F1", "F2"], choices=list(MODELS))
    ap.add_argument("--configs", nargs="+", default=["ORIG"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--late", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    DEVICE = a.device; os.environ["P6A_DEVICE"] = a.device
    if a.estimate:
        estimate(a.jobs); return
    L.X_of("aligned"); L.X_of("rhythm")
    if a.late:
        late(a.jobs); return
    for cfg in a.configs:
        assert cfg in ("ORIG", "T55", "N300", "N270", "N240", "N200"), f"{cfg}: T50 / XUNC / A270X are the 6A B2 / B3 / A270 runs"
    jobs = [(mdl, c, s, r, k) for mdl in a.models for c in a.configs for s in a.seeds for r in (0, 1, 2) for k in range(5)]
    for mdl in a.models:
        for c in a.configs:
            for s in a.seeds:
                for r in (0, 1, 2):
                    if not done(f"{exp_id(mdl, c)}__s{s}_r{r}"):
                        mark_running(f"{exp_id(mdl, c)}__s{s}_r{r}", CMD)
    with Pool(a.jobs) as pool:
        list(pool.imap_unordered(_job, jobs))
    for mdl in a.models:
        for c in a.configs:
            for s in a.seeds:
                for r in (0, 1, 2):
                    assemble(mdl, c, s, r)


if __name__ == "__main__":
    main()
