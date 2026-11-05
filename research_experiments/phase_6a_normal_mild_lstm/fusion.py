"""Phase 6A conditional ECGFounder-LSTM fusion (PHASE6A_PLAN.md section 5): run only if the pre-registered
complementarity criterion holds (adding LSTM-sel as a 4th member to the C1 vote raises accuracy or macro-F1 in all three
repeats; computed by compare.complementarity).
  voteLSTM  equal-weight mean of [MLP fusion seed-mean, ECGF linear distance-CE, header+clinical LR, LSTM-sel seed-mean]
  c3LSTM    voteLSTM + the C3 nested class-prior rule (protected objective); the LSTM member's nested inner OOF uses
            LSTM-fixed with seed 42 (Phase 5 nested protocol: 4-fold patient-grouped inner CV, own 80/20 split per inner model)
    python -m research_experiments.phase_6a_normal_mild_lstm.fusion --jobs 4
"""
from __future__ import annotations

import argparse
import copy
import json
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
from research_experiments.phase_6a_normal_mild_lstm.existing import MEMBERS, inner_oof as inner_member
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import (CONF, INNER, PAT, RES, SEEDS, Y, decide, done, fold_vec, guard,
                                                                     load_run, nD, save_run)


def train_predict(v, tr, va, pred_rows, seed_key):
    """Same recipe as lstm.fit_fold, on arbitrary (train, val) rows; returns probabilities for pred_rows."""
    torch.set_num_threads(1)
    spec = L.VARIANTS[v]; X = L.X_of(spec["input"]); yt = torch.tensor(Y)
    torch.manual_seed(seed_key)
    m = L.LSTMNet(X.shape[2], spec["layers"], spec["bi"], spec["inter_dropout"], spec["cnn"])
    opt = torch.optim.AdamW(m.parameters(), lr=L.LR, weight_decay=L.WD); cw = class_w(Y[tr], "balanced")
    g = torch.Generator().manual_seed(seed_key); best, bad = None, 0
    for ep in range(L.MAX_EP):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), L.BS):
            j = perm[i:i + L.BS]
            loss = F.cross_entropy(m(X[j]), yt[j], weight=cw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), L.CLIP); opt.step()
        sv = s3(evaluate(Y[va], L.predict(m, X, va)))
        if best is None or sv > best[0] + 1e-9:
            best = (sv, copy.deepcopy(m.state_dict())); bad = 0
        else:
            bad += 1
        if ep + 1 >= L.MIN_EP and bad >= L.PATIENCE:
            break
    m.load_state_dict(best[1])
    return L.predict(m, X, pred_rows)


def lstm_inner(v, r, k):
    f = guard(INNER / f"lstm{v}_B1" / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    TR = np.flatnonzero(fold_vec(r) != k); P = np.zeros((len(TR), 4))
    for j, (ia, ib) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], PAT[TR])):
        fit = TR[ia]
        a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=2000 + 10 * k + j).split(fit, Y[fit], PAT[fit]))
        P[ib] = train_predict(v, fit[a], fit[b], TR[ib], 42 * 1000 + 100 * r + 10 * k + j)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, TR=TR, P=P)
    print(f"[inner LSTM {v} r{r} k{k}] done", flush=True)
    return TR, P


def _job(a):
    return lstm_inner(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args()
    comp = json.loads((RES / "phase6a_summary.json").read_text()).get("complementarity") or {}
    if not comp.get("criterion_met"):
        print("complementarity criterion not met -> fusion not run (pre-registered)"); return
    v = json.loads((CONF / "lstm_fixed.json").read_text())["lstm_fixed"]
    L.X_of("aligned"); L.X_of("rhythm")
    with Pool(a.jobs) as pool:
        list(pool.imap_unordered(_job, [(v, r, k) for r in (0, 1, 2) for k in range(5)]))
    for r in (0, 1, 2):
        Pm = np.mean([load_run(f"P6A-mlpfusion-B1-v1__s{s}_r{r}")[0] for s in SEEDS], 0)
        Pl = load_run(f"P6A-ecgflindist-B1-v1__s42_r{r}")[0]; Ph = load_run(f"P6A-hdrclinlr-B1-v1__s0_r{r}")[0]
        Ps = np.mean([load_run(f"P6A-lstmsel-B1-v1__s{s}_r{r}")[0] for s in SEEDS], 0)
        Pv = np.mean([Pm, Pl, Ph, Ps], 0)
        if not done(f"P6A-voteLSTM-B1-v1__s0_r{r}"):
            save_run(f"P6A-voteLSTM-B1-v1__s0_r{r}", Pv, Pv.argmax(1), r, "B1", dict(model="vote of C1 members + LSTM-sel", repeat=r))
        if done(f"P6A-c3LSTM-B1-v1__s0_r{r}"):
            continue
        fo = fold_vec(r); pred = np.zeros(nD, int); alphas = []
        for k in range(5):
            inn = [inner_member(b, "B1", r, k) for b in MEMBERS] + [lstm_inner(v, r, k)]
            TR = inn[0][0]; assert all((x[0] == TR).all() for x in inn)
            Pin = np.mean([x[1] for x in inn], 0); prior = np.bincount(Y[TR], minlength=4) / len(TR)
            al = choose_alpha(Pin, Y[TR], prior, "protected"); alphas.append(al)
            pred[fo == k] = decide(Pv[fo == k], "prior", al, prior)
        save_run(f"P6A-c3LSTM-B1-v1__s0_r{r}", Pv, pred, r, "B1", dict(model="vote + LSTM + nested prior rule", repeat=r, lstm_inner=v),
                 extra=dict(alpha_per_fold=alphas))


if __name__ == "__main__":
    main()
