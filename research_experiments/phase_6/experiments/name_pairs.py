"""N: name-pair sensitivity analysis (PHASE6_PLAN.md section 4.6). Development ECGs only.

1. Near-identical names under different patient keys among DEVELOPMENT rows only (Phase-5 criterion: difflib ratio
   >= 0.90, or one normalised name of >= 8 letters contained in the other). Names are read from the manifest file paths
   into memory, compared and discarded; nothing name-derived is printed or written except counts.
2. Union-find merges each pair into one group (private mapping: data/private/name_groups.csv - hashed keys only).
3. Folds regenerated exactly as phase_2/protocol.make_folds (outer StratifiedGroupKFold(5, shuffle, 2025 + r); inner
   split StratifiedGroupKFold(5, shuffle, 100 + k), first split) with the merged groups; private copy only.
4. The C1 recipe (MLP fusion 3 seeds, ECGFounder linear distance-CE, header + clinical LR, equal-weight vote) and the
   C3 rule (nested prior rule, protected; nested inner CV and inner 80/20 splits also grouped by the merged groups) are
   re-trained on the merged folds and compared with the same code on the original folds (Phase 6A B1 runs).
No ECG is moved between development and locked test, dropped or relabelled. Output: results/name_pair_sensitivity.json
    python -m research_experiments.phase_6.experiments.name_pairs
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.experiments.nested import BASES
from research_experiments.phase_6.p6common import IDS, PAT, PRIV6, RES6, ROOT, SEEDS, Y, decide, guard6, make_X, metrics, nD

MEMBERS = ("mlpfusion", "ecgflindist", "hdrclinlr")
NP = PRIV6 / "name_pairs"


def norm_name(src):
    m = re.match(r"EK__\[(\d+)\]_(.*?)_*\[\]_", Path(src).name)
    return re.sub(r"[^A-Z]", "", m.group(2).upper()) if m else ""


def merged_groups():
    f = NP / "name_groups.csv"
    if f.exists():
        g = pd.read_csv(f).set_index("ecg_id").loc[IDS]; return g.group.values, json.loads((NP / "pairs.json").read_text())
    man = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "patient", "split", "source"])
    man = man[man.split != "test"].set_index("ecg_id").loc[IDS]                  # development rows only
    names = man.source.map(norm_name)                                            # in memory only
    key_name = pd.Series(names.values, index=man.patient.values).groupby(level=0).first()
    keys, nms = key_name.index.values, key_name.values
    pairs = 0; parent = {k: k for k in keys}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = nms[i], nms[j]
            if not a or not b:
                continue
            contained = min(len(a), len(b)) >= 8 and (a in b or b in a)
            sm = difflib.SequenceMatcher(None, a, b)
            if not contained and (sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9):
                continue
            if contained or sm.ratio() >= 0.9:
                pairs += 1; parent[find(keys[i])] = find(keys[j])
    del names, key_name, nms
    grp = np.array([hashlib.sha256(find(p).encode()).hexdigest()[:12] for p in PAT])
    info = dict(pairs_within_development=pairs, groups_before=int(len(set(PAT))), groups_after=int(len(set(grp))),
                ecgs_in_merged_groups=int(pd.Series(grp).map(pd.Series(grp).value_counts()).gt(pd.Series(PAT).map(pd.Series(PAT).value_counts())).sum()))
    guard6(NP).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(ecg_id=IDS, group=grp)).to_csv(f, index=False)
    (NP / "pairs.json").write_text(json.dumps(info))
    return grp, info


def make_folds(grp):
    """phase_2/protocol.make_folds with merged groups -> per repeat: fold vector and inner-train / inner-val masks."""
    out = {}
    for r in (0, 1, 2):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2025 + r)
        fold = np.full(nD, -1); splits = {}
        for k, (tr, te) in enumerate(outer.split(np.zeros(nD), Y, grp)):
            fold[te] = k
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + k)
            a, b = next(inner.split(np.zeros(len(tr)), Y[tr], grp[tr]))
            splits[k] = (tr[a], tr[b], te)
            assert not set(grp[tr[a]]) & set(grp[te]) and not set(grp[tr[b]]) & set(grp[te]) and not set(grp[tr[a]]) & set(grp[tr[b]])
        out[r] = (fold, splits)
    return out


def fit_member(base, Xtr, ytr, Xva, yva, seed, pre):
    from research_experiments.phase_6a_normal_mild_lstm.existing import fit_member as fm
    return fm(base, Xtr, ytr, Xva, yva, seed, pre)


def outer(base, seed, splits):
    P = np.zeros((nD, 4))
    for k, (itr, iva, ote) in splits.items():
        fm = np.zeros(nD, bool); fm[itr] = True
        X, pre = make_X(BASES[base][0], fm)
        P[ote] = fit_member(base, X[itr], Y[itr], X[iva], Y[iva], seed, pre)(X[ote])
    return P


def inner_oof(base, r, k, fold, grp):
    f = guard6(NP / "inner" / base / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    TR = np.flatnonzero(fold != k); P = np.zeros((len(TR), 4))
    for j, (ia, ib) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], grp[TR])):
        fit = TR[ia]
        a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=2000 + 10 * k + j).split(fit, Y[fit], grp[fit]))
        tr, va = fit[a], fit[b]
        fm = np.zeros(nD, bool); fm[tr] = True
        X, pre = make_X(BASES[base][0], fm)
        P[ib] = np.mean([fit_member(base, X[tr], Y[tr], X[va], Y[va], s, pre)(X[TR[ib]]) for s in BASES[base][3]], 0)
    f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, TR=TR, P=P)
    return TR, P


def main():
    torch.set_num_threads(1)
    grp, info = merged_groups()
    folds = make_folds(grp)
    res = dict(info=info, merged={}, original={})
    from research_experiments.phase_6a_normal_mild_lstm.p6acommon import load_run as load6a
    for r in (0, 1, 2):
        fold, splits = folds[r]
        Pm = np.mean([outer("mlpfusion", s, splits) for s in SEEDS], 0)
        Pv = np.mean([Pm, outer("ecgflindist", 42, splits), outer("hdrclinlr", 0, splits)], 0)
        pred = np.zeros(nD, int)
        for k in range(5):
            inn = [inner_oof(b, r, k, fold, grp) for b in MEMBERS]
            TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
            prior = np.bincount(Y[TR], minlength=4) / len(TR)
            pred[fold == k] = decide(Pv[fold == k], "prior", choose_alpha(Pin, Y[TR], prior, "protected"), prior)
        res["merged"][r] = dict(mlpfusion=metrics(Y, Pm, Pm.argmax(1)), vote=metrics(Y, Pv, Pv.argmax(1)), c3=metrics(Y, Pv, pred))
        Pm0 = np.mean([load6a(f"P6A-mlpfusion-B1-v1__s{s}_r{r}")[0] for s in SEEDS], 0)
        Pv0 = load6a(f"P6A-vote-B1-v1__s0_r{r}")[0]; P30, p30 = load6a(f"P6A-c3-B1-v1__s0_r{r}")
        res["original"][r] = dict(mlpfusion=metrics(Y, Pm0, Pm0.argmax(1)), vote=metrics(Y, Pv0, Pv0.argmax(1)), c3=metrics(Y, P30, p30))
        print(f"[r{r}] merged: vote {res['merged'][r]['vote']['accuracy']:.3f} c3 {res['merged'][r]['c3']['accuracy']:.3f} | "
              f"original: vote {res['original'][r]['vote']['accuracy']:.3f} c3 {res['original'][r]['c3']['accuracy']:.3f}", flush=True)
    summ = {}
    for m in ("mlpfusion", "vote", "c3"):
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "min_recall"):
            a = [res["merged"][r][m][k] for r in (0, 1, 2)]; b = [res["original"][r][m][k] for r in (0, 1, 2)]
            summ[f"{m}_{k}"] = dict(merged=float(np.mean(a)), original=float(np.mean(b)), delta=float(np.mean(a) - np.mean(b)),
                                    merged_sd_repeats=float(np.std(a)))
    out = dict(info=info, summary=summ, note="different folds -> no paired cell test; compare pooled metrics (mean over 3 repeats) "
               "against the repeat-to-repeat SD. Same code on both fold sets.")
    (guard6(RES6) / "name_pair_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(info=info, **{k: v for k, v in summ.items() if k.endswith(("accuracy", "macro_f1"))}), indent=1))


if __name__ == "__main__":
    main()
