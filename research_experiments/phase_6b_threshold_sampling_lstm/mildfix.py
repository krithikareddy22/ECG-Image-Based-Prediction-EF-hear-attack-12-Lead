"""MILDFIX (AMENDMENT_MILDFIX.md): one fix per reason Mild is not predicted, on the existing nested members.
  F1  Mild-aware decision rule (per-class logit offsets maximising macro-F1 on inner OOF)
  F2a ordinal decision (expected grade + cut points, floor objective)
  F2b ordinal model (Frank & Hall logistic regressions on header + ECGFounder f6 + clinical)
  F3a EF stacking (nested ridge EF + vote probabilities -> balanced multinomial LR + nested prior rule)
  F3b EF grading (classify by nested ridge EF with tuned cut points)
  F4  abstention (nested confidence threshold, target 0.70) for every model
  F5  noise-free training (members / ridge / ordinal trained on NFS-kept rows)
Every tuned parameter is chosen per outer fold on the fold's training portion only.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mildfix --run
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mildfix --report
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.p5common import ALPHAS, BIASES, cells, corr_ci, corr_t, decide, holm, make_X
from research_experiments.phase_6b_threshold_sampling_lstm import mmclean as M
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

TRAININGS = ("ORIG", "NFS")
METHODS = ("C3", "F1", "F2a", "F2b", "F3a", "F3b")
NAMES = {"C3": "C3 (reference: vote + prior rule)", "F1": "F1 Mild-aware rule", "F2a": "F2a ordinal decision",
         "F2b": "F2b ordinal model", "F3a": "F3a EF stacking", "F3b": "F3b EF grading"}
FLOOR, TARGET = 0.15, 0.70
BLOCKS = ["hdr", "ecgf", "feat:clin3"]
BASES = ("mlpfusion", "ecgflindist", "hdrclinlr")
PRIVD = B.PRIV / "mildfix"


def quick(y, pred):
    cm = np.bincount(4 * y + pred, minlength=16).reshape(4, 4)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1); prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return np.trace(cm) / len(y), rec, f1.mean()


def floor_best(cands, y):
    """cands: iterable of (param, pred). Highest accuracy with every recall >= FLOOR; None if none qualifies."""
    best = None
    for p, pr in cands:
        acc, rec, _ = quick(y, pr)
        if rec.min() >= FLOOR and (best is None or acc > best[1] + 1e-12):
            best = (p, acc)
    return None if best is None else best[0]


def ord_cut(E, c):
    return np.digitize(E, c)


def ef_grade(e, bnd, d0, t1, t2):
    return np.where(e > bnd + d0, 0, np.where(e >= t1, 1, np.where(e >= t2, 2, 3)))


def threshold(conf_in, ok_in):
    ts = [t for t in np.unique(np.round(conf_in, 3)) if (conf_in >= t).sum() >= 20 and ok_in[conf_in >= t].mean() >= TARGET]
    return ts[0] if ts else np.inf


def ridge_fit_predict(fit_rows, pred_rows):
    fm = np.zeros(B.nD, bool); fm[fit_rows] = True
    X, _ = make_X(BLOCKS, fm)
    m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[fit_rows], B.table().ef.values[fit_rows])
    return m.predict(X[pred_rows])


def ordinal_probs(models, X):
    g = np.column_stack([m.predict_proba(X)[:, 1] for m in models])          # P(y > 0), P(y > 1), P(y > 2)
    g = np.minimum.accumulate(g, axis=1)
    P = np.column_stack([1 - g[:, 0], g[:, 0] - g[:, 1], g[:, 1] - g[:, 2], g[:, 2]]).clip(0)
    return P / P.sum(1, keepdims=True).clip(1e-9)


def ordinal_fit(X, y, rows, C):
    return [LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000).fit(X[rows], (y[rows] > k).astype(int))
            for k in range(3)]


def job(arg):
    src, r = arg
    warnings.filterwarnings("ignore")
    Y, T = B.Y, B.table(); bnd = T.sex_boundary.values; fv = B.fold_vec(r)
    Pv, _ = M.preds("vote", src, r); _, pc3 = M.preds("c3", src, r)
    al = M._alphas(src, r)
    out = {m: dict(P=np.zeros((B.nD, 4)), pred=np.zeros(B.nD, int), conf=np.zeros(B.nD), keep=np.zeros(B.nD, bool),
                   params=[]) for m in METHODS}
    for k in range(5):
        ote = np.flatnonzero(fv == k)
        inn = [M._inner(b, src, r, k) for b in BASES]
        TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
        excl = M._excl(src, r, k) if src != "ORIG" else np.zeros(B.nD, bool)
        kp = ~excl[TR]; yTR = Y[TR]
        prior = np.bincount(yTR[kp], minlength=4) / kp.sum()
        res = {}
        # C3 reference (its own alpha)
        res["C3"] = (Pv[ote], pc3[ote], Pv[ote].max(1), Pin.max(1), decide(Pin, "prior", al[k], prior) == yTR, al[k])
        # F1 Mild-aware bias (macro-F1)
        best = max(range(len(BIASES)), key=lambda i: (quick(yTR[kp], decide(Pin[kp], "bias", BIASES[i], None))[2], -i))
        b = BIASES[best]
        res["F1"] = (Pv[ote], decide(Pv[ote], "bias", b, None), Pv[ote].max(1), Pin.max(1),
                     decide(Pin, "bias", b, None) == yTR, b.tolist())
        # F2a ordinal decision on expected grade
        g = np.arange(4.0); Ein, Eout = Pin @ g, Pv[ote] @ g
        grid = [(c1, c2, c3) for c1 in np.arange(0.2, 1.6, 0.05) for c2 in np.arange(c1 + 0.1, 2.45, 0.05)
                for c3 in np.arange(c2 + 0.1, 2.95, 0.05)]
        c = floor_best(((cc, ord_cut(Ein[kp], cc)) for cc in grid), yTR[kp])
        if c is None:
            pa_out, pa_in, c = Pv[ote].argmax(1), Pin.argmax(1), "argmax (no cut met the floor)"
        else:
            pa_out, pa_in = ord_cut(Eout, c), ord_cut(Ein, c); c = [round(x, 2) for x in c]
        res["F2a"] = (Pv[ote], pa_out, Pv[ote].max(1), Pin.max(1), pa_in == yTR, c)
        # F2b ordinal model (C on inner-val)
        itr, iva, _ = B.split_indices(r, k); itr = itr[~excl[itr]]
        fm = np.zeros(B.nD, bool); fm[itr] = True
        X, _ = make_X(BLOCKS, fm)
        Cb = max((0.001, 0.01, 0.1, 1.0), key=lambda C: np.mean(ordinal_probs(ordinal_fit(X, Y, itr, C), X[iva]).argmax(1) == Y[iva]))
        Po = ordinal_probs(ordinal_fit(X, Y, itr, Cb), X[ote])
        inner_split = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, yTR, B.PAT[TR]))
        Pob_in = np.zeros((len(TR), 4))
        for ia, ib in inner_split:
            fr = TR[ia][kp[ia]]; fm2 = np.zeros(B.nD, bool); fm2[fr] = True; X2, _ = make_X(BLOCKS, fm2)
            Pob_in[ib] = ordinal_probs(ordinal_fit(X2, Y, fr, Cb), X2[TR[ib]])
        res["F2b"] = (Po, Po.argmax(1), Po.max(1), Pob_in.max(1), Pob_in.argmax(1) == yTR, Cb)
        # F3 ridge EF: outer and inner OOF
        ef_out = ridge_fit_predict(TR[kp], ote)
        ef_in = np.zeros(len(TR))
        for ia, ib in inner_split:
            ef_in[ib] = ridge_fit_predict(TR[ia][kp[ia]], TR[ib])
        # F3a stacking + nested prior rule (cross-fitted inner stacker)
        Zin = np.column_stack([np.log(Pin.clip(1e-6)), ef_in / 10]); Zout = np.column_stack([np.log(Pv[ote].clip(1e-6)), ef_out / 10])
        st = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Zin[kp], yTR[kp])
        Pst_out = st.predict_proba(Zout)
        Pst_in = np.zeros((len(TR), 4))
        for ia, ib in inner_split:
            Pst_in[ib] = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Zin[ia][kp[ia]], yTR[ia][kp[ia]]).predict_proba(Zin[ib])
        a = floor_best(((a, decide(Pst_in[kp], "prior", a, prior)) for a in ALPHAS), yTR[kp])
        a = 0.0 if a is None else float(a)
        res["F3a"] = (Pst_out, decide(Pst_out, "prior", a, prior), Pst_out.max(1), Pst_in.max(1),
                      decide(Pst_in, "prior", a, prior) == yTR, a)
        # F3b EF grading
        bTR = bnd[TR]
        grid = [(d0, t1, t2) for d0 in np.arange(-15, 6.5, 0.5) for t1 in np.arange(30, 55.5, 0.5) for t2 in np.arange(20, t1, 0.5)]
        p = floor_best(((q, ef_grade(ef_in[kp], bTR[kp], *q)) for q in grid), yTR[kp])
        p = p if p is not None else (0.0, 41.0, 31.0)
        cuts_out = np.column_stack([bnd[ote] + p[0], np.full(len(ote), p[1]), np.full(len(ote), p[2])])
        cuts_in = np.column_stack([bTR + p[0], np.full(len(TR), p[1]), np.full(len(TR), p[2])])
        pr_out, pr_in = ef_grade(ef_out, bnd[ote], *p), ef_grade(ef_in, bTR, *p)
        res["F3b"] = (np.eye(4)[pr_out], pr_out, np.abs(ef_out[:, None] - cuts_out).min(1), np.abs(ef_in[:, None] - cuts_in).min(1),
                      pr_in == yTR, [float(x) for x in p])
        for m, (P, pred, conf, conf_in, ok_in, prm) in res.items():
            o = out[m]; o["P"][ote] = P; o["pred"][ote] = pred; o["conf"][ote] = conf
            o["keep"][ote] = conf >= threshold(conf_in, ok_in); o["params"].append(prm)
    d = B.guard(PRIVD); d.mkdir(parents=True, exist_ok=True)
    for m, o in out.items():
        np.savez(d / f"{m}_{src}_r{r}.npz", P=o["P"], pred=o["pred"], conf=o["conf"], keep=o["keep"])
    (d / f"params_{src}_r{r}.json").write_text(json.dumps({m: o["params"] for m, o in out.items()}, default=float))
    print(f"[{src} r{r}] " + " ".join(f"{m} {np.mean(o['pred'] == Y):.3f}" for m, o in out.items()), flush=True)
    return src, r


def load(m, src, r):
    z = np.load(PRIVD / f"{m}_{src}_r{r}.npz")
    return z["P"], z["pred"], z["conf"], z["keep"]


def summary(m, src, mask):
    ms, cl = [], []
    for r in (0, 1, 2):
        P, pred, _, _ = load(m, src, r)
        ms.append(B.metrics(B.Y[mask], P[mask], pred[mask])); cl += [c["accuracy"] for c in cells(P, pred, r, mask=mask)]
    o = {k: float(np.mean([x[k] for x in ms])) for k in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Normal",
                                                          "recall_Mild", "recall_Moderate", "recall_Severe", "precision_Mild")}
    o["min_recall"] = min(o[f"recall_{c}"] for c in B.CLASSES); o["admissible"] = bool(o["min_recall"] >= FLOOR)
    o["accuracy_ci95_corrected"] = corr_ci(cl); o["accuracy_per_repeat"] = [round(x["accuracy"], 4) for x in ms]
    o["confusion_mean"] = np.mean([B.confusion(B.Y[mask], load(m, src, r)[1][mask]) for r in (0, 1, 2)], 0).round(1).tolist()
    return o


def selective(m, src):
    reps, cl = [], []
    for r in (0, 1, 2):
        _, pred, _, keep = load(m, src, r); fv = B.fold_vec(r); y = B.Y
        for k in range(5):
            s = keep & (fv == k)
            if s.any():
                cl.append(float(np.mean(pred[s] == y[s])))
        reps.append(dict(coverage=float(keep.mean()), accuracy=float(np.mean(pred[keep] == y[keep])) if keep.any() else None,
                         coverage_by_class={c: float(keep[y == i].mean()) for i, c in enumerate(B.CLASSES)},
                         recall_covered={c: (float(np.mean(pred[keep & (y == i)] == i)) if (keep & (y == i)).any() else None)
                                         for i, c in enumerate(B.CLASSES)},
                         predicted_mix={c: int((pred[keep] == i).sum()) for i, c in enumerate(B.CLASSES)}))
    ok = [x for x in reps if x["accuracy"] is not None]
    agg = lambda key, c: float(np.mean([x[key][c] for x in reps if x[key][c] is not None])) if any(x[key][c] is not None for x in reps) else None  # noqa: E731
    return dict(model=m, trained=src, target=TARGET, coverage=float(np.mean([x["coverage"] for x in reps])),
                accuracy=float(np.mean([x["accuracy"] for x in ok])) if ok else None,
                accuracy_ci95_corrected=corr_ci(cl) if len(cl) > 2 else None, n_fold_cells=len(cl),
                coverage_by_class={c: agg("coverage_by_class", c) for c in B.CLASSES},
                recall_covered={c: agg("recall_covered", c) for c in B.CLASSES},
                predicted_mix={c: float(np.mean([x["predicted_mix"][c] for x in reps])) for c in B.CLASSES})


def paired(m, src, metric):
    d, dr = [], []
    for r in (0, 1, 2):
        Pa, pa, _, _ = load(m, src, r); Pb, pb, _, _ = load("C3", "ORIG", r)
        d += [x[metric] - z[metric] for x, z in zip(cells(Pa, pa, r), cells(Pb, pb, r))]
        dr.append(B.metrics(B.Y, Pa, pa)[metric] - B.metrics(B.Y, Pb, pb)[metric])
    _, p = corr_t(d)
    return dict(model=m, trained=src, metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr],
                ci95_corrected_delta=corr_ci(d), p_corrected_t=p, same_sign_all_repeats=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)))


def report():
    F = M.flags()
    subsets = {"all_837": np.ones(B.nD, bool), "NF_clear": ~F.NF.values & ~F.near_boundary.values, "HC": M.hc_mask(),
               "NFS_set": ~F.NFS.values}
    out = dict(subsets={k: int(v.sum()) for k, v in subsets.items()}, results=[], comparisons=[], selective=[], params={})
    for src in TRAININGS:
        for m in METHODS:
            for s, mask in subsets.items():
                out["results"].append(dict(model=NAMES[m], key=m, trained=src, evaluated_on=s, n_ecgs=int(mask.sum()), **summary(m, src, mask)))
            out["selective"].append(dict(name=NAMES[m], **selective(m, src)))
            if not (m == "C3" and src == "ORIG"):
                for metric in ("accuracy", "macro_f1", "recall_Mild"):
                    out["comparisons"].append(paired(m, src, metric))
        for r in (0, 1, 2):
            out["params"][f"{src}_r{r}"] = json.loads((PRIVD / f"params_{src}_r{r}.json").read_text())
    adj = holm([x["p_corrected_t"] for x in out["comparisons"]])
    for x, a in zip(out["comparisons"], adj):
        x["p_holm"] = float(a)
    main = [x for x in out["results"] if x["evaluated_on"] == "all_837"]
    for x in out["comparisons"]:
        adm = next(z for z in main if z["key"] == x["model"] and z["trained"] == x["trained"])["admissible"]
        x["genuine_improvement"] = bool(x["delta"] > 0 and x["p_holm"] < 0.05 and x["same_sign_all_repeats"] and adm)
    cand = [x for x in main if x["admissible"] and not (x["key"] == "C3" and x["trained"] == "ORIG")]
    fin = max(cand, key=lambda x: x["accuracy"]) if cand else None
    sel = [x for x in out["selective"] if x["accuracy"] is not None and x["coverage"] >= 0.20]
    out["final"] = dict(four_class=fin and {k: fin[k] for k in ("model", "trained", "accuracy", "accuracy_ci95_corrected", "macro_f1",
                                                                 "recall_Mild", "f1_Mild", "min_recall")},
                        abstention=max(sel, key=lambda x: x["accuracy"]) if sel else None,
                        best_mild=max(main, key=lambda x: x["f1_Mild"]))
    (B.guard(B.RES) / "mildfix_results.json").write_text(json.dumps(out, indent=1, default=float))
    rows = pd.DataFrame(out["results"]); rows["accuracy_ci95_corrected"] = rows.accuracy_ci95_corrected.map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    rows.drop(columns=["confusion_mean", "accuracy_per_repeat"]).to_csv(B.guard(B.RES / "mildfix_results.csv"), index=False, float_format="%.4f")
    pd.DataFrame(out["comparisons"]).to_csv(B.guard(B.RES / "mildfix_comparisons.csv"), index=False, float_format="%.4f")
    pd.DataFrame(out["selective"]).to_csv(B.guard(B.RES / "mildfix_selective.csv"), index=False, float_format="%.4f")
    plot(out)
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [x for x in out["results"] if x["evaluated_on"] == "all_837"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    lab = [f"{x['key']}\n{x['trained']}" for x in main]; xs = np.arange(len(main))
    col = ["#9aa5b1" if x["trained"] == "ORIG" else "#4c78a8" for x in main]
    acc = [x["accuracy"] for x in main]; ci = np.array([x["accuracy_ci95_corrected"] for x in main])
    axes[0].bar(xs, acc, color=col, yerr=[np.array(acc) - ci[:, 0], ci[:, 1] - np.array(acc)], capsize=2)
    axes[0].set_title("Four-class accuracy, all 837 ECGs (corrected 95 % CI)", fontsize=9); axes[0].axhline(0.7, ls=":", c="k", lw=0.8)
    axes[1].bar(xs - 0.2, [x["recall_Mild"] for x in main], 0.4, color=col, label="Mild recall")
    axes[1].bar(xs + 0.2, [x["f1_Mild"] for x in main], 0.4, color=col, alpha=0.5, hatch="//", label="Mild F1")
    axes[1].axhline(FLOOR, ls=":", c="r", lw=0.8); axes[1].set_title("Mild recall (solid) and Mild F1 (hatched), all 837", fontsize=9)
    S = out["selective"]
    axes[2].bar(xs, [x["accuracy"] or 0 for x in S], color=col)
    for i, x in enumerate(S):
        axes[2].text(i, (x["accuracy"] or 0) + 0.01, f"{x['coverage'] * 100:.0f}%", ha="center", fontsize=7)
    axes[2].axhline(0.7, ls=":", c="k", lw=0.8)
    axes[2].set_title("Abstention (target 0.70): accuracy on answered ECGs; label = % answered", fontsize=9)
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7); ax.set_ylim(0, 1)
    fig.suptitle("MILDFIX: one fix per reason (grey = trained on all labels, blue = noise-free NFS training); development CV, exploratory",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(B.guard(B.PLOTS / "mildfix_summary.png"), dpi=150); plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--trainings", nargs="+", default=list(TRAININGS)); ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.run:
        B.table(); M.flags()
        with Pool(a.jobs) as pool:
            for x in pool.imap_unordered(job, [(s, r) for s in a.trainings for r in (0, 1, 2)]):
                print("done", x, flush=True)
    if a.report:
        out = report()
        for x in out["results"]:
            if x["evaluated_on"] == "all_837":
                print(f"{x['key']:4s} {x['trained']:4s} acc {x['accuracy']:.3f} {np.round(x['accuracy_ci95_corrected'], 3).tolist()} "
                      f"mF1 {x['macro_f1']:.3f} Mild rec {x['recall_Mild']:.3f} F1 {x['f1_Mild']:.3f} min {x['min_recall']:.3f}")
        print(json.dumps(out["final"], indent=1, default=float))


if __name__ == "__main__":
    main()
