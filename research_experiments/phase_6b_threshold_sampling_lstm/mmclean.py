"""Exploratory MMC / MMNC configurations (AMENDMENT_MMCLEAN.md): the top-3 existing recipes (MLP fusion, C1 vote, C3)
trained without suspect Mild / Moderate labels (MMC), and without suspect Normal / Mild / Moderate labels (MMNC).
Severe ECGs are never removed or relabelled (owner request).
Grid addition (owner request "70 %"; AMENDMENT_MMCLEAN.md "Grid"): MN240 / MN200 = MMNC + Normal outer-training portion
reduced to 240 / 200 (whole Normal patients removed from inner-train, seeded); ECHO = Normal / Mild / Moderate ECGs
relabelled to the class of the nearest echo where it is Normal / Mild / Moderate and differs (no ECG removed).
Suspect = echo evidence against the label only (never a model prediction): folder label != nearest-echo class, a linked
echo of another class, EF spread > 10 points across links, or nearest-echo EF within 3 points of a class boundary.
Name-match score < 100 and |ECG - echo| > 30 days are counted but NOT used (they flag 47 % / 23 % of ECGs evenly across
classes: weak-link indicators, not evidence of a wrong label).
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --counts
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --run --configs MMC MMNC
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --run --configs MN240 MN200 ECHO
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --run --configs NFS NF
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --report
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research_experiments.phase_5.p5common import cells, corr_ci, corr_t, holm
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

CONFIGS = ("MMC", "MMNC")
GRID = ("MN240", "MN200", "ECHO")
NOISE = ("NFS", "NF")                      # noise-free training (AMENDMENT_MMCLEAN.md "Noise-free training")
ALL = CONFIGS + GRID + NOISE
NSIZE = {"MN240": 240, "MN200": 200}
BAND, SPREAD, DAYS = 3.0, 10.0, 30
MODELS = (("mlpfusion", "MLP fusion"), ("vote", "C1 vote"), ("c3", "C3 (vote + nested prior rule)"))
ORIG_ID = {"mlpfusion": "P6A-mlpfusion-B1-v1__s{s}_r{r}", "vote": "P6A-vote-B1-v1__s0_r{r}", "c3": "P6A-c3-B1-v1__s0_r{r}"}
_F = None


def flags():
    """Per-ECG boolean flags (development set). Echo evidence only; computed once."""
    global _F
    if _F is not None:
        return _F
    from research_experiments.phase_6a_normal_mild_lstm.p6acommon import _links_dev
    from research_experiments.phase_6b_threshold_sampling_lstm.audit6b import true_interval
    T = B.table(); ef = T.ef.values; lab = T.label.values; IDS = B.IDS
    L = _links_dev()
    other = set(L.assign(lab=L.ecg_id.map(pd.Series(lab, index=IDS))).query("derived != 'Unknown' and derived != lab").ecg_id)
    days = true_interval()
    F = pd.DataFrame(dict(
        mild_moderate=np.isin(lab, ["Mild", "Moderate"]),
        normal=lab == "Normal",
        inconsistent=T.derived.values != lab,
        near_boundary=(np.abs(ef - T.sex_boundary.values) <= BAND) | (np.abs(ef - 41) <= BAND) | (np.abs(ef - 31) <= BAND),
        name_score_lt_100=T.score.values < 100,
        interval_gt_30d=~(days <= DAYS),
        other_class_link=np.isin(IDS, list(other)),
        ef_spread_gt_10=T.ef_spread.values > SPREAD))
    F["suspect_any"] = F.inconsistent | F.near_boundary | F.other_class_link | F.ef_spread_gt_10
    F["border_any"] = F.inconsistent | F.near_boundary                     # = AMENDMENT_CLEAR border definition
    F["MMC"] = F.mild_moderate & F.suspect_any
    F["MMNC"] = (F.mild_moderate | F.normal) & F.suspect_any               # Severe never removed
    strong = ~F.name_score_lt_100 & ~F.interval_gt_30d                       # exact name match, echo <= 30 d (true dates)
    consistent = ~(F.inconsistent | F.other_class_link | F.ef_spread_gt_10)
    F["NF"] = ~(strong & consistent)                                         # removed: any link noise (all classes)
    F["NFS"] = F.NF & (lab != "Severe")                                      # removed: link noise, Severe untouched
    _F = F
    return F


def removed(cfg):
    return flags()[cfg].values.copy()


def y_echo():
    """ECHO labels: Normal / Mild / Moderate ECGs take the nearest-echo class where it is Normal / Mild / Moderate."""
    T = B.table(); lab, der = T.label.values, T.derived.values; nmm = ["Normal", "Mild", "Moderate"]
    y = B.Y.copy(); m = np.isin(lab, nmm) & np.isin(der, nmm) & (der != lab)
    y[m] = [B.CLASSES.index(d) for d in der[m]]
    return y


def nsize_after(r, k, target, base):
    """MMNC removal + whole Normal patients removed from inner-train (seeded as N-sizes) until the outer-training portion
    (inner-train after removal + inner-val) holds <= target Normal ECGs."""
    itr, iva, _ = B.split_indices(r, k)
    rem = base.copy(); Y = B.Y; PAT = B.PAT
    n_out = int(((Y[itr] == 0) & ~rem[itr]).sum() + (Y[iva] == 0).sum())
    rng = np.random.default_rng(6200 + 10 * r + k)
    for p in rng.permutation(pd.unique(PAT[itr])):
        if n_out <= target:
            break
        rows = itr[(PAT[itr] == p) & ~rem[itr]]
        if len(rows) == 0 or (Y[rows] != 0).any():
            continue
        if n_out - len(rows) >= target:
            rem[rows] = True; n_out -= len(rows)
    return rem


def _excl(cfg, r, k, _orig=B.train_exclude):
    if cfg in CONFIGS or cfg in NOISE:
        return removed(cfg)
    if cfg in NSIZE:
        return nsize_after(r, k, NSIZE[cfg], removed("MMNC"))
    if cfg == "ECHO":
        return np.zeros(B.nD, bool)
    return _orig(cfg, r, k)


def _ydef(cfg, _orig=B.y_def):
    return y_echo() if cfg == "ECHO" else _orig(cfg)


def work(cfg):
    from research_experiments.phase_6b_threshold_sampling_lstm import existing6b as E
    B.train_exclude = _excl; E.train_exclude = _excl                      # patched inside the worker (spawned process)
    B.y_def = _ydef; E.y_def = _ydef
    flags()
    return E.run_config(cfg)


def counts():
    F = flags(); lab = B.table().label.values
    out = {}
    for c in ("suspect_any", "border_any") + CONFIGS + ("inconsistent", "near_boundary", "name_score_lt_100", "interval_gt_30d",
                                                       "other_class_link", "ef_spread_gt_10"):
        m = F[c].values
        out[c] = dict(total=int(m.sum()), **{k: int((m & (lab == k)).sum()) for k in B.CLASSES})
    out["kept"] = {c: dict(total=int((~F[c].values).sum()), **{k: int((~F[c].values & (lab == k)).sum()) for k in B.CLASSES})
                   for c in CONFIGS + NOISE}
    ye = y_echo()
    out["ECHO_relabelled"] = {f"{B.CLASSES[a]}->{B.CLASSES[b]}": int(((B.Y == a) & (ye == b)).sum())
                              for a in range(4) for b in range(4) if a != b and ((B.Y == a) & (ye == b)).any()}
    out["ECHO_class_counts"] = {k: int((ye == i).sum()) for i, k in enumerate(B.CLASSES)}
    nrm = {}
    for c, t in NSIZE.items():
        v = []
        for r in (0, 1, 2):
            for k in range(5):
                itr, iva, _ = B.split_indices(r, k); ex = nsize_after(r, k, t, removed("MMNC"))
                v.append(int(((B.Y[itr] == 0) & ~ex[itr]).sum() + (B.Y[iva] == 0).sum()))
        nrm[c] = [min(v), max(v)]
    out["normal_outer_training_range"] = nrm
    return out


# ------------------------------------------------------------------ evaluation
def preds(m, src, r):
    """Probabilities and predictions of model m; src = 'ORIG' (Phase 6A B1 runs, regenerated on this machine) or a config."""
    if src == "ORIG":
        ids = [ORIG_ID[m].format(s=s, r=r) for s in (B.SEEDS if m == "mlpfusion" else (0,))]
    else:
        ids = [f"P6B-{m}-{src}-v1__s{s}_r{r}" for s in (B.SEEDS if m == "mlpfusion" else (0,))]
    Ps = [B.load_run(i) for i in ids]
    P = np.mean([p[0] for p in Ps], 0)
    return P, (Ps[0][1] if m == "c3" else P.argmax(1))


def summary(m, src, mask, y=None):
    y = B.Y if y is None else y
    ms, cl = [], []
    for r in (0, 1, 2):
        P, pred = preds(m, src, r)
        ms.append(B.metrics(y[mask], P[mask], pred[mask]))
        cl += [c["accuracy"] for c in cells(P, pred, r, y=y, mask=mask)]
    o = {k: float(np.mean([x[k] for x in ms])) for k in ("accuracy", "balanced_accuracy", "macro_f1", "recall_Normal", "recall_Mild",
                                                          "recall_Moderate", "recall_Severe", "min_recall", "f1_Mild")}
    present = [k for i, k in enumerate(B.CLASSES) if (y[mask] == i).any()]
    o["classes_present"] = present
    o["min_recall"] = float(min(o[f"recall_{k}"] for k in present))          # over the classes present in the subset
    o["accuracy_ci95_corrected"] = corr_ci(cl)
    o["accuracy_per_repeat"] = [round(x["accuracy"], 4) for x in ms]
    o["admissible"] = bool(o["min_recall"] >= 0.15)
    o["confusion_mean"] = np.mean([B.confusion(y[mask], preds(m, src, r)[1][mask]) for r in (0, 1, 2)], 0).round(1).tolist()
    return o


def paired(m, src, mask, metric="accuracy", y=None):
    y = B.Y if y is None else y
    d, dr = [], []
    for r in (0, 1, 2):
        Pa, pa = preds(m, src, r); Pb, pb = preds(m, "ORIG", r)
        d += [x[metric] - z[metric] for x, z in zip(cells(Pa, pa, r, y=y, mask=mask), cells(Pb, pb, r, y=y, mask=mask))]
        dr.append(B.metrics(y[mask], Pa[mask], pa[mask])[metric] - B.metrics(y[mask], Pb[mask], pb[mask])[metric])
    _, p = corr_t(d)
    return dict(model=m, config=src, metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr],
                ci95_corrected_delta=corr_ci(d), p_corrected_t=p,
                same_sign_all_repeats=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)))


def hc_mask():
    """AMENDMENT_MMCLEAN.md addition: MMNC-kept minus weak-link Normal / Mild / Moderate ECGs (Severe untouched)."""
    F = flags(); lab = B.table().label.values
    return ~F.MMNC.values & ~((F.name_score_lt_100 | F.interval_gt_30d).values & (lab != "Severe"))


def coverage_curve(m, src, levels=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)):
    out = {}
    for c in levels:
        acc, mix = [], []
        for r in (0, 1, 2):
            P, pred = preds(m, src, r); o = np.argsort(-P.max(1), kind="stable")[: int(round(c * B.nD))]
            acc.append(np.mean(pred[o] == B.Y[o])); mix.append(np.bincount(B.Y[o], minlength=4) / len(o))
        out[f"{c:.1f}"] = dict(accuracy=float(np.mean(acc)), accuracy_per_repeat=[round(x, 4) for x in acc],
                               true_class_mix={k: round(float(v), 3) for k, v in zip(B.CLASSES, np.mean(mix, 0))})
    return out


# ------------------------------------------------------------------ selective prediction (nested threshold; deployable)
TARGETS = (0.65, 0.70, 0.75)


def _inner(base, src, r, k):
    from research_experiments.phase_6.p6common import INNER6
    f = INNER6 / base / f"r{r}_k{k}.npz" if src == "ORIG" else B.INNER / f"{base}_{src}" / f"r{r}_k{k}.npz"
    z = np.load(f)
    return z["TR"], z["P"]


def _alphas(src, r):
    rid = f"P6A-c3-B1-v1__s0_r{r}" if src == "ORIG" else f"P6B-c3-{src}-v1__s0_r{r}"
    d = (B.ROOT / "research_experiments" / "phase_6a_normal_mild_lstm" / "results" / "experiments") if src == "ORIG" else B.EXP
    return json.loads((d / f"{rid}.json").read_text())["alpha_per_fold"]


def selective(m, src, target):
    """The model answers only when its confidence (max probability; the vote probability for C3) reaches a threshold.
    Per outer fold, the threshold is the lowest one at which the model's nested inner out-of-fold predictions on the
    fold's training portion (all of its ECGs, original labels) reach `target` accuracy with >= 20 answered ECGs; it is
    then applied unchanged to the outer test fold. No test-fold label is used to set it."""
    from research_experiments.phase_5.p5common import decide
    yt = y_echo() if src == "ECHO" else B.Y
    bases = {"mlpfusion": ("mlpfusion",), "vote": ("mlpfusion", "ecgflindist", "hdrclinlr"),
             "c3": ("mlpfusion", "ecgflindist", "hdrclinlr")}[m]
    reps, cl = [], []
    for r in (0, 1, 2):
        P, pred = preds(m, src, r); conf = P.max(1); fv = B.fold_vec(r); keep = np.zeros(B.nD, bool)
        al = _alphas(src, r) if m == "c3" else None
        for k in range(5):
            inn = [_inner(b, src, r, k) for b in bases]
            TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
            if m == "c3":
                kp = ~(_excl(src, r, k)[TR] if src != "ORIG" else np.zeros(len(TR), bool))
                prior = np.bincount(yt[TR][kp], minlength=4) / kp.sum()
                pin = decide(Pin, "prior", al[k], prior)
            else:
                pin = Pin.argmax(1)
            ci, ok = Pin.max(1), pin == B.Y[TR]
            ts = [t for t in np.unique(np.round(ci, 3)) if (ci >= t).sum() >= 20 and ok[ci >= t].mean() >= target]
            keep[fv == k] = conf[fv == k] >= (ts[0] if ts else 1.01)
            sel = (fv == k) & keep
            if sel.any():
                cl.append(float(np.mean(pred[sel] == B.Y[sel])))
        y, p = B.Y[keep], pred[keep]
        reps.append(dict(coverage=float(keep.mean()), accuracy=float(np.mean(p == y)) if keep.any() else None,
                         coverage_by_class={c: round(float(keep[B.Y == i].mean()), 3) for i, c in enumerate(B.CLASSES)},
                         recall_covered={c: (round(float(np.mean(p[y == i] == i)), 3) if (y == i).any() else None)
                                         for i, c in enumerate(B.CLASSES)},
                         predicted_mix={c: int((p == i).sum()) for i, c in enumerate(B.CLASSES)}))
    ok = [x for x in reps if x["accuracy"] is not None]
    return dict(model=m, trained=src, target=target, coverage=float(np.mean([x["coverage"] for x in reps])),
                accuracy=float(np.mean([x["accuracy"] for x in ok])) if ok else None,
                accuracy_ci95_corrected=corr_ci(cl) if len(cl) > 2 else None, n_fold_cells=len(cl), per_repeat=reps)


def report(configs=ALL, tag=""):
    F = flags(); hc = hc_mask(); ye = y_echo()
    if tag == "_nf":
        subsets = {"all_837": np.ones(B.nD, bool), "NF_set": ~F.NF.values, "NFS_set": ~F.NFS.values,
                   "not_NFS": F.NFS.values, "NF_clear": ~F.NF.values & ~F.near_boundary.values, "HC": hc}
    else:
        subsets = {"all_837": np.ones(B.nD, bool), "kept_MMC": ~F.MMC.values, "removed_MMC": F.MMC.values,
                   "kept_MMNC": ~F.MMNC.values, "removed_MMNC": F.MMNC.values, "HC": hc}
    out = dict(counts=counts(), subsets={k: int(v.sum()) for k, v in subsets.items()}, results=[], comparisons=[])
    for c in configs:
        labs = [("ORIG", None)] + ([("ECHO", ye)] if c == "ECHO" else [])
        for sname, mask in subsets.items():
            for lname, y in labs:
                for m, label in MODELS:
                    for src in ("ORIG", c):
                        out["results"].append(dict(config=c, evaluated_on=sname, labels=lname, n_ecgs=int(mask.sum()), model=label,
                                                   trained=src, **summary(m, src, mask, y)))
                    for metric in ("accuracy", "macro_f1", "recall_Mild"):
                        out["comparisons"].append(dict(evaluated_on=sname, labels=lname, n_ecgs=int(mask.sum()),
                                                       **paired(m, c, mask, metric, y)))
    out["coverage_curve"] = {f"{label} [{src}]": coverage_curve(m, src) for m, label in MODELS
                             for src in ("ORIG",) + tuple(configs)}
    out["selective"] = [selective(m, src, t) for m, _ in MODELS for src in ("ORIG",) + tuple(configs) for t in TARGETS]
    adj = holm([x["p_corrected_t"] for x in out["comparisons"]])
    for x, a in zip(out["comparisons"], adj):
        x["p_holm"] = float(a)
        x["genuine_improvement"] = bool(x["delta"] > 0 and a < 0.05 and x["same_sign_all_repeats"])
    (B.guard(B.RES) / f"mmclean{tag}_exploratory.json").write_text(json.dumps(out, indent=1))
    rows = pd.DataFrame(out["results"])
    rows["accuracy_ci95_corrected"] = rows.accuracy_ci95_corrected.map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    rows.drop(columns=["confusion_mean", "accuracy_per_repeat"]).to_csv(B.guard(B.RES / f"mmclean{tag}_results.csv"), index=False,
                                                                        float_format="%.4f")
    pd.DataFrame(out["comparisons"]).to_csv(B.guard(B.RES / f"mmclean{tag}_comparisons.csv"), index=False, float_format="%.4f")
    pd.DataFrame([{k: v for k, v in x.items() if k != "per_repeat"} for x in out["selective"]]).to_csv(
        B.guard(B.RES / f"mmclean{tag}_selective.csv"), index=False, float_format="%.4f")
    plot(out, tag)
    return out


def plot(out, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    R_ = [z for z in out["results"] if z["labels"] == "ORIG"]
    srcs = ["ORIG"] + [c for c in ALL if any(z["config"] == c for z in R_)]
    cols = dict(zip(srcs, ["#9aa5b1", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2", "#eeca3b"]))
    panels = (("all_837", "NFS_set", "NF_clear") if tag == "_nf" else ("all_837", "kept_MMNC", "HC"))
    titles = (("All 837 ECGs (original labels; primary)", "Noise-free link + all Severe (NFS set; descriptive)",
               "Noise-free link, not near a boundary (descriptive)") if tag == "_nf" else
              ("All 837 ECGs (original labels; primary)", "MMNC-kept ECGs (descriptive)", "HC subset (descriptive)"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, sname, title in zip(axes, panels, titles):
        x = 0; ticks, tl = [], []
        for m, label in MODELS:
            for j, src in enumerate(srcs):
                rr = next(z for z in R_ if z["evaluated_on"] == sname and z["model"] == label and z["trained"] == src
                          and (src != "ORIG" or z["config"] == srcs[1]))
                lo, hi = rr["accuracy_ci95_corrected"]
                ax.bar(x + 0.16 * j, rr["accuracy"], 0.15, color=cols[src], yerr=[[rr["accuracy"] - lo], [hi - rr["accuracy"]]],
                       capsize=1.5)
                ax.text(x + 0.16 * j, hi + 0.01, f"{rr['accuracy']:.3f}", ha="center", fontsize=6, rotation=90)
            ticks.append(x + 0.08 * (len(srcs) - 1)); tl.append(f"{label.split(' (')[0]}\n(n={rr['n_ecgs']})")
            x += 0.16 * len(srcs) + 0.35
        ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=8); ax.set_title(title, fontsize=9)
        ax.axhline(0.7, ls=":", c="k", lw=0.8); ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy, original labels (mean of 3 repeats, corrected 95 % CI)")
    axes[0].legend(handles=[Patch(color=v, label=f"trained: {k}") for k, v in cols.items()], fontsize=7, loc="upper left")
    fig.suptitle("Top-3 recipes trained on cleaned / reduced / echo-relabelled data; development CV, exploratory", fontsize=10)
    fig.tight_layout(); B.guard(B.PLOTS).mkdir(parents=True, exist_ok=True)
    fig.savefig(B.PLOTS / f"mmclean{tag}_accuracy.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, cur in out["coverage_curve"].items():
        if not name.startswith("C3"):
            continue
        xs = [float(k) for k in cur]; ys = [v["accuracy"] for v in cur.values()]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.6 if "[ORIG]" in name else 0.9, ls="-" if "[ORIG]" in name else "--", label=name)
    ax.axhline(0.7, ls=":", c="k", lw=0.8); ax.invert_xaxis()
    ax.set_xlabel("coverage (share of the 837 ECGs answered; the rest referred to echo)")
    ax.set_ylabel("accuracy on answered ECGs, original labels (mean of 3 repeats)")
    ax.set_title("C3: accuracy vs coverage, ranked by model confidence (descriptive)", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(B.PLOTS / f"mmclean{tag}_coverage_curve.png", dpi=150); plt.close(fig)


def main():
    warnings.filterwarnings("ignore", message="Only one class is present")   # per-fold AUROC on small subsets (unused)
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true"); ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--tag", default="", help='"_nf" for the noise-free batch (separate output files)')
    a = ap.parse_args()
    if a.counts:
        print(json.dumps(counts(), indent=1))
    if a.run:
        B.table()
        with Pool(len(a.configs)) as pool:
            for c in pool.imap_unordered(work, a.configs):
                print("done", c, flush=True)
    if a.report:
        out = report([c for c in ALL if c in a.configs], a.tag)
        for x in out["results"]:
            print(f"{x['config']:5s} {x['evaluated_on']:12s} {x['labels']:4s} n={x['n_ecgs']:3d} {x['model'][:10]:10s} "
                  f"trained {x['trained']:5s} "
                  f"acc {x['accuracy']:.3f} {np.round(x['accuracy_ci95_corrected'], 3).tolist()} macroF1 {x['macro_f1']:.3f} "
                  f"Mild rec {x['recall_Mild']:.3f} min rec {x['min_recall']:.3f}")


if __name__ == "__main__":
    main()
