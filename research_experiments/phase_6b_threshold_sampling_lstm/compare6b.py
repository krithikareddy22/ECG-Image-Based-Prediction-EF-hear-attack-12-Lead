"""Phase 6B section 4 (+ Amendment 1): consolidated comparison, statistics, conclusions and plots.

Every model is reduced to seed-combined per-repeat predictions (probability mean over seeds; stored decision-rule
predictions for rule models) on all 837 development ECGs and scored against the ORIGINAL labels (primary), and against
the T50 and T55 labels. Paired tests use the identical 15 (repeat, fold) cells (corrected repeated-CV t-test); Holm
within each family and metric; cluster bootstrap + exact McNemar (repeat 0) for the top candidates of each family.
Outputs (aggregates only):
  PHASE6B_MODEL_COMPARISON.csv            one row per model (means over repeats, CIs, eligibility, time, parameters)
  results/phase6b_results_per_repeat.csv  per model x repeat x label set
  results/phase6b_comparisons.csv         every paired test
  results/phase6b_summary.json            conclusions, decompositions, bootstrap / McNemar, 60 % assessment
  plots/*.png
    python -m research_experiments.phase_6b_threshold_sampling_lstm.compare6b
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from research_experiments.phase_6.p6common import EXP6, RUNS6  # noqa: E402
from research_experiments.phase_6a_normal_mild_lstm import p6acommon as A  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm.p6bcommon import (  # noqa: E402
    CLASSES, EXP, P6B, PAT, PLOTS, RES, ROOT, RUNS, SEEDS, Y, corr_ci, corr_t, fold_vec, guard, holm, load_oof_csv, metrics,
    y_def)

P2RES = ROOT / "research_experiments" / "phase_2" / "results"
A6 = {"ORIG": "B1", "T50": "B2", "XUNC": "B3", "A270X": "A270"}           # 6A run names of the re-used configurations
CONFIGS = ("ORIG", "T50", "T55", "XUNC", "A270X", "N300", "N270", "N240", "N200")
DEFN = {"T50": "T50", "T55": "T55"}                                       # training-label definition of a configuration
NORMAL_N = {"ORIG": 307.5, "N300": 300, "N270": 270, "N240": 240, "N200": 200}
LABELS = {"ORIG": Y, "T50": y_def("T50"), "T55": y_def("T55")}
METRICS = ("accuracy", "macro_f1", "f1_Mild", "recall_Mild")
REF = ("C1 Phase 3 vote ensemble", "C3 Phase 5 selected (vote + nested prior rule)")


def sfx(r):
    return "" if r == 0 else f"_r{r}"


def rid_path(rid):
    for base in (RUNS, A.RUNS, RUNS6):
        if (base / rid / "oof.csv").exists():
            return base / rid / "oof.csv"
    return None


def agg_of(rid):
    for base in (EXP, A.EXP, EXP6):
        if (base / f"{rid}.json").exists():
            return json.loads((base / f"{rid}.json").read_text())
    return None


def lstm_fixed():
    f = A.CONF / "lstm_fixed.json"
    return json.loads(f.read_text())["lstm_fixed"] if f.exists() else None


def cfg_rid(prefix, model, cfg):
    return f"P6A-{model}-{A6[cfg]}-v1" if cfg in A6 else f"P6B-{model}-{cfg}-v1"


# ------------------------------------------------------------------ registry
def models():
    """name -> spec(rids(r) | paths(r), family, config, definition, arch, rule, eligible, note)."""
    M = {}

    def add(name, rids=None, family="existing", config="ORIG", arch="", rule=False, eligible=True, note="", paths=None, timing=None):
        get = paths or (lambda r: [rid_path(x) for x in rids(r)])
        ok = all(all(p is not None and Path(p).exists() for p in get(r)) for r in (0, 1, 2))
        if not ok:
            return
        M[name] = dict(paths=get, rids=rids, family=family, config=config, definition=DEFN.get(config, "ORIG"), arch=arch, rule=rule,
                       eligible=eligible, note=note, timing=timing)

    add("ECGFounder frozen, linear", paths=lambda r: [P2RES / f"ecgf_aligned_f6_lin{sfx(r)}" / "oof.csv"], arch="ECGFounder f6 + linear",
        note="deterministic")
    add("ECGFounder fine-tuned (Phase 6 A1)", lambda r: [f"P6A1-ft-s6-joint-v1__s42_r{r}"], arch="ECGFounder stage 6 fine-tuned + joint head",
        eligible=False, note="seed 42 only (regenerated locally)")
    add("Header + ECGFounder MLP fusion", lambda r: [f"P5A-import-mlpfusion-v1__s{s}_r{r}" for s in SEEDS], arch="MLP fusion")
    add(REF[0], lambda r: [f"P5A-import-vote-v1__s0_r{r}"], arch="vote (MLP fusion, ECGF linear, header+clinical LR)")
    add("C2 Phase 3 calibrated MLP", lambda r: [f"P5A-calmlp-protected-v1__s0_r{r}"], arch="MLP fusion + nested prior rule", rule=True)
    add(REF[1], lambda r: [f"P5D-E5-vote-prior-protected-v1__s0_r{r}"], arch="vote + nested prior rule", rule=True)
    fx = lstm_fixed()
    for c in CONFIGS:
        add(f"MLP fusion [{c}]", lambda r, c=c: [f"{cfg_rid('', 'mlpfusion', c)}__s{s}_r{r}" for s in SEEDS], "config", c, "MLP fusion")
        add(f"Vote, C1 recipe [{c}]", lambda r, c=c: [f"{cfg_rid('', 'vote', c)}__s0_r{r}"], "config", c, "vote")
        add(f"C3 recipe [{c}]", lambda r, c=c: [f"{cfg_rid('', 'c3', c)}__s0_r{r}"], "config", c, "vote + nested prior rule", rule=True)
        if fx:
            three = all(rid_path(f"{cfg_rid('', 'lstm' + fx, c)}__s{s}_r{r}") for s in SEEDS for r in (0, 1, 2))
            seeds = SEEDS if three else (42,)
            add(f"LSTM-fixed {fx} [{c}]" if three else f"LSTM-fixed {fx} seed 42 [{c}]",
                rids=lambda r, c=c, seeds=seeds: [f"{cfg_rid('', 'lstm' + fx, c)}__s{s}_r{r}" for s in seeds],
                family="lstm" if c == "ORIG" else "config", config=c, arch=f"LSTM {fx}", eligible=three,
                note="" if three else "seed 42 only (compute rule)", timing="runs")
            if not three and c != "ORIG":
                add(f"LSTM-fixed {fx} seed 42 [ORIG]", lambda r: [f"P6A-lstm{fx}-B1-v1__s42_r{r}"], "lstm", "ORIG", f"LSTM {fx}",
                    eligible=False, note="seed-42 comparator for the compute rule", timing="runs")
    for v in ("L0", "L1", "L2", "L3", "L4"):
        add(f"LSTM {v} [ORIG]", lambda r, v=v: [f"P6A-lstm{v}-B1-v1__s{s}_r{r}" for s in SEEDS], "lstm", "ORIG", f"LSTM {v}", timing="runs")
    add("LSTM-sel (nested) [ORIG]", lambda r: [f"P6A-lstmsel-B1-v1__s{s}_r{r}" for s in SEEDS], "lstm", "ORIG", "LSTM nested selection over L0-L4",
        timing="lstmsel")
    for m, arch in (("L5", "CNN-LSTM"), ("L6", "CNN-LSTM + header"), ("F1", "LSTM-fixed + header + ECGFounder f6 (early fusion)"),
                    ("F2", "CNN-LSTM + header + ECGFounder f6 (early fusion)")):
        add(f"{m} {arch} [ORIG]", lambda r, m=m: [f"P6B-{m}-ORIG-v1__s{s}_r{r}" for s in SEEDS], "lstm", "ORIG", arch, timing="runs")
    add("F3 vote + LSTM-sel (late fusion) [ORIG]", lambda r: [f"P6B-F3-ORIG-v1__s0_r{r}"], "fusion", "ORIG", "C1 members + LSTM-sel vote", timing="F3")
    add("F4 F3 + nested prior rule [ORIG]", lambda r: [f"P6B-F4-ORIG-v1__s0_r{r}"], "fusion", "ORIG", "late fusion + C3 rule", rule=True, timing="F4")
    return M


def preds(spec, r):
    Ps, pr = [], []
    for p in spec["paths"](r):
        P, pred = load_oof_csv(p); Ps.append(P); pr.append(pred)
    P = np.mean(Ps, 0)
    return P, (pr[0] if spec["rule"] else P.argmax(1))


def cells(y, P, pred, r):
    fv = fold_vec(r)
    return [metrics(y[fv == k], P[fv == k], pred[fv == k]) for k in range(5)]


# ------------------------------------------------------------------ time / parameters (new models: from their aggregates)
def timing(name, spec):
    t = spec["timing"]
    if t is None:
        return dict(n_params=None, train_minutes_per_run=None, infer_ms_per_ecg=None)
    if t == "runs":
        ag = [agg_of(x) for r in (0, 1, 2) for x in spec["rids"](r)]
        return dict(n_params=ag[0].get("n_params"), train_minutes_per_run=float(np.mean([a["train_minutes"] for a in ag])),
                    infer_ms_per_ecg=float(np.mean([a["infer_ms_per_ecg"] for a in ag])))
    var = [f"P6A-lstm{v}-B1-v1__s{s}_r{r}" for v in ("L0", "L1", "L2", "L3", "L4") for s in SEEDS for r in (0, 1, 2)]
    ag = {x: agg_of(x) for x in var}
    sel_cost = float(np.sum([a["train_minutes"] for a in ag.values()]) / 9)        # all five variants trained per (seed, repeat)
    sel = json.loads((A.RES / "lstm_selection.json").read_text())
    chosen = list(sel["choice_per_fold"].values())
    out = dict(n_params=f"{min(sel['params'][v] for v in chosen)}-{max(sel['params'][v] for v in chosen)}",
               train_minutes_per_run=sel_cost,
               infer_ms_per_ecg=float(np.mean([a["infer_ms_per_ecg"] for x, a in ag.items() if x.split("-")[1][4:] in set(chosen)])))
    if t in ("F3", "F4"):
        out["n_params"] = f"LSTM part {out['n_params']} + C1 members"
        out["note"] = "train time = LSTM-sel part (C1 members re-used)"
        if t == "F4":
            out["train_minutes_per_run"] += float(np.mean([agg_of(f"P6B-F4-ORIG-v1__s0_r{r}").get("lstm_inner_train_minutes", 0) for r in (0, 1, 2)]))
    return out


# ------------------------------------------------------------------ summaries
def summarise(name, spec, per_rep):
    out = dict(model=name, family=spec["family"], config=spec["config"], definition=spec["definition"], architecture=spec["arch"],
               eligible=spec["eligible"], note=spec["note"])
    own = LABELS[spec["definition"]]
    rows = {k: [] for k in LABELS}; cm, cm_own, cl = [], [], []
    for r in (0, 1, 2):
        P, pred = preds(spec, r)
        for lab, y in LABELS.items():
            m = metrics(y, P, pred); rows[lab].append(m)
            per_rep.append(dict(model=name, repeat=r, scored_on=lab, **m))
        cm.append(np.bincount(4 * Y + pred, minlength=16).reshape(4, 4)); cm_own.append(np.bincount(4 * own + pred, minlength=16).reshape(4, 4))
        cl.extend([c["accuracy"] for c in cells(Y, P, pred, r)])
    for k in rows["ORIG"][0]:
        out[k] = float(np.mean([m[k] for m in rows["ORIG"]]))
    out["accuracy_sd_repeats"] = float(np.std([m["accuracy"] for m in rows["ORIG"]]))
    out["accuracy_ci95_corrected"] = corr_ci(cl)
    out["admissible"] = bool(min(out[f"recall_{c}"] for c in CLASSES) >= 0.15)
    for lab in ("T50", "T55"):
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "min_recall"):
            out[f"{k}_vs_{lab}"] = float(np.mean([m[k] for m in rows[lab]]))
    if spec["definition"] != "ORIG":
        cl_own = []
        for r in (0, 1, 2):
            P, pred = preds(spec, r); cl_own.extend([c["accuracy"] for c in cells(own, P, pred, r)])
        out["accuracy_own_ci95_corrected"] = corr_ci(cl_own)
        out["admissible_own"] = bool(min(np.mean([m[f"recall_{c}"] for m in rows[spec["definition"]]]) for c in CLASSES) >= 0.15)
    out["confusion_mean"] = np.mean(cm, 0).round(1).tolist()
    out["confusion_mean_own_labels"] = np.mean(cm_own, 0).round(1).tolist()
    out.update(timing(name, spec))
    return out


def paired(a, b, M, metric, y=Y):
    d, dr = [], []
    for r in (0, 1, 2):
        Pa, pa = preds(M[a], r); Pb, pb = preds(M[b], r)
        d += [x[metric] - z[metric] for x, z in zip(cells(y, Pa, pa, r), cells(y, Pb, pb, r))]
        dr.append(metrics(y, Pa, pa)[metric] - metrics(y, Pb, pb)[metric])
    _, p = corr_t(d)
    return dict(candidate=a, comparator=b, metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr],
                ci95_corrected_delta=corr_ci(d), p_corrected_t=p,
                same_sign_all_repeats=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)), n_cells=len(d))


def boot_mcnemar(a, b, M, y=Y, n=2000):
    Pa, pa = preds(M[a], 0); Pb, pb = preds(M[b], 0)
    ca, cb = pa == y, pb == y
    groups = [np.flatnonzero(PAT == p) for p in pd.unique(PAT)]
    rng = np.random.default_rng(0); bs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        bs.append(ca[idx].mean() - cb[idx].mean())
    b01, b10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = binomtest(b10, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return dict(boot_delta_acc_r0=float(ca.mean() - cb.mean()), boot95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                mcnemar_p_r0=float(p), discordant_candidate_right_comparator_right=[b10, b01])


def families(M):
    fx = lstm_fixed(); F = {"a": [], "b": [], "c": []}
    for c in [m for m in M if M[m]["family"] in ("lstm", "fusion") and M[m]["config"] == "ORIG" and not m.startswith("LSTM L")
              and "seed 42" not in m]:
        for ref in REF:
            if ref in M:
                F["a"].append((c, ref, Y))
    bases = ["MLP fusion", "Vote, C1 recipe", "C3 recipe"] + ([f"LSTM-fixed {fx}"] if fx else [])
    for base in bases:
        for c in CONFIGS[1:]:
            cand = next((m for m in (f"{base} [{c}]", f"{base} seed 42 [{c}]") if m in M), None)
            comp = f"{base} seed 42 [ORIG]" if cand and "seed 42" in cand else f"{base} [ORIG]"
            if cand and comp in M:
                F["b"].append((cand, comp, Y))
                if c in DEFN:
                    F["c"].append((cand, comp, LABELS[c]))
    return F


# ------------------------------------------------------------------ plots
COL = {"existing": "#4C78A8", "config": "#F58518", "lstm": "#54A24B", "fusion": "#B279A2"}


def bars(S, metric, fname, title):
    T = S.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9.5, 0.28 * len(T) + 1.6))
    ax.barh(range(len(T)), T[metric], color=[COL[f] for f in T.family])
    if metric == "accuracy":
        for i, ci in enumerate(T.accuracy_ci95_corrected):
            ax.plot(ci, [i, i], color="k", lw=0.8)
        ax.axvline(0.60, color="r", ls="--", lw=1)
    ax.set_yticks(range(len(T)))
    ax.set_yticklabels([m + ("" if a else "  (not admissible)") + ("" if e else "  (not eligible)")
                        for m, a, e in zip(T.model, T.admissible, T.eligible)], fontsize=6.5)
    lo = max(0.0, float(T[metric].min()) - 0.05)
    ax.set_xlim(lo, max(0.66, float(T[metric].max()) + 0.03))
    ax.set_xlabel(f"{metric} on the ORIGINAL labels (mean over 3 repeats)" + ("; bars: corrected 95 % CI; dashed: 60 %" if metric == "accuracy" else ""))
    ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in COL.items()], fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=150); plt.close(fig)


def plots(S, key):
    guard(PLOTS).mkdir(parents=True, exist_ok=True)
    bars(S, "accuracy", "accuracy_comparison.png", "Phase 6B: four-class accuracy (original labels)")
    bars(S, "macro_f1", "macro_f1_comparison.png", "Phase 6B: macro-F1 (original labels)")
    T = S.set_index("model").loc[[k for k in key if k in set(S.model)]]
    fig, axs = plt.subplots(1, 3, figsize=(16, 0.42 * len(T) + 2), sharey=True)
    for ax, what in zip(axs, ("precision", "recall", "f1")):
        h = 0.2
        for j, c in enumerate(CLASSES):
            ax.barh(np.arange(len(T)) + (j - 1.5) * h, T[f"{what}_{c}"], h, label=c)
        ax.set_title(f"{what} per class (original labels)"); ax.set_xlim(0, 1)
        if what == "recall":
            ax.axvline(0.15, color="k", ls=":", lw=1)
    axs[0].set_yticks(range(len(T))); axs[0].set_yticklabels(T.index, fontsize=7); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(PLOTS / "per_class_precision_recall_f1.png", dpi=150); plt.close(fig)
    n = len(T); cols = 4; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3.6 * rows))
    for ax, m in zip(np.ravel(axs), list(T.index) + [None] * (rows * cols - n)):
        if m is None:
            ax.axis("off"); continue
        own = T.loc[m, "definition"] != "ORIG"
        C = np.array(T.loc[m, "confusion_mean_own_labels" if own else "confusion_mean"])
        ax.imshow(C / C.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(["N", "Mi", "Mo", "S"]); ax.set_yticklabels(["N", "Mi", "Mo", "S"])
        ax.set_title(m, fontsize=7); ax.set_xlabel("predicted"); ax.set_ylabel(f"{T.loc[m, 'definition']} label")
    fig.suptitle("Confusion matrices (mean counts over repeats; colour = row-normalised)"); fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrices.png", dpi=150); plt.close(fig)
    # threshold sensitivity: ORIG / T50 / T55-trained, scored on original and on own labels
    fx = lstm_fixed(); bases = ["MLP fusion", "Vote, C1 recipe", "C3 recipe"] + ([f"LSTM-fixed {fx}"] if fx else [])
    Si = S.set_index("model")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for b_i, base in enumerate(bases):
        xs = np.arange(3) + (b_i - 1.5) * 0.2
        get = lambda c, col: Si.loc[f"{base} [{c}]", col] if f"{base} [{c}]" in Si.index else np.nan
        axs[0].bar(xs, [get(c, "accuracy") for c in ("ORIG", "T50", "T55")], 0.2, label=base)
        axs[1].bar(xs, [get("ORIG", "accuracy"), get("T50", "accuracy_vs_T50"), get("T55", "accuracy_vs_T55")], 0.2, label=base)
        axs[2].bar(xs, [get("ORIG", "recall_Mild"), get("T50", "recall_Mild"), get("T55", "recall_Mild")], 0.2, label=base)
    for ax, t in zip(axs, ("accuracy on the ORIGINAL labels", "accuracy on the model's OWN definition", "Mild recall (original labels)")):
        ax.set_xticks(range(3)); ax.set_xticklabels(["ORIG (M>52/F>=54)", "T50 (>=50)", "T55 (>=55)"]); ax.set_title(t, fontsize=9)
        ax.set_ylim(0, 0.7 if "recall" not in t else 0.6)
    axs[2].axhline(0.15, color="k", ls=":", lw=1); axs[0].legend(fontsize=7)
    fig.suptitle("Normal/Mild threshold sensitivity (training definition on the x-axis)"); fig.tight_layout()
    fig.savefig(PLOTS / "threshold_sensitivity.png", dpi=150); plt.close(fig)
    # Normal-size curve
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for base in bases:
        xs, rows_ = [], []
        for c, nn in NORMAL_N.items():
            nm = next((m for m in (f"{base} [{c}]", f"{base} seed 42 [{c}]") if m in Si.index), None)
            if nm:
                xs.append(nn); rows_.append(Si.loc[nm])
        if not xs:
            continue
        for ax, col in zip(np.ravel(axs), ["accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASSES]):
            ax.plot(xs, [r_[col] for r_ in rows_], "o-", label=base)
            if col == "accuracy":
                ax.fill_between(xs, [r_["accuracy_ci95_corrected"][0] for r_ in rows_], [r_["accuracy_ci95_corrected"][1] for r_ in rows_], alpha=0.08)
    for ax, col in zip(np.ravel(axs), ["accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASSES]):
        ax.set_title(col.replace("_", " ") + " (original labels)", fontsize=9)
        ax.set_xlabel("Normal ECGs in the outer-training portion (307-308 = original)", fontsize=8)
        ax.set_xticks(list(NORMAL_N.values())); ax.set_xticklabels(["307", "300", "270", "240", "200"], fontsize=8); ax.invert_xaxis()
    np.ravel(axs)[0].legend(fontsize=7)
    fig.suptitle("Normal training-size curve (inner-val and outer test unchanged)"); fig.tight_layout()
    fig.savefig(PLOTS / "normal_size_curve.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    M = models()
    per_rep = []
    S = pd.DataFrame([summarise(n, s, per_rep) for n, s in M.items()])
    guard(RES).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_rep).to_csv(RES / "phase6b_results_per_repeat.csv", index=False, float_format="%.4f")
    adm = S.set_index("model").admissible.to_dict(); elig = S.set_index("model").eligible.to_dict()
    adm_own = S.set_index("model").get("admissible_own", pd.Series(dtype=object)).to_dict()
    tests = []
    for fam, pairs in families(M).items():
        for metric in METRICS:
            rows = [dict(paired(a, b, M, metric, y), family=fam, scored_on=("ORIG" if y is Y else M[a]["definition"])) for a, b, y in pairs]
            for x, h in zip(rows, holm([x["p_corrected_t"] for x in rows]) if rows else []):
                x["p_holm"] = float(h)
                ok_adm = adm_own.get(x["candidate"]) if fam == "c" else adm[x["candidate"]]
                x["candidate_admissible"] = bool(ok_adm); x["candidate_eligible"] = bool(elig[x["candidate"]])
                x["genuine_improvement"] = bool(x["delta"] > 0 and h < 0.05 and x["same_sign_all_repeats"] and ok_adm and elig[x["candidate"]])
            tests += rows
    for fam in ("a", "b", "c"):
        top = sorted([x for x in tests if x["family"] == fam and x["metric"] == "accuracy"], key=lambda x: -x["delta"])[:3]
        for x in top:
            x.update(boot_mcnemar(x["candidate"], x["comparator"], M, LABELS[x["scored_on"]]))
    TT = pd.DataFrame(tests)
    TT.to_csv(RES / "phase6b_comparisons.csv", index=False, float_format="%.4f")
    # decomposition (definition effect / training effect)
    Si = S.set_index("model"); fx = lstm_fixed(); dec = []
    for base in ["MLP fusion", "Vote, C1 recipe", "C3 recipe"] + ([f"LSTM-fixed {fx}"] if fx else []):
        for d in ("T50", "T55"):
            a, b = f"{base} [{d}]", f"{base} [ORIG]"
            if a in Si.index and b in Si.index:
                t = next((x for x in tests if x["family"] == "c" and x["candidate"] == a and x["metric"] == "accuracy"), {})
                dec.append(dict(model=base, definition=d, ORIG_model_on_original=Si.loc[b, "accuracy"], ORIG_model_on_alt=Si.loc[b, f"accuracy_vs_{d}"],
                                alt_model_on_alt=Si.loc[a, f"accuracy_vs_{d}"], alt_model_on_original=Si.loc[a, "accuracy"],
                                definition_effect=Si.loc[b, f"accuracy_vs_{d}"] - Si.loc[b, "accuracy"],
                                training_effect_on_alt=Si.loc[a, f"accuracy_vs_{d}"] - Si.loc[b, f"accuracy_vs_{d}"],
                                training_effect_p_holm=t.get("p_holm"), mild_recall_ORIG=Si.loc[b, "recall_Mild"],
                                mild_recall_alt_model_on_alt=Si.loc[a, f"recall_Mild_vs_{d}"], mild_recall_alt_model_on_original=Si.loc[a, "recall_Mild"]))
    # conclusions
    orig_trained = S[(S.definition == "ORIG") & S.admissible & S.eligible].sort_values(["accuracy", "macro_f1"], ascending=False)
    best1 = orig_trained.iloc[0].to_dict() if len(orig_trained) else None
    alt = S[S.definition.isin(["T50", "T55"])].copy()
    if len(alt):
        alt["acc_own"] = [r_[f"accuracy_vs_{r_['definition']}"] for _, r_ in alt.iterrows()]
        alt["mf1_own"] = [r_[f"macro_f1_vs_{r_['definition']}"] for _, r_ in alt.iterrows()]
        alt = alt[alt.admissible_own.fillna(False).astype(bool) & alt.eligible].sort_values(["acc_own", "mf1_own"], ascending=False)
    best2 = alt.iloc[0].to_dict() if len(alt) else None

    def sixty(acc, ci):
        return dict(mean=acc, ci95=ci, reached_mean_ge_0_60=bool(acc >= 0.60), robust_ci_lower_ge_0_60=bool(ci[0] >= 0.60))
    keep = ["model", "config", "definition", "accuracy", "accuracy_ci95_corrected", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild",
            "min_recall", "qwk"]
    summary = dict(
        n_models=len(S),
        conclusion_1_original_labels=dict(best={k: best1[k] for k in keep} if best1 else None,
                                          sixty_percent=sixty(best1["accuracy"], best1["accuracy_ci95_corrected"]) if best1 else None,
                                          genuine_improvements_vs_C1_C3=[x for x in tests if x["family"] == "a" and x["genuine_improvement"]],
                                          genuine_improvements_configs_vs_ORIG=[x for x in tests if x["family"] == "b" and x["genuine_improvement"]]),
        conclusion_2_alternative_definition=dict(
            best={**{k: best2[k] for k in keep}, "accuracy_own_labels": best2["acc_own"], "macro_f1_own_labels": best2["mf1_own"],
                  "accuracy_own_ci95_corrected": best2["accuracy_own_ci95_corrected"]} if best2 else None,
            sixty_percent_own_labels=sixty(best2["acc_own"], best2["accuracy_own_ci95_corrected"]) if best2 else None,
            genuine_training_effects=[x for x in tests if x["family"] == "c" and x["genuine_improvement"]],
            note="scored on its own definition; NOT comparable with conclusion 1"),
        decomposition=dec, bootstrap_mcnemar=[x for x in tests if "boot95" in x],
        n_tests={f"{f}/{m}": int(n) for (f, m), n in TT.groupby(["family", "metric"]).size().items()} if len(TT) else {})
    (RES / "phase6b_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    cols = ["model", "family", "config", "definition", "architecture", "eligible", "admissible", "accuracy", "accuracy_ci95_corrected",
            "accuracy_sd_repeats", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "precision_Mild", "min_recall", "qwk", "macro_auroc"] + \
           [f"{m}_{c}" for c in CLASSES for m in ("precision", "recall", "f1")] + \
           ["accuracy_vs_T50", "macro_f1_vs_T50", "recall_Mild_vs_T50", "accuracy_vs_T55", "macro_f1_vs_T55", "recall_Mild_vs_T55",
            "n_params", "train_minutes_per_run", "infer_ms_per_ecg", "confusion_mean", "note"]
    S2 = S.reindex(columns=cols).copy()
    S2["accuracy_ci95_corrected"] = S2.accuracy_ci95_corrected.map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
    S2["confusion_mean"] = S2.confusion_mean.map(json.dumps)
    S2.sort_values(["family", "accuracy"], ascending=[True, False]).to_csv(P6B / "PHASE6B_MODEL_COMPARISON.csv", index=False, float_format="%.4f")
    key = [m for m in [REF[1], REF[0], "LSTM-sel (nested) [ORIG]", f"LSTM-fixed {fx} [ORIG]", "L5 CNN-LSTM [ORIG]", "L6 CNN-LSTM + header [ORIG]",
                       "F1 LSTM-fixed + header + ECGFounder f6 (early fusion) [ORIG]", "F2 CNN-LSTM + header + ECGFounder f6 (early fusion) [ORIG]",
                       "F3 vote + LSTM-sel (late fusion) [ORIG]", "F4 F3 + nested prior rule [ORIG]",
                       best2["model"] if best2 else None, best1["model"] if best1 else None] if m and m in set(S.model)]
    plots(S, list(dict.fromkeys(key)))
    pd.set_option("display.width", 250)
    print(S[["model", "accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "min_recall", "admissible", "eligible"]]
          .sort_values("accuracy", ascending=False).round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("conclusion_1_original_labels", "conclusion_2_alternative_definition")}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
