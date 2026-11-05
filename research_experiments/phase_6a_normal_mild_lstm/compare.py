"""Phase 6A sections 6-8: consolidated comparison, statistics, complementarity, plots.

Every model is reduced to seed-combined per-repeat predictions (probability mean over seeds; stored decision-rule
predictions for C2 / C3-type models) on all 837 development ECGs, scored against the ORIGINAL labels (primary) and,
for label-configuration analyses, also against the B2 labels. Paired tests use the identical 15 (repeat, fold) cells.
Outputs (aggregates only): PHASE6A_MODEL_COMPARISON.csv, results/phase6a_comparisons.csv, results/phase6a_summary.json,
plots/*.png.
    python -m research_experiments.phase_6a_normal_mild_lstm.compare
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

from research_experiments.phase_6.p6common import RUNS6  # noqa: E402
from research_experiments.phase_6a_normal_mild_lstm.p6acommon import (  # noqa: E402
    CLASSES, CONF, P6A, PAT, PLOTS, RES, ROOT, RUNS, SEEDS, Y, corr_ci, corr_t, fold_vec, guard, holm, label_table,
    load_oof_csv, metrics)

P2RES = ROOT / "research_experiments" / "phase_2" / "results"
YB2 = None


def src6(rid):
    return RUNS6 / rid / "oof.csv"


def src6a(rid):
    return RUNS / rid / "oof.csv"


def sfx(r):
    return "" if r == 0 else f"_r{r}"


def lstm_fixed():
    f = CONF / "lstm_fixed.json"
    return json.loads(f.read_text())["lstm_fixed"] if f.exists() else None


def models():
    """name -> dict(paths per repeat (list over seeds), rule (use stored pred), family, config, arch)."""
    M = {}
    def add(name, paths, family, config="B1", arch="", rule=False, repeats=(0, 1, 2), note=""):
        if all(all(Path(p).exists() for p in paths(r)) for r in repeats):
            M[name] = dict(paths=paths, family=family, config=config, arch=arch, rule=rule, repeats=repeats, note=note)
    add("Header + ResNet-18 (original default)", lambda r: [P2RES / "R_hdrall+img_wavg" / "oof.csv"], "existing", arch="ResNet-18 image + header",
        repeats=(0,), note="repeat 0 only (descriptive)")
    add("ECGFounder frozen, linear", lambda r: [P2RES / f"ecgf_aligned_f6_lin{sfx(r)}" / "oof.csv"], "existing", arch="ECGFounder f6 + linear")
    add("ECGFounder fine-tuned (stage 6, Phase 6 A1)", lambda r: [src6(f"P6A1-ft-s6-joint-v1__s42_r{r}")], "existing",
        arch="ECGFounder stage 6 fine-tuned + joint head", note="seed 42 only")
    add("Header + ECGFounder MLP fusion", lambda r: [src6(f"P5A-import-mlpfusion-v1__s{s}_r{r}") for s in SEEDS], "existing", arch="MLP fusion")
    add("C1 Phase 3 vote ensemble", lambda r: [src6(f"P5A-import-vote-v1__s0_r{r}")], "existing", arch="vote (MLP fusion, ECGF linear, header+clinical LR)")
    add("C2 Phase 3 calibrated MLP", lambda r: [src6(f"P5A-calmlp-protected-v1__s0_r{r}")], "existing", arch="MLP fusion + nested prior rule", rule=True)
    add("C3 Phase 5 selected (vote + nested prior rule)", lambda r: [src6(f"P5D-E5-vote-prior-protected-v1__s0_r{r}")], "existing",
        arch="vote + nested prior rule", rule=True)
    for cfg in ("B1", "A270", "B2", "B3"):
        add(f"MLP fusion [{cfg}]", lambda r, c=cfg: [src6a(f"P6A-mlpfusion-{c}-v1__s{s}_r{r}") for s in SEEDS], "config", cfg, "MLP fusion")
        add(f"Vote, C1 recipe [{cfg}]", lambda r, c=cfg: [src6a(f"P6A-vote-{c}-v1__s0_r{r}")], "config", cfg, "vote")
        add(f"C3 recipe [{cfg}]", lambda r, c=cfg: [src6a(f"P6A-c3-{c}-v1__s0_r{r}")], "config", cfg, "vote + nested prior rule", rule=True)
    for v in ("L0", "L1", "L2", "L3", "L4"):
        add(f"LSTM {v} [B1]", lambda r, v=v: [src6a(f"P6A-lstm{v}-B1-v1__s{s}_r{r}") for s in SEEDS], "lstm", "B1", f"LSTM {v}")
    add("LSTM-sel (nested) [B1]", lambda r: [src6a(f"P6A-lstmsel-B1-v1__s{s}_r{r}") for s in SEEDS], "lstm", "B1", "LSTM nested selection")
    fx = lstm_fixed()
    if fx:
        for cfg in ("B1", "A270", "B2", "B3"):
            add(f"LSTM-fixed {fx} [{cfg}]", lambda r, c=cfg: [src6a(f"P6A-lstm{fx}-{c}-v1__s{s}_r{r}") for s in SEEDS], "config", cfg, f"LSTM {fx}")
    # CNN-LSTM and the late fusions are identical models that Phase 6B runs unconditionally (owner; PHASE6B_PLAN.md
    # Amendment 1); Phase 6A uses those runs and applies its own pre-registered conditions in main().
    add("CNN-LSTM (L5) [B1]", lambda r: [src6b(f"P6B-L5-ORIG-v1__s{s}_r{r}") for s in SEEDS], "lstm", "B1", "CNN-LSTM",
        note="run in Phase 6B")
    add("Vote + LSTM [B1]", lambda r: [src6b(f"P6B-F3-ORIG-v1__s0_r{r}")], "fusion", "B1", "C1 members + LSTM-sel vote", note="run in Phase 6B (F3)")
    add("Vote + LSTM + nested prior rule [B1]", lambda r: [src6b(f"P6B-F4-ORIG-v1__s0_r{r}")], "fusion", "B1", "fusion + C3 rule", rule=True,
        note="run in Phase 6B (F4)")
    return M


def src6b(rid):
    return ROOT / "research_experiments" / "phase_6b_threshold_sampling_lstm" / "data" / "private" / "runs" / rid / "oof.csv"


def preds(spec, r):
    Ps, preds_ = [], []
    for p in spec["paths"](r):
        P, pred = load_oof_csv(p); Ps.append(P); preds_.append(pred)
    P = np.mean(Ps, 0)
    return P, (preds_[0] if spec["rule"] else P.argmax(1))


def cells(y, P, pred, r):
    fv = fold_vec(r)
    return [metrics(y[fv == k], P[fv == k], pred[fv == k]) for k in range(5)]


def summarise(name, spec):
    yb2 = label_table().y_b2.values.astype(int)
    rows = {"orig": [], "b2": []}; cm = []; cl = []
    for r in spec["repeats"]:
        P, pred = preds(spec, r)
        rows["orig"].append(metrics(Y, P, pred)); rows["b2"].append(metrics(yb2, P, pred))
        cm.append(np.bincount(4 * Y + pred, minlength=16).reshape(4, 4)); cl.extend([c["accuracy"] for c in cells(Y, P, pred, r)])
    out = dict(model=name, family=spec["family"], label_config=spec["config"], architecture=spec["arch"], repeats=len(spec["repeats"]),
               note=spec["note"])
    for k in rows["orig"][0]:
        out[k] = float(np.mean([m[k] for m in rows["orig"]]))
    out["accuracy_sd_repeats"] = float(np.std([m["accuracy"] for m in rows["orig"]]))
    out["accuracy_ci95_corrected"] = corr_ci(cl) if len(cl) >= 10 else None
    out["admissible"] = bool(min(out[f"recall_{c}"] for c in CLASSES) >= 0.15)
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild"):
        out[f"{k}_vs_B2_labels"] = float(np.mean([m[k] for m in rows["b2"]]))
    out["confusion_mean"] = (np.mean(cm, 0)).round(1).tolist()
    return out


def paired(a, b, M, metric="accuracy", y=None):
    """Candidate a minus comparator b on the 15 shared (repeat, fold) cells; per-repeat pooled deltas."""
    y = Y if y is None else y
    reps = sorted(set(M[a]["repeats"]) & set(M[b]["repeats"]))
    d, dr = [], []
    for r in reps:
        Pa, pa = preds(M[a], r); Pb, pb = preds(M[b], r)
        ca, cb = cells(y, Pa, pa, r), cells(y, Pb, pb, r)
        d += [x[metric] - z[metric] for x, z in zip(ca, cb)]
        dr.append(metrics(y, Pa, pa)[metric] - metrics(y, Pb, pb)[metric])
    mean, p = corr_t(d)
    return dict(candidate=a, comparator=b, metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr],
                p_corrected_t=p, same_sign_all_repeats=bool(len(dr) == 3 and (all(x > 0 for x in dr) or all(x < 0 for x in dr))),
                n_cells=len(d))


def boot_mcnemar(a, b, M, n=2000):
    Pa, pa = preds(M[a], 0); Pb, pb = preds(M[b], 0)
    ca, cb = (pa == Y), (pb == Y)
    groups = [np.flatnonzero(PAT == p) for p in pd.unique(PAT)]
    rng = np.random.default_rng(0); bs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        bs.append(ca[idx].mean() - cb[idx].mean())
    b01, b10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = binomtest(b10, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return dict(boot_delta_acc_r0=float(ca.mean() - cb.mean()), boot95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                mcnemar_p_r0=float(p), discordant=[b10, b01])


def complementarity(M):
    """Error overlap of LSTM-sel with the MLP fusion and the 4-member vote check (pre-registered criterion)."""
    if "LSTM-sel (nested) [B1]" not in M:
        return None
    out = dict(per_repeat=[])
    for r in (0, 1, 2):
        Pl, pl = preds(M["LSTM-sel (nested) [B1]"], r); Pm, pm = preds(M["Header + ECGFounder MLP fusion"], r)
        Pv, pv = preds(M["C1 Phase 3 vote ensemble"], r)
        Pv4 = (3 * Pv + Pl) / 4                                   # vote of the 3 C1 members + LSTM-sel (equal weights)
        mv, mv4 = metrics(Y, Pv, pv), metrics(Y, Pv4, Pv4.argmax(1))
        lw = pm != Y
        out["per_repeat"].append(dict(lstm_correct_overall=float(np.mean(pl == Y)), lstm_correct_where_mlp_wrong=float(np.mean(pl[lw] == Y[lw])),
                                      both_wrong=float(np.mean((pl != Y) & lw)), corr_of_errors=float(np.corrcoef(pl != Y, lw)[0, 1]),
                                      vote4_minus_vote_acc=mv4["accuracy"] - mv["accuracy"], vote4_minus_vote_mf1=mv4["macro_f1"] - mv["macro_f1"]))
    pr = out["per_repeat"]
    out["criterion_met"] = bool(all(x["vote4_minus_vote_acc"] > 0 for x in pr) or all(x["vote4_minus_vote_mf1"] > 0 for x in pr))
    return out


def plots(S):
    guard(PLOTS).mkdir(parents=True, exist_ok=True)
    order = S.sort_values("accuracy").model.tolist()
    col = {"existing": "#4C78A8", "config": "#F58518", "lstm": "#54A24B", "fusion": "#B279A2"}
    for metric, fname, title in (("accuracy", "accuracy_comparison.png", "Four-class accuracy (original labels)"),
                                 ("macro_f1", "macro_f1_comparison.png", "Macro-F1 (original labels)")):
        T = S.set_index("model").loc[order]
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(T) + 1.5))
        ax.barh(range(len(T)), T[metric], color=[col[f] for f in T.family])
        if metric == "accuracy":
            for i, ci in enumerate(T.accuracy_ci95_corrected):
                if isinstance(ci, list):
                    ax.plot(ci, [i, i], color="k", lw=1)
        ax.set_yticks(range(len(T))); ax.set_yticklabels([m + ("" if a else "  (not admissible)") for m, a in zip(T.index, T.admissible)], fontsize=7)
        ax.set_xlabel(metric + " (mean over repeats; bars: corrected 95 % CI)" if metric == "accuracy" else metric)
        ax.set_title(title); ax.set_xlim(max(0.0, float(T[metric].min()) - 0.05), max(0.65, float(T[metric].max()) + 0.03))
        fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=150); plt.close(fig)
    T = S.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(T) + 1.5)); h = 0.27
    for j, (k, c) in enumerate((("precision_Mild", "#9ecae9"), ("recall_Mild", "#4C78A8"), ("f1_Mild", "#F58518"))):
        ax.barh(np.arange(len(T)) + (j - 1) * h, T[k], h, color=c, label=k.replace("_Mild", ""))
    ax.axvline(0.15, color="k", ls=":", lw=1); ax.set_yticks(range(len(T))); ax.set_yticklabels(T.index, fontsize=7)
    ax.set_title("Mild precision / recall / F1 (original labels; dotted: 0.15 recall floor)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "mild_precision_recall_f1.png", dpi=150); plt.close(fig)
    key = [m for m in ("C3 Phase 5 selected (vote + nested prior rule)", "C1 Phase 3 vote ensemble", "C3 recipe [B2]", "C3 recipe [B3]",
                       "C3 recipe [A270]", "LSTM-sel (nested) [B1]", "CNN-LSTM (L5) [B1]", "Vote + LSTM + nested prior rule [B1]") if m in set(S.model)]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, m in zip(axs.ravel(), key + [None] * (8 - len(key))):
        if m is None:
            ax.axis("off"); continue
        C = np.array(S.set_index("model").loc[m, "confusion_mean"])
        ax.imshow(C / C.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(["N", "Mi", "Mo", "S"]); ax.set_yticklabels(["N", "Mi", "Mo", "S"])
        ax.set_title(m, fontsize=8); ax.set_xlabel("predicted"); ax.set_ylabel("original label")
    fig.suptitle("Confusion matrices (mean counts over repeats; colour = row-normalised)"); fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrices.png", dpi=150); plt.close(fig)


def main():
    M = models()
    S = pd.DataFrame([summarise(n, s) for n, s in M.items()])
    # ---- confirmatory family (pre-registered), accuracy primary; macro-F1 and Mild F1 secondary
    fam = []
    lsel = "LSTM-sel (nested) [B1]"
    for ex in ("ECGFounder frozen, linear", "ECGFounder fine-tuned (stage 6, Phase 6 A1)", "Header + ECGFounder MLP fusion",
               "C1 Phase 3 vote ensemble", "C2 Phase 3 calibrated MLP", "C3 Phase 5 selected (vote + nested prior rule)"):
        if lsel in M and ex in M:
            fam.append((lsel, ex, "LSTM vs existing"))
    fx = lstm_fixed()
    for cfg in ("A270", "B2", "B3"):
        for base in ("MLP fusion", "Vote, C1 recipe", "C3 recipe") + ((f"LSTM-fixed {fx}",) if fx else ()):
            a, b = f"{base} [{cfg}]", f"{base} [B1]"
            if a in M and b in M:
                fam.append((a, b, f"{cfg} vs B1"))
    # pre-registered conditions (PHASE6A_PLAN.md section 5): CNN-LSTM only if LSTM-sel is admissible and its macro-F1
    # >= 0.389; fusion only if the complementarity criterion holds. Otherwise the models are reported descriptively.
    Si = S.set_index("model")
    cond = dict(cnn_lstm=bool(lsel in Si.index and Si.loc[lsel, "admissible"] and Si.loc[lsel, "macro_f1"] >= 0.389),
                fusion=bool((complementarity(M) or {}).get("criterion_met")))
    for a, b, c in (("CNN-LSTM (L5) [B1]", lsel, "cnn_lstm"), ("Vote + LSTM [B1]", "C1 Phase 3 vote ensemble", "fusion"),
                    ("Vote + LSTM + nested prior rule [B1]", "C3 Phase 5 selected (vote + nested prior rule)", "fusion")):
        if a in M and b in M and cond[c]:
            fam.append((a, b, "conditional"))
    tests = []
    for metric in ("accuracy", "macro_f1", "f1_Mild"):
        rows = [dict(paired(a, b, M, metric), family=f) for a, b, f in fam]
        adj = holm([x["p_corrected_t"] for x in rows]) if rows else []
        for x, h in zip(rows, adj):
            x["p_holm_family"] = float(h)
            adm = bool(S.set_index("model").loc[x["candidate"], "admissible"])
            x["candidate_admissible"] = adm
            x["genuine_improvement"] = bool(x["delta"] > 0 and h < 0.05 and x["same_sign_all_repeats"] and adm)
        tests += rows
    # B2 vs B1 against the B2 labels (definition-level, secondary) + decomposition
    yb2 = label_table().y_b2.values.astype(int)
    decomp = []
    for base in ("MLP fusion", "Vote, C1 recipe", "C3 recipe") + ((f"LSTM-fixed {fx}",) if fx else ()):
        a, b = f"{base} [B2]", f"{base} [B1]"
        if a in M and b in M:
            sa, sb = S.set_index("model").loc[a], S.set_index("model").loc[b]
            t = paired(a, b, M, "accuracy", y=yb2)
            decomp.append(dict(model=base, B1_on_original=sb.accuracy, B1_on_B2_labels=sb.accuracy_vs_B2_labels,
                               B2_on_B2_labels=sa.accuracy_vs_B2_labels, B2_on_original=sa.accuracy,
                               definition_effect=sb.accuracy_vs_B2_labels - sb.accuracy,
                               training_effect_on_B2_labels=sa.accuracy_vs_B2_labels - sb.accuracy_vs_B2_labels,
                               training_effect_p=t["p_corrected_t"], mild_recall_B1=sb.recall_Mild, mild_recall_B2_on_B2=sa.recall_Mild_vs_B2_labels,
                               mild_recall_B2_on_original=sa.recall_Mild))
    top = [x for x in tests if x["metric"] == "accuracy"]
    top = sorted(top, key=lambda x: -x["delta"])[:6]
    for x in top:
        x.update(boot_mcnemar(x["candidate"], x["comparator"], M))
    T = pd.DataFrame(tests)
    T.to_csv(guard(RES) / "phase6a_comparisons.csv", index=False, float_format="%.4f")
    cols = ["model", "family", "label_config", "architecture", "repeats", "admissible", "accuracy", "accuracy_ci95_corrected",
            "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "precision_Mild", "min_recall", "qwk", "macro_auroc"] + \
           [f"{m}_{c}" for c in CLASSES for m in ("precision", "recall", "f1")] + \
           ["accuracy_vs_B2_labels", "macro_f1_vs_B2_labels", "f1_Mild_vs_B2_labels", "recall_Mild_vs_B2_labels", "confusion_mean", "note"]
    S2 = S[cols].copy()
    S2["accuracy_ci95_corrected"] = S2.accuracy_ci95_corrected.map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if isinstance(c, list) else "")
    S2["confusion_mean"] = S2.confusion_mean.map(json.dumps)
    S2.sort_values(["family", "accuracy"], ascending=[True, False]).to_csv(P6A / "PHASE6A_MODEL_COMPARISON.csv", index=False, float_format="%.4f")
    comp = complementarity(M)
    best_orig = S[S.admissible & (S.repeats == 3) & S.label_config.isin(["B1"])].sort_values(["accuracy", "macro_f1"], ascending=False)
    best_any = S[S.admissible & (S.repeats == 3)].sort_values(["accuracy", "macro_f1"], ascending=False)
    summary = dict(n_models=len(S), best_admissible_original_definition=best_orig.iloc[0][["model", "accuracy", "macro_f1"]].to_dict() if len(best_orig) else None,
                   best_admissible_any_configuration_scored_on_original_labels=best_any.iloc[0][["model", "accuracy", "macro_f1"]].to_dict() if len(best_any) else None,
                   b2_decomposition=decomp, complementarity=comp, bootstrap_mcnemar_top=top, conditions_preregistered=cond,
                   genuine_improvements=[x for x in tests if x["genuine_improvement"]])
    (RES / "phase6a_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    plots(S)
    pd.set_option("display.width", 250)
    print(S[["model", "accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "min_recall", "admissible"]]
          .sort_values("accuracy", ascending=False).round(3).to_string(index=False))
    print(T[T.metric == "accuracy"][["candidate", "comparator", "delta", "delta_per_repeat", "p_corrected_t", "p_holm_family", "genuine_improvement"]].round(4).to_string(index=False))
    print(json.dumps(dict(decomp=decomp, complementarity=comp), indent=1, default=float))


if __name__ == "__main__":
    main()
