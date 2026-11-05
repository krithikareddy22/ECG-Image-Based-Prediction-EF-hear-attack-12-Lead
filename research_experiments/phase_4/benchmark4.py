"""Phase 4F fusion + 4H benchmark (pre-registered; development CV only, locked test untouched).

1. Regression-assisted ensembles (no tuned weights): vote ensemble + one equal-weight extra member
     fus_vote+ridgeA   Gaussian class probabilities of ridge (header || ECGF, cohort A)
     fus_vote+mlpregA  Gaussian class probabilities of the MLP regression (header || ECGF, cohort A; seed mean)
     fus_vote+mtsel    class head of the multitask model with lambda selected on inner-val (seed mean)
2. Seed combination within a repeat: probabilistic models -> mean probabilities, argmax; hard EF decoders
   (tuned / clin / recal) -> ordinal median of the seed predictions (probabilities averaged for AUROC).
3. Primary comparison: candidate - vote ensemble on the 15 matched (repeat, fold) cells, all 837 ECGs, folder labels,
   corrected repeated-CV t-test; Holm over the confirmatory family (primary decoder `tuned` for regression, all
   multitask / label-quality / fusion models); sign consistency over repeats; patient-cluster paired bootstrap (r0).
   Secondary: the same on G (folder and EF-derived labels).
Outputs (aggregates only): results/phase4_benchmark.{csv,json}, results/phase4_all_runs.csv, results/phase4_results.xlsx

    python -m research_experiments.phase_4.benchmark4
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from research_experiments.common import paired_bootstrap
from research_experiments.phase_2.p2common import holm
from research_experiments.phase_4.p4common import (AGG, COHORTS, PAT, RES4, REPEATS, SEEDS, Y, YD, corr_ci, corr_t,
                                                     done, eval_sets, fold_vec, load_run, metrics, nD, save_run)

HARD = ("__tuned", "__clin", "__recal")
KEYM = ["accuracy", "balanced_accuracy", "macro_f1", "s3", "f1_Mild", "qwk", "macro_auroc"]


def fusions():
    for r in REPEATS:
        Pv = load_run(f"bench_vote_r{r}")[0]
        extra = {"fus_vote+ridgeA": [f"reg_ridge_hdr+ecgf_A_s42_r{r}__gauss"],
                 "fus_vote+mlpregA": [f"reg_mlp_hdr+ecgf_A_s{s}_r{r}__gauss" for s in SEEDS],
                 "fus_vote+mtsel": [f"mt_sel_s{s}_r{r}" for s in SEEDS]}
        for name, tags in extra.items():
            if done(f"{name}_r{r}") or not all(done(t) for t in tags):
                continue
            Pm = np.mean([load_run(t)[0] for t in tags], 0)
            P = (3 * Pv + Pm) / 4
            save_run(f"{name}_r{r}", P, P.argmax(1), r, dict(family="4F_fusion", members="vote(3) + " + name.split("+")[1],
                                                            extra_member_runs=tags, weights="equal (1/4 each)"))


def model_groups():
    """name -> {repeat: [run tags]} for every model with all runs present."""
    tags = sorted(p.stem for p in AGG.glob("*.json"))
    groups = {}
    for t in tags:
        m = re.match(r"(.+?)(?:_s(\d+))?_r(\d)(__\w+)?$", t)
        if not m:
            continue
        name = m.group(1) + (m.group(4) or "")
        groups.setdefault(name, {}).setdefault(int(m.group(3)), []).append(t)
    def n_seeds(n):          # deterministic models have one run per repeat, seeded models three
        return 1 if n.startswith(("reg_ridge", "bench_vote", "bench_ecgflin", "bench_hdrclin", "fus_")) else len(SEEDS)
    return {n: g for n, g in groups.items() if set(g) == set(REPEATS) and all(len(g[r]) == n_seeds(n) for r in REPEATS)}


def combined(name, runs):
    """Seed-combined (P, pred, ef_hat) for one repeat."""
    L = [load_run(t) for t in runs]
    P = np.mean([x[0] for x in L], 0)
    if name.endswith(HARD):
        pred = np.median(np.stack([x[1] for x in L]), 0).round().astype(int)
    else:
        pred = P.argmax(1)
    ef = np.mean([x[2] for x in L], 0) if L[0][2] is not None else None
    return P, pred, ef


def family(name):
    if name.startswith("bench_"):
        return "benchmark"
    if name.startswith("reg_"):
        return "4D_regression"
    if name.startswith("mt_"):
        return "4E_multitask"
    if name.startswith("lq_"):
        return "4F_label_quality"
    return "4F_fusion"


def confirmatory(name):
    if name.startswith("reg_"):
        return name.endswith("__tuned")
    return (family(name) in ("4E_multitask", "4F_label_quality", "4F_fusion") and not name.startswith("mt_lam0_")
            and not name.endswith("_dist"))       # lambda-0 control and exploratory variants are not confirmatory


def cells(P, pred, r, y=None, mask=None):
    y = Y if y is None else y; mask = np.ones(nD, bool) if mask is None else mask
    fo = fold_vec(r)
    return [metrics(y[(fo == k) & mask], P[(fo == k) & mask], pred[(fo == k) & mask]) for k in range(5)]


def _boot(pa, preda, pb, predb):
    b = paired_bootstrap(Y, pa, pb, PAT, preda, predb)
    return {k: b[k] for k in b if k in ("accuracy", "balanced_accuracy", "macro_f1", "mild_f1", "qwk", "mild_auroc", "mcnemar_accuracy")}


def forest(B, stats):
    """Model comparison figure: difference vs the vote ensemble (all 837 ECGs, folder labels) with corrected 95 % CIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from research_experiments.phase_4.p4common import PLOT4
    sel = B[(B.confirmatory | (B.family == "benchmark") | B.model.str.endswith("_dist") | B.model.str.startswith("mt_lam0_"))
            & (B.model != "bench_vote")].sort_values("all_folder:s3")
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.28 * len(sel) + 1.5), sharey=True)
    col = {"benchmark": "#7f7f7f", "4D_regression": "#4c78a8", "4E_multitask": "#54a24b", "4F_label_quality": "#e45756",
           "4F_fusion": "#b279a2"}
    for a, k, t in zip(axes, ("s3", "accuracy", "balanced_accuracy"), ("S3", "accuracy", "balanced accuracy")):
        for i, (_, r) in enumerate(sel.iterrows()):
            st = stats[r.model][f"all_folder:{k}"]; lo, hi = st["ci95_corrected"]
            a.plot([lo, hi], [i, i], c=col.get(r.family, "k"), lw=1.5); a.plot(st["diff"], i, "o", c=col.get(r.family, "k"), ms=4)
        a.axvline(0, c="k", lw=0.8); a.set_xlabel(f"difference in {t} vs vote ensemble")
    axes[0].set_yticks(range(len(sel)), sel.model, fontsize=7)
    fig.suptitle("Phase 4 models vs vote ensemble: 837 development ECGs, folder labels, 3 CV repeats "
                 "(corrected repeated-CV 95 % CI; uncorrected for multiplicity)", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOT4 / "model_comparison_vs_vote.png", dpi=130); plt.close(fig)


def main():
    fusions()
    G = model_groups()
    comb = {n: {r: combined(n, g[r]) for r in REPEATS} for n, g in G.items()}
    vote = comb["bench_vote"]
    rows, stats = [], {}
    setdefs = {"all_folder": (Y, np.ones(nD, bool)), "G_folder": (Y, COHORTS["G"]), "G_derived": (YD, COHORTS["G"]),
               "nonG_folder": (Y, ~COHORTS["G"])}
    for n, rr in comb.items():
        row = dict(model=n, family=family(n), confirmatory=confirmatory(n), runs=sum(len(G[n][r]) for r in REPEATS))
        for sname, (yy, mk) in setdefs.items():
            ms = [metrics(yy[mk], rr[r][0][mk], rr[r][1][mk]) for r in REPEATS]
            for k in KEYM + ["f1_Normal", "f1_Moderate", "f1_Severe", "recall_Mild"]:
                row[f"{sname}:{k}"] = float(np.mean([m[k] for m in ms]))
                if sname == "all_folder":
                    row[f"{sname}:{k}:sd_repeats"] = float(np.std([m[k] for m in ms], ddof=1))
            row[f"{sname}:n_ecgs"] = int(mk.sum()); row[f"{sname}:n_patients"] = int(pd.Series(PAT[mk]).nunique())
        cc = [c for r in REPEATS for c in cells(*rr[r][:2], r)]
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "s3"):
            row[f"all_folder:{k}:ci95_corrected"] = corr_ci([c[k] for c in cc])
        ev = [eval_sets(rr[r][0], rr[r][1], rr[r][2]) for r in REPEATS]
        for k in ev[0]["binary_auroc"]:
            row[f"auroc:{k}"] = float(np.mean([e["binary_auroc"][k] for e in ev]))
        for blk in ("mild_auroc_all_folder", "mild_auroc_G_derived"):
            for k in ev[0][blk]:
                row[f"{blk}:{k}"] = float(np.mean([e[blk][k] for e in ev]))
        if rr[0][2] is not None:
            for co in ("A", "C", "G"):
                for k in ("mae", "rmse", "r2", "pearson", "spearman", "bias", "calib_slope"):
                    row[f"reg_{co}:{k}"] = float(np.mean([e["regression"][co][k] for e in ev]))
        # comparisons vs vote ensemble
        if n != "bench_vote":
            st = {}
            for sname, (yy, mk) in setdefs.items():
                cv = [c for r in REPEATS for c in cells(*vote[r][:2], r, yy, mk)]
                cn = [c for r in REPEATS for c in cells(*rr[r][:2], r, yy, mk)]
                for k in ("accuracy", "balanced_accuracy", "macro_f1", "s3", "f1_Mild"):
                    dl = [b[k] - a[k] for a, b in zip(cv, cn)]
                    d, p = corr_t(dl)
                    per_rep = [metrics(yy[mk], rr[r][0][mk], rr[r][1][mk])[k] - metrics(yy[mk], vote[r][0][mk], vote[r][1][mk])[k]
                               for r in REPEATS]
                    st[f"{sname}:{k}"] = dict(diff=d, p_corrected_t=p, ci95_corrected=corr_ci(dl), diff_per_repeat=per_rep,
                                              same_sign_all_repeats=bool(np.all(np.sign(per_rep) == np.sign(d)) and d != 0))
            stats[n] = st
            for k in ("accuracy", "s3", "balanced_accuracy", "macro_f1"):
                row[f"vs_vote:all_folder:{k}:diff"] = st[f"all_folder:{k}"]["diff"]
                row[f"vs_vote:all_folder:{k}:p"] = st[f"all_folder:{k}"]["p_corrected_t"]
                row[f"vs_vote:all_folder:{k}:same_sign"] = st[f"all_folder:{k}"]["same_sign_all_repeats"]
                row[f"vs_vote:G_folder:{k}:diff"] = st[f"G_folder:{k}"]["diff"]
                row[f"vs_vote:G_derived:{k}:diff"] = st[f"G_derived:{k}"]["diff"]
        from sklearn.metrics import confusion_matrix
        row["confusion_all_folder_mean_counts"] = json.dumps(np.mean([confusion_matrix(Y, rr[r][1], labels=range(4))
                                                                      for r in REPEATS], 0).round(1).tolist())
        for k in ("precision", "recall"):
            for c in ("Normal", "Mild", "Moderate", "Severe"):
                row[f"all_folder:{k}_{c}"] = float(np.mean([metrics(Y, rr[r][0], rr[r][1])[f"{k}_{c}"] for r in REPEATS]))
        rows.append(row)
    B = pd.DataFrame(rows)
    # Holm within metric over the confirmatory family (all-D folder labels)
    conf = B.confirmatory & (B.model != "bench_vote")
    for k in ("accuracy", "s3", "balanced_accuracy", "macro_f1"):
        col = f"vs_vote:all_folder:{k}:p"
        B.loc[conf, col.replace(":p", ":p_holm")] = holm(B.loc[conf, col].values)
        B[f"genuine_improvement:{k}"] = conf & (B[f"vs_vote:all_folder:{k}:diff"] > 0) & \
            (B.get(col.replace(":p", ":p_holm"), 1.0) < 0.05) & B[f"vs_vote:all_folder:{k}:same_sign"].fillna(False)
    # paired cluster bootstrap on repeat 0 for the confirmatory family + benchmarks
    names = [n for n in B.model[conf | (B.family == "benchmark")] if n != "bench_vote"]
    from multiprocessing import Pool
    with Pool(4) as pool:
        res = pool.starmap(_boot, [(vote[0][0], vote[0][1], comb[n][0][0], comb[n][0][1]) for n in names])
    boots = dict(zip(names, res))
    B = B.sort_values("all_folder:s3", ascending=False)
    RES4.mkdir(exist_ok=True)
    B.to_csv(RES4 / "phase4_benchmark.csv", index=False, float_format="%.4f")
    (RES4 / "phase4_benchmark.json").write_text(json.dumps(dict(comparisons_vs_vote=stats, bootstrap_r0_vs_vote=boots,
                                                                n_confirmatory=int(conf.sum())), indent=1, default=float))
    # every individual run (filterable spreadsheet)
    R = []
    for p in sorted(AGG.glob("*.json")):
        a = json.loads(p.read_text()); e = a["eval"]; inf = a["info"]
        r = dict(run=a["tag"], repeat=a["repeat"], family=inf.get("family"), model=inf.get("model", inf.get("variant", "")),
                 features=inf.get("features", "hdr+ecgf" if inf.get("family") != "benchmark" else ""),
                 train_cohort=inf.get("train_cohort", inf.get("ef_cohort", "")), decoder=inf.get("decoder", "argmax"),
                 seed=inf.get("seed"))
        for s in ("all_folder", "G_folder", "G_derived", "nonG_folder"):
            for k in KEYM + ["f1_Normal", "f1_Moderate", "f1_Severe"]:
                r[f"{s}:{k}"] = e[s][k]
            if "n_ecgs" in e[s]:
                r[f"{s}:n_ecgs"], r[f"{s}:n_patients"] = e[s]["n_ecgs"], e[s]["n_patients"]
        for co, v in e.get("regression", {}).items():
            for k in ("mae", "rmse", "r2", "pearson", "spearman"):
                r[f"reg_{co}:{k}"] = v[k]
        R.append(r)
    R = pd.DataFrame(R); R.to_csv(RES4 / "phase4_all_runs.csv", index=False, float_format="%.4f")
    finalize()


def finalize():
    """Spreadsheet + forest plot + console summary from the saved aggregate outputs (no recomputation)."""
    B = pd.read_csv(RES4 / "phase4_benchmark.csv"); R = pd.read_csv(RES4 / "phase4_all_runs.csv")
    stats = json.loads((RES4 / "phase4_benchmark.json").read_text())["comparisons_vs_vote"]
    with pd.ExcelWriter(RES4 / "phase4_results.xlsx", engine="openpyxl") as xw:
        B.to_excel(xw, sheet_name="models (seed-combined)", index=False)
        R.to_excel(xw, sheet_name="all runs", index=False)
        for ws in xw.sheets.values():
            ws.auto_filter.ref = ws.dimensions; ws.freeze_panes = "B2"
    forest(B, stats)
    show = B[["model", "family", "runs", "all_folder:accuracy", "all_folder:balanced_accuracy", "all_folder:macro_f1",
              "all_folder:f1_Mild", "all_folder:s3", "G_folder:accuracy", "G_derived:accuracy"]].copy()
    show["dS3"] = B["vs_vote:all_folder:s3:diff"]; show["p"] = B["vs_vote:all_folder:s3:p"]
    show["p_holm"] = B.get("vs_vote:all_folder:s3:p_holm")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    import sys
    finalize() if "--finalize" in sys.argv else main()
