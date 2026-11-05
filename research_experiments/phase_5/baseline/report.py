"""5A baseline table: every imported / reproduced model, accuracy-first, with patient-grouped CIs and paired tests.

Seed combination within a repeat: probabilistic models -> mean probabilities, argmax; models whose stored decisions are
not the argmax (calibrated rules, EF-threshold decoders) -> ordinal median of the seed decisions.
CIs: corrected repeated-CV 95 % CI over the (repeat, fold) cells, and patient-cluster bootstrap 95 % CI (repeat 0).
Comparisons (candidate - comparator, 15 matched cells; r0-only models: 5 cells): C1 vote ensemble, C2 calibrated MLP
(protected rule, reproduced). Admissible = every class recall >= RECALL_FLOOR (mean over repeats).
Also used by later milestones: summarise(exp_ids) -> DataFrame.
    python -m research_experiments.phase_5.baseline.report
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from research_experiments.phase_5.p5common import (CLASSES, EXP5, PAT, RECALL_FLOOR, RES5, Y, bootstrap_ci, cells, confusion,
                                                   corr_ci, corr_t, guard5, load_run, metrics, paired_bootstrap)

C1, C2 = "P5A-import-vote-v1", "P5A-calmlp-protected-v1"
KEY = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "qwk", "min_recall"] + \
      [f"{m}_{c}" for c in CLASSES for m in ("precision", "recall", "f1")]


def groups():
    g = {}
    for p in EXP5.glob("*.json"):
        m = re.match(r"(.+)__s(\d+)_r(\d)$", p.stem)
        if m:
            g.setdefault(m.group(1), {}).setdefault(int(m.group(3)), []).append(p.stem)
    return g


def combined(runs):
    L = [load_run(r) for r in runs]
    P = np.mean([x[0] for x in L], 0)
    hard = any((x[1] != x[0].argmax(1)).any() for x in L)
    pred = np.median(np.stack([x[1] for x in L]), 0).round().astype(int) if hard else P.argmax(1)
    return P, pred, hard


def summarise(exp_ids=None, comparators=(C1, C2), with_boot=True):
    G = groups()
    exp_ids = exp_ids or sorted(G)
    comb = {e: {r: combined(G[e][r]) for r in sorted(G[e])} for e in set(exp_ids) | set(c for c in comparators if c in G)}
    rows, detail = [], {}
    for e in exp_ids:
        rr = comb[e]; reps = sorted(rr)
        ms = [metrics(Y, *rr[r][:2]) for r in reps]
        row = dict(model=e, repeats=len(reps), runs=sum(len(G[e][r]) for r in reps), decision="rule/median" if rr[reps[0]][2] else "argmax")
        for k in KEY:
            row[k] = float(np.mean([m[k] for m in ms]))
        row["accuracy_sd_repeats"] = float(np.std([m["accuracy"] for m in ms], ddof=1)) if len(ms) > 1 else np.nan
        row["min_class_recall_mean"] = float(min(row[f"recall_{c}"] for c in CLASSES))
        row["admissible"] = row["min_class_recall_mean"] >= RECALL_FLOOR
        cc = [c for r in reps for c in cells(*rr[r][:2], r)]
        row["accuracy_ci95_corrected"] = json.dumps([round(x, 4) for x in corr_ci([c["accuracy"] for c in cc])])
        cm = np.mean([confusion(Y, rr[r][1]) for r in reps], 0)
        row["confusion_mean"] = json.dumps(cm.round(1).tolist())
        d = dict(confusion_mean=cm.round(1).tolist(), per_repeat=[{k: m[k] for k in KEY} for m in ms])
        if with_boot and 0 in rr:
            b = bootstrap_ci(Y, rr[0][0], PAT, rr[0][1])
            row["accuracy_ci95_cluster_boot_r0"] = json.dumps([round(x, 4) for x in b["accuracy"]])
            d["cluster_bootstrap_r0"] = b
        for cname in comparators:
            if cname not in comb or cname == e:
                continue
            common = [r for r in reps if r in comb[cname]]
            dl = {k: [] for k in ("accuracy", "balanced_accuracy", "macro_f1", "recall_Severe")}
            for r in common:
                for a, b_ in zip(cells(*comb[cname][r][:2], r), cells(*rr[r][:2], r)):
                    for k in dl:
                        dl[k].append(b_[k] - a[k])
            tag = "vs_C1_vote" if cname == C1 else "vs_C2_calmlp"
            for k, v in dl.items():
                dd, p = corr_t(v)
                row[f"{tag}:{k}:diff"], row[f"{tag}:{k}:p"] = dd, p
            per_rep = [metrics(Y, *rr[r][:2])["accuracy"] - metrics(Y, *comb[cname][r][:2])["accuracy"] for r in common]
            row[f"{tag}:accuracy:same_sign_all_repeats"] = bool(len(per_rep) == 3 and np.all(np.sign(per_rep) == np.sign(np.mean(per_rep))) and np.mean(per_rep) != 0)
            if with_boot and 0 in common:
                pb = paired_bootstrap(Y, comb[cname][0][0], rr[0][0], PAT, comb[cname][0][1], rr[0][1])
                row[f"{tag}:accuracy:boot_ci95_r0"] = json.dumps([round(x, 4) for x in pb["accuracy"]["ci95"]])
                row[f"{tag}:mcnemar_p_r0"] = pb["mcnemar_accuracy"]["p_exact"]
        rows.append(row); detail[e] = d
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False), detail


def md_table(B, cols=("accuracy", "balanced_accuracy", "macro_f1", "recall_Severe", "precision_Severe", "min_class_recall_mean")):
    L = ["| Model | Repeats (runs) | " + " | ".join(cols) + " | Acc. 95 % CI (corrected) | Admissible | Δacc vs C1 (p) | Δacc vs C2 (p) |",
         "|---|---|" + "---|" * len(cols) + "---|---|---|---|"]
    for _, r in B.iterrows():
        f = lambda c: f"{r[c]:+.3f} ({r[c.replace(':diff', ':p')]:.2f})" if c in r and pd.notna(r[c]) else "–"
        L.append(f"| `{r.model}` | {r.repeats} ({r.runs}) | " + " | ".join(f"{r[c]:.3f}" for c in cols) +
                 f" | {r.accuracy_ci95_corrected} | {'yes' if r.admissible else '**no**'} | {f('vs_C1_vote:accuracy:diff')} | {f('vs_C2_calmlp:accuracy:diff')} |")
    return "\n".join(L)


def main():
    B, detail = summarise([e for e in groups() if e.startswith("P5A-")])
    guard5(RES5).mkdir(exist_ok=True)
    B.to_csv(RES5 / "phase5_baseline_results.csv", index=False, float_format="%.4f")
    (RES5 / "phase5_baseline_results.json").write_text(json.dumps(dict(
        note="development CV, 837 ECGs / 794 name-based patient keys, folder labels; locked test not used",
        recall_floor=RECALL_FLOOR, comparators=dict(C1=C1, C2=C2), models=B.to_dict("records"), detail=detail), indent=1, default=float))
    (RES5 / "phase5_baseline_tables.md").write_text(md_table(B) + "\n")
    pd.set_option("display.width", 250)
    print(B[["model", "repeats", "runs", "accuracy", "balanced_accuracy", "macro_f1", "recall_Severe", "precision_Severe",
             "min_class_recall_mean", "admissible"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
