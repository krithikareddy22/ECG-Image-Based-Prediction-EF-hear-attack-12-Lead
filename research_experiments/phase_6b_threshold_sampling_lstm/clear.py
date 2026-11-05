"""Exploratory CLEAR configuration (AMENDMENT_CLEAR.md): existing model recipe trained without border ECGs.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.clear
"""
import json

import numpy as np

from research_experiments.phase_6b_threshold_sampling_lstm import existing6b as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B


def border():
    T = B.table(); ef = T.ef.values
    return ((np.abs(ef - T.sex_boundary.values) <= 3) | (np.abs(ef - 41) <= 3) | (np.abs(ef - 31) <= 3)
            | (T.derived.values != T.label.values))


BORDER = border()
_orig = B.train_exclude


def excl(cfg, r, k):
    return BORDER.copy() if cfg == "CLEAR" else _orig(cfg, r, k)


def main():
    B.train_exclude = excl; E.train_exclude = excl
    E.run_config("CLEAR")
    out = dict(border_ecgs=int(BORDER.sum()), clear_ecgs=int((~BORDER).sum()), models={})
    for m, seeds, rule in (("mlpfusion", B.SEEDS, False), ("vote", (0,), False), ("c3", (0,), True)):
        ref = {"mlpfusion": "P6A-mlpfusion-B1-v1__s{s}_r{r}", "vote": "P6A-vote-B1-v1__s0_r{r}", "c3": "P6A-c3-B1-v1__s0_r{r}"}[m]
        res = {}
        for tag, ids in (("CLEAR", [f"P6B-{m}-CLEAR-v1__s{s}_r{{r}}" for s in seeds]),
                         ("ORIG", [ref.format(s=s, r="{r}") for s in seeds])):
            acc = {"all": [], "clear": [], "border": []}; ms = []
            for r in (0, 1, 2):
                Ps = [B.load_run(i.format(r=r)) for i in ids]
                P = np.mean([p[0] for p in Ps], 0); pred = Ps[0][1] if rule else P.argmax(1)
                ms.append(B.metrics(B.Y, P, pred))
                acc["all"].append(np.mean(pred == B.Y)); acc["clear"].append(np.mean(pred[~BORDER] == B.Y[~BORDER]))
                acc["border"].append(np.mean(pred[BORDER] == B.Y[BORDER]))
            res[tag] = dict(accuracy_all=float(np.mean(acc["all"])), accuracy_clear_cut=float(np.mean(acc["clear"])),
                            accuracy_border=float(np.mean(acc["border"])),
                            **{k: float(np.mean([x[k] for x in ms])) for k in ("balanced_accuracy", "macro_f1", "recall_Normal",
                                                                               "recall_Mild", "recall_Moderate", "recall_Severe", "min_recall")})
        out["models"][m] = res
    (B.guard(B.RES) / "clear_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
