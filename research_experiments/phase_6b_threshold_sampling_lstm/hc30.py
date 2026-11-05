"""HC30 cohort study + Mild-factor study + HC30+F model (AMENDMENT_HC30.md).
    python -m research_experiments.phase_6b_threshold_sampling_lstm.hc30 --study      (descriptive factors)
    python -m research_experiments.phase_6b_threshold_sampling_lstm.hc30 --run        (HC30 and HC30+F models)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.experiments.nested import BASES
from research_experiments.phase_6a_normal_mild_lstm.data_check import YEARS, load, rule
from research_experiments.phase_6b_threshold_sampling_lstm import existing6b as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

FAMS = ["clin3", "qrs", "st", "hrv", "freq", "wavelet", "leadstat", "sq", "logits150"]
BASES["allfact"] = (["hdr", "feat:" + "+".join(FAMS)], "sklr", None, (42,))
SRC = B.ROOT / "research_experiments" / "phase_4" / "data" / "private" / "ecg_echo"


def hc30():
    man = pd.read_csv(B.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "split"]); man = man[man.split != "test"]
    man["ts"] = man.ecg_id.str.split("_").str[1]
    L = pd.concat([load(SRC, y)[1] for y in YEARS], ignore_index=True)
    L = L[L.ts.isin(set(man.ts)) & L.ef_raw.notna()].merge(man[["ts", "ecg_id"]], on="ts")
    L["ad"] = (L.echo_date.dt.normalize() - pd.to_datetime(L.ts.str[:8], format="%Y%m%d")).dt.days.abs()
    L["cls"] = [rule(e, s) for e, s in zip(L.ef_raw, L.sex)]
    near = L[L.ad == L.groupby("ecg_id").ad.transform("min")].groupby("ecg_id").agg(ad=("ad", "first"), score=("score", "min")).reindex(B.IDS)
    n30 = L[L.ad <= 30].groupby("ecg_id").cls.nunique().reindex(B.IDS).fillna(0).values
    return (near.score.values >= 100) & (near.ad.values <= 30) & (n30 <= 1)


HC = hc30()
_orig = B.train_exclude


def excl(cfg, r, k):
    return ~HC if cfg in ("HC30", "HC30F") else _orig(cfg, r, k)


def study():
    from research_experiments.phase_5.p5common import feature_family, header_numeric
    H = header_numeric(np.ones(B.nD, bool)); hn = ["age", "male", "Rate", "RR", "PR", "QRSD", "QT", "QTcB", "QTcF", "PR_missing",
                                                   "P_axis_sin", "P_axis_cos", "QRS_axis_sin", "QRS_axis_cos", "T_axis_sin", "T_axis_cos"]
    blocks = [("header", H, hn)]
    lab150 = open(B.ROOT / "research_experiments" / "ecgfounder" / "tasks_150_labels.txt").read().splitlines()
    for f in FAMS:
        V, cols = feature_family(f)
        if f == "logits150":
            cols = ["ECGFounder: " + lab150[int(c.split("_")[1])] for c in cols]
        blocks.append((f, V, cols))
    rows = []
    for fam, V, cols in blocks:
        for j, c in enumerate(cols):
            x = V[:, j]
            for name, a, b in (("mild_vs_normal", 1, 0), ("mild_vs_moderate", 1, 2)):
                m = np.isin(B.Y, [a, b]) & np.isfinite(x)
                if m.sum() < 50 or np.nanstd(x[m]) == 0:
                    continue
                auc = roc_auc_score(B.Y[m] == a, x[m])
                rows.append(dict(family=fam, feature=c, contrast=name, auroc=auc, strength=abs(auc - 0.5),
                                 direction="higher in Mild" if auc > 0.5 else "lower in Mild"))
    R = pd.DataFrame(rows)
    out = {}
    for con in ("mild_vs_normal", "mild_vs_moderate"):
        T = R[R.contrast == con].sort_values("strength", ascending=False)
        out[con] = dict(top15=T.head(15)[["family", "feature", "auroc", "direction"]].round(3).to_dict("records"),
                        best_by_family=T.groupby("family").strength.max().add(0.5).round(3).sort_values(ascending=False).to_dict(),
                        n_features=int(len(T)))
    (B.guard(B.RES) / "mild_factor_study.json").write_text(json.dumps(out, indent=1))
    for con, v in out.items():
        print("==", con, "best AUROC by family:", v["best_by_family"])
        for x in v["top15"]:
            print(f"   {x['auroc']:.3f}  {x['direction']:14s} [{x['family']}] {x['feature']}")


def run():
    torch.set_num_threads(1)
    B.train_exclude = excl; E.train_exclude = excl
    yt = B.Y; out = {}
    for cfg, members in (("HC30", ("mlpfusion", "ecgflindist", "hdrclinlr")), ("HC30F", ("mlpfusion", "ecgflindist", "hdrclinlr", "allfact"))):
        res = {"vote": [], "c3": [], "vote_all837": [], "c3_all837": []}
        for r in (0, 1, 2):
            Ps = []
            for b in members:
                seeds = B.SEEDS if b == "mlpfusion" else ((42,) if b in ("ecgflindist", "allfact") else (0,))
                bcfg = "HC30"                                     # members shared between HC30 and HC30F (same fitting rows)
                Ps.append(np.mean([E.outer_member(b, bcfg, s, r) for s in seeds], 0))
            Pv = np.mean(Ps, 0)
            fo = B.fold_vec(r); pred = np.zeros(B.nD, int)
            for k in range(5):
                inn = [E.inner_oof(b, "HC30", r, k) for b in members]
                TR, Pin = inn[0][0], np.mean([x[1] for x in inn], 0)
                keep = HC[TR]
                prior = np.bincount(yt[TR][keep], minlength=4) / keep.sum()
                a = choose_alpha(Pin[keep], yt[TR][keep], prior, "protected")
                pred[fo == k] = B.decide(Pv[fo == k], "prior", a, prior)
            res["vote"].append(B.metrics(yt[HC], Pv[HC], Pv.argmax(1)[HC])); res["c3"].append(B.metrics(yt[HC], Pv[HC], pred[HC]))
            res["vote_all837"].append(B.metrics(yt, Pv, Pv.argmax(1))); res["c3_all837"].append(B.metrics(yt, Pv, pred))
        out[cfg] = {k: {m: float(np.mean([x[m] for x in v])) for m in ("accuracy", "balanced_accuracy", "macro_f1", "recall_Normal", "recall_Mild",
                                                                        "recall_Moderate", "recall_Severe", "f1_Mild", "min_recall")} | {"accuracy_per_repeat": [round(x["accuracy"], 4) for x in v]}
                    for k, v in res.items()}
    # reference: original C3 (trained on all 837) evaluated on the HC30 ECGs
    ref = []
    for r in (0, 1, 2):
        P, p = B.load_run(f"P6A-c3-B1-v1__s0_r{r}"); ref.append(B.metrics(yt[HC], P[HC], p[HC]))
    out["C3_trained_on_all_scored_on_HC30"] = {m: float(np.mean([x[m] for x in ref])) for m in ("accuracy", "balanced_accuracy", "macro_f1", "recall_Mild", "min_recall")}
    out["n_HC30"] = int(HC.sum()); out["HC30_class_counts"] = [int((yt[HC] == c).sum()) for c in range(4)]
    (B.guard(B.RES) / "hc30_exploratory.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--study", action="store_true"); ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.study:
        study()
    if a.run:
        run()
