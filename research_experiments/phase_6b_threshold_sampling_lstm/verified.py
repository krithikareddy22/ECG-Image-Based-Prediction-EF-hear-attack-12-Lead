"""Exploratory EF-verified label configurations (AMENDMENT_VERIFIED.md). Development ECGs only; model-independent.
High-confidence link = nearest echo by TRUE acquisition date (file-name timestamp; corrects the 2021 sheet date swap),
exact name score 100, |ECG-echo| <= 30 days, no linked echo of a different class within 30 days.
  VER   folder label replaced by the high-confidence echo's class where they disagree
  VERB  VER + borderline Mild resolved to the nearest class (within 2 points of the sex-specific upper boundary ->
        Normal; within 2 points of 41 % -> Moderate) ONLY for high-confidence links
Category counts of borderline Mild are written as aggregates; per-ECG labels stay private.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.verified
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research_experiments.phase_6a_normal_mild_lstm.data_check import YEARS, load, rule
from research_experiments.phase_6b_threshold_sampling_lstm import existing6b as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

SRC = B.ROOT / "research_experiments" / "phase_4" / "data" / "private" / "ecg_echo"
CLS = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}


def verified_labels():
    man = pd.read_csv(B.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "split"]); man = man[man.split != "test"]
    man["ts"] = man.ecg_id.str.split("_").str[1]
    L = pd.concat([load(SRC, y)[1] for y in YEARS], ignore_index=True)
    L = L[L.ts.isin(set(man.ts)) & L.ef_raw.notna()].merge(man[["ts", "ecg_id"]], on="ts")
    L["ad"] = (L.echo_date.dt.normalize() - pd.to_datetime(L.ts.str[:8], format="%Y%m%d")).dt.days.abs()
    L["cls"] = [rule(e, s) for e, s in zip(L.ef_raw, L.sex)]
    near = L[L.ad == L.groupby("ecg_id").ad.transform("min")].groupby("ecg_id").agg(
        ef=("ef_raw", "mean"), sex=("sex", "first"), ad=("ad", "first"), score=("score", "min")).reindex(B.IDS)
    near["cls"] = [rule(e, s) for e, s in zip(near.ef, near.sex)]
    nconf = L[L.ad <= 30].groupby("ecg_id").cls.nunique().reindex(B.IDS).fillna(0).values
    hc = (near.score.values >= 100) & (near.ad.values <= 30) & (nconf <= 1)
    yd = near.cls.map(CLS).values.astype(float)
    ok = ~np.isnan(yd)
    dis = ok & (yd != B.Y)
    y_ver = np.where(dis & hc, yd, B.Y).astype(int)
    ef, male = near.ef.values, near.sex.values == "M"
    mild = (y_ver == 1) & (near.cls.values == "Mild")
    up = mild & (np.abs(ef - np.where(male, 52, 54)) <= 2); lo = mild & (np.abs(ef - 41) <= 2)
    y_verb = y_ver.copy(); y_verb[up & hc] = 0; y_verb[lo & hc] = 2
    border = (B.Y == 1) & (up | lo)
    cats = dict(borderline_mild=int(border.sum()), high_confidence=int((border & hc).sum()), uncertain_link=int((border & ~hc).sum()),
                hc_upper_to_normal=int((up & hc).sum()), hc_lower_to_moderate=int((lo & hc).sum()),
                label_disagrees_with_nearest_echo=int(dis.sum()), of_which_high_confidence=int((dis & hc).sum()),
                VER_changes=int((y_ver != B.Y).sum()), VERB_changes=int((y_verb != B.Y).sum()),
                VER_counts=np.bincount(y_ver, minlength=4).tolist(), VERB_counts=np.bincount(y_verb, minlength=4).tolist())
    return y_ver, y_verb, cats


Y_VER, Y_VERB, CATS = verified_labels()


def run(cfg):
    E.y_def = lambda c: Y_VER if c == "VER" else Y_VERB if c == "VERB" else B.y_def(c)
    E.run_config(cfg)
    return cfg


def summarise():
    out = dict(categories=CATS, models={})
    labs = {"original": B.Y, "VER": Y_VER, "VERB": Y_VERB}
    for cfg in ("VER", "VERB"):
        for m, seeds, rl in (("mlpfusion", B.SEEDS, False), ("vote", (0,), False), ("c3", (0,), True)):
            res = {k: [] for k in labs}; b1 = {k: [] for k in labs}
            for r in (0, 1, 2):
                Ps = [B.load_run(f"P6B-{m}-{cfg}-v1__s{s}_r{r}") for s in seeds]
                P = np.mean([p[0] for p in Ps], 0); pred = Ps[0][1] if rl else P.argmax(1)
                qid = {"mlpfusion": [f"P6A-mlpfusion-B1-v1__s{s}_r{r}" for s in B.SEEDS], "vote": [f"P6A-vote-B1-v1__s0_r{r}"],
                       "c3": [f"P6A-c3-B1-v1__s0_r{r}"]}[m]
                Qs = [B.load_run(x) for x in qid]; Q = np.mean([q[0] for q in Qs], 0); qp = Qs[0][1] if rl else Q.argmax(1)
                for k, y in labs.items():
                    res[k].append(B.metrics(y, P, pred)); b1[k].append(B.metrics(y, Q, qp))
            keys = ("accuracy", "balanced_accuracy", "macro_f1", "f1_Mild", "recall_Mild", "min_recall")
            out["models"][f"{m} [{cfg}]"] = {f"scored_on_{k}": {mm: float(np.mean([x[mm] for x in v])) for mm in keys} for k, v in res.items()}
            out["models"][f"{m} [B1 original model]"] = {f"scored_on_{k}": {mm: float(np.mean([x[mm] for x in v])) for mm in keys} for k, v in b1.items()}
    (B.guard(B.RES) / "verified_labels_exploratory.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(CATS))
    with Pool(2) as pool:
        list(pool.imap_unordered(run, ["VER", "VERB"]))
    o = summarise()
    for k, v in o["models"].items():
        print(k, {s: (round(x["accuracy"], 3), round(x["macro_f1"], 3), round(x["recall_Mild"], 3)) for s, x in v.items()})
