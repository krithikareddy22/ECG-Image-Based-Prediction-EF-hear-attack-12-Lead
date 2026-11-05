"""FIXES (AMENDMENT_FIXES.md): re-check worklist for re-linking / Simpson re-reads, and the ingest pipeline that
retrains and re-evaluates the header-free model once verified echo data are returned. Development ECGs only.
    python -m research_experiments.phase_6b_threshold_sampling_lstm.fixes --worklist
    python -m research_experiments.phase_6b_threshold_sampling_lstm.fixes --template
    python -m research_experiments.phase_6b_threshold_sampling_lstm.fixes --ingest [path/to/filled.csv] [--selftest]
Private outputs (git-ignored): data/private/fixes/. Committed: aggregate JSON only.
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5.p5common import corr_ci, corr_t, feature_family
from research_experiments.phase_6.p6common import PAT_GROUP
from research_experiments.phase_6a_normal_mild_lstm.data_check import YEARS, load, rule
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

warnings.filterwarnings("ignore")
OUT = B.PRIV / "fixes"
SRC = B.ROOT / "research_experiments" / "phase_4" / "data" / "private" / "ecg_echo"
LT = E.LT; Y = B.Y; nD = B.nD
CLS = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
PROTOCOL = ("Re-read protocol: reader blinded to the folder label and to any model output; confirm the ECG-echo link by "
            "MRN/UHID; re-measure EF by biplane Simpson; use every re-read value whatever it shows.")


def links():
    """All matching-sheet rows linked to development ECGs, with true dates (C6A-1 day/month correction via the timestamp)."""
    man = pd.read_csv(B.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "split", "source"]); man = man[man.split != "test"]
    man["ts"] = man.ecg_id.str.split("_").str[1]
    L = pd.concat([load(SRC, y)[1] for y in YEARS], ignore_index=True)
    L = L[L.ts.isin(set(man.ts)) & L.ef_raw.notna()].merge(man[["ts", "ecg_id"]], on="ts")
    L["ecg_true_date"] = pd.to_datetime(L.ts.str[:8], format="%Y%m%d")
    L["gap"] = (L.echo_date.dt.normalize() - L.ecg_true_date).dt.days.abs()
    L["cls"] = [rule(e, s) for e, s in zip(L.ef_raw, L.sex)]
    return L, man.set_index("ecg_id").loc[B.IDS]


def worklist():
    L, man = links()
    near = L[L.gap == L.groupby("ecg_id").gap.transform("min")].sort_values("score").groupby("ecg_id").first().reindex(B.IDS)
    conflict30 = (L[L.gap <= 30].groupby("ecg_id").cls.nunique().reindex(B.IDS).fillna(0).values > 1)
    nb = np.where(LT.male.values, 52.0, 54.0); ef = LT.ef.values
    dist = np.min(np.abs(ef[:, None] - np.c_[np.full(nD, 31.0), np.full(nD, 41.0), nb]), 1)
    lab150 = open(B.ROOT / "research_experiments" / "ecgfounder" / "tasks_150_labels.txt").read().splitlines()
    V, cols = feature_family("logits150")
    names = [lab150[int(c.split("_")[1])] for c in cols]
    mi = (V[:, [i for i, n in enumerate(names) if "INFARCT" in n]] >= 0).any(1)             # sigmoid >= 0.5 <=> logit >= 0
    lbbb = V[:, names.index("LEFT BUNDLE BRANCH BLOCK")] >= 0
    flags = {
        "label_differs_from_nearest_echo": (LT.label.values != LT.derived.values, 3),
        "ef_within_3_of_limit": (dist < 3, 3),
        "ef_within_3_to_6_of_limit": ((dist >= 3) & (dist < 6), 1),
        "name_score_below_100": (near.score.values < 100, 2),
        "echo_more_than_30_days": (near.gap.values > 30, 2),
        "echo_8_to_30_days": ((near.gap.values > 7) & (near.gap.values <= 30), 1),
        "other_class_echo_within_30_days": (conflict30, 2),
        "ef_spread_over_10": (LT.ef_spread.values > 10, 1),
        "teichholz_unreliable_pattern_mi_or_lbbb": (mi | lbbb, 1),
    }
    prio = sum(v.astype(int) * w for v, w in flags.values())
    W = pd.DataFrame(dict(ecg_id=B.IDS, pdf=man.source.values, ecg_date=near.ecg_true_date.dt.date.values,
                          nearest_echo_date=near.echo_date.dt.date.values, gap_days=near.gap.values, name_score=near.score.values,
                          ef_teichholz=ef, folder_label=LT.label.values, nearest_echo_class=LT.derived.values,
                          ef_distance_to_limit=dist.round(1), priority=prio,
                          reasons=["; ".join(k for k, (v, _) in flags.items() if v[i]) for i in range(nD)]))
    W = W.sort_values(["priority", "ef_distance_to_limit"], ascending=[False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    f = B.guard(OUT / "recheck_worklist.csv")
    with open(f, "w") as fh:
        fh.write("# " + PROTOCOL + "\n")
        W.to_csv(fh, index=False)
    tiers = {"high (>= 5)": W.priority >= 5, "medium (3-4)": W.priority.between(3, 4), "low (1-2)": W.priority.between(1, 2), "none (0)": W.priority == 0}
    summ = dict(n_ecgs=nD, flags={k: int(v.sum()) for k, (v, _) in flags.items()},
                priority_tiers={t: dict(n=int(m.sum()), by_folder_label=W[m].folder_label.value_counts().to_dict()) for t, m in tiers.items()},
                any_flag=int((W.priority > 0).sum()), protocol=PROTOCOL)
    (B.guard(B.RES) / "fixes_worklist_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def template(selftest=False):
    L, man = links()
    near = L[L.gap == L.groupby("ecg_id").gap.transform("min")].sort_values("score").groupby("ecg_id").first().reindex(B.IDS)
    T = pd.DataFrame(dict(ecg_id=B.IDS, mrn_verified="yes" if selftest else "", echo_date=near.echo_date.dt.date.values,
                          ef_simpson="", ef_teichholz=LT.ef.values if selftest else "", regional_wma="", notes=""))
    OUT.mkdir(parents=True, exist_ok=True)
    f = B.guard(OUT / ("selftest_filled.csv" if selftest else "verified_echo_template.csv"))
    T.to_csv(f, index=False)
    print("written", f.relative_to(B.ROOT))
    return f


# ------------------------------------------------------------------ ingest: retrain + evaluate on the verified labels
def classes(ef, male):
    return np.array([CLS[rule(e, "M" if m else "F")] for e, m in zip(ef, male)])


def folds_kept(r, k, keep):
    fv = B.fold_vec(r)
    TR = np.flatnonzero((fv != k) & keep); TE = np.flatnonzero((fv == k) & keep)
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], PAT_GROUP()[TR]))
    return TR, TE, inner


def fit_fold(y, ef, keep, r, k):
    TR, TE, inner = folds_kept(r, k, keep)
    fm = np.zeros(nD, bool); fm[TR] = True
    probs_in, probs_out = [], []
    for m in ("M2", "M5"):
        X = E.features(m, r, k, fm)
        oof = {C: np.zeros((len(TR), 4)) for C in E.CS}
        for ia, ib in inner:
            lr = LogisticRegression(class_weight="balanced", max_iter=2000, warm_start=True)
            for C in E.CS:
                lr.set_params(C=C).fit(X[TR[ia]], y[TR[ia]]); oof[C][ib] = lr.predict_proba(X[TR[ib]])
        C = min(E.CS, key=lambda c: -np.mean(np.log(np.clip(oof[c][np.arange(len(TR)), y[TR]], 1e-9, 1))))
        probs_in.append(oof[C]); probs_out.append(LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X[TR], y[TR]).predict_proba(X[TE]))
    X = E.features("M6", r, k, fm); best = None
    for a in E.ALPHAS_R:
        mu = np.zeros(len(TR))
        for ia, ib in inner:
            mu[ib] = Ridge(alpha=a).fit(X[TR[ia]], ef[TR[ia]]).predict(X[TR[ib]])
        rmse = float(np.sqrt(np.mean((mu - ef[TR]) ** 2)))
        if best is None or rmse < best[1]:
            best = (a, rmse, mu)
    nc = np.where(LT.male.values, 52.0, 53.95)

    def band(mu, idx):
        cN, cM, cS = (norm.cdf((c - mu) / best[1]) for c in (nc[idx], 41.0, 31.0))
        return np.c_[1 - cN, cN - cM, cM - cS, cS]
    probs_in.append(band(best[2], TR)); probs_out.append(band(Ridge(alpha=best[0]).fit(X[TR], ef[TR]).predict(X[TE]), TE))
    vi, vo = np.mean(probs_in, 0), np.mean(probs_out, 0)
    w = E.tune(vi, y[TR])
    return TE, (vo * w).argmax(1)


def evaluate(y, ef, keep):
    preds, cells = [], []
    for r in (0, 1, 2):
        p = np.full(nD, -1)
        for k in range(5):
            TE, pk = fit_fold(y, ef, keep, r, k); p[TE] = pk
        preds.append(p); fo = B.fold_vec(r)
        cells += [float(np.mean(p[(fo == k) & keep] == y[(fo == k) & keep])) for k in range(5)]
    m = keep
    s = dict(n=int(m.sum()), class_counts=[int((y[m] == c).sum()) for c in range(4)],
             accuracy=float(np.mean([np.mean(p[m] == y[m]) for p in preds])), accuracy_ci95_corrected=corr_ci(cells),
             kappa=float(np.mean([cohen_kappa_score(y[m], p[m]) for p in preds])),
             **{f"recall_{c}": float(np.mean([np.mean(p[m & (y == i)] == i) for p in preds])) for i, c in enumerate(B.CLASSES)})
    return s, cells


def ingest(path, selftest=False):
    F = pd.read_csv(path).set_index("ecg_id").reindex(B.IDS)
    keep = (F.mrn_verified.astype(str).str.lower() != "no").values
    ef_new = pd.to_numeric(F.ef_simpson, errors="coerce").fillna(pd.to_numeric(F.ef_teichholz, errors="coerce")).fillna(pd.Series(LT.ef.values, index=B.IDS)).values
    y_new = classes(ef_new, LT.male.values)
    y_cur = classes(LT.ef.values, LT.male.values)
    s_new, c_new = evaluate(y_new, ef_new, keep)
    same = np.array_equal(y_new, y_cur) and np.allclose(ef_new, LT.ef.values)
    s_cur, c_cur = (s_new, c_new) if same else evaluate(y_cur, LT.ef.values, keep)     # identical inputs -> identical model
    d, p = corr_t(np.array(c_new) - np.array(c_cur))
    out = dict(selftest=selftest, rows_excluded_not_verified=int((~keep).sum()), rows_with_simpson=int(pd.to_numeric(F.ef_simpson, errors="coerce").notna().sum()),
               labels_changed=int((y_new != y_cur)[keep].sum()), new_labels=s_new, current_labels_same_ecgs=s_cur,
               delta_accuracy=d, p=p)
    name = "fixes_selftest.json" if selftest else "fixes_ingest.json"
    (B.guard(B.RES) / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", action="store_true"); ap.add_argument("--template", action="store_true")
    ap.add_argument("--ingest", nargs="?", const="", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.worklist:
        worklist()
    if a.template:
        template()
    if a.ingest is not None:
        f = template(selftest=True) if a.selftest else a.ingest
        ingest(f, selftest=a.selftest)
