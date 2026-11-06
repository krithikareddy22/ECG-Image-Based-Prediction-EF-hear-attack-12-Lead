"""5B: four-class error analysis (development CV; aggregates only; cells < 5 ECGs suppressed).

Models: C1 vote ensemble, C2 calibrated MLP (protected), MLP fusion (seed-combined), each per repeat, then averaged.
  1. confusion pairs Normal-Mild, Mild-Moderate, Moderate-Severe, Normal-Severe (both directions)
  2. errors near EF boundaries: for each true class, distance of the nearest-echo EF to the boundary the error crossed
  3. ECG-echo interval        4. signal quality (phase-5 'sq' family)
  5. repeated ECGs of one patient key; possible split identities (near-identical names under different keys)
  6. lead-wise informativeness (univariate AUROC of per-lead features for EF < 41 vs Normal)
  7. confidence and calibration (ECE, reliability, accuracy vs coverage)
  8. Severe class separately
Echo EF is used for ANALYSIS only (never as a model input).
    python -m research_experiments.phase_5.error_analysis.analyse
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from research_experiments.phase_5.baseline.report import combined, groups  # noqa: E402
from research_experiments.phase_5.p5common import (CLASSES, FEAT5, IDS, P4, P5, PAT, RES5, ROOT, Y, guard5, metrics, nD)  # noqa: E402

SUP = 5
MODELS = {"C1 vote ensemble": "P5A-import-vote-v1", "C2 calibrated MLP (56.8 %)": "P5A-calmlp-protected-v1",
          "MLP fusion": "P5A-import-mlpfusion-v1"}
PL = P5 / "results" / "plots"


def sup(n, v):
    return None if n < SUP else (round(float(v), 4) if v is not None and np.isfinite(v) else None)


def possible_split_identity_mask():
    """ECGs of development patient keys that have a near-identical name under another development key (in memory)."""
    man = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "patient", "split", "source"])
    man = man[man.split != "test"]
    nm = man.source.map(lambda s: re.sub(r"[^A-Z]", "", (re.match(r"EK__\[(\d+)\]_(.*?)_*\[\]_", Path(s).name) or [None, None, ""])[2].upper()))
    kn = pd.Series(nm.values, index=man.patient.values).groupby(level=0).first()
    keys, names = kn.index.values, kn.values; flagged = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = names[i], names[j]
            if not a or not b:
                continue
            cont = min(len(a), len(b)) >= 8 and (a in b or b in a)
            sm = difflib.SequenceMatcher(None, a, b)
            if cont or (sm.real_quick_ratio() >= 0.9 and sm.quick_ratio() >= 0.9 and sm.ratio() >= 0.9):
                flagged |= {keys[i], keys[j]}
    del names, kn, nm
    return np.isin(PAT, list(flagged))


def main():
    G = groups()
    comb = {lab: {r: combined(G[e][r]) for r in (0, 1, 2)} for lab, e in MODELS.items()}
    coh = pd.read_csv(P4 / "data" / "private" / "cohorts.csv").set_index("ecg_id").loc[IDS]
    EF, male, days = coh.ef.values, coh.male.values, coh.abs_days.values
    up = np.where(male == 1, 52.0, 54.0)
    sq = pd.read_csv(FEAT5 / "sq.csv").set_index("ecg_id").loc[IDS]
    out = {"n_ecgs": nD, "models": MODELS}
    # 1 + 2: confusion pairs and boundary distance of the errors
    pairs = [(0, 1, "Normal->Mild", lambda: np.abs(EF - up)), (1, 0, "Mild->Normal", lambda: np.abs(EF - up)),
             (1, 2, "Mild->Moderate", lambda: np.abs(EF - 41)), (2, 1, "Moderate->Mild", lambda: np.abs(EF - 41)),
             (2, 3, "Moderate->Severe", lambda: np.abs(EF - 31)), (3, 2, "Severe->Moderate", lambda: np.abs(EF - 31)),
             (0, 3, "Normal->Severe", lambda: np.abs(EF - 31)), (3, 0, "Severe->Normal", lambda: np.abs(EF - up))]
    out["confusion_pairs"] = {}
    for lab in MODELS:
        rows = []
        for t, p, name, dist in pairs:
            d = dist(); n_t = int((Y == t).sum())
            err = [comb[lab][r][1] == p for r in (0, 1, 2)]
            k = np.mean([(e & (Y == t)).sum() for e in err])
            ok = [(comb[lab][r][1] == t) & (Y == t) for r in (0, 1, 2)]
            d_err = np.nanmedian(np.concatenate([d[e & (Y == t)] for e in err])) if k else np.nan
            d_ok = np.nanmedian(np.concatenate([d[o] for o in ok]))
            within5_err = np.mean(np.concatenate([d[e & (Y == t)] < 5 for e in err])) if k else np.nan
            within5_ok = np.mean(np.concatenate([d[o] < 5 for o in ok]))
            rows.append(dict(pair=name, true_class_n=n_t, errors_mean=round(float(k), 1), share_of_true_class=round(float(k / n_t), 3),
                             median_ef_distance_to_crossed_boundary_errors=sup(k, d_err),
                             median_ef_distance_same_boundary_correct=round(float(d_ok), 2),
                             share_within_5_points_errors=sup(k, within5_err), share_within_5_points_correct=round(float(within5_ok), 3)))
        out["confusion_pairs"][lab] = rows
    # 3 interval, 4 quality
    def acc_by(mask_fn_list, lab):
        res = []
        for name, m in mask_fn_list:
            acc = np.mean([(comb[lab][r][1][m] == Y[m]).mean() for r in (0, 1, 2)]) if m.sum() else np.nan
            res.append(dict(group=name, ecgs=int(m.sum()), accuracy=sup(int(m.sum()), acc)))
        return res
    ib = [("same day", days == 0), ("1-14 d", (days >= 1) & (days <= 14)), ("15-90 d", (days >= 15) & (days <= 90)), ("> 90 d", days > 90)]
    npl = sq.n_poor_leads.values
    qb = [("0 poor leads", npl == 0), ("1 poor lead", npl == 1), (">= 2 poor leads", npl >= 2)]
    hf = sq.hf_max.values; t1, t2 = np.percentile(hf, [33.3, 66.7])
    hb = [("HF noise low tertile", hf <= t1), ("mid", (hf > t1) & (hf <= t2)), ("high tertile", hf > t2)]
    out["accuracy_by_interval"] = {lab: acc_by(ib, lab) for lab in MODELS}
    out["accuracy_by_quality"] = {lab: acc_by(qb, lab) + acc_by(hb, lab) for lab in MODELS}
    # 5 repeated ECGs / split identities
    cnt = pd.Series(PAT).map(pd.Series(PAT).value_counts()).values
    split_id = possible_split_identity_mask()
    out["repeated_ecgs"] = {}
    for lab in MODELS:
        rows = acc_by([("single-ECG patient keys", cnt == 1), ("patient keys with >= 2 ECGs", cnt >= 2),
                       ("ECGs in possible split identities", split_id), ("all other ECGs", ~split_id)], lab)
        same = []
        for r in (0, 1, 2):
            pr = pd.Series(comb[lab][r][1], index=PAT)[cnt >= 2]
            same.append(float((pr.groupby(level=0).nunique() == 1).mean()))
        rows.append(dict(group="multi-ECG keys: all ECGs of the key get the same prediction", ecgs=int((cnt >= 2).sum()),
                         accuracy=round(float(np.mean(same)), 3)))
        out["repeated_ecgs"][lab] = rows
    out["repeated_ecgs_note"] = ("'possible split identities' = development ECGs whose key has a near-identical name under another "
                                 "key (REPOSITORY_AUDIT.md section 6). Higher accuracy there would suggest leakage across folds.")
    # 6 lead-wise informativeness (label-based univariate AUROC; descriptive)
    q = pd.read_csv(FEAT5 / "qrs.csv").set_index("ecg_id").loc[IDS]; st = pd.read_csv(FEAT5 / "st.csv").set_index("ecg_id").loc[IDS]
    red = np.isin(Y, [2, 3]); norm = Y == 0; m = red | norm
    lead_rows = []
    for ld in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        row = dict(lead=ld)
        for nm_, col in (("qrs_area", q[f"area_{ld}"]), ("qrs_width", q[f"width_{ld}"]), ("R_amp", q[f"R_{ld}"]), ("t_amp", st[f"t_amp_{ld}"])):
            v = col.values; ok = m & np.isfinite(v)
            a = roc_auc_score(red[ok], v[ok]); row[nm_] = round(float(max(a, 1 - a)), 3)
        lead_rows.append(row)
    out["lead_wise_univariate_auroc_reducedEF_vs_normal"] = lead_rows
    # 7 confidence / calibration
    out["calibration"] = {}
    PL.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for lab, c in zip(MODELS, ("#4c78a8", "#e45756", "#54a24b")):
        eces, covs = [], []
        for r in (0, 1, 2):
            P, pred = comb[lab][r][:2]; conf = P.max(1); corr = pred == Y
            bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
            eces.append(sum(abs(conf[idx == b].mean() - corr[idx == b].mean()) * (idx == b).mean() for b in range(10) if (idx == b).any()))
            order = np.argsort(-conf)
            covs.append([float(corr[order[:int(f * nD)]].mean()) for f in (0.25, 0.5, 0.75, 1.0)])
        cv = np.mean(covs, 0)
        out["calibration"][lab] = dict(ece_argmax_confidence=round(float(np.mean(eces)), 4),
                                       accuracy_at_coverage={f"{int(f * 100)}%": round(float(a), 3) for f, a in zip((0.25, 0.5, 0.75, 1.0), cv)})
        ax[0].plot([25, 50, 75, 100], cv, "o-", c=c, label=lab)
        P = comb[lab][0][0]; conf = P.max(1); corr = comb[lab][0][1] == Y
        bins = np.linspace(0.2, 1, 9); idx = np.digitize(conf, bins)
        xs = [conf[idx == b].mean() for b in range(1, 9) if (idx == b).sum() >= SUP]; ys = [corr[idx == b].mean() for b in range(1, 9) if (idx == b).sum() >= SUP]
        ax[1].plot(xs, ys, "o-", c=c, label=lab)
    ax[0].set_xlabel("coverage (% most confident ECGs kept)"); ax[0].set_ylabel("accuracy"); ax[0].legend(fontsize=7)
    ax[0].set_title("Accuracy vs coverage (mean over 3 repeats)", fontsize=9)
    ax[1].plot([0.2, 1], [0.2, 1], "k:", lw=0.8); ax[1].set_xlabel("max class probability"); ax[1].set_ylabel("observed accuracy")
    ax[1].set_title("Reliability (repeat 0; bins with >= 5 ECGs)", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(guard5(PL / "confidence_calibration.png"), dpi=130); plt.close(fig)
    # 8 Severe
    out["severe"] = {}
    qw = q.width_median.values; qrsd = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[IDS].QRSD.values
    for lab in MODELS:
        ms = [metrics(Y, *comb[lab][r][:2]) for r in (0, 1, 2)]
        sev = Y == 3
        hit = [comb[lab][r][1] == 3 for r in (0, 1, 2)]
        det = np.mean(hit, 0) >= 0.5
        out["severe"][lab] = dict(n=int(sev.sum()), recall=round(float(np.mean([m["recall_Severe"] for m in ms])), 3),
                                  precision=round(float(np.mean([m["precision_Severe"] for m in ms])), 3),
                                  missed_predicted_as={CLASSES[c]: round(float(np.mean([((comb[lab][r][1] == c) & sev).sum() for r in (0, 1, 2)])), 1)
                                                       for c in range(3)},
                                  echo_ef_median_detected=sup(int((sev & det).sum()), np.median(EF[sev & det])),
                                  echo_ef_median_missed=sup(int((sev & ~det).sum()), np.median(EF[sev & ~det])),
                                  printed_qrsd_median_detected=sup(int((sev & det).sum()), np.nanmedian(qrsd[sev & det])),
                                  printed_qrsd_median_missed=sup(int((sev & ~det).sum()), np.nanmedian(qrsd[sev & ~det])),
                                  false_severe_true_class={CLASSES[c]: round(float(np.mean([((comb[lab][r][1] == 3) & (Y == c)).sum() for r in (0, 1, 2)])), 1)
                                                           for c in range(3)})
    _ = qw
    (RES5 / "error_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("accuracy_by_quality", "repeated_ecgs", "calibration", "severe")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
