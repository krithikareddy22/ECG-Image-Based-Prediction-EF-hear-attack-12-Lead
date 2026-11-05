"""One-shot locked-test evaluation of the frozen header-free triage model (PHASE7_PLAN.md Amendment 6; owner
authorisation 2026-11-06). Scores the 210 locked-test ECGs ONCE; refuses a second run (ledger).
Names are read from file paths into memory only for the pre-declared name-pair exclusion, compared and discarded;
nothing name-derived and no identifier is printed or written. Per-ECG outputs carry test row numbers only (private).
    python -m research_experiments.phase_7.testkit.locked_test --run --owner-authorised
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time

import numpy as np
import pandas as pd

from research_experiments.phase_5.data_audit.grouping_audit import norm_name
from research_experiments.phase_5.p5common import ecgf_f6
from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.testkit import kit as K
from research_experiments.phase_7.testkit import score as S

NAME = "locked_test"
AUDIT = C.ROOT / "research_experiments" / "phase_5" / "data_audit" / "grouping_audit.json"


def name_pairs(man):
    """-> (test keys paired with a development key, within-test key pairs, number of straddling pairs).
    Phase 5 grouping_audit criterion, names in memory only."""
    names = man.source.map(norm_name).values
    key_name = pd.Series(names, index=man.patient.values).groupby(level=0).first()
    keys, nms = key_name.index.values, key_name.values
    split_of = man.groupby("patient").split.first().map(lambda s: "test" if s == "test" else "dev")
    straddle, within, n_straddle = set(), [], 0
    for i in range(len(keys)):
        a = nms[i]
        for j in range(i + 1, len(keys)):
            b = nms[j]
            if not a or not b:
                continue
            sm = difflib.SequenceMatcher(None, a, b); contained = min(len(a), len(b)) >= 8 and (a in b or b in a)
            if not contained and (sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9):
                continue
            if contained or sm.ratio() >= 0.9:
                sa, sb = split_of[keys[i]], split_of[keys[j]]
                if sa != sb:
                    n_straddle += 1; straddle.add(keys[i] if sa == "test" else keys[j])
                elif sa == "test":
                    within.append((keys[i], keys[j]))
    del names, key_name, nms
    return straddle, within, n_straddle


def clusters(pat, within):
    parent = {k: k for k in set(pat)}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for a, b in within:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return np.array([find(p) for p in pat])


def assess(y, c3, t, groups):
    tr, fc = S.triage_stats(y, t), S.fourclass(y, c3)
    tr["answered_share_ci95_wilson"] = S.wilson(tr["answered"], tr["n"])
    tr["accuracy_answered_ci95_wilson"] = S.wilson(int(np.sum((t == y) & (t >= 0))), tr["answered"])
    fc["accuracy_ci95_wilson"] = S.wilson(int(np.sum(c3 == y)), fc["n"])
    return dict(n_ecgs=int(len(y)), n_patient_clusters=int(pd.unique(groups).size),
                true_class_counts={c: int(np.sum(y == i)) for i, c in enumerate(K.CLASSES)},
                imgc3_four_class_all=fc, triage=tr, bootstrap_ci95_patient_cluster=S.bootstrap(y, t, c3, groups))


def run():
    L = json.loads(K.LEDGER.read_text()) if K.LEDGER.exists() else []
    if any(x["name"] == NAME for x in L) or (K.RUNSK / NAME).exists():
        S.refuse("the locked test has already been scored once (ledger); it is not re-scored")
    fr = K.Frozen.load(); card = json.loads(K.CARD.read_text())
    if not {"pdf_reproduction", "recipe_reproduction"} <= set(card.get("checks", {})):
        S.refuse("the frozen model has not passed its checks")
    man = pd.read_csv(C.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "label", "patient", "split", "source"])
    audit = json.loads(AUDIT.read_text())["similar_name_pairs_under_different_keys"]
    straddle_keys, within, n_str = name_pairs(man)
    if n_str != audit["straddling_development_and_locked_test"] or len(within) != audit["pairs"] - n_str - audit["both_in_development"]:
        S.refuse("name-pair counts differ from the Phase 5 audit: nothing scored")
    T = man[man.split == "test"].reset_index(drop=True)
    y = np.array([K.CLASSES.index(x) for x in T.label])
    H = pd.read_csv(C.ROOT / "data" / "header_features.csv", usecols=["ecg_id", "male"]).set_index("ecg_id")
    male = H.male.loc[T.ecg_id].values == 1.0
    excl_name = T.patient.isin(straddle_keys).values
    t0 = time.time(); n = len(T); cols, st = fr.d["clin_cols"], fr.d["mildfeat"]
    F6 = np.full((n, 1024), np.nan, np.float32); IM = np.full((n, len(cols)), np.nan); bad = np.zeros(n, bool)
    for i, src in enumerate(T.source):
        f, c, e = K.inputs_from_pdf(C.ROOT / src, cols, st)
        if e:
            bad[i] = True
        else:
            F6[i], IM[i] = f, c
        if (i + 1) % 30 == 0 or i + 1 == n:
            print(f"read {i + 1}/{n} test PDFs", flush=True)
    ok = ~bad
    Fd = ecgf_f6().astype(float); Fd /= np.linalg.norm(Fd, axis=1, keepdims=True)
    Fn = F6[ok].astype(float); Fn /= np.linalg.norm(Fn, axis=1, keepdims=True)
    dup = np.zeros(n, bool); dup[ok] = (Fn @ Fd.T).max(1) >= S.DUP_COS
    o = {k: np.full(n, -1) for k in ("c3", "grade", "triage")}; o.update(ef_est=np.full(n, np.nan), dist=np.full(n, np.nan), P=np.full((n, 4), np.nan))
    r = fr.run(F6[ok], IM[ok], male[ok])
    for k in o:
        o[k][ok] = r[k]
    grp = clusters(T.patient.values, within)
    prim = ok & ~excl_name & ~dup
    cv = card["cv_expectation"]
    res = dict(run=NAME, created=time.strftime("%F %T"), git=C.B.git_head(), frozen_sha256=card["frozen_sha256"],
               authorisation="owner, 2026-11-06: chose 'Score locked 210 (Recommended)' - 'I authorise using the locked test set once'",
               rows=dict(total=int(n), unreadable_pdf=int(bad.sum()), excluded_name_pair_with_development=int((excl_name & ok).sum()),
                         excluded_near_duplicate_of_development=int((dup & ~excl_name).sum()), primary=int(prim.sum()),
                         within_test_name_pairs_merged_for_ci=len(within)),
               labels="folder labels (data/manifest.csv), as in all development results",
               primary=assess(y[prim], o["c3"][prim], o["triage"][prim], grp[prim]),
               sensitivity_all_readable=assess(y[ok], o["c3"][ok], o["triage"][ok], grp[ok]),
               cv_expectation=cv, minutes=round((time.time() - t0) / 60, 1))
    p, b = res["primary"], res["primary"]["bootstrap_ci95_patient_cluster"]
    res["primary_vs_cv"] = [dict(metric=k, cv_expectation=float(v), test=tv, test_ci95=b.get(k),
                                 cv_inside_test_ci=None if b.get(k) is None or tv is None else bool(b[k][0] <= v <= b[k][1]))
                            for k, v, tv in (("imgc3.accuracy", cv["imgc3_all"]["accuracy"], p["imgc3_four_class_all"]["accuracy"]),
                                             ("imgc3.macro_f1", cv["imgc3_all"]["macro_f1"], p["imgc3_four_class_all"]["macro_f1"]),
                                             ("imgc3.balanced_accuracy", cv["imgc3_all"]["balanced_accuracy"], p["imgc3_four_class_all"]["balanced_accuracy"]),
                                             ("imgc3.recall_Mild", cv["imgc3_all"]["recall_Mild"], p["imgc3_four_class_all"]["per_class"]["Mild"]["recall"]),
                                             ("triage.answered_share", cv["triage"]["answered_share"], p["triage"]["answered_share"]),
                                             ("triage.accuracy_answered", cv["triage"]["accuracy_answered"], p["triage"]["accuracy_answered"]))]
    out = C.guard7(K.RUNSK / NAME); out.mkdir(parents=True, exist_ok=True)
    pe = pd.DataFrame(dict(test_row=np.arange(n), label=y, primary=prim, unreadable=bad, name_pair_excluded=excl_name, dev_duplicate=dup,
                           imgc3=o["c3"], ef_estimate=o["ef_est"], ef_grade=o["grade"], distance_to_cut=o["dist"], triage=o["triage"]))
    pe.to_csv(out / "per_ecg.csv", index=False)                                           # private (git-ignored)
    C.guard7(K.RESK).mkdir(parents=True, exist_ok=True)
    (K.RESK / "locked_test_scores.json").write_text(json.dumps(res, indent=1, default=float))
    L.append(dict(name=NAME, time=res["created"], manifest_sha256="data/manifest.csv split == test", frozen_sha256=card["frozen_sha256"], n=int(n),
                  reason="owner-authorised one-shot locked-test evaluation (Amendment 6)"))
    C.guard7(K.LEDGER).write_text(json.dumps(L, indent=1))
    summary(res)


def summary(res):
    print("\nrows:", res["rows"])
    for part in ("primary", "sensitivity_all_readable"):
        p = res[part]; fc, tr, b = p["imgc3_four_class_all"], p["triage"], p["bootstrap_ci95_patient_cluster"]
        print(f"\n[{part}] {p['n_ecgs']} ECGs, true {p['true_class_counts']}")
        print(f"  FOUR-CLASS (IMG-C3, all): accuracy {S.fmt(fc['accuracy'], b['imgc3.accuracy'])}  macro-F1 {S.fmt(fc['macro_f1'], b['imgc3.macro_f1'])}  "
              f"balanced {S.fmt(fc['balanced_accuracy'], b['imgc3.balanced_accuracy'])}")
        print("  per class P/R/F1: " + "; ".join(f"{c} {v['precision']:.2f}/{v['recall']:.2f}/{v['f1']:.2f}" for c, v in fc["per_class"].items()))
        print(f"  confusion (rows true N/Mi/Mo/S): {fc['confusion_rows_true_cols_pred']}")
        print(f"  TRIAGE: answered {S.fmt(tr['answered_share'], b['triage.answered_share'])}  accuracy on answered "
              f"{S.fmt(tr['accuracy_answered'], b['triage.accuracy_answered'])}  mix {tr['predicted_mix_answered']}  "
              f"Normal calls w/o EF<41 {S.fmt(tr['normal_calls_without_ef_below_41'])}  Mild caught {S.fmt(tr['mild_caught_of_all_mild'])}")
    print()
    for r in res["primary_vs_cv"]:
        print(f"  {r['metric']:26s} CV {r['cv_expectation']:.3f}  test {S.fmt(r['test'], r['test_ci95'])}  CV inside test CI: {r['cv_inside_test_ci']}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--owner-authorised", action="store_true")
    a = ap.parse_args()
    if not (a.run and a.owner_authorised):
        print("needs --run --owner-authorised (PHASE7_PLAN.md Amendment 6)"); sys.exit(1)
    run()


if __name__ == "__main__":
    main()
