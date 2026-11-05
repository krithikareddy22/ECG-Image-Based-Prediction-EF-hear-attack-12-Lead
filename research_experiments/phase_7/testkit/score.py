"""One-shot scorer for a test set the owner collects (PHASE7_PLAN.md Amendment 5; instructions in TESTKIT.md).
Runs the FROZEN header-free triage model on every ECG PDF in a manifest and reports aggregate results with 95 % CIs
next to the cross-validation expectation. Per-ECG outputs stay in phase_7/data/private/testkit/runs/<name>/ (manifest
row numbers only, never file names); the aggregate JSON goes to phase_7/results/testkit/<name>_scores.json.
    python -m research_experiments.phase_7.testkit.score --skeleton <folder of PDFs> --out <private manifest.csv>
    python -m research_experiments.phase_7.testkit.score --manifest <manifest.csv> --name <run name>
Nothing that identifies a patient is printed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.phase_5.p5common import ecgf_f6
from research_experiments.phase_6a_normal_mild_lstm.data_check import rule, sex_of
from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.testkit import kit as K

COLS = ["ecg_file", "patient_id", "sex", "ef", "echo_gap_days", "ef_method"]
MAX_GAP, DUP_COS, NBOOT, EF_RANGE = 30, 0.999, 2000, (5.0, 90.0)
CL = K.CLASSES


def refuse(msg):
    print("REFUSED:", msg); sys.exit(2)


def in_ignored_place(path):
    """True when the file is outside the repository or under a git-ignored data/ folder (.gitignore: data/)."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(C.ROOT.resolve())
    except ValueError:
        return True
    return "data" in rel.parts


def rows_text(mask):
    r = np.flatnonzero(np.asarray(mask)); return ", ".join(map(str, r[:20])) + (" ..." if len(r) > 20 else "")


# ------------------------------------------------------------------ skeleton
def skeleton(folder, out):
    if not in_ignored_place(out):
        refuse("write the manifest outside the repository or under a git-ignored data/ folder (file names can contain patient names)")
    pdfs = sorted(Path(folder).rglob("*.pdf"))
    D = pd.DataFrame({"ecg_file": [str(p.resolve()) for p in pdfs]})
    for c in COLS[1:]:
        D[c] = ""
    Path(out).parent.mkdir(parents=True, exist_ok=True); D.to_csv(out, index=False)
    print(f"{len(D)} PDFs listed. Fill patient_id (pseudonymised), sex (M/F), ef (echo, numeric), echo_gap_days, ef_method.")


# ------------------------------------------------------------------ manifest
def read_manifest(path):
    if not in_ignored_place(path):
        refuse("keep the manifest outside the repository or under a git-ignored data/ folder")
    D = pd.read_csv(path, dtype=str, keep_default_na=False)
    miss = [c for c in COLS if c not in D]
    if miss:
        refuse(f"manifest lacks columns {miss}; template: phase_7/testkit/manifest_template.csv")
    if not len(D):
        refuse("manifest is empty")
    base = Path(path).resolve().parent
    D["path"] = [(Path(p) if Path(p).is_absolute() else base / p).resolve() for p in D.ecg_file]
    if D.path.duplicated().any():
        refuse(f"the same PDF appears more than once (rows {rows_text(D.path.duplicated())})")
    D["sex_n"] = D.sex.map(sex_of)
    D["ef_n"] = pd.to_numeric(D.ef, errors="coerce")
    D["gap_n"] = pd.to_numeric(D.echo_gap_days, errors="coerce").abs()
    D["label"] = [rule(e, s) if s else "Unknown" for e, s in zip(D.ef_n, D.sex_n)]
    D["y"] = [CL.index(x) if x in CL else -1 for x in D.label]
    D["status"] = np.select([~D.path.map(Path.exists), D.ef_n.isna() | ~D.ef_n.between(*EF_RANGE), D.sex_n.isna(), D.y < 0],
                            ["missing_file", "bad_ef", "bad_sex", "no_label"], "ok")
    pid = D.patient_id.str.strip()
    D["group"] = np.where(pid != "", pid, "row" + D.index.astype(str))
    return D


def guards(D, authorised):
    man = pd.read_csv(C.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source", "split"])        # metadata only
    ts = man.ecg_id.str.split("_").str[1]; test = man.split == "test"
    key = lambda p: str(Path(p).resolve()).lower()                                                     # noqa: E731
    paths = man.source.map(lambda s: key(C.ROOT / s))
    mts = D.path.map(K.timestamp); mpath = D.path.map(key)
    in_dev = mts.isin(set(ts[~test])) | mpath.isin(set(paths[~test]))
    in_test = mts.isin(set(ts[test])) | mpath.isin(set(paths[test]))
    if in_dev.any():
        refuse(f"{int(in_dev.sum())} manifest rows are development ECGs (rows {rows_text(in_dev)}); a test set must hold new ECGs")
    if in_test.any() and not authorised:
        refuse(f"{int(in_test.sum())} manifest rows are locked-test ECGs (rows {rows_text(in_test)}). They may be scored only with the "
               "owner's explicit authorisation (--locked-test-authorised) after the 11 + 3 near-identical name pairs are resolved "
               "(phase_6/PHASE6_HANDOFF.md, known issue 2)")
    return in_test.values


def ledger(msha, name, again):
    L = json.loads(K.LEDGER.read_text()) if K.LEDGER.exists() else []
    if any(x["name"] == name for x in L) or (K.RUNSK / name).exists():
        refuse(f"a run named '{name}' exists; choose another --name")
    prev = [x for x in L if x["manifest_sha256"] == msha]
    if prev and not again:
        refuse(f"this manifest was already scored ({prev[0]['time']}, run '{prev[0]['name']}'). Scoring is one-shot; a re-run needs "
               '--again "<reason>", and the reason is logged and reported')
    return L, prev


# ------------------------------------------------------------------ metrics
def wilson(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, 1.959964
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def safe(f, m):
    return float(f(m)) if np.any(m) else None


def fourclass(y, p):
    cm = np.bincount(4 * y + p, minlength=16).reshape(4, 4)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1); prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0); present = cm.sum(1) > 0
    return dict(n=int(len(y)), accuracy=float(np.trace(cm) / max(len(y), 1)), balanced_accuracy=float(rec[present].mean()) if present.any() else None,
                macro_f1=float(f1.mean()), per_class={c: dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]), n_true=int(cm[i].sum()),
                                                            n_predicted=int(cm[:, i].sum())) for i, c in enumerate(CL)},
                confusion_rows_true_cols_pred=cm.tolist())


def triage_stats(y, t):
    ans = t >= 0; nrm = ans & (t == 0); red = ans & (t >= 2)
    return dict(n=int(len(y)), answered=int(ans.sum()), answered_share=float(ans.mean()) if len(y) else None,
                accuracy_answered=safe(lambda m: np.mean(t[m] == y[m]), ans),
                predicted_mix_answered={c: int(np.sum(t == i)) for i, c in enumerate(CL)},
                true_mix_answered={c: int(np.sum(ans & (y == i))) for i, c in enumerate(CL)},
                recall_among_answered={c: safe(lambda m: np.mean(t[m] == i), ans & (y == i)) for i, c in enumerate(CL)},
                normal_calls_truly={c: safe(lambda m: np.mean(y[m] == i), nrm) for i, c in enumerate(CL)},
                normal_calls_without_ef_below_41=safe(lambda m: np.mean(y[m] <= 1), nrm),
                all_ef_below_41_given_normal=safe(lambda m: np.mean(nrm[m]), y >= 2),
                all_mild_given_normal=safe(lambda m: np.mean(nrm[m]), y == 1),
                mild_caught_of_all_mild=safe(lambda m: np.mean(ans[m] & (t[m] == 1)), y == 1),
                modsev_calls_truly_ef_below_41=safe(lambda m: np.mean(y[m] >= 2), red))


def key_scalars(y, t, c3):
    tr, fc = triage_stats(y, t), fourclass(y, c3)
    return {"triage.answered_share": tr["answered_share"], "triage.accuracy_answered": tr["accuracy_answered"],
            "triage.normal_calls_without_ef_below_41": tr["normal_calls_without_ef_below_41"],
            "triage.all_ef_below_41_given_normal": tr["all_ef_below_41_given_normal"], "triage.mild_caught_of_all_mild": tr["mild_caught_of_all_mild"],
            "imgc3.accuracy": fc["accuracy"], "imgc3.balanced_accuracy": fc["balanced_accuracy"], "imgc3.macro_f1": fc["macro_f1"],
            "imgc3.recall_Mild": fc["per_class"]["Mild"]["recall"], "imgc3.precision_Mild": fc["per_class"]["Mild"]["precision"]}


def bootstrap(y, t, c3, groups, nboot=NBOOT, seed=0):
    """Patient-cluster bootstrap percentile CIs of the key scalars."""
    ug = pd.unique(groups); pos = {g: np.flatnonzero(groups == g) for g in ug}; rng = np.random.default_rng(seed)
    draws = []
    for _ in range(nboot):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), replace=True)])
        draws.append(key_scalars(y[idx], t[idx], c3[idx]))
    out = {}
    for k in draws[0]:
        v = np.array([d[k] for d in draws if d[k] is not None], float)
        out[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 0.9 * nboot else None
    return out


def analyse(D, o, m):
    y, t, c3 = D.y.values[m], o["triage"][m], o["c3"][m]
    tr, fc = triage_stats(y, t), fourclass(y, c3)
    tr["answered_share_ci95_wilson"] = wilson(tr["answered"], tr["n"])
    tr["accuracy_answered_ci95_wilson"] = wilson(int(np.sum((t == y) & (t >= 0))), tr["answered"])
    fc["accuracy_ci95_wilson"] = wilson(int(np.sum(c3 == y)), fc["n"])
    err = o["ef_est"][m] - D.ef_n.values[m]
    efe = dict(n=int(m.sum()), mean_error=float(err.mean()), sd_error=float(err.std(ddof=1)) if m.sum() > 1 else None,
               mae=float(np.abs(err).mean()), pearson_r=float(np.corrcoef(o["ef_est"][m], D.ef_n.values[m])[0, 1]) if m.sum() > 2 else None)
    return dict(n_ecgs=int(m.sum()), n_patients=int(pd.unique(D.group.values[m]).size), true_class_counts={c: int(np.sum(y == i)) for i, c in enumerate(CL)},
                triage=tr, imgc3_all_patients=fc, ef_estimate_error=efe, bootstrap_ci95_patient_cluster=bootstrap(y, t, c3, D.group.values[m]))


def versus_cv(res, cv):
    b = res["bootstrap_ci95_patient_cluster"]; rows = []
    for k, cvv in (("triage.answered_share", cv["triage"]["answered_share"]), ("triage.accuracy_answered", cv["triage"]["accuracy_answered"]),
                   ("imgc3.accuracy", cv["imgc3_all"]["accuracy"]), ("imgc3.macro_f1", cv["imgc3_all"]["macro_f1"]),
                   ("imgc3.recall_Mild", cv["imgc3_all"]["recall_Mild"])):
        a, f = k.split("."); sec = res["triage"] if a == "triage" else res["imgc3_all_patients"]
        test = sec[f] if f in sec else sec["per_class"]["Mild"]["recall"]
        ci = b.get(k)
        rows.append(dict(metric=k, cv_expectation=float(cvv), test=test, test_ci95=ci,
                         cv_inside_test_ci=None if ci is None or test is None else bool(ci[0] <= cvv <= ci[1])))
    return rows


# ------------------------------------------------------------------ main
def pipeline_frame(n):
    """Development ECGs for the end-to-end pipeline test (training data: numbers are NOT estimates)."""
    man = pd.read_csv(C.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id")
    idx = np.sort(np.random.default_rng(1).choice(C.nD, n, replace=False)); T = C.B.table()
    D = pd.DataFrame(dict(ecg_file=[str(C.ROOT / man.loc[C.IDS[i], "source"]) for i in idx], patient_id=[f"g{i}" for i in pd.factorize(C.PAT)[0][idx]],
                          sex=np.where(T.male.values[idx], "M", "F"), ef=T.ef.values[idx].astype(str), echo_gap_days="0", ef_method="development label"))
    tmp = C.guard7(K.KIT / "pipeline_test" / "manifest.csv"); tmp.parent.mkdir(parents=True, exist_ok=True); D.to_csv(tmp, index=False)
    return tmp


def score(manifest, name, authorised, again, pipeline_test=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", name):
        refuse("--name: letters, digits, _ and - only")
    fr = K.Frozen.load(); card = json.loads(K.CARD.read_text())
    if "pdf_reproduction" not in card.get("checks", {}):
        refuse("the frozen model has not passed its PDF self-test (freeze --selftest)")
    D = read_manifest(manifest)
    msha = K.sha256(manifest)
    if pipeline_test:                                            # development ECGs on purpose: skip guards and ledger
        locked, L, prev = np.zeros(len(D), bool), None, []
    else:
        locked = guards(D, authorised); L, prev = ledger(msha, name, again)
    t0 = time.time(); n = len(D); cols, st = fr.d["clin_cols"], fr.d["mildfeat"]
    F6 = np.full((n, 1024), np.nan, np.float32); IM = np.full((n, len(cols)), np.nan); err = np.zeros(n, bool)
    for i, p in enumerate(D.path):
        if D.status[i] == "missing_file":
            continue
        f, c, e = K.inputs_from_pdf(p, cols, st)
        if e:
            err[i] = True
        else:
            F6[i], IM[i] = f, c
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"read {i + 1}/{n} PDFs", flush=True)
    D.loc[err, "status"] = "unreadable_pdf"
    ok = (D.status == "ok").values
    if not ok.any():
        refuse(f"no usable ECG: {D.status.value_counts().to_dict()}")
    Fd = ecgf_f6().astype(float); Fd /= np.linalg.norm(Fd, axis=1, keepdims=True)
    Fn = F6[ok].astype(float); Fn /= np.linalg.norm(Fn, axis=1, keepdims=True)
    dup_dev = np.zeros(n, bool); dup_dev[ok] = (Fn @ Fd.T).max(1) >= DUP_COS
    if dup_dev.any() and not pipeline_test:
        refuse(f"{int(dup_dev.sum())} ECGs are near-identical to development ECGs (rows {rows_text(dup_dev)}): remove them; a test set "
               "must hold new ECGs")
    S = Fn @ Fn.T; np.fill_diagonal(S, 0); dup_within = int((np.triu(S, 1) >= DUP_COS).sum())
    male = (D.sex_n == "M").values
    o = {k: np.full(n, -1) for k in ("c3", "grade", "triage")}; o.update(ef_est=np.full(n, np.nan), dist=np.full(n, np.nan),
                                                                        P=np.full((n, 4), np.nan), answered=np.zeros(n, bool))
    r = fr.run(F6[ok], IM[ok], male[ok])
    for k in o:
        o[k][ok] = r[k]
    gap_ok = (D.gap_n <= MAX_GAP).values
    prim = ok & gap_ok
    res = dict(run=name, created=time.strftime("%F %T"), git=C.B.git_head(), frozen_sha256=card["frozen_sha256"], manifest_sha256=msha,
               locked_test_rows=int(locked.sum()), rescore_reason=again or None, previous_scorings_of_this_manifest=len(prev),
               rows=dict(total=n, **{s: int((D.status == s).sum()) for s in ("ok", "missing_file", "unreadable_pdf", "bad_ef", "bad_sex", "no_label")},
                         echo_gap_over_30_or_unknown=int((ok & ~gap_ok).sum()), near_duplicate_pairs_within_test=dup_within,
                         without_patient_id=int((D.patient_id.str.strip() == "").sum())),
               ef_methods=D.ef_method[prim].str.strip().replace("", "unspecified").value_counts().to_dict(),
               primary=analyse(D, o, prim) if prim.any() else None,
               sensitivity_all_readable_any_gap=analyse(D, o, ok) if ok.any() else None,
               cv_expectation=card["cv_expectation"], minutes=round((time.time() - t0) / 60, 1))
    res["primary_vs_cv"] = versus_cv(res["primary"], card["cv_expectation"]) if prim.any() else None
    if pipeline_test:
        res["WARNING"] = "pipeline test on development (training) ECGs: numbers are NOT estimates"
        res["development_duplicates_detected"] = int(dup_dev.sum())
    out = C.guard7((K.KIT / "pipeline_test") if pipeline_test else (K.RUNSK / name)); out.mkdir(parents=True, exist_ok=True)
    pe = pd.DataFrame(dict(manifest_row=np.arange(n), group=D.group, sex=D.sex_n, ef=D.ef_n, label=D.label, status=D.status,
                           echo_gap_days=D.gap_n, primary=prim, ef_estimate=o["ef_est"], ef_grade=o["grade"], imgc3=o["c3"],
                           distance_to_cut=o["dist"], answered=o["answered"], triage=o["triage"]))
    for j, c in enumerate(CL):
        pe[f"P_vote_{c}"] = o["P"][:, j]
    pe.to_csv(out / "per_ecg.csv", index=False)                                     # private (git-ignored)
    if pipeline_test:
        (out / "scores.json").write_text(json.dumps(res, indent=1, default=float)); report(res); print(res["WARNING"])
        return res
    C.guard7(K.RESK).mkdir(parents=True, exist_ok=True)
    (K.RESK / f"{name}_scores.json").write_text(json.dumps(res, indent=1, default=float))
    L.append(dict(name=name, time=res["created"], manifest_sha256=msha, frozen_sha256=card["frozen_sha256"], n=n, reason=again or None))
    C.guard7(K.LEDGER).write_text(json.dumps(L, indent=1))
    report(res)


def fmt(v, ci=None):
    if v is None:
        return "n/a"
    s = f"{v:.3f}"
    return s + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] is not None else "")


def report(res):
    print(f"\nrun {res['run']}: {res['rows']}")
    p = res["primary"]
    if p is None:
        print("no ECG qualifies for the primary analysis"); return
    tr, fc, b = p["triage"], p["imgc3_all_patients"], p["bootstrap_ci95_patient_cluster"]
    print(f"primary set: {p['n_ecgs']} ECGs, {p['n_patients']} patients, true classes {p['true_class_counts']}")
    print(f"TRIAGE  answered {fmt(tr['answered_share'], b['triage.answered_share'])}  accuracy on answered "
          f"{fmt(tr['accuracy_answered'], b['triage.accuracy_answered'])}  mix {tr['predicted_mix_answered']}")
    print(f"        Normal calls without EF < 41: {fmt(tr['normal_calls_without_ef_below_41'])}; EF < 41 given Normal: "
          f"{fmt(tr['all_ef_below_41_given_normal'])}; Mild caught: {fmt(tr['mild_caught_of_all_mild'])}")
    print(f"IMG-C3  all patients accuracy {fmt(fc['accuracy'], b['imgc3.accuracy'])}  macro-F1 {fmt(fc['macro_f1'], b['imgc3.macro_f1'])}  "
          f"balanced {fmt(fc['balanced_accuracy'])}  Mild P/R {fmt(fc['per_class']['Mild']['precision'])}/{fmt(fc['per_class']['Mild']['recall'])}")
    print(f"EF estimate error: {p['ef_estimate_error']}")
    for r in res["primary_vs_cv"]:
        print(f"  {r['metric']:28s} CV {r['cv_expectation']:.3f}  test {fmt(r['test'], r['test_ci95'])}  CV inside test CI: {r['cv_inside_test_ci']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton"); ap.add_argument("--out")
    ap.add_argument("--manifest"); ap.add_argument("--name")
    ap.add_argument("--locked-test-authorised", action="store_true"); ap.add_argument("--again", default=None)
    ap.add_argument("--pipeline-test", type=int, default=0, help="end-to-end run on N development ECGs (private output; not an estimate)")
    a = ap.parse_args()
    if a.pipeline_test:
        score(pipeline_frame(a.pipeline_test), "pipeline_test", False, None, pipeline_test=True)
    elif a.skeleton:
        if not a.out:
            refuse("--skeleton needs --out")
        skeleton(a.skeleton, a.out)
    elif a.manifest and a.name:
        score(a.manifest, a.name, a.locked_test_authorised, a.again)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
