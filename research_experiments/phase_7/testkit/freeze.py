"""Freeze the header-free triage model (PHASE7_PLAN.md Amendment 5) on all 837 development ECGs, and run the pre-use
checks. Development data only; the locked test set is never touched. The committed card holds aggregates only.
    python -m research_experiments.phase_7.testkit.freeze --check         # recipe reproduction on outer repeat r0
    python -m research_experiments.phase_7.testkit.freeze --freeze        # choose on pooled outer OOF, fit on all 837
    python -m research_experiments.phase_7.testkit.freeze --selftest 25   # PDF reader vs cached inputs, same answers
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_2 import p2heads
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.experiments.nested import LIN, MLPF
from research_experiments.phase_5.p5common import decide, ecgf_f6
from research_experiments.phase_6b_threshold_sampling_lstm import mildfix as F1
from research_experiments.phase_7 import e6_cascade as E6
from research_experiments.phase_7 import p7common as C
from research_experiments.phase_7.testkit import kit as K

B = C.B
Y, PAT, nD = C.Y, C.PAT, C.nD
LINDIST = dict(LIN, loss="ce_dist", lam=1.0)
SPLIT_MEMBERS, SPLIT_EF = 5000, 5001
GRID = [(d0, t1, t2) for d0 in np.arange(-15, 6.5, 0.5) for t1 in np.arange(30, 55.5, 0.5) for t2 in np.arange(20, t1, 0.5)]
VOTE_RUN = "P6B-imgvote-IMGF-v1__s0_r{r}"


def Ef():
    return ecgf_f6().astype(float)


def clin_cols():
    from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import clin
    return list(clin()[1])


# ------------------------------------------------------------------ model parts
def member(cfg, seed, tr, va):
    X = Ef()
    f = p2heads.fit(X[tr], Y[tr], X[va], Y[va], dict(cfg, no_std=False), seed=seed)
    return dict(model=f["model"], mu=f["mu"], sd=f["sd"], seed=seed, cfg=dict(cfg), best_epoch=f.get("best_epoch"))


def members(tr, va):
    return dict(mlp=[member(MLPF, s, tr, va) for s in C.SEEDS], lin=member(LINDIST, 42, tr, va))


def vote(mem, X):
    return (np.mean([K.member_probs(m, X) for m in mem["mlp"]], 0) + K.member_probs(mem["lin"], X)) / 2


def fit_ef(model, wv, fit_rows, clean, split_seed):
    """E6 fitpred recipe, returning the fitted parts (applied to new ECGs by kit.ef_predict)."""
    if model == "H3":
        return dict(kind="H3", parts=[fit_ef("H1", wv, fit_rows, clean, split_seed), fit_ef("H2", wv, fit_rows, clean, split_seed)])
    ef = B.table().ef.values; Em = Ef(); Im = E6.img()
    if wv == 0.0:
        fit_rows = fit_rows[clean[fit_rows]]; w = np.ones(len(fit_rows))
    else:
        w = np.where(clean[fit_rows], 1.0, wv)
    fm = np.zeros(nD, bool); fm[fit_rows] = True
    if model == "H1":
        med = np.nanmedian(Im[fm], 0); med = np.where(np.isnan(med), 0.0, med)
        Ii = np.where(np.isnan(Im), med[None], Im)
        part = dict(kind="H1", med=med, muE=Em[fm].mean(0), sdE=Em[fm].std(0) + 1e-6, muI=Ii[fm].mean(0), sdI=Ii[fm].std(0) + 1e-6)
        X = np.concatenate([(Em - part["muE"]) / part["sdE"] / np.sqrt(Em.shape[1]), (Ii - part["muI"]) / part["sdI"] / np.sqrt(Ii.shape[1])], 1)
        part["ridge"] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[fit_rows], ef[fit_rows], sample_weight=w)
        return part
    import lightgbm as lgb
    mu, sd = Em[fm].mean(0), Em[fm].std(0) + 1e-6
    pca = PCA(32, random_state=0).fit(((Em - mu) / sd)[fm])
    X = np.column_stack([Im, pca.transform((Em - mu) / sd)])
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=split_seed).split(fit_rows, Y[fit_rows], PAT[fit_rows]))
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, random_state=0, n_jobs=1, verbose=-1)
    m.fit(X[fit_rows[a]], ef[fit_rows[a]], sample_weight=w[a], eval_set=[(X[fit_rows[b]], ef[fit_rows[b]])], eval_sample_weight=[w[b]],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return dict(kind="H2", mu=mu, sd=sd, pca=pca, lgbm=m)


def dist_to_cuts(e, bnd, p):
    return np.abs(e[:, None] - np.column_stack([bnd + p[0], np.full(len(e), p[1]), np.full(len(e), p[2])])).min(1)


# ------------------------------------------------------------------ card
def card_read():
    return json.loads(K.CARD.read_text()) if K.CARD.exists() else {}


def card_write(c):
    C.guard7(K.CARD).write_text(json.dumps(c, indent=1, default=float))


def cv_expectation():
    t = json.loads((C.RES7 / "triage_scores.json").read_text())
    e = pd.read_csv(C.RES7 / "e6_cascades.csv").set_index("model").loc["IMG-C3 (header-free reference)"]
    return dict(source="patient-grouped 5-fold CV x 3 repeats on the 837 development ECGs (results/triage_scores.json, e6_cascades.csv)",
                triage=dict(answered_share=t["answered_share"], answered_share_ci95=[0.48, 0.63], accuracy_answered=t["accuracy"],
                            accuracy_answered_ci95=t["accuracy_ci95_corrected"], macro_f1_answered=t["macro_f1"],
                            balanced_accuracy_answered=t["balanced_accuracy"], recall_Mild_answered=t["recall_Mild"]),
                imgc3_all=dict(accuracy=float(e.accuracy), accuracy_ci95=e.accuracy_ci95_corrected, macro_f1=float(e.macro_f1),
                               balanced_accuracy=float(e.balanced_accuracy), recall_Mild=float(e.recall_Mild), precision_Mild=float(e.precision_Mild)))


# ------------------------------------------------------------------ --check
def check():
    warnings.filterwarnings("ignore"); torch.set_num_threads(4)
    fv = B.fold_vec(0); P = np.zeros((nD, 4)); X = Ef()
    for k in range(5):
        itr, iva, ote = B.split_indices(0, k)
        P[ote] = vote(members(itr, iva), X[ote])
        print(f"members fold {k} done", flush=True)
    Pc = B.load_run(VOTE_RUN.format(r=0))[0]
    out = dict(imgvote_r0=dict(accuracy_recomputed=float(np.mean(P.argmax(1) == Y)), accuracy_committed=float(np.mean(Pc.argmax(1) == Y)),
                               argmax_agreement=float(np.mean(P.argmax(1) == Pc.argmax(1))), mean_abs_prob_diff=float(np.abs(P - Pc).mean())))
    clean = C.verified()[0]; TR, ote = np.flatnonzero(fv != 0), np.flatnonzero(fv == 0); Im = E6.img()
    diffs = {}
    for m in ("H1", "H2", "H3"):
        for w, wv in E6.WEIGHTS.items():
            mine = K.ef_predict(fit_ef(m, wv, TR, clean, 3000), X[ote], Im[ote])
            ref = (E6.fitpred("H1", wv, TR, ote, clean, 3000) + E6.fitpred("H2", wv, TR, ote, clean, 3000)) / 2 if m == "H3" \
                else E6.fitpred(m, wv, TR, ote, clean, 3000)
            diffs[f"{m}_{w}"] = float(np.abs(mine - ref).max())
    out["ef_codepath_max_abs_diff_vs_e6_r0_fold0"] = diffs
    c = card_read(); c.setdefault("checks", {})["recipe_reproduction"] = out; card_write(c)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ --freeze
def oof_ef(r):
    warnings.filterwarnings("ignore")
    clean = C.verified()[0]; fv = B.fold_vec(r); cur = {v: np.zeros(nD) for v in E6.VARIANTS}
    for k in range(5):
        TR, ote = np.flatnonzero(fv != k), np.flatnonzero(fv == k)
        for m in E6.MODELS_H:
            for w, wv in E6.WEIGHTS.items():
                cur[(m, w)][ote] = E6.fitpred(m, wv, TR, ote, clean, 3000 + 10 * r + k)
        for w in E6.WEIGHTS:
            cur[("H3", w)][ote] = (cur[("H1", w)][ote] + cur[("H2", w)][ote]) / 2
    print(f"EF OOF r{r} done", flush=True)
    return r, cur


def freeze():
    warnings.filterwarnings("ignore"); t0 = time.time()
    T = B.table(); clean = C.verified()[0]; ef = T.ef.values; bnd = T.sex_boundary.values
    prior = np.bincount(Y, minlength=4) / nD
    # 1. prior-rule alpha on the committed IMG-vote outer OOF, pooled over the 3 repeats
    Ppool = np.concatenate([B.load_run(VOTE_RUN.format(r=r))[0] for r in (0, 1, 2)]); ypool = np.tile(Y, 3)
    alpha = choose_alpha(Ppool, ypool, prior, "protected")
    # 2. EF variant by pooled outer-OOF RMSE on the verified ECGs
    with Pool(3) as pool:
        res = dict(pool.imap_unordered(oof_ef, (0, 1, 2)))
    efo = {v: np.concatenate([res[r][v] for r in (0, 1, 2)]) for v in E6.VARIANTS}
    cl3 = np.tile(clean, 3); ef3 = np.tile(ef, 3)
    rmse = {f"{m}_{w}": float(np.sqrt(np.mean((efo[(m, w)] - ef3)[cl3] ** 2))) for m, w in E6.VARIANTS}
    vs = min(E6.VARIANTS, key=lambda v: rmse[f"{v[0]}_{v[1]}"])
    e6sel = []
    for r in (0, 1, 2):                                   # agreement with E6 where E6 picked the same variant in a fold
        z = np.load(E6.OUTD / f"r{r}.npz"); prm = json.loads((E6.OUTD / f"params_r{r}.json").read_text()); fv = B.fold_vec(r)
        for p in prm:
            if p["selected"] == f"{vs[0]}_{vs[1]}":
                s = fv == p["fold"]; e6sel.append(float(np.abs(res[r][vs][s] - z["SEL"][s]).max()))
    # 3. grading cuts (floor objective) and answer threshold on the pooled OOF
    e_pool = efo[vs]; b_pool = np.tile(bnd, 3)
    p = F1.floor_best(((q, F1.ef_grade(e_pool, b_pool, *q)) for q in GRID), ypool) or (0.0, 41.0, 31.0)
    g = F1.ef_grade(e_pool, b_pool, *p); d = dist_to_cuts(e_pool, b_pool, p)
    c3 = decide(Ppool, "prior", alpha, prior); ag = c3 == g
    t_ans = float(F1.threshold(d[ag], (g == ypool)[ag]))
    keep = ag & (d >= t_ans)
    sel = dict(note="computed on the same pooled OOF used to choose these values: optimistic, NOT a performance estimate",
               answered_share=float(keep.mean()), accuracy_answered=float(np.mean(g[keep] == ypool[keep])) if keep.any() else None,
               imgc3_accuracy_all=float(np.mean(c3 == ypool)), grade_accuracy_all=float(np.mean(g == ypool)))
    print(f"alpha {alpha} | EF {vs} rmse {rmse[f'{vs[0]}_{vs[1]}']:.2f} | cuts {p} | t {t_ans:.2f} | sel {sel}", flush=True)
    # 4. final fits on all 837
    allr = np.arange(nD)
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=SPLIT_MEMBERS).split(allr, Y, PAT))
    mem = members(allr[a], allr[b])
    efm = fit_ef(vs[0], E6.WEIGHTS[vs[1]], allr, clean, SPLIT_EF)
    frozen = dict(mlp=mem["mlp"], lin=mem["lin"], alpha=float(alpha), prior=prior.tolist(), ef=efm, cuts=[float(x) for x in p],
                  t_answer=t_ans, clin_cols=clin_cols(), mildfeat=K.mildfeat_settings(), ef_variant=f"{vs[0]}_{vs[1]}",
                  created=time.strftime("%F %T"), git=B.git_head())
    C.guard7(K.KIT).mkdir(parents=True, exist_ok=True)
    torch.save(frozen, C.guard7(K.FROZEN))
    c = card_read()
    c.update(dict(
        model="header-free triage (AGT-img): answer when the image EF grade agrees with IMG-C3 and the EF estimate is far from every cut",
        created=frozen["created"], git_head_at_freeze=frozen["git"], frozen_file="phase_7/data/private/testkit/frozen_triage.pt (git-ignored)",
        frozen_sha256=K.sha256(K.FROZEN), n_development_ecgs=int(nD),
        recipe=dict(members="ECGFounder-f6 MLP seeds 42/43/44 + linear distance-CE seed 42; fit on all 837 with a grouped 80/20 selection split "
                            f"(random_state {SPLIT_MEMBERS})",
                    alpha="choose_alpha(protected) on committed IMG-vote outer OOF pooled over 3 repeats",
                    ef="E6 variant with lowest pooled outer-OOF RMSE on verified ECGs; refit on all 837 "
                       f"(H2 split random_state {SPLIT_EF})",
                    cuts_and_threshold="floor-objective cuts and distance threshold (answered accuracy >= 0.70, >= 20 answered) on the pooled outer OOF"),
        chosen=dict(alpha=float(alpha), prior=dict(zip(K.CLASSES, prior.round(4).tolist())), ef_variant=frozen["ef_variant"],
                    ef_rmse_pooled_oof_verified=rmse, cuts=dict(d0=float(p[0]), t1=float(p[1]), t2=float(p[2])), t_answer_ef_points=t_ans,
                    normal_mild_cut="sex boundary (52 men / 54 women) + d0", mildfeat=frozen["mildfeat"], n_image_features=len(frozen["clin_cols"])),
        selection_data=sel, ef_oof_vs_e6_max_abs_diff_same_variant=max(e6sel) if e6sel else None,
        members=[dict(kind=m["cfg"].get("head"), loss=m["cfg"].get("loss"), seed=m["seed"], best_epoch=m["best_epoch"]) for m in mem["mlp"] + [mem["lin"]]],
        cv_expectation=cv_expectation(), minutes=round((time.time() - t0) / 60, 1)))
    c.get("checks", {}).pop("pdf_reproduction", None)                       # a new frozen file needs a new self-test
    card_write(c)
    print("frozen:", K.FROZEN.name, c["frozen_sha256"][:12], f"{c['minutes']} min")


# ------------------------------------------------------------------ --selftest
def selftest(n):
    warnings.filterwarnings("ignore"); torch.set_num_threads(4)
    fr = K.Frozen.load()
    man = pd.read_csv(C.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id")
    idx = np.sort(np.random.default_rng(0).choice(nD, n, replace=False))
    cols, st = fr.d["clin_cols"], fr.d["mildfeat"]
    assert cols == clin_cols(), "frozen feature columns differ from the development feature table"
    F6c, Ic = Ef()[idx], E6.img()[idx]
    F6p, Ip, errors = np.full_like(F6c, np.nan), np.full_like(Ic, np.nan), 0
    for j, i in enumerate(idx):
        f, c, e = K.inputs_from_pdf(C.ROOT / man.loc[C.IDS[i], "source"], cols, st)          # path never printed
        if e:
            errors += 1; continue
        F6p[j], Ip[j] = f, c
    ok = ~np.isnan(F6p).any(1)
    cos = np.sum(F6p[ok] * F6c[ok], 1) / (np.linalg.norm(F6p[ok], axis=1) * np.linalg.norm(F6c[ok], axis=1))
    eq = np.isclose(Ip[ok], Ic[ok], rtol=1e-5, atol=1e-6, equal_nan=True)
    male = B.table().male.values[idx]
    a, b = fr.run(F6p[ok], Ip[ok], male[ok]), fr.run(F6c[ok], Ic[ok], male[ok])
    full = fr.run(Ef(), E6.img(), B.table().male.values); keep = full["answered"]
    out = dict(n_sampled=int(n), sample_seed=0, unreadable=int(errors),
               f6_min_cosine=float(cos.min()), f6_max_abs_diff=float(np.abs(F6p[ok] - F6c[ok]).max()),
               image_features_equal_fraction=float(eq.mean()), rows_all_image_features_equal=float(eq.all(1).mean()),
               same_triage_decision=float(np.mean(a["triage"] == b["triage"])), same_imgc3_class=float(np.mean(a["c3"] == b["c3"])),
               ef_estimate_max_abs_diff=float(np.abs(a["ef_est"] - b["ef_est"]).max()),
               in_sample_sanity=dict(note="frozen model on its own 837 training ECGs (cached inputs): NOT a performance estimate",
                                     answered_share=float(keep.mean()), accuracy_answered=float(np.mean(full["triage"][keep] == Y[keep])),
                                     imgc3_accuracy_all=float(np.mean(full["c3"] == Y))))
    c = card_read(); c.setdefault("checks", {})["pdf_reproduction"] = out; card_write(c)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true"); ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--selftest", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        check()
    if a.freeze:
        freeze()
    if a.selftest:
        selftest(a.selftest)


if __name__ == "__main__":
    main()
