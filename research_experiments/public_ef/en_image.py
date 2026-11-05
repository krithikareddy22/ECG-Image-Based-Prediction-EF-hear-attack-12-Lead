"""Public-EF Amendment 1 (AMENDMENT_1_IMAGE_ONLY.md): image-only, all-feature EchoNext members and stacks.
Laptop only. Per-ECG arrays stay in the git-ignored data/private/; results are aggregates. Never prints EchoNext rows.
Used by en_stream.py after every download batch; `python -m research_experiments.public_ef.en_image` runs it once on the
current en_cache.npz (which must carry the `tab` features).
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import medfilt, resample_poly
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from research_experiments.phase_5.features import extract as X5
from research_experiments.phase_5.p5common import P2, corr_ci, corr_t, holm
from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
from research_experiments.phase_6b_threshold_sampling_lstm import mildfeat as MF
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import outer_ecgfmlp, sfx
from research_experiments.public_ef import en_prep as P
from research_experiments.public_ef import en_train as T

warnings.filterwarnings("ignore")
MF.TEMODE, MF.PFRAC = "median", 0.5                      # validated MILDFEAT settings (QRS threshold 0.1 below)
FAMS4 = ("qrs", "st", "hrv", "freq", "wavelet", "leadstat")
QUALITY = ("valid_leads", "min_beats_per_column", "n_beats")
CS_ALL = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GBM = dict(num_leaves=7, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
           min_child_samples=15, class_weight="balanced", random_state=0, verbose=-1)
HOSP = P.PRIV / "hospital_inputs"
Y, EF, nD = B.Y, E.EF, B.nD


# ------------------------------------------------------------------------------------------------ features (G3, G4)
def tab_features(leads, rhythm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = X5.one((np.asarray(leads, np.float64), np.asarray(rhythm, np.float64)))
        d = {f"{f}:{k}": v for f in FAMS4 for k, v in o[f].items()}
        try:
            c = MF.features(np.asarray(leads, np.float32), np.asarray(rhythm, np.float32), 0.1)
        except Exception:                                                # noqa: BLE001  (row stays missing)
            c = {}
    d.update({f"clin:{k}": v for k, v in c.items() if k not in QUALITY and not k.startswith("_")})
    return d


def tab_frame(rows, cols=None):
    F = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return F if cols is None else F.reindex(columns=list(cols))


# ------------------------------------------------------------------------------------- harmonised hospital inputs (H)
def _bounds():
    p = P.norm_params()
    return np.array(p["lowerbound"]) * P.MUSE_UV / 1000.0, np.array(p["upperbound"]) * P.MUSE_UV / 1000.0


def _h(x, lo, hi, n):
    y = np.asarray(x, float)[::2]                                       # 500 -> 250 Hz as IntroECG parse_xml
    y = y - medfilt(medfilt(y, 51), 151)                                # baseline removal: 0.2 s then 0.6 s median
    return resample_poly(np.clip(y, lo, hi), 2, 1)[:n]                  # published clipping; back to 500 Hz


def harmonise(leads, rhythm):
    lo, hi = _bounds()
    return (np.stack([_h(leads[i], lo[i], hi[i], 1250) for i in range(12)]).astype(np.float32),
            _h(rhythm, lo[1], hi[1], 5000).astype(np.float32))


def hospital_inputs(variant):
    """G1-G4 of the 837 development ECGs: 'plain' (existing caches + features) or 'H' (harmonised, recomputed)."""
    f = HOSP / f"{variant}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    sig = np.load(B.ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(sig["ecg_id"])), index=sig["ecg_id"]).loc[B.IDS].values     # development rows only
    L, R = sig["leads"][pos], sig["rhythm"][pos]
    rows, A, S = [], [], []
    for i in range(nD):
        lv, rv = harmonise(L[i], R[i]) if variant == "H" else (L[i], R[i])
        rows.append(tab_features(lv, rv))
        if variant == "H":
            A.append(P.build(lv, rv, "aligned")); S.append(P.synth_input(lv, rv))
    cols = None if variant == "plain" else hospital_inputs("plain")["cols"]
    tab = tab_frame(rows, cols)
    if variant == "H":
        model = P.load_model(False)
        oa, os_ = P.run(model, np.stack(A).astype(np.float32)), P.run(model, np.stack(S).astype(np.float32))
        f6a, la, f6s, ls = oa["f6"], oa["logits"], os_["f6"], os_["logits"]
    else:
        z = np.load(P2 / "models" / "ecgf_aligned_features.npz")
        p = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"]).loc[B.IDS].values
        b = E.build()
        f6a, la, f6s, ls = z["f6"][p], z["logits"][p], b["f6"], b["logits"]
    out = dict(f6a=np.asarray(f6a, np.float32), la=np.asarray(la, np.float32), f6s=np.asarray(f6s, np.float32),
               ls=np.asarray(ls, np.float32), tab=tab.values.astype(np.float32), cols=np.asarray(tab.columns, dtype="U64"))
    HOSP.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return out


# ----------------------------------------------------------------------------------------- EchoNext-trained members
class BlockStd:
    """Median imputation of the tabular block + per-column z-score + 1/sqrt(p) block weight (Phase 2 recipe)."""

    def fit(self, blocks):
        self.med = np.nanmedian(blocks[-1], 0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        bl = self._imp(blocks)
        self.mu, self.sd = [b.mean(0) for b in bl], [b.std(0) + 1e-6 for b in bl]
        return self

    def _imp(self, blocks):
        return list(blocks[:-1]) + [np.where(np.isnan(blocks[-1]), self.med[None], blocks[-1])]

    def transform(self, blocks):
        return np.concatenate([(b - m) / s / np.sqrt(b.shape[1]) for b, m, s in zip(self._imp(blocks), self.mu, self.sd)], 1)


def _blocks(d):
    return [np.asarray(d[k], float) for k in ("f6a", "f6s", "la", "ls", "tab")]


def _lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def fit_en_all(en, hosp):
    """en: EchoNext dict (f6a, f6s, la, ls, tab, y, ef, g). hosp: {variant: hospital dict}. -> internal, outputs."""
    from research_experiments.public_ef.en_split import locked          # Amendment 3: EchoNext locked test patients
    y, ef, g = en["y"], en["ef"], en["g"]
    L = locked(g); D = np.flatnonzero(~L)
    bl_all = _blocks(en)
    G_all = np.c_[bl_all[2], bl_all[3], bl_all[4]]
    bl, G, y_all, ef_all = [b[D] for b in bl_all], G_all[D], y, ef
    y, ef, g = y[D], ef[D], g[D]
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(G, y, g))
    bs = BlockStd().fit([b[tr] for b in bl])
    Xtr, Xva = bs.transform([b[tr] for b in bl]), bs.transform([b[va] for b in bl])
    C = min(CS_ALL, key=lambda c: U.logloss(y[va], _lr(c).fit(Xtr, y[tr]).predict_proba(Xva)))
    a = min(T.ALPHAS, key=lambda al: float(np.mean((Ridge(alpha=al).fit(Xtr, ef[tr]).predict(Xva) - ef[va]) ** 2)))
    pv = _lr(C).fit(Xtr, y[tr]).predict_proba(Xva)
    muv = Ridge(alpha=a).fit(Xtr, ef[tr]).predict(Xva)
    gb = lgb.LGBMClassifier(n_estimators=400, **GBM).fit(G[tr], y[tr], eval_set=[(G[va], y[va])],
                                                         callbacks=[lgb.early_stopping(50, verbose=False)])
    nr = int(gb.best_iteration_ or 400)
    pg = gb.predict_proba(G[va])
    internal = dict(n_train=int(len(tr)), n_heldout=int(len(va)), C=float(C), alpha=float(a), gbm_rounds=nr,
                    n_tab_features=int(bl[4].shape[1]),
                    heldout_cls_all=T.metrics(y[va], pv.argmax(1)), heldout_gbm=T.metrics(y[va], pg.argmax(1)),
                    heldout_mean=T.metrics(y[va], ((pv + pg) / 2).argmax(1)),
                    heldout_ef_r_all=float(pearsonr(muv, ef[va])[0]), heldout_ef_rmse_all=float(np.sqrt(np.mean((muv - ef[va]) ** 2))))
    bs = BlockStd().fit(bl)
    X = bs.transform(bl)
    lr, rg = _lr(C).fit(X, y), Ridge(alpha=a).fit(X, ef)
    gbf = lgb.LGBMClassifier(n_estimators=nr, **GBM).fit(G, y)
    if L.any():
        Xl = bs.transform([b[L] for b in bl_all])
        pc, pg_, ml = lr.predict_proba(Xl), gbf.predict_proba(G_all[L]), rg.predict(Xl)
        internal["echonext_locked_test"] = dict(n=int(L.sum()), cls_all=T.metrics(y_all[L], pc.argmax(1)),
                                                gbm=T.metrics(y_all[L], pg_.argmax(1)), mean=T.metrics(y_all[L], ((pc + pg_) / 2).argmax(1)),
                                                ef_r_all=float(pearsonr(ml, ef_all[L])[0]),
                                                ef_rmse_all=float(np.sqrt(np.mean((ml - ef_all[L]) ** 2))))
    outs = {}
    for v, d in hosp.items():
        hb = _blocks(d)
        Xh = bs.transform(hb)
        outs[v] = dict(P_cls=lr.predict_proba(Xh), mu=rg.predict(Xh), P_gbm=gbf.predict_proba(np.c_[hb[2], hb[3], hb[4]]))
    return internal, outs


# ------------------------------------------------------------------------------------------------------------ stacks
def fold_any(r, k, Pv, members, mus):
    """The plan's stack (en_train.fold) with arbitrary fixed-function members and EF-mu band members."""
    from sklearn.model_selection import StratifiedGroupKFold
    from research_experiments.phase_6.p6common import PAT_GROUP, inner_oof
    from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
    TR, TE, _ = Bd.splits(r, k)
    inn = [inner_oof(b, r, k, H.SPEC[b]) for b in ("ecgfmlp", "ecgflindist")]
    Ins, Outs = {"M1": np.mean([x[1] for x in inn], 0)}, {"M1": Pv[TE]}
    for m in ("M5", "M6"):
        j = E.member_job(m, r, k)
        Ins[m], Outs[m] = j["inner"], j["outer"]
    mu = Bd.mu_job(r, k)
    zi, zo = [mu["inner"] / 50], [mu["outer"] / 50]
    for n, Pm in members.items():
        Ins[n], Outs[n] = Pm[TR], Pm[TE]
    for n, m_ in mus.items():
        sd = float(np.std(m_[TR] - EF[TR]))
        Ins[n], Outs[n] = T.band(m_[TR], sd, TR), T.band(m_[TE], sd, TE)
        zi.append(m_[TR] / 50); zo.append(m_[TE] / 50)
    names, yt = list(Ins), Y[TR]
    Zi = np.column_stack([np.concatenate([U.lp(Ins[n]) for n in names], 1), *zi])
    Zo = np.column_stack([np.concatenate([U.lp(Outs[n]) for n in names], 1), *zo])
    sp = list(StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, yt, PAT_GROUP()[TR]))
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        mo = np.zeros((len(TR), 4))
        for ia, ib in sp:
            mo[ib] = LogisticRegression(C=C, max_iter=5000).fit(Zi[ia], yt[ia]).predict_proba(Zi[ib])
        ll = U.logloss(yt, mo)
        if best is None or ll < best[1]:
            best = (C, ll, mo)
    so = LogisticRegression(C=best[0], max_iter=5000).fit(Zi, yt).predict_proba(Zo)
    return TE, (so * U.tune_floor(best[2], yt, 0.18)).argmax(1)


def transfer(mu, P_):
    m01, m12 = np.isin(Y, (0, 1)), np.isin(Y, (1, 2))
    return dict(auroc_ef_le_45=float(roc_auc_score(EF <= 45, -mu)), auroc_normal_vs_mild=float(roc_auc_score(Y[m01] == 1, -mu[m01])),
                auroc_mild_vs_moderate=float(roc_auc_score(Y[m12] == 2, -mu[m12])), ef_r_vs_echo=float(pearsonr(mu, EF)[0]),
                zero_shot_argmax=T.metrics(Y, P_.argmax(1)))


def evaluate_all(out_path):
    z = np.load(P.PRIV / "en_cache.npz", allow_pickle=False)
    en = dict(f6a=z["f6_aligned"], f6s=z["f6_synth"], la=z["logits_aligned"], ls=z["logits_synth"], tab=z["tab"],
              y=z["y"], ef=z["ef"], g=z["patient_key"])
    hosp = {v: hospital_inputs(v) for v in ("H", "plain")}
    int_plan, Pen, mu_en = T.train_en()
    int_all, outs = fit_en_all(en, hosp)
    oh, op = outs["H"], outs["plain"]
    out = dict(n_echonext=int(len(en["y"])), echonext_class_counts=np.bincount(en["y"], minlength=4).tolist(),
               internal_plan=int_plan, internal_all=int_all,
               transfer=dict(plan=transfer(mu_en, Pen), all_H=transfer(oh["mu"], (oh["P_cls"] + oh["P_gbm"]) / 2),
                             all_plain=transfer(op["mu"], (op["P_cls"] + op["P_gbm"]) / 2)),
               models={}, comparisons=[])
    cfg = {"stack-ALL": oh, "stack-ALL-plain": op}
    preds, cells = {}, {}
    for r in (0, 1, 2):
        Pm, _ = outer_ecgfmlp(r)
        Pv = (Pm + B.load_oof_csv(T.P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]) / 2
        names = ("EN zero-shot + tuned decision", "EN-all zero-shot + tuned decision", "stack-BASE", "stack-PLUS", *cfg)
        pr = {n: np.zeros(nD, int) for n in names}
        P0 = (oh["P_cls"] + oh["P_gbm"]) / 2
        for k in range(5):
            TR, TE, _ = Bd.splits(r, k)
            pr["EN zero-shot + tuned decision"][TE] = (Pen[TE] * U.tune_floor(Pen[TR], Y[TR], 0.18)).argmax(1)
            pr["EN-all zero-shot + tuned decision"][TE] = (P0[TE] * U.tune_floor(P0[TR], Y[TR], 0.18)).argmax(1)
            for n, plus in (("stack-BASE", False), ("stack-PLUS", True)):
                te, p = T.fold(r, k, Pv, Pen, mu_en, plus); pr[n][te] = p
            for n, o in cfg.items():
                te, p = fold_any(r, k, Pv, {"EN-cls": Pen, "EN-cls-all": o["P_cls"], "EN-gbm": o["P_gbm"]},
                                 {"EN-band": mu_en, "EN-band-all": o["mu"]})
                pr[n][te] = p
        pr["C3 (with header; reference)"] = B.load_run(f"P6A-c3-B1-v1__s0_r{r}")[1]
        fo = B.fold_vec(r)
        for n, p in pr.items():
            preds.setdefault(n, []).append(p)
            cells.setdefault(n, []).extend(float(np.mean(p[fo == k] == Y[fo == k])) for k in range(5))
    for n, ps in preds.items():
        ms = [T.metrics(Y, p) for p in ps]
        s = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        s.update(accuracy_ci95_corrected=corr_ci(cells[n]), accuracy_per_repeat=[round(m["accuracy"], 4) for m in ms],
                 admissible=bool(s["min_recall"] >= 0.15), image_only=not n.startswith("C3"))
        out["models"][n] = s
    pv = []
    for a, b in (("stack-PLUS", "stack-BASE"), ("stack-PLUS", "C3 (with header; reference)"), ("stack-ALL", "stack-BASE"),
                 ("stack-ALL", "stack-PLUS"), ("stack-ALL", "C3 (with header; reference)")):
        d = np.array(cells[a]) - np.array(cells[b])
        md, p = corr_t(d)
        pv.append(p)
        out["comparisons"].append(dict(model=a, vs=b, delta_accuracy=md, p=p,
                                       same_sign_all_repeats=bool(len({np.sign(d[5 * r:5 * r + 5].mean()) for r in range(3)}) == 1)))
    for c, h in zip(out["comparisons"], holm(pv)):
        c["p_holm"] = float(h)
    adm = {n: s for n, s in out["models"].items() if s["image_only"] and s["admissible"]}
    out["best_image_only_admissible_selected_after_the_fact"] = max(adm, key=lambda n: adm[n]["accuracy"]) if adm else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    o = evaluate_all(P.RES / "en_image_results.json")
    for n, s in o["models"].items():
        print(f"{n:36s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} min-rec {s['min_recall']:.2f} adm {s['admissible']}")
