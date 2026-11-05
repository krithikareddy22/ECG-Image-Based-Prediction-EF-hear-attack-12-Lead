"""Interim streaming run of the public-EF study (owner, 2026-11-06: "start with whatever data is downloaded and keep
updating every 50 mb of download").

While EchoNext_<split>_waveforms.npy is still downloading, every --step-mb of new data the complete ECG rows go through
exactly the en_prep steps (to_mv -> printout -> aligned + synth ECGFounder inputs) and are appended to a private cache.
en_cache.npz is then rewritten with the rows so far and en_train is re-run:
  stage A  EchoNext models (internal held-out check) + transfer AUROCs + zero-shot accuracy on the 837 hospital ECGs;
  stage B  (once the hospital inner caches exist) the full en_train.main(): stack-BASE / stack-PLUS / C3.
Row-order check: the heart rate measured on each waveform is correlated with the metadata ventricular_rate (rows that
belong together agree; shifted rows do not).
Interim outputs go only to the git-ignored data/private/interim/. They are NOT the pre-registered result, which is
en_prep + en_train on the finished val + test files. Never prints EchoNext rows.

    set ECHONEXT_DIR=C:\\krithika_major\\public_data\\echonext            (metadata)
    set ECHONEXT_MINI_DIR=...\\IntroECG\\7-EchoNext Minimodel
    python -m research_experiments.public_ef.en_stream --wave C:\\EchoNext\\echonext-1.1.1\\EchoNext_test_waveforms.npy --split test
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from research_experiments.public_ef import en_prep as P

INT = P.PRIV / "interim"
NS = 2500 * 12


def header(path):
    with open(path, "rb") as f:
        v = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        return shape, fortran, dtype, f.tell()


def read_rows(path, off, dtype, a, b):
    rb = NS * dtype.itemsize
    with open(path, "rb") as f:
        f.seek(off + a * rb)
        buf = f.read((b - a) * rb)
    assert len(buf) == (b - a) * rb, "short read"
    return np.frombuffer(buf, dtype=dtype).reshape(b - a, 1, 2500, 12)


def heart_rate(ii, fs=250):
    e = np.abs(sosfiltfilt(butter(2, [5, 30], btype="band", fs=fs, output="sos"), ii))
    pk, _ = find_peaks(e, distance=int(0.3 * fs), height=0.4 * np.percentile(e, 99))
    return 60.0 / (np.median(np.diff(pk)) / fs) if len(pk) >= 3 else np.nan


def inner_ready():
    from research_experiments.phase_6.p6common import INNER6
    fs = [INNER6 / "ecgfmlp" / f"r{r}_k{k}.npz" for r in range(3) for k in range(5)]
    return all(f.exists() and time.time() - f.stat().st_mtime > 10 for f in fs)


def evaluate(split, st, M, keep_idx):
    """Rewrite en_cache.npz with the rows so far and re-run en_train (stage A, or stage B when ready); with the
    tabular features present, run Amendment 1 (en_image.evaluate_all) instead of en_train.main."""
    from research_experiments.public_ef import en_train as T
    idx = st["row"]
    sel = M.iloc[idx]
    y = np.array([P.CLS[P.rule(e, "M" if s == "male" else "F")] for e, s in zip(sel.lvef_value.values, sel.sex.values)])
    np.savez(P.PRIV / "en_cache.npz", split=np.asarray([split] * len(idx), dtype="U8"), row=idx,
             patient_key=np.asarray([str(v) for v in sel.patient_key.to_numpy()], dtype="U64"),
             ef=sel.lvef_value.to_numpy(dtype=float), male=np.asarray(sel.sex.to_numpy() == "male", dtype=bool), y=y,
             **{k: st[k] for k in ("f6_aligned", "logits_aligned", "f6_synth", "logits_synth", "tab") if k in st})
    counts = np.bincount(y, minlength=4)
    res = dict(rows_used=int(len(idx)), class_counts=counts.tolist())
    if counts.min() < 10:
        res["note"] = "fewer than 10 ECGs in some class: models not fitted yet"
        return res
    if inner_ready() and "tab" in st:
        from research_experiments.public_ef import en_image as I
        out = I.evaluate_all(INT / "en_image_results.json")
        res.update(internal=out["internal_plan"], internal_all=out["internal_all"], transfer=out["transfer"]["plan"],
                   transfer_all=out["transfer"]["all_H"], models=out["models"], comparisons=out["comparisons"],
                   best=out["best_image_only_admissible_selected_after_the_fact"])
        return res
    if inner_ready():
        T.RES = INT
        T.main()
        out = json.loads((INT / "en_train_results.json").read_text())
        res.update(internal=out["internal_echonext"], transfer=out["transfer"], models=out["models"], comparisons=out["comparisons"])
        return res
    internal, Pen, mu = T.train_en()
    Y, EF = T.Y, T.EF
    m01, m12 = np.isin(Y, (0, 1)), np.isin(Y, (1, 2))
    res.update(internal=internal, transfer=dict(
        auroc_ef_le_45=float(roc_auc_score(EF <= 45, -mu)), auroc_normal_vs_mild=float(roc_auc_score(Y[m01] == 1, -mu[m01])),
        auroc_mild_vs_moderate=float(roc_auc_score(Y[m12] == 2, -mu[m12])), ef_r_vs_echo=float(pearsonr(mu, EF)[0]),
        zero_shot_argmax=T.metrics(Y, Pen.argmax(1))), note="stack evaluation waits for the hospital inner caches")
    return res


def line(mb, total_mb, n_raw, hr_r, hr_mad, res):
    s = f"UPDATE {mb:.0f}/{total_mb:.0f} MB | EchoNext ECGs {n_raw} read, {res['rows_used']} labelled " \
        f"(N/Mi/Mo/S {'/'.join(map(str, res['class_counts']))}) | row-order check: HR r={hr_r:.3f}, median |diff| {hr_mad:.1f} bpm"
    if "internal" in res:
        i, t = res["internal"], res["transfer"]
        s += f" | EN held-out acc {i['heldout']['accuracy']:.3f}, EF r {i['heldout_ef_r']:.3f}" \
             f" | hospital: EF<=45 AUROC {t['auroc_ef_le_45']:.3f}, N-vs-Mi {t['auroc_normal_vs_mild']:.3f}," \
             f" Mi-vs-Mo {t['auroc_mild_vs_moderate']:.3f}, EF r {t['ef_r_vs_echo']:.3f}, zero-shot acc {t['zero_shot_argmax']['accuracy']:.3f}"
    if "internal_all" in res:
        i, t = res["internal_all"], res["transfer_all"]
        s += f" | EN-all held-out acc {i['heldout_mean']['accuracy']:.3f}, EF r {i['heldout_ef_r_all']:.3f}" \
             f" | EN-all on hospital: EF<=45 AUROC {t['auroc_ef_le_45']:.3f}, N-vs-Mi {t['auroc_normal_vs_mild']:.3f}," \
             f" Mi-vs-Mo {t['auroc_mild_vs_moderate']:.3f}, EF r {t['ef_r_vs_echo']:.3f}"
    lt = res.get("internal_all", {}).get("echonext_locked_test")
    if lt:
        lp = res.get("internal", {}).get("echonext_locked_test", {})
        s += f"\n    EchoNext test set (locked 20 % of patients, n={lt['n']}): EN-all acc {lt['mean']['accuracy']:.3f}" \
             f" (rec {'/'.join(f'{lt["mean"][k]:.2f}' for k in ('recall_Normal', 'recall_Mild', 'recall_Moderate', 'recall_Severe'))})," \
             f" EF r {lt['ef_r_all']:.3f}; plan EN-cls acc {lp.get('accuracy', float('nan')):.3f}, EF r {lp.get('ef_r', float('nan')):.3f}"
    if "models" in res:
        for n, m in res["models"].items():
            s += f"\n    {n:36s} {m['accuracy']:.3f} [{m['accuracy_ci95_corrected'][0]:.3f}, {m['accuracy_ci95_corrected'][1]:.3f}]" \
                 f" rec {'/'.join(f'{m[k]:.2f}' for k in ('recall_Normal', 'recall_Mild', 'recall_Moderate', 'recall_Severe'))}" \
                 f"{'' if m.get('admissible', True) else ' (not admissible)'}"
        for c in res.get("comparisons", []):
            s += f"\n    {c['model']} vs {c['vs']}: {c['delta_accuracy']:+.3f}, Holm p {c.get('p_holm', float('nan')):.3f}"
        if res.get("best"):
            s += f"\n    best image-only admissible (selected after the fact): {res['best']}"
    if "note" in res:
        s += f" | {res['note']}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--step-mb", type=float, default=50)
    ap.add_argument("--poll", type=float, default=30)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    INT.mkdir(parents=True, exist_ok=True)
    src = P.source_dir()
    params = P.norm_params()
    M = pd.read_csv(src / "echonext_metadata_100k.csv")
    M = M[M.split == a.split].reset_index(drop=True)
    shape, fortran, dtype, off = header(a.wave)
    assert not fortran and shape[1:] == (1, 2500, 12) and shape[0] == len(M), f"{shape} vs {len(M)} metadata rows"
    total = off + int(np.prod(shape)) * dtype.itemsize
    keep = (M.lvef_value.notna() & M.sex.isin(["male", "female"])).values
    sf = INT / f"stream_{a.split}.npz"
    st = dict(np.load(sf)) if sf.exists() else dict(n_raw=np.array(0), normalised=np.array(-1), hr=np.zeros(0), row=np.zeros(0, int),
                                                     f6_aligned=np.zeros((0, 1024), np.float32), logits_aligned=np.zeros((0, 150), np.float32),
                                                     f6_synth=np.zeros((0, 1024), np.float32), logits_synth=np.zeros((0, 150), np.float32))
    model = P.load_model(False)
    from research_experiments.public_ef import en_image as I
    cols = I.hospital_inputs("plain")["cols"]
    I.hospital_inputs("H")
    if "tab" not in st and len(st["row"]):                              # Amendment 1 catch-up for rows done before it
        rows_ = st["row"]
        tabs = []
        for s in range(0, len(rows_), 256):
            r_ = rows_[s:s + 256]
            W = read_rows(a.wave, off, dtype, int(r_.min()), int(r_.max()) + 1)
            tabs += [I.tab_features(*P.printout(P.to_mv(W[i - r_.min(), 0], params, bool(st["normalised"])))) for i in r_]
        st["tab"] = I.tab_frame(tabs, cols).values.astype(np.float32)
        np.savez(sf, **st)
        print(f"INFO Amendment 1 features computed for the {len(rows_)} ECGs already processed", flush=True)
    elif "tab" not in st:
        st["tab"] = np.zeros((0, len(cols)), np.float32)
    last_bytes = off + int(st["n_raw"]) * NS * dtype.itemsize
    print(f"START {a.split}: {shape[0]} ECGs, {total / 1e6:.0f} MB; resuming at {int(st['n_raw'])} rows", flush=True)
    while True:
        size = os.path.getsize(a.wave)
        n_av = min(shape[0], (size - off) // (NS * dtype.itemsize))
        done = n_av == shape[0]
        if (size - last_bytes < a.step_mb * 1e6 and not done) or n_av <= int(st["n_raw"]):
            if done:
                print(f"DONE {a.split}: all {shape[0]} ECGs processed", flush=True)
                return
            time.sleep(a.poll)
            continue
        a0, b0 = int(st["n_raw"]), int(n_av)
        if int(st["normalised"]) < 0:
            if b0 < 200 and not done:
                time.sleep(a.poll); continue
            sd = float(np.asarray(read_rows(a.wave, off, dtype, 0, min(200, b0)), float).std())   # as en_prep
            st["normalised"] = np.array(int(sd < 5))
            print(f"INFO normalised input detected: {bool(st['normalised'])} (sample SD {sd:.3f})", flush=True)
        W = read_rows(a.wave, off, dtype, a0, b0)
        hr, rows, A, S, tb = [], [], [], [], []
        for j in range(b0 - a0):
            x = P.to_mv(W[j, 0], params, bool(st["normalised"]))
            hr.append(heart_rate(x[1]))
            if keep[a0 + j]:
                leads, rhythm = P.printout(x)
                A.append(P.build(leads, rhythm, "aligned")); S.append(P.synth_input(leads, rhythm)); rows.append(a0 + j)
                tb.append(I.tab_features(leads, rhythm))
        if rows:
            st["tab"] = np.concatenate([st["tab"], I.tab_frame(tb, cols).values.astype(np.float32)])
            oa, os_ = P.run(model, np.stack(A).astype(np.float32)), P.run(model, np.stack(S).astype(np.float32))
            for k, v in (("f6_aligned", oa["f6"]), ("logits_aligned", oa["logits"]), ("f6_synth", os_["f6"]), ("logits_synth", os_["logits"])):
                st[k] = np.concatenate([st[k], v.astype(np.float32)])
            st["row"] = np.concatenate([st["row"], np.array(rows, int)])
        st["hr"] = np.concatenate([st["hr"], np.array(hr)]); st["n_raw"] = np.array(b0)
        np.savez(sf, **st)
        last_bytes = off + b0 * NS * dtype.itemsize
        vr = M.ventricular_rate.to_numpy(float)[:b0]
        ok = np.isfinite(st["hr"]) & np.isfinite(vr)
        hr_r = float(pearsonr(st["hr"][ok], vr[ok])[0]) if ok.sum() > 10 else float("nan")
        hr_mad = float(np.median(np.abs(st["hr"][ok] - vr[ok]))) if ok.any() else float("nan")
        if ok.sum() > 50 and hr_r < 0.6:
            print(f"STOP row-order check failed: HR r={hr_r:.3f} (waveform rows do not match the metadata rows)", flush=True)
            return
        res = evaluate(a.split, st, M, keep)
        res.update(n_raw=b0, mb=size / 1e6, hr_r=hr_r, hr_mad=hr_mad, time=time.strftime("%H:%M:%S"))
        with open(INT / f"updates_{a.split}.jsonl", "a") as f:
            f.write(json.dumps(res, default=float) + "\n")
        print(line(size / 1e6, total / 1e6, b0, hr_r, hr_mad, res), flush=True)


if __name__ == "__main__":
    main()
