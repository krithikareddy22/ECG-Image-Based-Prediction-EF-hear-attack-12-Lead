"""Test kit shared code (PHASE7_PLAN.md Amendment 5): PDF -> image inputs, the frozen header-free triage model, guards.

Inputs of the model, all from the ECG drawing in the PDF (no printed machine text, age or header fields):
  * ECGFounder f6 (1,024-d) of the digitised waveform: ecg_ef.signals.extract_signals -> ecgfounder 'aligned' input with
    the undrawn gaps masked -> frozen ECGFounder, stage-6 time mean (the code path of phase_2/ecgf_features.py);
  * the image clinician features (phase_6b mildfeat.features with its calibrated settings; the 118 model columns =
    mildfeat's 121 minus 3 beat-count quality columns), rounded to 5 significant digits like the development table.
The patient's sex places the Normal / Mild cut (52 men, 54 women), exactly as in the label definition.
The patient name is never read or printed; a file name is used only for the 14-digit timestamp in square brackets.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import torch

from research_experiments.phase_5.p5common import decide
from research_experiments.phase_6b_threshold_sampling_lstm import mildfix as F1
from research_experiments.phase_7 import p7common as C

KIT = C.PRIV7 / "testkit"
FROZEN = KIT / "frozen_triage.pt"
LEDGER = KIT / "ledger.json"
RUNSK = KIT / "runs"
CARD = C.RES7 / "testkit_frozen_card.json"
RESK = C.RES7 / "testkit"
TS = re.compile(r"\[(\d{14})\]")
MILDFEAT_SETTINGS = C.ROOT / "research_experiments" / "phase_6b_threshold_sampling_lstm" / "results" / "mildfeat_validation.json"
CLASSES = list(C.CLASSES)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def timestamp(path):
    m = TS.search(Path(path).name)
    return m.group(1) if m else None


# ------------------------------------------------------------------ PDF -> inputs
_NET = None


def ecgfounder():
    global _NET
    if _NET is None:
        from research_experiments.phase_2.ecgf_features import load_model
        _NET = load_model(False).eval()
    return _NET


def mildfeat_settings():
    return json.loads(MILDFEAT_SETTINGS.read_text())["chosen"]


@torch.no_grad()
def f6_from_signals(leads, rhythm, pdf):
    from research_experiments.ecgfounder.inputs import build
    from research_experiments.phase_2.audit_waveform import geometry
    from research_experiments.phase_2.ecgf_features import mask_gaps
    x = mask_gaps(build(leads, rhythm, "aligned"), geometry(str(Path(pdf).resolve())))
    m = ecgfounder(); out = m.first_activation(m.first_conv(torch.from_numpy(x[None])))
    for st in m.stage_list:
        out = st(out)
    return out.mean(-1).numpy()[0].astype(np.float32)


def clin_from_signals(leads, rhythm, cols, settings):
    from research_experiments.phase_6b_threshold_sampling_lstm import mildfeat as MF
    MF.TEMODE, MF.PFRAC = settings["t_end_mode"], settings["p_threshold"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            F = MF.features(leads, rhythm, settings["qrs_threshold"])
        except Exception:                                                  # noqa: BLE001 (as in mildfeat: all-NaN row)
            F = {}
    v = np.array([F.get(c, np.nan) for c in cols], float)
    # the development table (mildfeat.csv) was written with float_format "%.5g": round identically
    return np.array([float(f"{x:.5g}") if np.isfinite(x) else x for x in v])


def inputs_from_pdf(pdf, cols, settings):
    """-> (f6 [1024], clin [len(cols)], error or None). Never raises for an unreadable PDF."""
    from ecg_ef.signals import extract_signals
    try:
        leads, rhythm = extract_signals(pdf)
        return f6_from_signals(leads, rhythm, pdf), clin_from_signals(leads, rhythm, cols, settings), None
    except Exception as e:                                                 # noqa: BLE001
        return None, None, type(e).__name__


# ------------------------------------------------------------------ frozen model
def member_probs(mem, X):
    x = torch.tensor((np.asarray(X, float) - mem["mu"]) / mem["sd"], dtype=torch.float32)
    with torch.no_grad():
        return mem["model"].probs(x).numpy()


def ef_predict(part, E, I):
    E = np.asarray(E, float); I = np.asarray(I, float)
    if part["kind"] == "H3":
        return np.mean([ef_predict(p, E, I) for p in part["parts"]], 0)
    if part["kind"] == "H1":
        Ii = np.where(np.isnan(I), part["med"][None], I)
        X = np.concatenate([(E - part["muE"]) / part["sdE"] / np.sqrt(E.shape[1]),
                            (Ii - part["muI"]) / part["sdI"] / np.sqrt(I.shape[1])], 1)
        return part["ridge"].predict(X)
    X = np.column_stack([I, part["pca"].transform((E - part["mu"]) / part["sd"])])
    return part["lgbm"].predict(X)


class Frozen:
    """The header-free triage model (AGT-img) with every value fixed at freeze time."""

    def __init__(self, d):
        self.d = d

    @classmethod
    def load(cls, path=FROZEN, card=CARD):
        if not Path(path).exists():
            raise FileNotFoundError("frozen model missing: run python -m research_experiments.phase_7.testkit.freeze")
        want = json.loads(Path(card).read_text())["frozen_sha256"]
        if sha256(path) != want:
            raise PermissionError("frozen model hash differs from results/testkit_frozen_card.json: refusing to score")
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def vote(self, F6):
        pm = np.mean([member_probs(m, F6) for m in self.d["mlp"]], 0)
        return (pm + member_probs(self.d["lin"], F6)) / 2

    def run(self, F6, IMG, male):
        d = self.d; male = np.asarray(male, bool)
        P = self.vote(F6)
        c3 = decide(P, "prior", d["alpha"], np.asarray(d["prior"]))
        e = ef_predict(d["ef"], F6, IMG)
        bnd = np.where(male, 52.0, 54.0); d0, t1, t2 = d["cuts"]
        g = F1.ef_grade(e, bnd, d0, t1, t2)
        dist = np.abs(e[:, None] - np.column_stack([bnd + d0, np.full(len(e), t1), np.full(len(e), t2)])).min(1)
        ans = (c3 == g) & (dist >= d["t_answer"])
        return dict(P=P, c3=c3, ef_est=e, grade=g, dist=dist, answered=ans, triage=np.where(ans, g, -1))
