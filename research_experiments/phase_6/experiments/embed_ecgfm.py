"""Experiment B (PHASE6_PLAN.md section 4.2): frozen ECG-FM embeddings for the 837 DEVELOPMENT ECGs (label-free).

Encoder: ECG-FM (wanglab/ecg-fm @ 584219ea, mimic_iv_ecg_physionet_pretrained.pt, MIT licence; configs/encoders.json),
re-implemented here in plain PyTorch from the checkpoint's parameter layout (fairseq-signals wav2vec2_cmsc, base size):
  conv feature extractor 4 x [Conv1d(k2, s2, no bias)] 12 -> 256 ch (GroupNorm(256) after the first conv), GELU
  -> LayerNorm(256) -> Linear(256 -> 768) -> + weight-normed grouped positional conv (k128, 16 groups, SamePad, GELU)
  -> LayerNorm -> 12 post-LN transformer layers (768, 12 heads, FFN 3072, GELU).
The checkpoint is loaded with torch.load(weights_only=True) (tensors only; no code is unpickled).
Input: the project's "aligned" 12 x 5000 construction (phase_2 build_p2: true lead timing, zeros where a lead was not
recorded) at 500 Hz, standardised per lead over its recorded samples (ECG-FM preprocessing standardises each lead),
cut into two non-overlapping 5-s windows (ECG-FM's pre-training segment length). Embedding = mean over time of the
final-layer output, averaged over the two windows (768-d).
Sanity check (label-free, aggregates): ridge CV prediction of printed heart rate and QRS width from the embedding,
compared with a randomly initialised copy of the same architecture and with ECGFounder f6.
Output: phase_6/data/private/embeddings/ecgfm.npz (git-ignored); results/ecgfm_embedding_validation.json
    python -m research_experiments.phase_6.experiments.embed_ecgfm
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_experiments.common import ROOT
from research_experiments.phase_6.experiments.embed_ecgf_cache import dev_aligned_inputs
from research_experiments.phase_6.p6common import EMB6, IDS, RES6, ecgf_f6, guard6

CKPT = ROOT / "research_experiments" / "models" / "pretrained" / "ecg-fm" / "mimic_iv_ecg_physionet_pretrained.pt"
SHA = "4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e"
OUT = EMB6 / "ecgfm.npz"


class Layer(nn.Module):
    def __init__(self, d=768, h=12, ff=3072):
        super().__init__()
        self.h = h
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))
        self.self_attn_layer_norm, self.final_layer_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x):
        B, T, D = x.shape; hd = D // self.h
        q = (self.q_proj(x) * hd ** -0.5).view(B, T, self.h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2), -1) @ v
        x = self.self_attn_layer_norm(x + self.out_proj(a.transpose(1, 2).reshape(B, T, D)))
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))


class ECGFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(12 if i == 0 else 256, 256, 2, 2, bias=False) for i in range(4)])
        self.gn = nn.GroupNorm(256, 256)
        self.layer_norm, self.post_extract_proj = nn.LayerNorm(256), nn.Linear(256, 768)
        self.pos_g = nn.Parameter(torch.ones(1, 1, 128)); self.pos_v = nn.Parameter(torch.zeros(768, 48, 128))
        self.pos_b = nn.Parameter(torch.zeros(768))
        self.enc_ln = nn.LayerNorm(768)
        self.layers = nn.ModuleList([Layer() for _ in range(12)])

    def load(self, m):
        sd = {}
        for i in range(4):
            sd[f"convs.{i}.weight"] = m[f"feature_extractor.conv_layers.{i}.0.weight"]
        sd.update({"gn.weight": m["feature_extractor.conv_layers.0.2.weight"], "gn.bias": m["feature_extractor.conv_layers.0.2.bias"],
                   "layer_norm.weight": m["layer_norm.weight"], "layer_norm.bias": m["layer_norm.bias"],
                   "post_extract_proj.weight": m["post_extract_proj.weight"], "post_extract_proj.bias": m["post_extract_proj.bias"],
                   "pos_g": m["conv_pos.pos_conv.0.weight_g"], "pos_v": m["conv_pos.pos_conv.0.weight_v"], "pos_b": m["conv_pos.pos_conv.0.bias"],
                   "enc_ln.weight": m["encoder.layer_norm.weight"], "enc_ln.bias": m["encoder.layer_norm.bias"]})
        for j in range(12):
            for n in ("q_proj", "k_proj", "v_proj", "out_proj", "self_attn_layer_norm", "final_layer_norm", "fc1", "fc2"):
                for t in ("weight", "bias"):
                    src = f"encoder.layers.{j}.self_attn.{n}.{t}" if n.endswith("_proj") else f"encoder.layers.{j}.{n}.{t}"
                    sd[f"layers.{j}.{n}.{t}"] = m[src]
        self.load_state_dict(sd, strict=True)
        return self

    def forward(self, x):                                   # x: (B, 12, 2500)
        for i, c in enumerate(self.convs):
            x = c(x)
            if i == 0:
                x = self.gn(x)
            x = F.gelu(x)
        x = self.post_extract_proj(self.layer_norm(x.transpose(1, 2)))          # (B, T', 768)
        w = self.pos_g * self.pos_v / self.pos_v.norm(dim=(0, 1), keepdim=True)   # weight_norm(dim=2)
        xc = F.conv1d(x.transpose(1, 2), w, self.pos_b, padding=64, groups=16)[:, :, :-1]   # SamePad (even kernel)
        x = self.enc_ln(x + F.gelu(xc).transpose(1, 2))
        for layer in self.layers:
            x = layer(x)
        return x.mean(1)


def standardise(X):
    """Per lead over recorded (non-zero) samples; zeros (not recorded) stay zero."""
    out = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(12):
            m = X[i, j] != 0
            if m.sum() > 10:
                v = X[i, j, m]; out[i, j, m] = (v - v.mean()) / (v.std() + 1e-6)
    return out


@torch.no_grad()
def embed(model, X, bs=8):
    E = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs])
        E.append(((model(x[:, :, :2500]) + model(x[:, :, 2500:])) / 2).numpy())
    return np.concatenate(E)


def ridge_cv_r(E, y, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ok = np.isfinite(y)
    p = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))), E[ok], y[ok],
                          cv=KFold(5, shuffle=True, random_state=seed))
    return float(np.corrcoef(p, y[ok])[0, 1])


def main():
    if OUT.exists():
        print("exists"); return
    torch.set_num_threads(2)
    h = hashlib.sha256(CKPT.read_bytes()).hexdigest(); assert h == SHA, "ECG-FM checkpoint hash mismatch"
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = ECGFM().load(ck["model"]).eval()
    t0 = time.time(); X, _ = dev_aligned_inputs(); X = standardise(X).astype(np.float32); tb = time.time() - t0
    t0 = time.time(); E = embed(model, X); tf = time.time() - t0
    torch.manual_seed(0); rnd = ECGFM().eval()
    for p in rnd.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    Er = embed(rnd, X[:837])
    H = pd.read_csv(ROOT / "data" / "header_features.csv").set_index("ecg_id").loc[IDS]
    val = dict(n_ecgs=len(IDS), dim=int(E.shape[1]), build_minutes=tb / 60, embed_minutes=tf / 60, nan_embeddings=int(np.isnan(E).sum()),
               sanity_ridge_cv_r={tgt: dict(ecgfm_pretrained=ridge_cv_r(E, H[tgt].values.astype(float)),
                                            ecgfm_random_init=ridge_cv_r(Er, H[tgt].values.astype(float)),
                                            ecgfounder_f6=ridge_cv_r(ecgf_f6().astype(float), H[tgt].values.astype(float)))
                                  for tgt in ("Rate", "QRSD", "QTcB", "PR")})
    print(json.dumps(val, indent=1))
    guard6(EMB6).mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ecg_id=np.asarray(IDS, dtype="U32"), E=E.astype(np.float32))
    (guard6(RES6) / "ecgfm_embedding_validation.json").write_text(json.dumps(val, indent=1))


if __name__ == "__main__":
    main()
