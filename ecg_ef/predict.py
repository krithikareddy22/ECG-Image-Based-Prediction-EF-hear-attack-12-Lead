"""Predict LVEF category + heart-failure likelihood for new 12-lead ECG PDFs/images.

    python -m ecg_ef.predict path/to/ecg.pdf
    python -m ecg_ef.predict some_folder/ --csv predictions.csv --gradcam heatmaps/
    python -m ecg_ef.predict ecg.png --model models/resnet18_filtered.pt

By default the model chosen by `python -m ecg_ef.compare` (results/best_model.json) is
used: a single CNN, a soft-voting ensemble, or a CNN fused with the model on the PDF
header's machine measurements/statements (ecg_ef/header.py). The header part needs the
PDF's text layer; for PNG/JPG inputs (or with --no-header) only the image CNN is used,
with its own screening threshold.

Output per ECG:
  * LVEF category: Normal / Mild / Moderate / Severe (+ probability of each)
  * Heart-failure likelihood = P(Mild) + P(Moderate) + P(Severe) = 1 - P(Normal)
  * Further cardiac evaluation needed: YES if likelihood >= the model's screening
    threshold (tuned on the validation set for >= 80% sensitivity), else NO

This is a research screening aid, not a diagnosis; confirm with echocardiography.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import CLASSES, FURTHER_EVALUATION, HF_THRESHOLD, RESULTS_DIR, ROOT
from .metrics import ordinal_decode
from .data import Normalizer
from .models import build_model
from .preprocess import load_waveform, prepare_for_model

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], ckpt.get("n_outputs", len(ckpt["classes"])))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


class Predictor:
    """One checkpoint or a soft-voting ensemble of several."""

    def __init__(self, ckpt_paths: list[Path], hf_threshold: float | None = None, name: str = "",
                 cutpoints: list[float] | None = None, decoding: str = "argmax",
                 header_model: Path | None = None, header_weight: float = 0.0,
                 image_hf_threshold: float | None = None):
        self.members = []
        for p in ckpt_paths:
            model, ckpt = load_checkpoint(p)
            self.members.append((model, Normalizer(**ckpt["normalizer"]), ckpt["variant"],
                                 tuple(ckpt["img_hw"]), ckpt))
        if hf_threshold is None:
            hf_threshold = self.members[0][4].get("hf_threshold", HF_THRESHOLD)
        self.hf_threshold = float(hf_threshold)
        self.name = name or ", ".join(p.stem for p in ckpt_paths)
        if cutpoints is None and len(ckpt_paths) == 1:
            ordj = RESULTS_DIR / ckpt_paths[0].stem / "ordinal.json"
            if ordj.exists():
                cutpoints = json.loads(ordj.read_text())["cutpoints"]
        if decoding == "ordinal" and cutpoints is None:
            raise SystemExit("ordinal decoding needs cut-points (run python -m ecg_ef.compare)")
        self.cutpoints = cutpoints if decoding == "ordinal" else None   # None -> argmax
        self.header = None
        self.header_weight = float(header_weight)
        self.image_hf_threshold = float(image_hf_threshold if image_hf_threshold is not None
                                        else self.members[0][4].get("hf_threshold", HF_THRESHOLD))
        if header_model is not None and self.header_weight > 0:
            import joblib
            self.header = joblib.load(header_model)
            if self.cutpoints is not None:
                raise SystemExit("ordinal cut-points are fitted for image-only models; "
                                 "use --no-header with --decoding ordinal")

    def category(self, probs: np.ndarray) -> int:
        if self.cutpoints is None:
            return int(probs.argmax())
        return int(ordinal_decode(probs[None], self.cutpoints)[0])

    @classmethod
    def from_args(cls, model_args, decoding: str = "argmax", use_header: bool = True):
        if model_args:
            paths = [Path(m) for m in model_args]
            if len(paths) == 1 and paths[0].suffix == ".json":
                return cls._from_json(paths[0], decoding, use_header)
            return cls(paths, decoding=decoding)
        best = RESULTS_DIR / "best_model.json"
        if not best.exists():
            raise SystemExit("No results/best_model.json - pass --model models/<name>.pt")
        return cls._from_json(best, decoding, use_header)

    @classmethod
    def _from_json(cls, path: Path, decoding: str, use_header: bool = True):
        spec = json.loads(path.read_text())
        header = spec.get("header_model") if use_header else None
        ckpts = [ROOT / c for c in spec["checkpoints"]]
        if header:
            return cls(ckpts, spec["hf_threshold"], spec.get("model", spec.get("tag", "")),
                       None, decoding, ROOT / header, spec["header_weight"],
                       spec.get("image_hf_threshold"))
        # image-only: its own threshold / cut-points
        thr = spec.get("image_hf_threshold", spec.get("hf_threshold")) if spec.get("header_model") \
            else spec.get("hf_threshold")
        name = spec.get("image_model", spec.get("model", spec.get("tag", "")))
        cut = spec.get("image_ordinal_cutpoints", spec.get("ordinal_cutpoints"))
        return cls(ckpts, thr, name, cut, decoding)

    def fuse(self, p_img: np.ndarray, path: Path) -> tuple[np.ndarray, float, str]:
        """Combine with the header model when possible -> (probs, threshold, source)."""
        if self.header is None:
            return p_img, self.hf_threshold, "image"
        if Path(path).suffix.lower() != ".pdf":
            return p_img, self.image_hf_threshold, "image only (no PDF text layer)"
        from .header import header_probs
        w = self.header_weight
        return w * header_probs(path, self.header) + (1 - w) * p_img, self.hf_threshold, \
            "image + header"

    def inputs(self, path: Path):
        cache = {}
        xs = []
        for model, norm, variant, hw, _ in self.members:
            if (variant, hw) not in cache:
                cache[(variant, hw)] = prepare_for_model(path, variant, hw)
            xs.append(norm(torch.from_numpy(cache[(variant, hw)])[None, None].float() / 255))
        return xs

    @torch.no_grad()
    def probs(self, xs) -> np.ndarray:
        return np.mean([F.softmax(m(x)[:, :len(CLASSES)], 1)[0].numpy()
                        for (m, *_), x in zip(self.members, xs)], 0)

    def gradcam(self, xs, target: int) -> np.ndarray:
        cams = [cv2.resize(gradcam(m, x, target), (512, 256), interpolation=cv2.INTER_CUBIC)
                for (m, *_), x in zip(self.members, xs)]
        cam = np.clip(np.mean(cams, 0), 0, None)
        return cam / cam.max() if cam.max() > 0 else cam


def gradcam(model, x, target: int) -> np.ndarray:
    """Grad-CAM on the last convolutional feature map -> [h, w] in [0, 1]."""
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    model.zero_grad(set_to_none=True)
    logits[0, target].backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1))[0].detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def save_gradcam(src: Path, cam: np.ndarray, title: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = load_waveform(src)
    h, w = rgb.shape[:2]
    heat = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgb); ax.imshow(heat, cmap="jet", alpha=0.35, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11); ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def iter_inputs(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED)
        else:
            yield p


def main():
    ap = argparse.ArgumentParser(description="Predict LVEF category from 12-lead ECG images")
    ap.add_argument("inputs", nargs="+", help="ECG PDF/image files or folders")
    ap.add_argument("--model", nargs="+", default=None,
                    help="checkpoint(s) .pt (several = ensemble) or a best_model.json "
                         "(default: results/best_model.json)")
    ap.add_argument("--csv", type=Path, default=None, help="write all predictions to CSV")
    ap.add_argument("--gradcam", type=Path, default=None, help="folder for Grad-CAM heatmaps")
    ap.add_argument("--no-header", action="store_true",
                    help="ignore the PDF header text; use the image CNN only")
    ap.add_argument("--decoding", choices=["argmax", "ordinal"], default="argmax",
                    help="4-class rule: most probable class (default) or ordinal severity bands "
                         "(more Mild/Moderate calls, but misses more Severe)")
    args = ap.parse_args()

    torch.set_num_threads(4)
    pred = Predictor.from_args(args.model, args.decoding, use_header=not args.no_header)
    extra = (f" + header model (weight {pred.header_weight:.1f})" if pred.header is not None else "")
    print(f"model: {pred.name}  ({len(pred.members)} network(s){extra}); "
          f"category by {'ordinal' if pred.cutpoints else 'argmax'} decoding; "
          f"further-evaluation threshold: HF likelihood >= {pred.hf_threshold:.0%}\n")
    if args.gradcam:
        args.gradcam.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in iter_inputs(args.inputs):
        xs = pred.inputs(path)
        probs, thr, source = pred.fuse(pred.probs(xs), path)
        k = pred.category(probs)
        cat = CLASSES[k]
        hf = float(1 - probs[0])
        needs = hf >= thr
        rec = FURTHER_EVALUATION[cat] if needs == (cat != "Normal") else (
            "YES - borderline: most probable class is Normal but heart-failure likelihood "
            "is above the screening threshold; consider echo" if needs else
            "NO  - borderline: heart-failure likelihood below the screening threshold; "
            "use clinical judgement")
        rows.append(dict(file=str(path), category=cat, source=source, threshold=thr,
                         hf_likelihood=hf, further_evaluation="YES" if needs else "NO",
                         recommendation=rec, **{f"p_{c}": float(p) for c, p in zip(CLASSES, probs)}))
        print(f"{path.name}")
        print(f"  LVEF category           : {cat}      probabilities: "
              + "  ".join(f"{c} {p:.0%}" for c, p in zip(CLASSES, probs)))
        print(f"  Heart-failure likelihood: {hf:.0%}")
        print(f"  Further evaluation      : {rec}")
        print(f"  (based on {source}; threshold {thr:.0%})\n")

        if args.gradcam:
            cam = pred.gradcam(xs, k)
            save_gradcam(path, cam, f"Predicted {cat},  HF likelihood {hf:.0%},  "
                         f"further evaluation: {'YES' if needs else 'NO'}",
                         args.gradcam / f"{path.stem}_gradcam.png")

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"saved {len(rows)} predictions -> {args.csv}")


if __name__ == "__main__":
    main()
