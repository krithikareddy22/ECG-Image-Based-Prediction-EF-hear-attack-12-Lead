"""Convert the output of the ORIGINAL training code (Phase 0b retrain, written to
research_experiments/results/repro/) into the research metrics format (CIs, OvR AUROC ...),
and compare it with the original checkpoint's predictions.
    python -m research_experiments.collect_repro
"""
import json

import numpy as np
import pandas as pd

from research_experiments.common import CLASSES, RES, ROOT, load_meta, save_run

meta = load_meta()
mv = meta[meta.split == "val"].reset_index(drop=True); mt = meta[meta.split == "test"].reset_index(drop=True)
for d in sorted((RES / "repro").glob("R0_*")):
    P = {}
    for s, md in (("val", mv), ("test", mt)):
        df = pd.read_csv(d / f"{s}_predictions.csv").set_index("ecg_id").loc[md.ecg_id]
        P[s] = df[[f"p_{c}" for c in CLASSES]].values
    orig = json.loads((d / "metrics.json").read_text())
    s = save_run(d.name, mv, P["val"], mt, P["test"], dict(
        input="ECG image (filtered 256x512)", note="retrained with the original ecg_ef/train.py, same args and seed",
        params_total=orig["params"], params_trainable=orig["params"], best_epoch=orig["best_epoch"],
        epochs_run=orig["epochs_run"], train_minutes=orig["train_minutes"]), hf_threshold=orig["hf_threshold"])
    o = pd.read_csv(ROOT / "results" / "resnet18_filtered" / "test_predictions.csv").set_index("ecg_id").loc[mt.ecg_id]
    po = o[[f"p_{c}" for c in CLASSES]].values
    rep = dict(max_abs_prob_diff_vs_original=float(np.abs(po - P["test"]).max()),
               argmax_agreement_with_original=float((po.argmax(1) == P["test"].argmax(1)).mean()),
               original_best_epoch=14, retrain_best_epoch=orig["best_epoch"],
               original_test=dict(accuracy=0.4857, balanced_accuracy=0.3562, macro_f1=0.3233),
               retrain_test={k: s["test"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "qwk")})
    (RES / f"{d.name}_vs_original.json").write_text(json.dumps(rep, indent=2))
    print(d.name, rep)
