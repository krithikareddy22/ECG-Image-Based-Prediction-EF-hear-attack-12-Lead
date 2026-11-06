"""Phase 6 final tables and plot from the committed aggregates (no private data needed).
Outputs: results/phase6_all_models.csv, results/phase6_comparisons.csv, results/phase6_selection.json,
plots/phase6_stage1_screen.png
    python -m research_experiments.phase_6.validation.final6
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_experiments.phase_6.p6common import P6, RES6  # noqa: E402

GROUPS = {"P6R0-joint-frozen-v1": "R0 frozen reference (A)", "P6A1-ft-s6-joint-v1": "A1 fine-tune stage 6",
          "P6A2-ft-s56-joint-v1": "A2 fine-tune stages 5-6", "P6B0-ecgf-f6-v1": "B0 ECGFounder f6 (B pipeline)",
          "P6B-ecgfm-v1": "B ECG-FM", "P6B-f6+ecgfm-v1": "B ECGFounder + ECG-FM (exploratory)",
          "P6D2-leadII-2p5s-v1": "D2 lead II 2.5 s"}


def main():
    rows = []
    for pre, name in GROUPS.items():
        fs = sorted(glob.glob(str(RES6 / "experiments" / f"{pre}__s*_r*.json")))
        A = [json.load(open(f)) for f in fs]
        m = {k: float(np.mean([a["metrics"][k] for a in A])) for k in A[0]["metrics"]}
        r = [a["ef_regression"]["A"]["pearson"] for a in A if "ef_regression" in a]
        rows.append(dict(model=name, run_prefix=pre, runs=len(A), seeds=sorted({a["config"].get("seed") for a in A}),
                         ef_r_mean_per_run=float(np.mean(r)) if r else None, admissible=m["min_recall"] >= 0.15, **m))
    S = pd.DataFrame(rows)
    S.to_csv(RES6 / "phase6_all_models.csv", index=False, float_format="%.4f")
    scr = json.load(open(RES6 / "stage1_screen.json"))
    C = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in x.items()} for x in scr])
    C.to_csv(RES6 / "phase6_comparisons.csv", index=False, float_format="%.4f")
    sel = dict(stage1_passers=[x["candidate"] for x in scr if x["PASS"]], stage2_run=False,
               selected_model="P5D-E5-vote-prior-protected-v1 (C3, unchanged from Phase 5)",
               selected_accuracy=0.561, selected_ci95=[0.520, 0.603], genuine_improvements=0,
               note="No Phase 6 candidate passed the pre-registered Stage-1 screen (EF r >= 0.69 or macro-F1 >= 0.465 with "
                    "all recalls >= 0.15); Phase 6A / 6B (separate reports) found no genuine improvement either.")
    (RES6 / "phase6_selection.json").write_text(json.dumps(sel, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    d = [x for x in scr]
    ax[0].barh([x["candidate"] for x in d], [x["r_A"] for x in d], color="#4C78A8"); ax[0].axvline(0.69, ls=":", c="k")
    ax[0].set_xlabel("EF Pearson r (seed 42, mean of 3 repeats)"); ax[0].set_xlim(0.55, 0.72); ax[0].set_title("Stage-1 screen: pass mark r >= 0.69")
    ax[1].barh([x["candidate"] for x in d], [x["macro_f1"] for x in d], color="#F58518"); ax[1].axvline(0.465, ls=":", c="k")
    ax[1].set_xlabel("four-class macro-F1"); ax[1].set_xlim(0.35, 0.5); ax[1].set_title("alternative pass mark macro-F1 >= 0.465")
    fig.tight_layout(); fig.savefig(P6 / "plots" / "phase6_stage1_screen.png", dpi=150); plt.close(fig)
    print(S[["model", "runs", "accuracy", "balanced_accuracy", "macro_f1", "min_recall", "ef_r_mean_per_run"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
