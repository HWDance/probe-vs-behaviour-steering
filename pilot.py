"""Step 0 pilot: is there enough disagreement between B and Z, and does h_T know Z?

For each model and frame:
  1. The model's answer is B = logit(" are") - logit(" is") at the final position;
     sign(B) disagrees with the true subject number Z on attraction errors.
  2. Diagnostic for hypothesis (i): does the final-token residual h_T carry information
     about Z that the model's answer does not? Per layer, a linear probe for Z on h_T
     is fit with cross-validation grouped by subject noun (so it can't just memorise
     noun identity). We report
       - its accuracy on the items the model gets wrong (above 50% means h_T still
         encodes the right answer there), and
       - the bits of information about Z its score adds beyond B and the attractor's
         number. The baseline includes attractor number because B depends on it too, so
         conditioning on B alone would make attractor number informative about Z.

Layer k means the residual stream after k blocks (0 = embeddings).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm import run_model

DEFAULT_MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]


def oof_proba(X, y, groups, C, n_splits=5):
    """Out-of-fold P(y=1) from a standardised L2 logistic regression."""
    proba = np.zeros(len(y))
    for train, test in GroupKFold(n_splits).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        clf.fit(X[train], y[train])
        proba[test] = clf.predict_proba(X[test])[:, 1]
    return np.clip(proba, 1e-6, 1 - 1e-6)


def diagnostic(df, H, C):
    z = df.z.to_numpy()
    groups = df.subject.to_numpy()
    wrong = ~df.correct.to_numpy()

    # Three features only, so fit nearly unregularised.
    base = np.column_stack([df.B, df.attr_plural, df.B * df.attr_plural])
    nll_base = log_loss(z, oof_proba(base, z, groups, C=1e4))

    rows = []
    for layer in range(H.shape[1]):
        p = oof_proba(H[:, layer], z, groups, C)
        score = np.log(p / (1 - p))
        nll_full = log_loss(z, oof_proba(np.column_stack([base, score]), z, groups, C=1e4))
        rows.append({
            "layer": layer,
            "probe_acc": ((p > 0.5) == z).mean(),
            "probe_acc_on_model_errors": ((p[wrong] > 0.5) == z[wrong]).mean() if wrong.any() else np.nan,
            "bits_beyond_B": (nll_base - nll_full) / np.log(2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--stimuli", default="results/stimuli.csv")
    parser.add_argument("--frames", nargs="+", help="only run these frames (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--C", type=float, default=0.1, help="inverse L2 strength of the h_T probes")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    stimuli = pd.read_csv(args.stimuli)
    if args.frames:
        stimuli = stimuli[stimuli.frame.isin(args.frames)].reset_index(drop=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.models:
        tag = name.split("/")[-1]
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        B, H = run_model(model, tokenizer, stimuli.prompt.tolist(), args.batch_size)
        del model

        df = stimuli.assign(B=B, answer=(B > 0).astype(int))
        df["correct"] = df.answer == df.z
        df["mismatch"] = df.z != df.attr_plural
        df.to_csv(out_dir / f"{tag}_items.csv", index=False)

        for frame, fdf in df.groupby("frame"):
            print(f"\n=== {tag} / {frame} ===")
            cells = fdf.assign(
                subject_num=fdf.z.map({0: "sg", 1: "pl"}),
                attractor_num=fdf.attr_plural.map({0: "sg", 1: "pl"}),
            ).groupby(["subject_num", "attractor_num"]).agg(
                error_rate=("correct", lambda c: 1 - c.mean()), mean_B=("B", "mean"),
            )
            print(cells.round(3).to_string())

            diag = diagnostic(fdf, H[fdf.index], args.C)
            diag.to_csv(out_dir / f"{tag}_{frame}_diagnostic.csv", index=False)
            print(diag.round(3).to_string(index=False))

            best = diag.loc[diag.bits_beyond_B.idxmax()]
            summary.append({
                "model": tag,
                "frame": frame,
                "disagreement": 1 - fdf.correct.mean(),
                "disagreement_matched": 1 - fdf[~fdf.mismatch].correct.mean(),
                "disagreement_mismatched": 1 - fdf[fdf.mismatch].correct.mean(),
                "n_errors": int((~fdf.correct).sum()),
                "max_bits_beyond_B": best.bits_beyond_B,
                "at_layer": int(best.layer),
                "probe_acc_on_errors_there": best.probe_acc_on_model_errors,
            })

    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== summary ===")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
