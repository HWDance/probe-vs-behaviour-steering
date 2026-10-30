"""2SLS estimate of the effect of h_T on B, with the final token as the instrument.

Causal masking makes the final token (the embedded verb) a natural instrument for
h_T at any layer: earlier positions cannot attend to it, so it reaches B only
through h_T (exclusion), and its lemma is sampled independently of the rest of the
sentence (independence). Its form ("likes" / "like") depends on attractor number,
so the instruments are lemma dummies within each attractor-number stratum, with
subject number and attractor number partialled out as controls.

With r instrument dimensions, beta is identified only on the r-dimensional subspace
S of h_T along which the verb moves it. Within S we compare
  iv      2SLS: uses only verb-driven variation in h_T
  ols     OLS of B on the projection of h_T onto S: uses all variation in S,
          including variation driven by earlier positions (confounded)
  oracle  the mean gradient E[dB/dh_T] (from compare.py), projected onto S
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm import run_model


def residualise(Y, C):
    coef, *_ = np.linalg.lstsq(C, Y, rcond=None)
    return Y - C @ coef


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def instruments(df):
    """Verb-lemma dummies within attractor-number strata, with the controls partialled out."""
    controls = np.column_stack([np.ones(len(df)), df.z, df.attr_plural, df.z * df.attr_plural])
    stratum_lemma = df.link + "_" + df.attr_plural.astype(str)
    W = residualise(pd.get_dummies(stratum_lemma).to_numpy(float), controls)
    return W, controls


def two_sls(W, h, b):
    """First stage h on W, second stage b on the fitted h. Returns (h_hat, beta)."""
    h_hat = W @ np.linalg.lstsq(W, h, rcond=None)[0]
    return h_hat, np.linalg.lstsq(h_hat, b, rcond=None)[0]  # min-norm, so beta lies in S


def verb_iv(df, H_layer):
    """The verb-instrumented estimate of dB/dh_T at one layer."""
    W, controls = instruments(df)
    h = residualise(H_layer.astype(float), controls)
    b = residualise(df.B.to_numpy(float), controls)
    return two_sls(W, h, b)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--stimuli", default="results/compare/stimuli.csv")
    parser.add_argument("--directions", default="results/compare/directions.npz")
    parser.add_argument("--out", default="results/compare/iv.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.stimuli)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    B, H = run_model(model, tokenizer, df.prompt.tolist())
    oracle = np.load(args.directions)

    W, controls = instruments(df)
    b = residualise(B.astype(float), controls)

    # Reduced form: how much the verb moves B at all. IV can only be as informative as this.
    rf_coef, *_ = np.linalg.lstsq(W, b, rcond=None)
    rank_W = np.linalg.matrix_rank(W)
    rss, tss = np.sum((b - W @ rf_coef) ** 2), np.sum(b**2)
    f_stat = ((tss - rss) / rank_W) / (rss / (len(b) - rank_W - controls.shape[1]))
    print(f"reduced form: R^2 of B on the verb = {1 - rss / tss:.4f}, F({rank_W}, n) = {f_stat:.1f}")

    halves = np.random.default_rng(0).permutation(len(df)) % 2 == 0

    rows = []
    for layer in range(H.shape[1]):
        h = residualise(H[:, layer].astype(float), controls)
        # First stage: the part of h_T explained by the verb.
        h_hat, beta_iv = two_sls(W, h, b)
        _, s, Vt = np.linalg.svd(h_hat, full_matrices=False)
        S = Vt[: int((s > s[0] * 1e-8).sum())]  # orthonormal basis of the identified subspace
        split_half = cosine(*(two_sls(W[m], h[m], b[m])[1] for m in (halves, ~halves)))
        ols_coef, *_ = np.linalg.lstsq(h @ S.T, b, rcond=None)
        beta_ols = S.T @ ols_coef
        g = oracle[f"oracle_layer{layer}"]
        g_S = S.T @ (S @ g)

        rows.append({
            "layer": layer,
            "rank_S": len(S),
            "first_stage_r2": np.sum(h_hat**2) / np.sum(h**2),
            "oracle_norm2_in_S": np.sum(g_S**2) / np.sum(g**2),
            "iv_split_half_cos": split_half,
            "cos(iv, oracle_S)": cosine(beta_iv, g_S),
            "cos(ols_S, oracle_S)": cosine(beta_ols, g_S),
            "|iv| / |oracle_S|": np.linalg.norm(beta_iv) / np.linalg.norm(g_S),
            "|ols_S| / |oracle_S|": np.linalg.norm(beta_ols) / np.linalg.norm(g_S),
        })

    result = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
