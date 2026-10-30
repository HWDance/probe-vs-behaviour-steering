"""Compare steering directions fit to the model's answer B vs the true answer Z.

For each layer, on the final-token residual h_T of the training items:
  probe_Z         logistic regression of Z (true subject number)
  probe_signB     logistic regression of sign(B) (the model's answer)
  probe_B         ridge regression of continuous B
  diffmean_Z      E[h | Z = pl] - E[h | Z = sg]
  diffmean_signB  E[h | answer = pl] - E[h | answer = sg]
  oracle          E[dB/dh_T], the local causal effect of h_T on B
  random          a fixed random direction, as a control
All are oriented so that adding them pushes the answer toward plural. Probe weights
are mapped back to raw residual coordinates (w / sigma) before comparison.

Train and test items have disjoint subject nouns. Steering is evaluated on test
items: add +/- alpha * unit(v) at h_T to push each item toward the opposite of its
current answer. On items the model gets right this induces an error; on items it
gets wrong it fixes the error. alpha is a fraction of the mean ||h_T|| at that layer.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm import final_logprobs, mean_gradient, run_model, steer

ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.8]


def load_items(model_name, stimuli, seed):
    """Run the model on every prompt and split train/test by subject noun."""
    df = pd.read_csv(stimuli)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    B, H = run_model(model, tokenizer, df.prompt.tolist())
    df["B"] = B
    df["answer"] = (B > 0).astype(int)
    df["correct"] = df.answer == df.z
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(df, groups=df.subject)
    )
    return model, tokenizer, df, H, train_idx, test_idx


def fit_directions(H, z, B, groups, seed):
    """Fit every direction on one layer's h_T. Returns directions and test-time predictors."""
    answer = (B > 0).astype(int)
    splits = list(GroupKFold(5).split(H, z, groups))
    scaler = StandardScaler().fit(H)
    X = scaler.transform(H)

    def logistic(y):
        return LogisticRegressionCV(
            Cs=[1e-3, 1e-2, 1e-1, 1], cv=splits, scoring="neg_log_loss", max_iter=5000,
        ).fit(X, y)

    models = {
        "probe_Z": logistic(z),
        "probe_signB": logistic(answer),
        "probe_B": RidgeCV(alphas=np.logspace(-1, 4, 6), cv=splits).fit(X, B),
    }
    directions = {name: m.coef_.ravel() / scaler.scale_ for name, m in models.items()}
    directions["diffmean_Z"] = H[z == 1].mean(0) - H[z == 0].mean(0)
    directions["diffmean_signB"] = H[answer == 1].mean(0) - H[answer == 0].mean(0)
    directions["random"] = np.random.default_rng(seed).standard_normal(H.shape[1])
    return directions, scaler, models


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def steering_sweep(model, tokenizer, test, scales, directions, alphas):
    """Push each test item toward the opposite of its current answer along each direction.

    directions: {layer: {name: vector}}; scales: {layer: mean ||h_T|| on training items}.
    """
    prompts = test.prompt.tolist()
    base_logprobs = final_logprobs(model, tokenizer, prompts)
    push = np.where(test.answer == 1, -1.0, 1.0)
    groups = {
        "all": np.ones(len(test), dtype=bool),
        "correct": test.correct.to_numpy(),
        "error": ~test.correct.to_numpy(),
    }
    rows = []
    for layer, dirs in directions.items():
        for name, v in dirs.items():
            print(f"steering layer {layer} (mean ||h_T|| = {scales[layer]:.1f}): {name}", flush=True)
            unit = v / np.linalg.norm(v)
            for alpha in alphas:
                deltas = torch.tensor(push[:, None] * alpha * scales[layer] * unit, dtype=torch.float32)
                B_new, kl = steer(model, tokenizer, prompts, layer, deltas, base_logprobs)
                shift = push * (B_new - test.B.to_numpy())
                flipped = (B_new > 0).astype(int) != test.answer.to_numpy()
                for group, mask in groups.items():
                    rows.append({
                        "layer": layer, "direction": name, "alpha": alpha, "group": group,
                        "n": int(mask.sum()), "flip_rate": flipped[mask].mean(),
                        "mean_shift": shift[mask].mean(), "off_target_kl": kl[mask].mean(),
                    })
    return pd.DataFrame(rows)


def show_tables(steering):
    def show(group, value):
        table = steering[steering.group == group].pivot_table(
            index=["layer", "direction"], columns="alpha", values=value,
        )
        print(table.round(3).to_string())

    print("\n=== flip rate on items the model gets right (steering induces an error) ===")
    show("correct", "flip_rate")
    print("\n=== flip rate on items the model gets wrong (steering fixes the error) ===")
    show("error", "flip_rate")
    print("\n=== off-target KL, all test items ===")
    show("all", "off_target_kl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--stimuli", default="results/compare/stimuli.csv")
    parser.add_argument("--steer-layers", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/compare")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, df, H, train_idx, test_idx = load_items(args.model, args.stimuli, args.seed)
    train, test = df.iloc[train_idx], df.iloc[test_idx].reset_index(drop=True)
    print(f"train {len(train)} items ({(~train.correct).sum()} errors), "
          f"test {len(test)} items ({(~test.correct).sum()} errors)")

    oracle = mean_gradient(model, tokenizer, train.prompt.tolist())

    # Directions, their agreement with the oracle, and how well each probe predicts held-out data.
    directions, geometry = {}, []
    for layer in range(H.shape[1]):
        dirs, scaler, models = fit_directions(
            H[train_idx, layer], train.z.to_numpy(), train.B.to_numpy(), train.subject.to_numpy(), args.seed,
        )
        dirs["oracle"] = oracle[layer]
        directions[layer] = dirs
        X_test = scaler.transform(H[test_idx, layer])
        row = {
            "layer": layer,
            # .score() would use the CV scoring (log-loss), so compute accuracy directly.
            "test_acc_probe_Z": (models["probe_Z"].predict(X_test) == test.z).mean(),
            "test_acc_probe_signB": (models["probe_signB"].predict(X_test) == test.answer).mean(),
            "test_r2_probe_B": models["probe_B"].score(X_test, test.B),
            "cos(probe_Z, probe_signB)": cosine(dirs["probe_Z"], dirs["probe_signB"]),
        }
        row.update({f"cos({name}, oracle)": cosine(v, oracle[layer]) for name, v in dirs.items() if name != "oracle"})
        geometry.append(row)
    geometry = pd.DataFrame(geometry)
    geometry.to_csv(out_dir / "geometry.csv", index=False)
    np.savez(out_dir / "directions.npz", **{
        f"{name}_layer{layer}": v for layer, dirs in directions.items() for name, v in dirs.items()
    })
    print("\n=== geometry (per layer) ===")
    print(geometry.round(3).T.to_string(header=False))

    scales = {layer: np.linalg.norm(H[train_idx, layer], axis=1).mean() for layer in args.steer_layers}
    steering = steering_sweep(
        model, tokenizer, test, scales, {layer: directions[layer] for layer in args.steer_layers}, args.alphas,
    )
    steering.to_csv(out_dir / "steering.csv", index=False)
    show_tables(steering)


if __name__ == "__main__":
    main()
