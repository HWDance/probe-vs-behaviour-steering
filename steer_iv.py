"""Steering with instrumental-variable estimates of the effect of h_T on behaviour.

Two instruments for h_T at each steering layer:
  verb_iv          the final token (the embedded verb), 2SLS of continuous B on the
                   training items (see iv.py). Identified only on the subspace the
                   verb moves h_T along.
  noise_<y>_<s>    random perturbations delta ~ N(0, s^2 ||h_T||^2 / d * I) added to
                   h_T. delta is independent of everything, reaches the output only
                   through h_T and varies in every direction, so 2SLS reduces to
                   regressing the change in behaviour on delta. s is in the same units
                   as the steering alpha. Antithetic pairs: (y(h + delta) - y(h - delta)) / 2.
                   Two behaviour readouts y, fit to the same perturbations:
                     cont  continuous B. By Stein's lemma the estimate is the gradient of
                           B averaged over the noise.
                     sign  the binary answer 1[B > 0] (a linear probability model), as
                           when only discrete outputs are observable. It carries signal
                           only from perturbations that flip the answer.

Steering uses the same test split, pushes and alphas as compare.py and is combined
with its steering.csv; the oracle is re-run as a consistency check.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare import ALPHAS, cosine, load_items, show_tables, steering_sweep
from iv import verb_iv
from lm import perturbed_behaviour


def noise_iv(model, tokenizer, prompts, layer, sigma, scale, n_draws, seed):
    """Regress the antithetic change in behaviour on random perturbations of h_T.

    Returns {readout: (estimate, split-half cosine)} for the continuous and binary
    readouts, both fit to the same perturbations, and the fraction of perturbation
    pairs whose answers differ.
    """
    d = model.config.hidden_size
    delta = np.random.default_rng(seed).standard_normal((len(prompts), n_draws, d)) * sigma * scale / np.sqrt(d)
    delta_t = torch.tensor(delta, dtype=torch.float32)
    B = perturbed_behaviour(model, tokenizer, prompts, layer, torch.cat([delta_t, -delta_t], dim=1))
    B_plus, B_minus = B[:, :n_draws], B[:, n_draws:]
    X = delta.reshape(-1, d)
    readouts = {
        "cont": (B_plus - B_minus).ravel() / 2,
        "sign": ((B_plus > 0).astype(float) - (B_minus > 0)).ravel() / 2,
    }

    half = np.arange(len(X)) % 2 == 0
    everything = np.ones(len(X), dtype=bool)

    def fit(y, mask):
        return np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]

    fits = {
        name: (fit(y, everything), cosine(fit(y, half), fit(y, ~half)))
        for name, y in readouts.items()
    }
    return fits, np.mean((B_plus > 0) != (B_minus > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--stimuli", default="results/compare/stimuli.csv")
    parser.add_argument("--steer-layers", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.4, 1.0])
    parser.add_argument("--n-draws", type=int, default=4, help="perturbations per training item")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/compare", help="must hold compare.py's outputs")
    parser.add_argument("--tag", default="binary", help="suffix for this run's output files")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    model, tokenizer, df, H, train_idx, test_idx = load_items(args.model, args.stimuli, args.seed)
    train, test = df.iloc[train_idx], df.iloc[test_idx].reset_index(drop=True)
    earlier = np.load(out_dir / "directions.npz")
    scales = {layer: np.linalg.norm(H[train_idx, layer], axis=1).mean() for layer in args.steer_layers}

    directions, geometry = {}, []
    for layer in args.steer_layers:
        oracle = earlier[f"oracle_layer{layer}"]
        dirs = {"verb_iv": verb_iv(train, H[train_idx, layer])}
        info = {"verb_iv": {}}
        for sigma in args.sigmas:
            print(f"noise IV: layer {layer}, sigma {sigma:g}", flush=True)
            fits, flip = noise_iv(
                model, tokenizer, train.prompt.tolist(), layer, sigma, scales[layer], args.n_draws, args.seed,
            )
            for readout, (beta, split_half) in fits.items():
                name = f"noise_{readout}_{sigma:g}"
                dirs[name] = beta
                info[name] = {"split_half_cos": split_half, "flip_fraction": flip}
        for name, v in dirs.items():
            geometry.append({
                "layer": layer, "direction": name,
                **info[name],
                "cos(., oracle)": cosine(v, oracle),
                "cos(., probe_B)": cosine(v, earlier[f"probe_B_layer{layer}"]),
                "cos(., probe_signB)": cosine(v, earlier[f"probe_signB_layer{layer}"]),
            })
        dirs["oracle"] = oracle  # re-run as a consistency check against compare.py
        directions[layer] = dirs

    geometry = pd.DataFrame(geometry)
    geometry.to_csv(out_dir / f"iv_geometry_{args.tag}.csv", index=False)
    np.savez(out_dir / f"iv_directions_{args.tag}.npz", **{
        f"{name}_layer{layer}": v for layer, dirs in directions.items() for name, v in dirs.items()
    })
    print("\n=== IV directions ===")
    print(geometry.round(3).to_string(index=False))

    new = steering_sweep(model, tokenizer, test, scales, directions, args.alphas)
    earlier_steering = pd.read_csv(out_dir / "steering.csv")
    check = earlier_steering[earlier_steering.direction == "oracle"].merge(
        new[new.direction == "oracle"], on=["layer", "alpha", "group"],
    )
    print(f"\noracle re-run vs compare.py: max |flip rate difference| = "
          f"{np.abs(check.flip_rate_x - check.flip_rate_y).max():.4f} over {len(check)} cells")
    combined = pd.concat([earlier_steering, new[new.direction != "oracle"]], ignore_index=True)
    combined.to_csv(out_dir / f"steering_{args.tag}.csv", index=False)
    show_tables(combined)


if __name__ == "__main__":
    main()
