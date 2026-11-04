"""Noise IV with and without variance preservation.

Adding noise to h_T changes two things at once: how much of the original activation the
perturbed one carries (information), and how large it is (scale). To separate them, at
each noise level s we compare, at the same signal-to-noise ratio,

  add_<s>    h~ = h + s xi
  vp_<s>     h~ = mu + (h - mu + s xi) / sqrt(1 + s^2)      (variance preserving)
  overwrite  h~ = mu + xi                                   (the s -> infinity limit of vp)

where mu is the mean of h_T on the training items and xi ~ N(0, v/d I), with v the total
variance of h_T about mu. So s is in units of the natural spread of h_T, and the vp
perturbations keep the mean and total variance of h_T, while additive noise inflates the
variance by 1 + s^2. All designs use the same draws of xi. Antithetic pairs replace xi
by -xi, and the change in behaviour is regressed on the noise actually added, so by
Stein's lemma each estimate is the gradient of B averaged over its perturbed activations.

Steering uses the same test split, pushes and alphas as compare.py. Steering results are
also summarised as flip rates at matched off-target KL.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare import ALPHAS, cosine, load_items, steering_sweep
from steer_iv import antithetic_iv


def designs(h, mu, xi, levels):
    """Yield (name, shift, noise) for each design; see the module docstring."""
    for s in levels:
        yield f"add_{s:g}", np.zeros_like(h), s * xi
    for s in levels:
        c = np.sqrt(1 + s ** 2)
        yield f"vp_{s:g}", (mu - h) * (1 - 1 / c), s / c * xi
    yield "overwrite", mu - h, xi


def matched_kl(steering, kls):
    """Flip rate on correct items, interpolated at fixed off-target KL on all items."""
    rows = []
    for (layer, name), g in steering.groupby(["layer", "direction"]):
        flips = g[g.group == "correct"].sort_values("alpha").flip_rate.to_numpy()
        kl = g[g.group == "all"].sort_values("alpha").off_target_kl.to_numpy()
        rows.append({"layer": layer, "direction": name, **{f"KL={t:g}": np.interp(t, kl, flips) for t in kls}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--stimuli", default="results/compare/stimuli.csv")
    parser.add_argument("--steer-layers", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--levels", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4],
                        help="noise levels s, in units of the natural spread of h_T")
    parser.add_argument("--n-draws", type=int, default=4, help="perturbations per training item")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/compare", help="must hold compare.py's outputs")
    parser.add_argument("--tag", default="vp", help="suffix for this run's output files")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    model, tokenizer, df, H, train_idx, test_idx = load_items(args.model, args.stimuli, args.seed)
    train, test = df.iloc[train_idx], df.iloc[test_idx].reset_index(drop=True)
    prompts = train.prompt.tolist()
    earlier = np.load(out_dir / "directions.npz")
    scales = {layer: np.linalg.norm(H[train_idx, layer], axis=1).mean() for layer in args.steer_layers}
    d = model.config.hidden_size
    rng = np.random.default_rng(args.seed)

    directions, geometry = {}, []
    for layer in args.steer_layers:
        h = H[train_idx, layer].astype(np.float64)
        mu = h.mean(0)
        v = ((h - mu) ** 2).sum(1).mean()
        print(f"layer {layer}: mean ||h_T|| = {scales[layer]:.1f}, natural spread sqrt(v) = {np.sqrt(v):.1f} "
              f"(steer_iv.py's sigma corresponds to s = sigma * {scales[layer] / np.sqrt(v):.2f})", flush=True)
        xi = rng.standard_normal((len(prompts), args.n_draws, d)) * np.sqrt(v / d)
        oracle = earlier[f"oracle_layer{layer}"]
        dirs = {}
        for name, shift, noise in designs(h, mu, xi, args.levels):
            print(f"noise IV: layer {layer}, {name}", flush=True)
            fits, flip = antithetic_iv(model, tokenizer, prompts, layer, shift, noise)
            for readout, (beta, split_half) in fits.items():
                dirs[f"{name}_{readout}"] = beta
                geometry.append({
                    "layer": layer, "direction": f"{name}_{readout}", "split_half_cos": split_half,
                    "flip_fraction": flip, "cos(., oracle)": cosine(beta, oracle),
                })
        dirs["oracle"] = oracle
        directions[layer] = dirs

    geometry = pd.DataFrame(geometry)
    geometry.to_csv(out_dir / f"iv_geometry_{args.tag}.csv", index=False)
    np.savez(out_dir / f"iv_directions_{args.tag}.npz", **{
        f"{name}_layer{layer}": v for layer, dirs in directions.items() for name, v in dirs.items()
    })
    print("\n=== noise IV directions ===")
    print(geometry.round(3).to_string(index=False))

    steering = steering_sweep(model, tokenizer, test, scales, directions, args.alphas)
    steering.to_csv(out_dir / f"steering_{args.tag}.csv", index=False)
    summary = matched_kl(steering, [0.1, 0.2, 0.3, 0.5])
    summary.to_csv(out_dir / f"matched_kl_{args.tag}.csv", index=False)
    print("\n=== flip rate on correct items at matched off-target KL ===")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
