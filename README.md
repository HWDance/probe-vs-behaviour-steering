# Why don't linear probes steer? An identification view

Linear probes decode concepts from activations well but make poor steering vectors
(e.g. AxBench). This repo tests why, on a small controlled task, by asking which
assumptions for "regression coefficient = causal effect" fail when a probe is fit to
observational activations.

## Task

Subject-verb agreement with an attractor, in Pythia-70m:

- prompts: "The author that the guards like" -> next token
- **Z**: number of the subject (from the template)
- **B**: logit(" are") - logit(" is") at the final position; sign(B) is the model's answer
- B and Z disagree on attraction errors (about 6% of items)

Every word is a single token, so positions line up across items. Train and test items
have disjoint subject nouns.

## Directions compared (on the final-token residual h_T)

| Name | What it is | Variation used |
|---|---|---|
| `probe_Z`, `probe_signB`, `probe_B` | logistic / ridge regression of Z, sign(B), continuous B on h_T | observational |
| `diffmean_Z`, `diffmean_signB` | difference of class means | observational |
| `verb_iv` | 2SLS with the final token (the embedded verb) as instrument; exclusion holds by causal masking | natural experiment |
| `noise_cont_<s>`, `noise_sign_<s>` | regress the change in B (or in the binary answer) on random Gaussian perturbations of h_T | randomised experiment |
| `oracle` | mean gradient E[dB/dh_T] | reference |
| `random` | random direction | control |

Steering adds a fixed-norm push at h_T toward the opposite of each test item's current
answer, and records flip rates (on items the model gets right and wrong) and the
off-target KL of the next-token distribution.

## Findings so far (Pythia-70m, one task)

- **The label barely matters.** Directions fit to the truth Z and to the model's answer
  sign(B) steer about equally (badly).
- **Observational directions miss the causal direction.** At layers 1-4 the gradient lies
  almost entirely outside the subspace in which h_T varies across inputs (a positivity
  failure), so no observational estimator can find it.
- **A natural instrument does not help.** The verb is a valid instrument, but it only
  moves h_T in 32 directions that contain about 5% of the gradient; `verb_iv` steers like
  a random direction.
- **A randomised instrument does.** Noise IV recovers the gradient (cosine 0.80-0.98) and
  steers near-oracle, with roughly 4x less push than the best observational direction.
  It also works with only the **binary** answer, provided the noise flips about 10-30% of
  answers, and it beats the probe trained on the same label at equal off-target KL.

Open: transfer to other readouts (was/were, has/have), noise restricted to the data's
principal subspace, a probe on concatenated positions, larger models and judge-scored
behaviour.

## Scripts

| Script | Does |
|---|---|
| `stimuli.py` | builds the prompts (`pp`, `orc`, `orc_pp` frames) |
| `pilot.py` | error rates per model/frame, and whether h_T still encodes Z on the model's errors |
| `lm.py` | residual capture, B, gradient, steering and cached perturbation passes |
| `compare.py` | fits the observational directions and the oracle; geometry and steering |
| `iv.py` | verb-instrumented 2SLS and its diagnostics |
| `steer_iv.py` | verb IV and noise IV (continuous and binary), steering against `compare.py` |

## Running

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# pilot
.venv\Scripts\python stimuli.py
.venv\Scripts\python pilot.py

# main comparison (Pythia-70m, relative-clause frame, 8000 prompts)
.venv\Scripts\python stimuli.py --frames orc --n-per-cell 2000 --out results/compare/stimuli.csv
.venv\Scripts\python compare.py
.venv\Scripts\python iv.py
.venv\Scripts\python steer_iv.py
```

Everything runs on a laptop CPU. `compare.py` takes about 40 minutes; `steer_iv.py`
about 20.

`results/compare/steering_cont.csv` comes from an earlier version of `steer_iv.py`
(continuous noise IV with sigma in {0.01, 0.1, 0.4} and 2 draws per item);
`steering_binary.csv` is from the current version. In `results/compare/geometry.csv` the
`test_acc_*` columns hold negative held-out log-loss, not accuracy (fixed in the code
since that run).
