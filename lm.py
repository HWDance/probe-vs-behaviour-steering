"""Model helpers: final-token residuals, the behaviour B, its gradient, and steering.

Layer k means the residual stream after k blocks (0 = embeddings). Everything acts
on the final token position, which is where the verb is predicted.
"""

from contextlib import contextmanager

import torch


def single_token(tokenizer, text):
    ids = tokenizer.encode(text)
    assert len(ids) == 1, f"{text!r} is {len(ids)} tokens"
    return ids[0]


def verb_ids(tokenizer):
    return single_token(tokenizer, " is"), single_token(tokenizer, " are")


def behaviour(logits, is_id, are_id):
    """B = logit(" are") - logit(" is") at the final position."""
    return logits[:, -1, are_id] - logits[:, -1, is_id]


def batches(tokenizer, prompts, batch_size):
    """Yield (indices, tokens) with BOS prepended; prompts of equal length are batched
    together so no padding is needed."""
    ids = [[tokenizer.bos_token_id] + tokenizer.encode(p) for p in prompts]
    by_length = {}
    for i, seq in enumerate(ids):
        by_length.setdefault(len(seq), []).append(i)
    for idx in by_length.values():
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            yield torch.tensor(chunk), torch.tensor([ids[i] for i in chunk])


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


@contextmanager
def record_resid(model, retain_grad=False):
    """Record the residual-stream tensor at every layer during forward passes."""
    layers = model.gpt_neox.layers
    acts = []

    def keep(t):
        if retain_grad:
            t.retain_grad()
        acts.append(t)

    hooks = [layers[0].register_forward_pre_hook(lambda mod, args: keep(args[0]))]
    for layer in layers:
        hooks.append(layer.register_forward_hook(lambda mod, args, out: keep(_hidden(out))))
    try:
        yield acts
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def run_model(model, tokenizer, prompts, batch_size=64):
    """Return B (n,) and final-token residuals h_T (n, n_layers + 1, d_model)."""
    is_id, are_id = verb_ids(tokenizer)
    B = torch.zeros(len(prompts))
    H = torch.zeros(len(prompts), model.config.num_hidden_layers + 1, model.config.hidden_size)
    with record_resid(model) as acts:
        for idx, tokens in batches(tokenizer, prompts, batch_size):
            acts.clear()
            B[idx] = behaviour(model(tokens, logits_to_keep=1).logits, is_id, are_id)
            H[idx] = torch.stack([a[:, -1] for a in acts], dim=1)
    return B.numpy(), H.numpy()


def mean_gradient(model, tokenizer, prompts, batch_size=64):
    """E[dB/dh_T] at every layer: to first order, the effect on B of steering h_T."""
    is_id, are_id = verb_ids(tokenizer)
    total = torch.zeros(model.config.num_hidden_layers + 1, model.config.hidden_size)
    with torch.enable_grad(), record_resid(model, retain_grad=True) as acts:
        for _, tokens in batches(tokenizer, prompts, batch_size):
            acts.clear()
            behaviour(model(tokens, logits_to_keep=1).logits, is_id, are_id).sum().backward()
            total += torch.stack([a.grad[:, -1] for a in acts], dim=1).sum(0)
    model.zero_grad(set_to_none=True)
    return (total / len(prompts)).numpy()


def _non_verb_logprobs(logits, tokenizer):
    """Next-token log-probs at the final position, renormalised without " is"/" are"."""
    keep = torch.ones(logits.shape[-1], dtype=torch.bool)
    keep[list(verb_ids(tokenizer))] = False
    return logits[:, -1, keep].log_softmax(-1)


@torch.inference_mode()
def final_logprobs(model, tokenizer, prompts, batch_size=256):
    """Unsteered non-verb log-probs (n, vocab - 2): the baseline for off-target KL."""
    out = None
    for idx, tokens in batches(tokenizer, prompts, batch_size):
        logprobs = _non_verb_logprobs(model(tokens, logits_to_keep=1).logits, tokenizer)
        if out is None:
            out = torch.zeros(len(prompts), logprobs.shape[-1])
        out[idx] = logprobs
    return out


@contextmanager
def _final_position_hook(model, layer):
    """While active, current["delta"] (if not None) is added to the last position's
    residual at `layer`."""
    layers = model.gpt_neox.layers
    current = {"delta": None}

    def add(h):
        if current["delta"] is None:
            return h
        h = h.clone()
        h[:, -1] += current["delta"]
        return h

    if layer == 0:
        hook = layers[0].register_forward_pre_hook(lambda mod, args: (add(args[0]),) + args[1:])
    else:
        hook = layers[layer - 1].register_forward_hook(
            lambda mod, args, out: (add(out[0]),) + out[1:] if isinstance(out, tuple) else add(out)
        )
    try:
        yield current
    finally:
        hook.remove()


@torch.inference_mode()
def steer(model, tokenizer, prompts, layer, deltas, base_logprobs=None, batch_size=256):
    """Add deltas[i] to h_T at `layer` for prompt i.

    Returns the steered B (n,) and the off-target KL(base || steered) (n,) of the
    next-token distribution without the verb tokens; base_logprobs is from
    final_logprobs(). Without base_logprobs the KL is NaN.
    """
    is_id, are_id = verb_ids(tokenizer)
    B = torch.zeros(len(prompts))
    kl = torch.full((len(prompts),), float("nan"))
    with _final_position_hook(model, layer) as current:
        for idx, tokens in batches(tokenizer, prompts, batch_size):
            current["delta"] = deltas[idx]
            logits = model(tokens, logits_to_keep=1).logits
            B[idx] = behaviour(logits, is_id, are_id)
            if base_logprobs is not None:
                p = base_logprobs[idx]
                kl[idx] = (p.exp() * (p - _non_verb_logprobs(logits, tokenizer))).sum(-1)
    return B.numpy(), kl.numpy()


@torch.inference_mode()
def perturbed_behaviour(model, tokenizer, prompts, layer, deltas, batch_size=256):
    """B with deltas[i, k] added to h_T at `layer`, for prompt i and perturbation k.

    deltas: (n, K, d). Earlier positions cannot attend to the final one, so their keys
    and values are computed once per prompt and reused for all K perturbations; each
    perturbation then costs a single-token forward pass. Returns B (n, K).
    """
    is_id, are_id = verb_ids(tokenizer)
    B = torch.zeros(deltas.shape[:2])
    with _final_position_hook(model, layer) as current:
        for idx, tokens in batches(tokenizer, prompts, batch_size):
            current["delta"] = None
            cache = model(tokens[:, :-1], use_cache=True).past_key_values
            for k in range(deltas.shape[1]):
                current["delta"] = deltas[idx, k]
                logits = model(tokens[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1).logits
                B[idx, k] = behaviour(logits, is_id, are_id)
                cache.crop(-1)
    return B.numpy()
