import hashlib

import numpy as np
import torch

VALID_MODES = ("none", "round", "topk", "label_only", "flip")


_FLIP_SECRET = b"project33-victim-secret"


def defense_round(probs, decimals=2):
    factor = 10 ** decimals
    rounded = torch.round(probs * factor) / factor
    return rounded / rounded.sum(dim=1, keepdim=True).clamp_min(1e-12)


def defense_top_k(probs, k=3):
    out = torch.zeros_like(probs)
    topk_vals, topk_idx = torch.topk(probs, k, dim=1)
    out.scatter_(1, topk_idx, topk_vals)
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)


def defense_labels_only(probs):
    out = torch.zeros_like(probs)
    idx = probs.argmax(dim=1, keepdim=True)
    out.scatter_(1, idx, 1.0)
    return out


def _row_seed(key_bytes):
    d = hashlib.blake2b(key_bytes, digest_size=8, key=_FLIP_SECRET).digest()
    return int.from_bytes(d, "little") % (2 ** 32)


def defense_flip(probs, epsilon=0.3, X=None):

    N, C = probs.shape
    true_labels = probs.argmax(dim=1)
    out = torch.zeros_like(probs)
    for i in range(N):
        true = int(true_labels[i])
        if X is not None:
            key = X[i].detach().cpu().numpy().astype("float32").tobytes()
        else:  # fallback: key on the prediction vector itself
            key = probs[i].detach().cpu().numpy().astype("float32").tobytes()
        rng = np.random.RandomState(_row_seed(key))
        if rng.random_sample() < epsilon:
            choices = [c for c in range(C) if c != true]
            label = int(rng.choice(choices))
        else:
            label = true
        out[i, label] = 1.0
    return out


def apply_defense(probs, spec, X=None):

    if spec is None:
        return probs
    mode = spec.get("mode", "none")
    if mode == "none":
        return probs
    if mode == "round":
        return defense_round(probs, int(spec.get("decimals", 2)))
    if mode == "topk":
        return defense_top_k(probs, int(spec.get("k", 3)))
    if mode == "label_only":
        return defense_labels_only(probs)
    if mode == "flip":
        return defense_flip(probs, float(spec.get("epsilon", 0.3)), X=X)
    raise ValueError(f"unknown defense mode: {mode!r} (valid: {VALID_MODES})")


def spec_label(spec):
    """Human-readable one-liner for logs/plots."""
    if spec is None or spec.get("mode", "none") == "none":
        return "none"
    mode = spec["mode"]
    if mode == "round":
        return f"round({spec.get('decimals', 2)}d)"
    if mode == "topk":
        return f"top-{spec.get('k', 3)}"
    if mode == "flip":
        return f"flip(eps={spec.get('epsilon', 0.3)})"
    return "label_only"
