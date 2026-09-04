"""
victim/defenses.py
-------------------
Output-perturbation defenses, applied SERVER-SIDE (at the API layer) before the
prediction ever leaves the victim machine.

Two families:

  A) CONFIDENCE-HIDING (stop equation-solving, NOT knockoff):
       * Tramer et al. 2016, Sec 7  : round confidences / return labels only.
       * Orekondy et al. 2019, Sec 6: reduce informativeness of predictions.
       * Jagielski et al. 2020      : truncated / label-only outputs (WSL).
     These preserve the top-1 label, so a hard-label knockoff attacker loses
     nothing (see attacker/defense_analysis.py: 0% label change).

  B) LABEL-PERTURBATION / "prediction poisoning" (the defense that actually
     hurts knockoff): with probability `epsilon` the API returns a WRONG label
     instead of the true top-1. This is the randomized-response / DP-on-output
     idea all three papers name as the open direction but none implements. It
     works against knockoff precisely because it changes the label the attacker
     trains on -- at a direct cost to honest users' accuracy (~epsilon), the
     utility/security trade-off.

     Made DETERMINISTIC per input (keyed by a hash of the image + a server
     secret): the same query always gets the same (possibly flipped) answer, so
     an attacker cannot average the noise away by re-querying the same point.

A defense is a small JSON-serialisable "spec" dict:
    {"mode": "none"}
    {"mode": "round", "decimals": 2}
    {"mode": "topk",  "k": 3}
    {"mode": "label_only"}
    {"mode": "flip",  "epsilon": 0.3}      # <-- the anti-knockoff defense

Every function takes and returns a torch tensor of shape (N, n_classes).
"""

import hashlib

import numpy as np
import torch

VALID_MODES = ("none", "round", "topk", "label_only", "flip")

# Fixed server-side secret so the deterministic flips are stable across runs
# (reproducible demo) but not guessable by the attacker.
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
    """Randomized-response label perturbation (deterministic per input).

    Returns a labels-only (one-hot) response whose label is the true top-1 with
    prob (1-epsilon) and a uniformly-random OTHER class with prob epsilon.
    Emitting one-hot (not the soft vector) keeps the defense honest: it hides
    confidences too, so the only information the attacker gets is a label that
    is wrong an epsilon fraction of the time.
    """
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
    """Apply the defense described by `spec` to a (N, n_classes) prob tensor.

    X (the (N,1,28,28) input batch) is used only by the 'flip' defense so its
    randomness can be keyed deterministically to each input image.
    """
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
