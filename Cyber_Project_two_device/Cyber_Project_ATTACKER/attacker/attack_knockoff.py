"""
attacker/attack_knockoff.py
---------------------------
Knockoff-style learning-based extraction (Orekondy et al. 2019 / Jagielski
et al. 2020) against the SmallCNN victim, reached over the HTTP API.

Attack idea:
  1. Adversary does NOT know the victim's architecture (trains its own
     SubstituteCNN), training data, or class semantics.
  2. Samples a "transfer set" of `budget` images from a pool P_A:
       - "closed": real MNIST images from a DIFFERENT split than the victim's
         own training subsample (adversary has pixels only, never the victim's
         labels) -- Orekondy's closed-world (P_V subset of P_A).
       - "open"  : pure random-noise images -- the weak, out-of-distribution
         adversary (Orekondy's open-world).
  3. Each transfer image is sent to the victim API for a pseudo-label
     (top-1 predicted class).
  4. A brand-new substitute CNN is trained from scratch on the
     (image, pseudo-label) pairs.
  5. Query budget is swept to reproduce an Orekondy-style Fig. 5 curve, with
     both accuracy (vs ground truth) and fidelity (vs victim's predictions).
"""

import torch

from common.data_mnist import load_fashion_pool, load_mnist_tensors
from common.train_utils import eval_accuracy, train_model
from attacker.substitute_def import SUBSTITUTES, SubstituteCNN

_ADVERSARY_POOL_CACHE = {}
_FASHION_POOL_CACHE = {}


def get_adversary_pool(seed=123, n_pool=5000):
    """A REAL MNIST pool disjoint from the victim's own training subsample
    (seed 123 != the victim's seed 42) -- the adversary's own unlabeled
    digit images for the closed-world scenario."""
    key = (seed, n_pool)
    if key not in _ADVERSARY_POOL_CACHE:
        X_pool, _y_unused, _, _ = load_mnist_tensors(n_train=n_pool, n_test=1, seed=seed)
        _ADVERSARY_POOL_CACHE[key] = X_pool
    return _ADVERSARY_POOL_CACHE[key]


def get_fashion_pool(seed=123, n_pool=5000):
    """A REAL but out-of-distribution pool: Fashion-MNIST (clothing images).
    Used for the stronger open-world scenario (a different real distribution,
    not noise) -- Orekondy et al.'s central open-world finding."""
    key = (seed, n_pool)
    if key not in _FASHION_POOL_CACHE:
        _FASHION_POOL_CACHE[key] = load_fashion_pool(n_pool=n_pool, seed=seed)
    return _FASHION_POOL_CACHE[key]


def sample_transfer_set(budget, mode, seed):
    """Build the attacker's transfer set of `budget` query images.

    mode = "closed"    : real MNIST, disjoint split (adversary has task-relevant
                         pixels, never the victim's labels).
    mode = "open"      : pure uniform-random NOISE images -- the weakest,
                         maximally-generic adversary (kept as promised in the
                         design report).
    mode = "open_real" : Fashion-MNIST -- REAL images from a DIFFERENT
                         distribution (clothes, not digits). Reproduces
                         Orekondy et al.'s stronger open-world result.
    """
    g = torch.Generator().manual_seed(seed)
    if mode == "closed":
        pool = get_adversary_pool()
        idx = torch.randint(0, pool.size(0), (budget,), generator=g)
        X = pool[idx].clone()
    elif mode == "open":
        X = torch.rand((budget, 1, 28, 28), generator=g)
    elif mode == "open_real":
        pool = get_fashion_pool()
        idx = torch.randint(0, pool.size(0), (budget,), generator=g)
        X = pool[idx].clone()
    else:
        raise ValueError(mode)
    return X


def knockoff_attack(oracle, X_test, y_test, budget, mode="closed", seed=0,
                    epochs=6, arch="cnn"):
    """Run one knockoff extraction at a fixed query budget.

    `arch` selects the ATTACKER's own substitute architecture ("cnn" or "mlp").
    The victim's architecture is never used -- picking "mlp" against the CNN
    victim demonstrates architecture-independent stealing (Orekondy Fig. 10).

    A per-call torch.manual_seed(seed) is set so substitute-network weight
    initialisation and batch shuffling are reproducible across re-runs (D3).
    """
    torch.manual_seed(seed)  # reproducible init + batch order for this run

    X_transfer = sample_transfer_set(budget, mode, seed)
    probs = oracle.query(X_transfer)
    pseudo_labels = probs.argmax(dim=1)

    if pseudo_labels.unique().numel() < 2:
        # Victim collapses to a single class on this (degenerate) query set:
        # no usable signal, so extraction cannot beat majority-guessing.
        return {"budget": budget, "mode": mode, "arch": arch,
                "accuracy": 0.0, "fidelity": 0.0}

    substitute_cls = SUBSTITUTES.get(arch, SubstituteCNN)
    knockoff = substitute_cls(n_classes=oracle.n_classes)
    knockoff = train_model(
        knockoff, X_transfer, pseudo_labels,
        epochs=epochs, batch_size=64, lr=1e-3, verbose=False,
    )

    accuracy = eval_accuracy(knockoff, X_test, y_test)

    victim_pred_on_test = oracle.query(X_test).argmax(dim=1)
    with torch.no_grad():
        knock_pred = knockoff(X_test).argmax(dim=1)
    fidelity = (knock_pred == victim_pred_on_test).float().mean().item()

    return {"budget": budget, "mode": mode, "arch": arch,
            "accuracy": float(accuracy), "fidelity": float(fidelity)}


def budget_sweep(oracle, X_test, y_test, budgets, mode="closed", seed=0, arch="cnn"):
    results = []
    for i, b in enumerate(budgets):
        r = knockoff_attack(oracle, X_test, y_test, budget=b, mode=mode,
                            seed=seed + i, arch=arch)
        results.append(r)
        print(f"[knockoff:{mode}:{arch}] budget={b:5d}  "
              f"accuracy={r['accuracy']:.4f}  fidelity={r['fidelity']:.4f}")
    return results
