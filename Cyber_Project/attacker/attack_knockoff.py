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

from common.data_mnist import load_mnist_tensors
from common.train_utils import eval_accuracy, train_model
from attacker.substitute_def import SubstituteCNN

_ADVERSARY_POOL_CACHE = {}


def get_adversary_pool(seed=123, n_pool=5000):
    """A REAL MNIST pool disjoint from the victim's own training subsample
    (seed 123 != the victim's seed 42) -- the adversary's own unlabeled
    digit images for the closed-world scenario."""
    key = (seed, n_pool)
    if key not in _ADVERSARY_POOL_CACHE:
        X_pool, _y_unused, _, _ = load_mnist_tensors(n_train=n_pool, n_test=1, seed=seed)
        _ADVERSARY_POOL_CACHE[key] = X_pool
    return _ADVERSARY_POOL_CACHE[key]


def sample_transfer_set(budget, mode, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    if mode == "closed":
        pool = get_adversary_pool()
        idx = torch.randint(0, pool.size(0), (budget,), generator=g)
        X = pool[idx].clone()
    elif mode == "open":
        X = torch.rand((budget, 1, 28, 28), generator=g)
    else:
        raise ValueError(mode)
    return X


def knockoff_attack(oracle, X_test, y_test, budget, mode="closed", seed=0, epochs=6):
    X_transfer = sample_transfer_set(budget, mode, seed)
    probs = oracle.query(X_transfer)
    pseudo_labels = probs.argmax(dim=1)

    if pseudo_labels.unique().numel() < 2:
        return {"budget": budget, "mode": mode, "accuracy": 0.0, "fidelity": 0.0}

    knockoff = SubstituteCNN(n_classes=oracle.n_classes)
    knockoff = train_model(
        knockoff, X_transfer, pseudo_labels,
        epochs=epochs, batch_size=64, lr=1e-3, verbose=False,
    )

    accuracy = eval_accuracy(knockoff, X_test, y_test)

    victim_pred_on_test = oracle.query(X_test).argmax(dim=1)
    fidelity = (knockoff(X_test).argmax(dim=1) == victim_pred_on_test).float().mean().item()

    return {"budget": budget, "mode": mode,
            "accuracy": float(accuracy), "fidelity": float(fidelity)}


def budget_sweep(oracle, X_test, y_test, budgets, mode="closed", seed=0):
    results = []
    for i, b in enumerate(budgets):
        r = knockoff_attack(oracle, X_test, y_test, budget=b, mode=mode, seed=seed + i)
        results.append(r)
        print(f"[knockoff:{mode}] budget={b:5d}  "
              f"accuracy={r['accuracy']:.4f}  fidelity={r['fidelity']:.4f}")
    return results
