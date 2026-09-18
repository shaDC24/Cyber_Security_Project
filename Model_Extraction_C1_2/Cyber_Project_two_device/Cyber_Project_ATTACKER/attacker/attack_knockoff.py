import torch

from common.data_mnist import load_fashion_pool, load_mnist_tensors
from common.train_utils import eval_accuracy, train_model
from attacker.substitute_def import SUBSTITUTES, SubstituteCNN

_ADVERSARY_POOL_CACHE = {}
_FASHION_POOL_CACHE = {}


def get_adversary_pool(seed=123, n_pool=5000):

    key = (seed, n_pool)
    if key not in _ADVERSARY_POOL_CACHE:
        X_pool, _y_unused, _, _ = load_mnist_tensors(n_train=n_pool, n_test=1, seed=seed)
        _ADVERSARY_POOL_CACHE[key] = X_pool
    return _ADVERSARY_POOL_CACHE[key]


def get_fashion_pool(seed=123, n_pool=5000):

    key = (seed, n_pool)
    if key not in _FASHION_POOL_CACHE:
        _FASHION_POOL_CACHE[key] = load_fashion_pool(n_pool=n_pool, seed=seed)
    return _FASHION_POOL_CACHE[key]


def sample_transfer_set(budget, mode, seed):

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

    torch.manual_seed(seed) 

    X_transfer = sample_transfer_set(budget, mode, seed)
    probs = oracle.query(X_transfer)
    pseudo_labels = probs.argmax(dim=1)

    if pseudo_labels.unique().numel() < 2:
 
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
