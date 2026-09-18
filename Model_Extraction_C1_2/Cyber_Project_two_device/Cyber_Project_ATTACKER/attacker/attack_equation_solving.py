
import numpy as np
import torch


def equation_solving_attack(oracle, d, n_classes, n_queries=None, seed=0):
    rng = np.random.RandomState(seed)
    if n_queries is None:
        n_queries = d + 1
    n_queries = max(n_queries, d + 1)

    X_query_flat = rng.uniform(0.0, 1.0, size=(n_queries, d)).astype(np.float64)
    X_query = torch.from_numpy(X_query_flat.astype(np.float32)).view(n_queries, 1, 28, 28)

    probs = oracle.query(X_query).cpu().numpy().astype(np.float64)  # (n_queries, n_classes)

    eps = 1e-38
    logp = np.log(np.clip(probs, eps, 1.0))
    Y = logp[:, 1:] - logp[:, 0:1]  # (n_queries, n_classes - 1)

    X_aug = np.hstack([X_query_flat, np.ones((n_queries, 1), dtype=np.float64)])
    theta, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)  # (d+1, n_classes-1)

    W_hat = np.zeros((n_classes, d), dtype=np.float64)
    b_hat = np.zeros(n_classes, dtype=np.float64)
    W_hat[1:] = theta[:-1].T
    b_hat[1:] = theta[-1]

    return W_hat, b_hat, n_queries


def predict_from_extracted(W_hat, b_hat, X):
    """X: (N,1,28,28) torch tensor. Returns numpy predicted labels."""
    X_flat = X.view(X.size(0), -1).cpu().numpy().astype(np.float64)
    logits = X_flat @ W_hat.T + b_hat
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs.argmax(axis=1)


def evaluate_extraction(W_hat, b_hat, X_test, y_test, victim_oracle):
    pred_hat = predict_from_extracted(W_hat, b_hat, X_test)
    y_test_np = y_test.cpu().numpy()
    accuracy = (pred_hat == y_test_np).mean()

    victim_pred = victim_oracle.query(X_test).argmax(dim=1).cpu().numpy()
    fidelity = (pred_hat == victim_pred).mean()
    return float(accuracy), float(fidelity)
