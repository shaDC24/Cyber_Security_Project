"""
common/train_utils.py
----------------------
Self-contained supervised training + evaluation helpers (no external training
library -- plain PyTorch loop with the Adam optimizer). Shared by:
  * victim/train_victims.py   (trains the victim models)
  * attacker/attack_knockoff.py (trains the attacker's substitute model)
"""

import torch
import torch.nn as nn


def train_model(model, X_train, y_train, epochs=5, batch_size=128, lr=1e-3,
                weight_decay=0.0, device="cpu", verbose=True):
    """Plain SGD-style loop with Adam + CrossEntropyLoss.

    y_train are integer class labels (hard labels)."""
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    n = X_train.size(0)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx].to(device), y_train[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}  "
                  f"loss={total_loss / n:.4f}  train_acc={correct / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def eval_accuracy(model, X, y, device="cpu", batch_size=256):
    model.eval()
    model.to(device)
    correct = 0
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
    return correct / X.size(0)
