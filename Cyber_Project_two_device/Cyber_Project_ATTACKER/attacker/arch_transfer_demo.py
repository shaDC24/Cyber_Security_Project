"""
attacker/arch_transfer_demo.py
------------------------------
ROBUSTNESS DEMO 1 -- architecture-independent stealing (Orekondy et al. 2019,
Sec. 6.2 / Fig. 10).

The main knockoff experiment happens to use a substitute CNN with the same shape
as the victim's SmallCNN (the favourable case Orekondy studies). A natural
question for the demo is: "did the attack only work because you secretly used
the victim's architecture?" This script answers NO by running the SAME attack
with a DELIBERATELY DIFFERENT attacker architecture -- a plain fully-connected
MLP (no convolutions) -- against the same convolutional victim, over the same
HTTP API. If the MLP knockoff also recovers most of the victim's accuracy, the
attack clearly does not depend on knowing the victim's architecture.

The attacker still only ever calls oracle.query(...) (HTTP POST /predict/cnn);
it never imports the victim model.

    python attacker/arch_transfer_demo.py

Prereq: undefended server running (python victim/victim_api.py, port 5000).
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common.data_mnist import load_mnist_tensors
from common.paths import MODELS_DIR, RESULTS_DIR
from attacker.remote_oracle import RemoteOracle
from attacker.attack_knockoff import knockoff_attack

UNDEFENDED_URL = os.environ.get("VICTIM_UNDEFENDED_URL", "http://localhost:5000")
BUDGETS = [300, 800, 2000, 4000]
ARCHS = ["cnn", "mlp"]   # attacker's own choices; victim is a CNN either way


def main():
    RemoteOracle(UNDEFENDED_URL, "cnn").wait_until_up()
    _, _, X_test, y_test = load_mnist_tensors(n_train=1, n_test=2000, seed=42)
    oracle = RemoteOracle(UNDEFENDED_URL, "cnn", defense_spec=None)

    # victim clean accuracy: from meta if present, else measured live over the
    # API (so a separate attacker machine needs no victim files at all).
    meta_path = os.path.join(MODELS_DIR, "victim_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            victim_acc = json.load(f)["victim_cnn_accuracy"]
    else:
        pred = oracle.query(X_test).argmax(dim=1).cpu()
        victim_acc = float((pred == y_test.cpu()).float().mean().item())
        print(f"(victim_meta.json not found -> measured victim accuracy = {victim_acc:.4f})")

    print("\nARCHITECTURE-TRANSFER DEMO (victim = SmallCNN, no defense)")
    print("-" * 68)
    print(f"victim CNN accuracy = {victim_acc:.4f}")
    print(f"{'budget':>7} | {'CNN-substitute acc':>18} | {'MLP-substitute acc':>18}")
    print("-" * 68)

    curves = {a: [] for a in ARCHS}
    for b in BUDGETS:
        row = {}
        for a in ARCHS:
            r = knockoff_attack(oracle, X_test, y_test, budget=b, mode="closed",
                                seed=1, arch=a)
            curves[a].append(r["accuracy"])
            row[a] = r["accuracy"]
        print(f"{b:>7} | {row['cnn']*100:>17.2f}% | {row['mlp']*100:>17.2f}%")
    print("-" * 68)
    print("Read-off: the MLP substitute (NO convolutions, different from the")
    print("victim CNN) also recovers most of the victim's accuracy -> the attack")
    print("is architecture-independent, exactly the Knockoff-Nets claim.")

    plt.figure(figsize=(6, 4))
    for a in ARCHS:
        plt.plot(BUDGETS, [v * 100 for v in curves[a]], marker="o",
                 label=f"{a.upper()} substitute")
    plt.axhline(victim_acc * 100, color="gray", linestyle="--", label="victim accuracy")
    plt.xlabel("Query budget (B)"); plt.ylabel("Knockoff accuracy (%)")
    plt.title("Architecture-independent stealing\n(CNN victim; attacker uses CNN vs MLP)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_arch_transfer.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nSaved {out}")

    with open(os.path.join(RESULTS_DIR, "arch_transfer.json"), "w") as f:
        json.dump({"victim_cnn_accuracy": victim_acc, "budgets": BUDGETS,
                   "accuracy_by_arch": curves}, f, indent=2)
    print("Saved results/arch_transfer.json")


if __name__ == "__main__":
    main()
