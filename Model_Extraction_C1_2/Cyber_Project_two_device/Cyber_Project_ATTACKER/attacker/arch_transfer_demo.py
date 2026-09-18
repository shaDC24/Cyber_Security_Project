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
ARCHS = ["cnn", "mlp"]   


def main():
    RemoteOracle(UNDEFENDED_URL, "cnn").wait_until_up()
    _, _, X_test, y_test = load_mnist_tensors(n_train=1, n_test=2000, seed=42)
    oracle = RemoteOracle(UNDEFENDED_URL, "cnn", defense_spec=None)
    
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
