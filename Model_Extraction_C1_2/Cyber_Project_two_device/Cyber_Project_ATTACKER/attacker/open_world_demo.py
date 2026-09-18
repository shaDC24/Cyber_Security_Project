import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data_mnist import load_mnist_tensors
from common.paths import MODELS_DIR, RESULTS_DIR
from attacker.remote_oracle import RemoteOracle
from attacker.attack_knockoff import budget_sweep

UNDEFENDED_URL = os.environ.get("VICTIM_UNDEFENDED_URL", "http://localhost:5000")
BUDGETS = [100, 300, 800, 2000, 4000]
MODES = ["closed", "open", "open_real"]
MODE_LABEL = {"closed": "closed (MNIST)",
              "open": "open (noise)",
              "open_real": "open (Fashion-MNIST)"}


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

    print("\nOPEN-WORLD DEMO (victim = SmallCNN, no defense)")
    results = {}
    for mode in MODES:
        print(f"\n-- transfer distribution: {MODE_LABEL[mode]} --")
        results[mode] = budget_sweep(oracle, X_test, y_test, BUDGETS, mode=mode, seed=1)

    plt.figure(figsize=(6, 4))
    for mode in MODES:
        r = results[mode]
        plt.plot([x["budget"] for x in r], [x["accuracy"] * 100 for x in r],
                 marker="o", label=MODE_LABEL[mode])
    plt.axhline(victim_acc * 100, color="gray", linestyle="--", label="victim accuracy")
    plt.axhline(10.0, color="red", linestyle=":", label="chance (10%)")
    plt.xlabel("Query budget (B)"); plt.ylabel("Knockoff accuracy (%)")
    plt.title("Knockoff transfer distribution matters\n(closed vs noise vs real-OOD)")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_open_world.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nSaved {out}")

    with open(os.path.join(RESULTS_DIR, "open_world.json"), "w") as f:
        json.dump({"victim_cnn_accuracy": victim_acc, "results": results}, f, indent=2)
    print("Saved results/open_world.json")
    print("\nRead-off: Fashion-MNIST (real, but not digits) beats the noise floor,")
    print("reproducing Orekondy's point that a realistic query distribution -- not")
    print("just more queries -- is what makes extraction work.")


if __name__ == "__main__":
    main()
