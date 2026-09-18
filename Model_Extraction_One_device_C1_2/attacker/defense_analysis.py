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
from common.paths import RESULTS_DIR
from attacker.remote_oracle import RemoteOracle
from attacker.attack_knockoff import knockoff_attack

UNDEFENDED_URL = os.environ.get("VICTIM_UNDEFENDED_URL", "http://localhost:5000")
DEFENDED_URL = os.environ.get("VICTIM_DEFENDED_URL", "http://localhost:5001")

DEFENSES = {
    "round(2 decimals)": {"mode": "round", "decimals": 2},
    "top-3 truncation":  {"mode": "topk", "k": 3},
    "labels-only":       {"mode": "label_only"},
    "flip(eps=0.30)":    {"mode": "flip", "epsilon": 0.30},
}

FLIP_EPSILONS = [0.0, 0.1, 0.2, 0.3, 0.5]
SWEEP_BUDGET = 1500


def table1_why(X_test):

    plain_lin = RemoteOracle(UNDEFENDED_URL, "linear", defense_spec=None)
    plain_lin_probs = plain_lin.query(X_test)

    plain_cnn = RemoteOracle(UNDEFENDED_URL, "cnn", defense_spec=None)
    plain_cnn_labels = plain_cnn.query(X_test).argmax(dim=1)

    print("\nTABLE 1 -- why each defense (does/doesn't) work, by attack")
    print("-" * 80)
    print(f"{'Defense':<20} | {'conf. distortion |Δp|':>23} | {'top-1 label change':>19}")
    print(f"{'':<20} | {'on LINEAR (eq-solving)':>23} | {'on CNN (knockoff)':>19}")
    print("-" * 80)
    for name, spec in DEFENSES.items():
        # distortion column: probability change on the linear (eq-solving) victim
        d_lin = RemoteOracle(DEFENDED_URL, "linear", defense_spec=spec).query(X_test)
        distortion = (d_lin - plain_lin_probs).abs().sum(dim=1).mean().item()
        # label-change column: argmax change on the CNN (knockoff) victim
        d_cnn = RemoteOracle(DEFENDED_URL, "cnn", defense_spec=spec).query(X_test)
        label_change = (d_cnn.argmax(dim=1) != plain_cnn_labels).float().mean().item()
        print(f"{name:<20} | {distortion:>23.4f} | {label_change * 100:>17.2f}%")
    print("-" * 80)
    print("Read-off: confidence-hiding defenses (round/top-k/labels-only) distort the")
    print("linear victim's probability vector (so equation-solving dies) but cause ~0%")
    print("label change on the CNN (so knockoff is untouched).")
    print("Only 'flip' moves the label-change column -> the first defense that hits knockoff.")


def table2_flip_tradeoff(X_test, y_test):
    y = y_test.cpu()
    honest_accs, knock_accs, knock_fids = [], [], []
    print("\nTABLE 2 -- flip defense: utility vs security trade-off")
    print("-" * 86)
    print(f"{'epsilon':>8} | {'honest user acc':>15} | "
          f"{'knockoff acc (B=%d)' % SWEEP_BUDGET:>20} | {'knockoff fidelity':>18}")
    print("-" * 86)
    for eps in FLIP_EPSILONS:
        spec = None if eps == 0.0 else {"mode": "flip", "epsilon": eps}
        url = UNDEFENDED_URL if eps == 0.0 else DEFENDED_URL
        oracle = RemoteOracle(url, "cnn", defense_spec=spec)

        honest = float((oracle.query(X_test).argmax(dim=1).cpu() == y).float().mean().item())
        r = knockoff_attack(oracle, X_test, y_test, budget=SWEEP_BUDGET,
                            mode="closed", seed=1, epochs=6)
        honest_accs.append(honest)
        knock_accs.append(r["accuracy"])
        knock_fids.append(r["fidelity"])
        print(f"{eps:>8.2f} | {honest * 100:>14.2f}% | {r['accuracy'] * 100:>19.2f}% | "
              f"{r['fidelity'] * 100:>16.2f}%")
    print("-" * 86)
    print("Key insight (Jagielski et al.'s accuracy-vs-fidelity split):")
    print("  * knockoff FIDELITY (agreement with the victim) drops sharply as epsilon")
    print("    rises -> flip DOES defeat a reconnaissance-motivated (fidelity) attacker.")
    print("  * knockoff ACCURACY barely drops -> a CNN tolerates minority label noise,")
    print("    so a theft-motivated (accuracy) attacker is only mildly hurt.")
    print("  * honest-user accuracy also falls by ~epsilon -> the inseparable")
    print("    utility/security trade-off all three papers name but none quantifies.")

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(FLIP_EPSILONS, [a * 100 for a in honest_accs], marker="o",
             label="honest user accuracy (utility)")
    plt.plot(FLIP_EPSILONS, [a * 100 for a in knock_accs], marker="s",
             label=f"knockoff accuracy (B={SWEEP_BUDGET})")
    plt.plot(FLIP_EPSILONS, [a * 100 for a in knock_fids], marker="^",
             label=f"knockoff fidelity (B={SWEEP_BUDGET})")
    plt.xlabel("flip probability  epsilon"); plt.ylabel("accuracy / fidelity (%)")
    plt.title("Label-flip defense: utility vs security\n"
              "(flip mainly hurts knockoff FIDELITY, at a cost to honest users)")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_flip_tradeoff.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nSaved {out}")


def main():
    RemoteOracle(UNDEFENDED_URL, "cnn").wait_until_up()
    RemoteOracle(DEFENDED_URL, "cnn").wait_until_up()

    _, _, X_test, y_test = load_mnist_tensors(n_train=1, n_test=2000, seed=42)

    table1_why(X_test)
    table2_flip_tradeoff(X_test, y_test)


if __name__ == "__main__":
    main()
