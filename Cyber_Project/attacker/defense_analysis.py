"""
attacker/defense_analysis.py
----------------------------
BONUS diagnostic -- now covering BOTH attacks and the label-flip defense.

The two attacks are governed by DIFFERENT quantities, so we report both in one
table:

  * Equation-solving is driven by CONFIDENCE DISTORTION -- how much the returned
    probability VECTOR is corrupted. Column: mean |Δp| (sum over classes of the
    absolute change vs the undefended probs, averaged over images), range 0..2.
  * Knockoff is driven by TOP-1 LABEL CHANGE -- it only uses argmax. Column:
    % of images whose returned label differs from the undefended label.

Table 1 makes the asymmetry explicit:
  - round / top-k / labels-only : large confidence distortion  -> kill
    equation-solving; but ~0% label change -> do NOTHING to knockoff.
  - flip (label perturbation)   : also distorts confidences AND finally moves
    the label-change column to ~epsilon -> the first defense that hurts
    knockoff.

Table 2 + the saved plot quantify the flip defense's utility/security
trade-off: as epsilon grows, the knockoff's stolen accuracy falls, but so does
the HONEST user's accuracy (they get a wrong label ~epsilon of the time). This
is exactly the open problem Tramer (Sec 7), Orekondy (Sec 6) and Jagielski
(Sec 8) all identify and none of them implements.

Prereqs: both servers running (see run_experiment.py / run_all.py).
"""

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

# defenses to profile in Table 1
DEFENSES = {
    "round(2 decimals)": {"mode": "round", "decimals": 2},
    "top-3 truncation":  {"mode": "topk", "k": 3},
    "labels-only":       {"mode": "label_only"},
    "flip(eps=0.30)":    {"mode": "flip", "epsilon": 0.30},
}

# epsilons for the flip utility/security sweep (Table 2 + plot)
FLIP_EPSILONS = [0.0, 0.1, 0.2, 0.3, 0.5]
SWEEP_BUDGET = 1500


def confidence_distortion_and_label_change(plain_probs, plain_labels, defended, X):
    dprobs = defended.query(X)
    distortion = (dprobs - plain_probs).abs().sum(dim=1).mean().item()   # mean |Δp|
    label_change = (dprobs.argmax(dim=1) != plain_labels).float().mean().item()
    return distortion, label_change


def table1_why(X_test):
    plain = RemoteOracle(UNDEFENDED_URL, "cnn", defense_spec=None)
    plain_probs = plain.query(X_test)
    plain_labels = plain_probs.argmax(dim=1)

    print("\nTABLE 1 -- why each defense (does/doesn't) work, by attack")
    print("-" * 78)
    print(f"{'Defense':<20} | {'conf. distortion |Δp|':>21} | {'top-1 label change':>19}")
    print(f"{'':<20} | {'(drives EQUATION-SOLVING)':>21} | {'(drives KNOCKOFF)':>19}")
    print("-" * 78)
    for name, spec in DEFENSES.items():
        defended = RemoteOracle(DEFENDED_URL, "cnn", defense_spec=spec)
        dist, lch = confidence_distortion_and_label_change(
            plain_probs, plain_labels, defended, X_test)
        print(f"{name:<20} | {dist:>21.4f} | {lch * 100:>17.2f}%")
    print("-" * 78)
    print("Read-off: confidence-hiding defenses (round/top-k/labels-only) have a big |Δp|")
    print("(so equation-solving dies) but ~0% label change (so knockoff is untouched).")
    print("Only 'flip' moves the label-change column -> the first defense that hits knockoff.")


def table2_flip_tradeoff(X_test, y_test):
    y = y_test.cpu()
    honest_accs, knock_accs = [], []
    print("\nTABLE 2 -- flip defense: utility vs security trade-off")
    print("-" * 70)
    print(f"{'epsilon':>8} | {'honest user acc':>15} | {'knockoff acc (B=%d)' % SWEEP_BUDGET:>20}")
    print("-" * 70)
    for eps in FLIP_EPSILONS:
        spec = None if eps == 0.0 else {"mode": "flip", "epsilon": eps}
        url = UNDEFENDED_URL if eps == 0.0 else DEFENDED_URL
        oracle = RemoteOracle(url, "cnn", defense_spec=spec)

        honest = float((oracle.query(X_test).argmax(dim=1).cpu() == y).float().mean().item())
        r = knockoff_attack(oracle, X_test, y_test, budget=SWEEP_BUDGET,
                            mode="closed", seed=1, epochs=6)
        honest_accs.append(honest)
        knock_accs.append(r["accuracy"])
        print(f"{eps:>8.2f} | {honest * 100:>14.2f}% | {r['accuracy'] * 100:>19.2f}%")
    print("-" * 70)
    print("As epsilon rises the stolen (knockoff) accuracy falls -- but so does the")
    print("honest user's accuracy. That inseparable cost is the open problem all three")
    print("papers name and none of them solves.")

    plt.figure(figsize=(6, 4))
    plt.plot(FLIP_EPSILONS, [a * 100 for a in honest_accs], marker="o", label="honest user accuracy")
    plt.plot(FLIP_EPSILONS, [a * 100 for a in knock_accs], marker="s", label=f"knockoff accuracy (B={SWEEP_BUDGET})")
    plt.xlabel("flip probability  epsilon"); plt.ylabel("accuracy (%)")
    plt.title("Label-flip defense: utility vs security\n(the defense that actually hurts knockoff)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
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
