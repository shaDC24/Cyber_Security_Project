"""
attacker/run_experiment.py
--------------------------
Runs BOTH extraction attacks against the victim over the HTTP API, with and
without the (server-side) defenses, sweeping query budget, then saves
results/results.json and 3 plots.

Prerequisites -- start the two victim servers first (separate terminals):

    python victim/victim_api.py                 # undefended,  port 5000
    python victim/victim_api_defended.py none   # defended,    port 5001

Then:

    python attacker/run_experiment.py

(Or just run `python run_all.py`, which starts the servers for you.)

Design of the sweep:
  * "no_defense"       -> queries the UNDEFENDED server (5000).
  * the 3 defense runs -> query the DEFENDED server (5001), passing the
    defense spec so the server applies that policy AT THE API LAYER.
The attacker never imports the victim model -- it only holds RemoteOracle
handles to the two URLs.
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
from attacker.attack_equation_solving import equation_solving_attack, evaluate_extraction
from attacker.attack_knockoff import budget_sweep

SEED = 42
D = 28 * 28
N_CLASSES = 10

UNDEFENDED_URL = os.environ.get("VICTIM_UNDEFENDED_URL", "http://localhost:5000")
DEFENDED_URL = os.environ.get("VICTIM_DEFENDED_URL", "http://localhost:5001")

# scenario name -> (server url, defense spec sent to that server)
SCENARIOS = {
    "no_defense":          (UNDEFENDED_URL, None),
    "defense_round2":      (DEFENDED_URL, {"mode": "round", "decimals": 2}),
    "defense_topk3":       (DEFENDED_URL, {"mode": "topk", "k": 3}),
    "defense_labels_only": (DEFENDED_URL, {"mode": "label_only"}),
    # label-perturbation defense: the one that actually hurts knockoff
    "defense_flip30":      (DEFENDED_URL, {"mode": "flip", "epsilon": 0.3}),
}


def make_oracle(scenario, model_name):
    url, spec = SCENARIOS[scenario]
    return RemoteOracle(url, model_name, n_classes=N_CLASSES, defense_spec=spec)


def measure_honest_accuracy(X_test, y_test):
    """Accuracy an HONEST user sees from the CNN API under each defense -- the
    utility side of the utility/security trade-off. Confidence-hiding defenses
    cost ~0; the flip defense costs ~epsilon."""
    y = y_test.cpu()
    honest = {}
    for scenario in SCENARIOS:
        oracle = make_oracle(scenario, "cnn")
        pred = oracle.query(X_test).argmax(dim=1).cpu()
        honest[scenario] = float((pred == y).float().mean().item())
        print(f"[honest-accuracy:{scenario}] {honest[scenario]:.4f}")
    return honest


# Fallbacks used when models/victim_meta.json is NOT present -- e.g. on a
# separate attacker machine that has no access to any victim file. In that
# fully-black-box case the attacker measures the victim's accuracy itself, by
# querying the API on a test set it holds, instead of reading it from meta.
DEFAULT_SEED = 42
DEFAULT_N_TEST = 2000


def load_meta_or_none():
    """Return the victim meta dict if the file is present, else None.

    The attacker does NOT need this file: it only carries the victim's clean
    accuracy (for the plot reference line) and the eval seed/size. When absent
    (separate attacker device, no models/ folder) we fall back to defaults and
    measure the victim accuracy live over the API -- see measure_victim_accuracy."""
    path = os.path.join(MODELS_DIR, "victim_meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def measure_victim_accuracy(model_name, X_test, y_test):
    """Measure the victim's clean accuracy the black-box way: query the
    UNDEFENDED API on the test set and compare its predictions to ground truth.
    Uses only what the attacker already has (the API + a labelled test set)."""
    oracle = RemoteOracle(UNDEFENDED_URL, model_name, n_classes=N_CLASSES,
                          defense_spec=None)
    pred = oracle.query(X_test).argmax(dim=1).cpu()
    return float((pred == y_test.cpu()).float().mean().item())


def check_servers():
    print("Checking victim API servers ...")
    RemoteOracle(UNDEFENDED_URL, "linear").wait_until_up()
    print(f"  undefended server up at {UNDEFENDED_URL}")
    RemoteOracle(DEFENDED_URL, "linear").wait_until_up()
    print(f"  defended  server up at {DEFENDED_URL}")


def run_equation_solving_experiment(X_test, y_test):
    budgets = [D + 1, 2 * (D + 1), 5 * (D + 1)]
    all_results = {}
    for scenario in SCENARIOS:
        oracle = make_oracle(scenario, "linear")
        scenario_results = []
        for b in budgets:
            oracle.reset_query_count()
            W_hat, b_hat, n_used = equation_solving_attack(oracle, D, N_CLASSES,
                                                           n_queries=b, seed=0)
            acc, fid = evaluate_extraction(W_hat, b_hat, X_test, y_test, oracle)
            scenario_results.append({"budget": b, "queries_used": n_used,
                                     "accuracy": acc, "fidelity": fid})
            print(f"[equation-solving:{scenario}] budget={b:5d}  "
                  f"accuracy={acc:.4f}  fidelity={fid:.4f}")
        all_results[scenario] = scenario_results
    return all_results


def run_knockoff_experiment(X_test, y_test):
    budgets = [100, 300, 800, 2000, 4000]
    all_results = {}
    for scenario in SCENARIOS:
        scenario_results = {}
        for mode in ["closed", "open"]:
            oracle = make_oracle(scenario, "cnn")
            results = budget_sweep(oracle, X_test, y_test, budgets, mode=mode, seed=1)
            scenario_results[mode] = results
        all_results[scenario] = scenario_results
    return all_results


def make_plots(eq_results, knock_results, meta):
    plt.figure(figsize=(6, 4))
    for name, results in eq_results.items():
        plt.plot([r["budget"] for r in results], [r["accuracy"] for r in results],
                 marker="o", label=name)
    plt.axhline(meta["victim_linear_accuracy"], color="gray", linestyle="--",
                label="victim accuracy")
    plt.xlabel("Query budget"); plt.ylabel("Extracted model accuracy")
    plt.title("Equation-Solving Attack vs. Defenses\n(LinearSoftmax victim, real MNIST)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_equation_solving.png"), dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    for name, results in knock_results.items():
        r = results["closed"]
        plt.plot([x["budget"] for x in r], [x["accuracy"] for x in r],
                 marker="o", label=name)
    plt.axhline(meta["victim_cnn_accuracy"], color="gray", linestyle="--",
                label="victim accuracy")
    plt.xlabel("Query budget (B)"); plt.ylabel("Knockoff model accuracy")
    plt.title("Knockoff Attack vs. Defenses (closed-world)\n(SmallCNN victim, real MNIST)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_knockoff_closed.png"), dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    for mode in ["closed", "open"]:
        r = knock_results["no_defense"][mode]
        plt.plot([x["budget"] for x in r], [x["accuracy"] for x in r],
                 marker="o", label=f"{mode}-world")
    plt.axhline(meta["victim_cnn_accuracy"], color="gray", linestyle="--",
                label="victim accuracy")
    plt.xlabel("Query budget (B)"); plt.ylabel("Knockoff model accuracy")
    plt.title("Knockoff Attack: Closed-world vs Open-world\n(no defense, real MNIST victim)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_knockoff_closed_vs_open.png"), dpi=150); plt.close()

    print("\nSaved plots to results/: plot_equation_solving.png, "
          "plot_knockoff_closed.png, plot_knockoff_closed_vs_open.png")


def main():
    torch.manual_seed(SEED)
    check_servers()
    meta = load_meta_or_none()
    eval_seed = meta["seed"] if meta else DEFAULT_SEED
    n_test = meta["n_test_victim"] if meta else DEFAULT_N_TEST

    print("\nLoading real MNIST test set for evaluation")
    _, _, X_test, y_test = load_mnist_tensors(n_train=1, n_test=n_test, seed=eval_seed)

    # Victim clean accuracy: from meta if we have it, otherwise measured live
    # over the API (fully black-box, no victim files needed on this machine).
    if meta:
        victim_linear_acc = meta["victim_linear_accuracy"]
        victim_cnn_acc = meta["victim_cnn_accuracy"]
        print(f"victim_linear accuracy (from meta): {victim_linear_acc:.4f}")
        print(f"victim_cnn    accuracy (from meta): {victim_cnn_acc:.4f}")
    else:
        print("victim_meta.json not found -> measuring victim accuracy over the API ...")
        victim_linear_acc = measure_victim_accuracy("linear", X_test, y_test)
        victim_cnn_acc = measure_victim_accuracy("cnn", X_test, y_test)
        print(f"victim_linear accuracy (measured): {victim_linear_acc:.4f}")
        print(f"victim_cnn    accuracy (measured): {victim_cnn_acc:.4f}")

    # a meta-shaped dict so make_plots / output stay unchanged downstream
    meta = {"victim_linear_accuracy": victim_linear_acc,
            "victim_cnn_accuracy": victim_cnn_acc,
            "seed": eval_seed, "n_test_victim": n_test}

    print("\n" + "=" * 70)
    print("Equation-Solving Attack (Tramer et al. 2016) -- over HTTP API")
    print("=" * 70)
    eq_results = run_equation_solving_experiment(X_test, y_test)

    print("\n" + "=" * 70)
    print("Knockoff Attack (Orekondy 2019 / Jagielski 2020) -- over HTTP API")
    print("=" * 70)
    knock_results = run_knockoff_experiment(X_test, y_test)

    print("\n" + "=" * 70)
    print("Honest-user accuracy under each defense (utility cost)")
    print("=" * 70)
    honest_accuracy = measure_honest_accuracy(X_test, y_test)

    output = {
        "victim_linear_accuracy": meta["victim_linear_accuracy"],
        "victim_cnn_accuracy": meta["victim_cnn_accuracy"],
        "input_dim": D, "n_classes": N_CLASSES,
        "transport": "http-flask-api",
        "honest_accuracy_cnn": honest_accuracy,
        "equation_solving_attack": eq_results,
        "knockoff_attack": knock_results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved results/results.json")

    make_plots(eq_results, knock_results, meta)
    print("\nDone.")


if __name__ == "__main__":
    main()
