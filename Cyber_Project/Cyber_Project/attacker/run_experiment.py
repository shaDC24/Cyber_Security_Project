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


SCENARIOS = {
    "no_defense":          (UNDEFENDED_URL, None),
    "defense_round2":      (DEFENDED_URL, {"mode": "round", "decimals": 2}),
    "defense_topk3":       (DEFENDED_URL, {"mode": "topk", "k": 3}),
    "defense_labels_only": (DEFENDED_URL, {"mode": "label_only"}),
    "defense_flip30":      (DEFENDED_URL, {"mode": "flip", "epsilon": 0.3}),
}


def make_oracle(scenario, model_name):
    url, spec = SCENARIOS[scenario]
    return RemoteOracle(url, model_name, n_classes=N_CLASSES, defense_spec=spec)


def measure_honest_accuracy(X_test, y_test):

    y = y_test.cpu()
    honest = {}
    for scenario in SCENARIOS:
        oracle = make_oracle(scenario, "cnn")
        pred = oracle.query(X_test).argmax(dim=1).cpu()
        honest[scenario] = float((pred == y).float().mean().item())
        print(f"[honest-accuracy:{scenario}] {honest[scenario]:.4f}")
    return honest


def load_meta():
    with open(os.path.join(MODELS_DIR, "victim_meta.json")) as f:
        return json.load(f)


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
    meta = load_meta()
    print(f"victim_linear accuracy (from training): {meta['victim_linear_accuracy']:.4f}")
    print(f"victim_cnn    accuracy (from training): {meta['victim_cnn_accuracy']:.4f}")

    print("\nLoading real MNIST test set for evaluation")
    _, _, X_test, y_test = load_mnist_tensors(
        n_train=1, n_test=meta["n_test_victim"], seed=meta["seed"]
    )

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
