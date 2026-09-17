"""
run_all.py
----------
One-command end-to-end run:
  1. (if missing) train the two victims.
  2. Start the UNDEFENDED API (5000) and DEFENDED API (5001) as subprocesses.
  3. Run both attacks with/without defenses -> results/results.json + plots.
  4. Run the bonus defense analysis.
  5. Shut the servers down.

    python run_all.py

If you would rather see the client/server split for your demo screenshots,
start the servers by hand in separate terminals instead (see README.md) and
run `python attacker/run_experiment.py`.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from attacker.remote_oracle import RemoteOracle
from common.paths import MODELS_DIR

UNDEFENDED_URL = "http://localhost:5000"
DEFENDED_URL = "http://localhost:5001"


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


def ensure_victims():
    linear = os.path.join(MODELS_DIR, "victim_linear.pth")
    cnn = os.path.join(MODELS_DIR, "victim_cnn.pth")
    if os.path.exists(linear) and os.path.exists(cnn):
        print("[run_all] victim checkpoints present -- skipping training.")
        return
    print("[run_all] training victims (first run) ...")
    subprocess.run([sys.executable, os.path.join("victim", "train_victims.py")],
                   cwd=ROOT, env=_env(), check=True)


def start_servers():
    procs = []
    print("[run_all] starting UNDEFENDED API on 5000 ...")
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join("victim", "victim_api.py")],
        cwd=ROOT, env=_env()))
    print("[run_all] starting DEFENDED API on 5001 (default: none) ...")
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join("victim", "victim_api_defended.py"), "none"],
        cwd=ROOT, env=_env()))

    RemoteOracle(UNDEFENDED_URL, "linear").wait_until_up()
    RemoteOracle(DEFENDED_URL, "linear").wait_until_up()
    print("[run_all] both servers are up.")
    return procs


def stop_servers(procs):
    print("[run_all] stopping servers ...")
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            p.kill()


def main():
    ensure_victims()
    procs = start_servers()
    try:
        from attacker import (run_experiment, defense_analysis,
                              arch_transfer_demo, open_world_demo,
                              flip_determinism_demo)
        run_experiment.main()
        print("\n" + "=" * 70)
        print("BONUS: defense analysis")
        print("=" * 70)
        defense_analysis.main()
        print("\n" + "=" * 70)
        print("ROBUSTNESS 1: architecture-transfer demo (CNN vs MLP substitute)")
        print("=" * 70)
        arch_transfer_demo.main()
        print("\n" + "=" * 70)
        print("ROBUSTNESS 2: open-world demo (closed vs noise vs Fashion-MNIST)")
        print("=" * 70)
        open_world_demo.main()
        print("\n" + "=" * 70)
        print("DEFENSE PROOF: flip determinism (re-query cannot average it away)")
        print("=" * 70)
        flip_determinism_demo.main()
    finally:
        time.sleep(0.5)
        stop_servers(procs)
    print("\n[run_all] complete. See results/ for results.json and the plots.")


if __name__ == "__main__":
    main()
