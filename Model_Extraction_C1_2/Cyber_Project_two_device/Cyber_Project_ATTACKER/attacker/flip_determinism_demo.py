import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from common.data_mnist import load_mnist_tensors
from attacker.remote_oracle import RemoteOracle

UNDEFENDED_URL = os.environ.get("VICTIM_UNDEFENDED_URL", "http://localhost:5000")
DEFENDED_URL = os.environ.get("VICTIM_DEFENDED_URL", "http://localhost:5001")

EPSILON = 0.30
REPEATS = 15
N_IMAGES = 500
N_CLASSES = 10


def main():
    RemoteOracle(UNDEFENDED_URL, "cnn").wait_until_up()
    RemoteOracle(DEFENDED_URL, "cnn").wait_until_up()

    _, _, X_test, y_test = load_mnist_tensors(n_train=1, n_test=N_IMAGES, seed=42)
    y_true = y_test.cpu().numpy()

    # Victim's TRUE (undefended) top-1 labels -- the target an attacker wants.
    plain = RemoteOracle(UNDEFENDED_URL, "cnn", defense_spec=None)
    victim_labels = plain.query(X_test).argmax(dim=1).cpu().numpy()

    flip = RemoteOracle(DEFENDED_URL, "cnn",
                        defense_spec={"mode": "flip", "epsilon": EPSILON})

    print("\n" + "=" * 70)
    print(f"FLIP-DETERMINISM PROOF  (epsilon={EPSILON}, repeats={REPEATS}, "
          f"images={N_IMAGES})")
    print("=" * 70)


    repeated = []
    for _ in range(REPEATS):
        repeated.append(flip.query(X_test).argmax(dim=1).cpu().numpy())
    repeated = np.stack(repeated, axis=0)            # (REPEATS, N_IMAGES)


    changed_per_image = (repeated != repeated[0:1]).any(axis=0)   # (N_IMAGES,)
    n_changed = int(changed_per_image.sum())
    first_resp = repeated[0]
    frac_flipped = float((first_resp != victim_labels).mean())

    print("\n[Check 1] Determinism across repeated identical queries")
    print(f"  images whose label changed over {REPEATS} repeats : "
          f"{n_changed} / {N_IMAGES}")
    print(f"  -> {'PASS: perfectly deterministic' if n_changed == 0 else 'FAIL'}")
    print(f"  fraction of labels flipped vs true victim label   : "
          f"{frac_flipped:.3f}  (target ~= epsilon = {EPSILON})")


    maj_real = _majority(repeated)
    acc_single = float((first_resp == victim_labels).mean())
    acc_maj_real = float((maj_real == victim_labels).mean())
    print("\n[Check 2] Majority vote over deterministic repeats (attacker's hope)")
    print(f"  single-query agreement with true victim label     : {acc_single:.3f}")
    print(f"  {REPEATS}-repeat majority agreement                     : {acc_maj_real:.3f}")
    print(f"  -> gain from re-querying: {acc_maj_real - acc_single:+.3f} "
          f"({'no gain -> defense holds' if acc_maj_real - acc_single < 1e-9 else 'gain!'})")

    rng = np.random.RandomState(0)
    sim = np.empty((REPEATS, N_IMAGES), dtype=np.int64)
    for r in range(REPEATS):
        for i in range(N_IMAGES):
            if rng.random_sample() < EPSILON:
                choices = [c for c in range(N_CLASSES) if c != victim_labels[i]]
                sim[r, i] = rng.choice(choices)
            else:
                sim[r, i] = victim_labels[i]
    maj_sim = _majority(sim)
    acc_maj_sim = float((maj_sim == victim_labels).mean())
    print("\n[Check 3] Contrast: NAIVE per-query-random flip (simulated)")
    print(f"  single noisy query agreement (~1-eps)             : "
          f"{float((sim[0] == victim_labels).mean()):.3f}")
    print(f"  {REPEATS}-repeat majority agreement                     : {acc_maj_sim:.3f}")
    print("  -> a non-deterministic flip is AVERAGED AWAY (accuracy jumps back up),")
    print("     which is exactly the attack our deterministic keying prevents.")

    print("\n" + "=" * 70)
    print("CONCLUSION: the flip defense is deterministic per input, so repeated")
    print("queries return identical answers and majority-voting yields no gain.")
    print("A naive random flip would be trivially defeated by the same trick.")
    print("=" * 70)


def _majority(mat):
    """Column-wise majority vote over a (R, N) int matrix -> (N,)."""
    R, N = mat.shape
    out = np.empty(N, dtype=np.int64)
    for i in range(N):
        vals, counts = np.unique(mat[:, i], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


if __name__ == "__main__":
    main()
