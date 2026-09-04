"""
victim/train_victims.py
-----------------------
Trains BOTH victim models on REAL MNIST digit images and saves them to
models/ as .pth checkpoints (plus victim_meta.json):

    models/victim_linear.pth  -- LinearSoftmax (target of equation-solving)
    models/victim_cnn.pth      -- SmallCNN      (target of the knockoff attack)

Run once before starting the API servers:

    python victim/train_victims.py

(The project already ships pretrained copies of these files, so you only need
to run this if you want to retrain from scratch.)
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from common.data_mnist import load_mnist_tensors
from common.paths import MODELS_DIR
from common.train_utils import eval_accuracy, train_model
from victim.model_def import LinearSoftmax, SmallCNN

SEED = 42
N_TRAIN_VICTIM = 8000
N_TEST_VICTIM = 2000


def main():
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Loading real MNIST data")
    print("=" * 70)
    X_train, y_train, X_test, y_test = load_mnist_tensors(
        n_train=N_TRAIN_VICTIM, n_test=N_TEST_VICTIM, seed=SEED
    )

    print("\n" + "=" * 70)
    print("Training victim 1/2: LinearSoftmax (equation-solving target)")
    print("=" * 70)
    # weight_decay keeps the softmax from becoming pathologically overconfident,
    # which would push probabilities to float32's precision floor and make the
    # equation-solving log-odds system ill-conditioned. Real deployed logistic
    # regression is L2-regularised for the same generalisation reasons.
    victim_linear = LinearSoftmax(n_classes=10)
    victim_linear = train_model(
        victim_linear, X_train, y_train, epochs=15, batch_size=128, lr=1e-2,
        weight_decay=1e-2,
    )
    acc_linear = eval_accuracy(victim_linear, X_test, y_test)
    print(f"  -> victim_linear test accuracy: {acc_linear:.4f}")
    torch.save(victim_linear.state_dict(), os.path.join(MODELS_DIR, "victim_linear.pth"))
    print("  Saved models/victim_linear.pth")

    print("\n" + "=" * 70)
    print("Training victim 2/2: SmallCNN (knockoff target)")
    print("=" * 70)
    victim_cnn = SmallCNN(n_classes=10)
    victim_cnn = train_model(
        victim_cnn, X_train, y_train, epochs=6, batch_size=128, lr=1e-3
    )
    acc_cnn = eval_accuracy(victim_cnn, X_test, y_test)
    print(f"  -> victim_cnn test accuracy: {acc_cnn:.4f}")
    torch.save(victim_cnn.state_dict(), os.path.join(MODELS_DIR, "victim_cnn.pth"))
    print("  Saved models/victim_cnn.pth")

    meta = {
        "seed": SEED,
        "n_train_victim": N_TRAIN_VICTIM,
        "n_test_victim": N_TEST_VICTIM,
        "victim_linear_accuracy": acc_linear,
        "victim_cnn_accuracy": acc_cnn,
        "input_shape": [1, 28, 28],
        "n_classes": 10,
    }
    with open(os.path.join(MODELS_DIR, "victim_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved models/victim_meta.json")
    print("\nDone. Start the API with:  python victim/victim_api.py")


if __name__ == "__main__":
    main()
