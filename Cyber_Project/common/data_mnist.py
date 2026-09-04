"""
common/data_mnist.py
---------------------
Loads the real MNIST handwritten-digit dataset (60,000 train / 10,000 test
28x28 grayscale images, 10 classes) as PyTorch tensors.

Tries the standard `torchvision.datasets.MNIST` auto-download first (this is
what will normally happen on your own laptop/lab machine with full internet
access, and the raw files are already shipped in data/MNIST/raw so no network
is needed at all). If that is blocked, it falls back to downloading the exact
same official MNIST files from a public GitHub mirror and parses the raw IDX
file format by hand -- no torchvision dataset classes in the fallback path,
only plain gzip + struct parsing, so this is still "our own code" reading the
data, not a borrowed data-loading pipeline.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np
import torch
from torch.utils.data import TensorDataset

from common.paths import DATA_DIR

MIRROR_BASE = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_mirror_files(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    for key, fname in FILES.items():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            url = MIRROR_BASE + fname
            print(f"[data_mnist] downloading {url} ...")
            urllib.request.urlretrieve(url, path)
    return cache_dir


def _parse_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return data


def _parse_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
    return data


def _load_via_mirror(cache_dir):
    _download_mirror_files(cache_dir)
    X_train = _parse_idx_images(os.path.join(cache_dir, FILES["train_images"]))
    y_train = _parse_idx_labels(os.path.join(cache_dir, FILES["train_labels"]))
    X_test = _parse_idx_images(os.path.join(cache_dir, FILES["test_images"]))
    y_test = _parse_idx_labels(os.path.join(cache_dir, FILES["test_labels"]))
    return X_train, y_train, X_test, y_test


def _load_via_torchvision(root):
    from torchvision import datasets  # local import: optional dependency path
    train_ds = datasets.MNIST(root=root, train=True, download=True)
    test_ds = datasets.MNIST(root=root, train=False, download=True)
    X_train = train_ds.data.numpy()
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.numpy()
    y_test = test_ds.targets.numpy()
    return X_train, y_train, X_test, y_test


def load_mnist_tensors(root=None, n_train=None, n_test=None, seed=0):
    """Returns (X_train, y_train, X_test, y_test) as torch tensors.

    X_* have shape (N, 1, 28, 28), float32, pixel values scaled to [0, 1].
    y_* have shape (N,), int64 class labels 0-9.

    n_train / n_test optionally subsample (with a fixed seed) to keep
    training times reasonable on CPU-only machines while still using 100%
    REAL MNIST images (never synthetic).
    """
    if root is None:
        root = DATA_DIR
    cache_dir = os.path.join(root, "MNIST_mirror_raw")
    try:
        print("[data_mnist] Trying torchvision (uses local data/MNIST/raw if present) ...")
        X_train, y_train, X_test, y_test = _load_via_torchvision(root)
        print("[data_mnist] torchvision load succeeded.")
    except Exception as e:
        print(f"[data_mnist] torchvision load failed ({type(e).__name__}: {e}).")
        print("[data_mnist] Falling back to GitHub mirror of the official MNIST files ...")
        X_train, y_train, X_test, y_test = _load_via_mirror(cache_dir)
        print("[data_mnist] GitHub mirror download succeeded.")

    rng = np.random.RandomState(seed)

    def subsample(X, y, n):
        if n is None or n >= len(X):
            return X, y
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx], y[idx]

    X_train, y_train = subsample(X_train, y_train, n_train)
    X_test, y_test = subsample(X_test, y_test, n_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
    y_train_t = torch.tensor(y_train, dtype=torch.int64)
    y_test_t = torch.tensor(y_test, dtype=torch.int64)

    print(f"[data_mnist] Train set: {tuple(X_train_t.shape)}, "
          f"Test set: {tuple(X_test_t.shape)}")
    return X_train_t, y_train_t, X_test_t, y_test_t


def make_loader(X, y, batch_size=128, shuffle=True):
    ds = TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
