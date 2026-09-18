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


def _load_via_local_raw(root):

    raw = os.path.join(root, "MNIST", "raw")
    fmap = {
        "train_images": "train-images-idx3-ubyte",
        "train_labels": "train-labels-idx1-ubyte",
        "test_images": "t10k-images-idx3-ubyte",
        "test_labels": "t10k-labels-idx1-ubyte",
    }

    def _pick(name):
        """Prefer the plain IDX file; fall back to the .gz next to it."""
        p = os.path.join(raw, fmap[name])
        return p if os.path.exists(p) else p + ".gz"

    def _open(path):
        return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")

    def _imgs(path):
        with _open(path) as f:
            _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    def _labs(path):
        with _open(path) as f:
            _magic, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    needed = [_pick(k) for k in fmap]
    if not all(os.path.exists(p) for p in needed):
        raise FileNotFoundError(f"shipped MNIST raw files not found under {raw}")

    X_train = _imgs(_pick("train_images"))
    y_train = _labs(_pick("train_labels"))
    X_test = _imgs(_pick("test_images"))
    y_test = _labs(_pick("test_labels"))
    return X_train, y_train, X_test, y_test


def _load_via_torchvision(root):
    from torchvision import datasets 
    train_ds = datasets.MNIST(root=root, train=True, download=True)
    test_ds = datasets.MNIST(root=root, train=False, download=True)
    X_train = train_ds.data.numpy()
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.numpy()
    y_test = test_ds.targets.numpy()
    return X_train, y_train, X_test, y_test


def load_mnist_tensors(root=None, n_train=None, n_test=None, seed=0):

    if root is None:
        root = DATA_DIR
    cache_dir = os.path.join(root, "MNIST_mirror_raw")

    try:
        print("[data_mnist] Trying shipped local data/MNIST/raw (offline IDX parser) ...")
        X_train, y_train, X_test, y_test = _load_via_local_raw(root)
        print("[data_mnist] Loaded shipped MNIST raw files (no network needed).")
    except Exception as e_local:
        print(f"[data_mnist] local raw load failed ({type(e_local).__name__}: {e_local}).")
        try:
            print("[data_mnist] Trying torchvision ...")
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


FASHION_MIRROR = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FASHION_TRAIN_IMAGES = "train-images-idx3-ubyte.gz"


def load_fashion_pool(root=None, n_pool=5000, seed=0):

    if root is None:
        root = DATA_DIR
    local_raw = os.path.join(root, "FashionMNIST", "raw")
    cache_dir = os.path.join(root, "FashionMNIST_mirror_raw")

    def _read(path):
        with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
            _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    plain = os.path.join(local_raw, "train-images-idx3-ubyte")
    gz_local = plain + ".gz"
    if os.path.exists(plain):
        imgs = _read(plain)
    elif os.path.exists(gz_local):
        imgs = _read(gz_local)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, FASHION_TRAIN_IMAGES)
        if not os.path.exists(dst):
            print(f"[data_mnist] downloading Fashion-MNIST {FASHION_MIRROR + FASHION_TRAIN_IMAGES} ...")
            urllib.request.urlretrieve(FASHION_MIRROR + FASHION_TRAIN_IMAGES, dst)
        imgs = _read(dst)

    rng = np.random.RandomState(seed)
    if n_pool < len(imgs):
        idx = rng.choice(len(imgs), size=n_pool, replace=False)
        imgs = imgs[idx]
    X = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    print(f"[data_mnist] Fashion-MNIST open-world pool: {tuple(X.shape)}")
    return X
