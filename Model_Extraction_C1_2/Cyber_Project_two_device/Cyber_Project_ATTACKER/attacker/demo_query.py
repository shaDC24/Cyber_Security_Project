import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests

from common.data_mnist import load_mnist_tensors

UNDEFENDED_URL = os.environ.get("VICTIM_UNDEFENDED_URL", "http://localhost:5000")
DEFENDED_URL = os.environ.get("VICTIM_DEFENDED_URL", "http://localhost:5001")


def show(title, url, model, image_2d, defense=None):
    payload = {"image": image_2d}
    if defense is not None:
        payload["defense"] = defense
    r = requests.post(f"{url}/predict/{model}", json=payload, timeout=30)
    resp = r.json()
    probs = [round(p, 4) for p in resp["probabilities"][0]]
    print(f"\n### {title}")
    print(f"POST {url}/predict/{model}")
    req_preview = {"image": "[[28x28 pixel floats in 0..1]]"}
    if defense is not None:
        req_preview["defense"] = defense
    print("request : " + json.dumps(req_preview))
    print("response: " + json.dumps({"probabilities": probs,
                                      "label": resp["labels"][0],
                                      "defense": resp["defense"]}))


def main():
    _, _, X_test, y_test = load_mnist_tensors(n_train=1, n_test=10, seed=42)
    img = X_test[0].squeeze(0).tolist()   # 28x28
    truth = int(y_test[0])
    print(f"Query image ground-truth digit = {truth}")

    show("UNDEFENDED (raw probabilities)", UNDEFENDED_URL, "cnn", img)
    show("DEFENSE: round to 2 decimals", DEFENDED_URL, "cnn", img,
         {"mode": "round", "decimals": 2})
    show("DEFENSE: top-3 truncation", DEFENDED_URL, "cnn", img,
         {"mode": "topk", "k": 3})
    show("DEFENSE: labels-only (one-hot)", DEFENDED_URL, "cnn", img,
         {"mode": "label_only"})
    show("DEFENSE: label-flip eps=0.3 (anti-knockoff)", DEFENDED_URL, "cnn", img,
         {"mode": "flip", "epsilon": 0.3})


if __name__ == "__main__":
    main()
