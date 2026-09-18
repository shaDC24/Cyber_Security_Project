import time

import requests
import torch

N_CLASSES = 10


class RemoteOracle:
    def __init__(self, base_url, model_name, n_classes=N_CLASSES,
                 defense_spec=None, batch_size=256, timeout=120):
        assert model_name in ("linear", "cnn")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.endpoint = f"{self.base_url}/predict/{model_name}"
        self.n_classes = n_classes
        self.defense_spec = defense_spec
        self.batch_size = batch_size
        self.timeout = timeout
        self.num_queries = 0


    def _post(self, imgs_list):
        payload = {"images": imgs_list}
        if self.defense_spec is not None:
            payload["defense"] = self.defense_spec
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["probabilities"]

    def health(self):
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def wait_until_up(self, retries=40, delay=0.5):
        last = None
        for _ in range(retries):
            try:
                return self.health()
            except Exception as e:  
                last = e
                time.sleep(delay)
        raise RuntimeError(f"victim API at {self.base_url} not reachable: {last}")

    def query(self, X):
        """X: (N,1,28,28) float tensor in [0,1]. Returns (N,n_classes) probs."""
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)
        X = X.view(X.size(0), 1, 28, 28)
        N = X.size(0)
        probs_all = []
        for i in range(0, N, self.batch_size):
            xb = X[i:i + self.batch_size]
            imgs = xb.squeeze(1).tolist()      # list of 28x28
            probs_all.extend(self._post(imgs))
        self.num_queries += N
        return torch.tensor(probs_all, dtype=torch.float32)

    def query_labels(self, X):
        return self.query(X).argmax(dim=1)

    def reset_query_count(self):
        self.num_queries = 0
