import os

import torch
from flask import Flask, jsonify, request

from common.paths import MODELS_DIR
from victim.defenses import apply_defense, spec_label
from victim.model_def import LinearSoftmax, SmallCNN

N_CLASSES = 10


def _load_victims():
    linear_path = os.path.join(MODELS_DIR, "victim_linear.pth")
    cnn_path = os.path.join(MODELS_DIR, "victim_cnn.pth")
    if not (os.path.exists(linear_path) and os.path.exists(cnn_path)):
        raise FileNotFoundError(
            "victim_linear.pth / victim_cnn.pth not found in models/. "
            "Run `python victim/train_victims.py` first."
        )
    linear = LinearSoftmax(n_classes=N_CLASSES)
    linear.load_state_dict(torch.load(linear_path, map_location="cpu"))
    linear.eval()

    cnn = SmallCNN(n_classes=N_CLASSES)
    cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
    cnn.eval()
    return {"linear": linear, "cnn": cnn}


def _parse_images(payload):
    if "images" in payload:
        imgs = payload["images"]
    elif "image" in payload:
        imgs = [payload["image"]]
    else:
        raise KeyError("request JSON must contain 'images' (batch) or 'image' (single)")
    x = torch.tensor(imgs, dtype=torch.float32)      # (N, 28, 28)
    if x.dim() == 2:                                  # a lone (28,28)
        x = x.unsqueeze(0)
    x = x.view(x.size(0), 1, 28, 28)                  # (N, 1, 28, 28)
    return x


def create_app(default_defense_spec=None, allow_request_override=False):
    app = Flask(__name__)
    models = _load_victims()
    default_spec = default_defense_spec or {"mode": "none"}

    @app.route("/predict/<model_name>", methods=["POST"])
    def predict(model_name):
        if model_name not in models:
            return jsonify({"error": f"unknown model '{model_name}' "
                                     f"(use 'linear' or 'cnn')"}), 404
        payload = request.get_json(force=True)
        try:
            x = _parse_images(payload)
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400

        probs = models[model_name].predict_proba(x)          # (N, 10) tensor


        spec = default_spec
        if allow_request_override and isinstance(payload.get("defense"), dict):
            spec = payload["defense"]
        probs = apply_defense(probs, spec, X=x)

        return jsonify({
            "probabilities": probs.tolist(),
            "labels": probs.argmax(dim=1).tolist(),
            "count": int(x.size(0)),
            "defense": spec_label(spec),
        })

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "victim API running",
            "default_defense": spec_label(default_spec),
            "allow_request_override": allow_request_override,
            "models": list(models.keys()),
        })

    return app
