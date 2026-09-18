import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from victim.server_core import create_app

PORT = 5001


def _spec_from_argv(argv):
    mode = argv[1] if len(argv) > 1 else "round"
    if mode == "round":
        return {"mode": "round", "decimals": int(argv[2]) if len(argv) > 2 else 2}
    if mode == "topk":
        return {"mode": "topk", "k": int(argv[2]) if len(argv) > 2 else 3}
    if mode == "label_only":
        return {"mode": "label_only"}
    if mode == "flip":
        return {"mode": "flip", "epsilon": float(argv[2]) if len(argv) > 2 else 0.3}
    if mode == "none":
        return {"mode": "none"}
    raise SystemExit(f"unknown defense mode '{mode}' "
                     f"(use: none | round [d] | topk [k] | label_only | flip [eps])")


if __name__ == "__main__":
    spec = _spec_from_argv(sys.argv)
    app = create_app(default_defense_spec=spec, allow_request_override=True)
    print(f"[victim_api_defended] DEFENDED victim API on http://localhost:{PORT}")
    print(f"  fixed default policy: {spec}")
    print("  POST /predict/linear   POST /predict/cnn   GET /health")
    app.run(host=os.environ.get("HOST","0.0.0.0"), port=PORT, debug=False, use_reloader=False)
