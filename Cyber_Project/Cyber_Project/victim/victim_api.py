import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from victim.server_core import create_app

PORT = 5000

if __name__ == "__main__":
    app = create_app(default_defense_spec={"mode": "none"},
                     allow_request_override=False)
    print(f"[victim_api] UNDEFENDED victim API on http://localhost:{PORT}")
    print("  POST /predict/linear   POST /predict/cnn   GET /health")
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)