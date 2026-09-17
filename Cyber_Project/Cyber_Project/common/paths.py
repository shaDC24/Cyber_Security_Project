import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
