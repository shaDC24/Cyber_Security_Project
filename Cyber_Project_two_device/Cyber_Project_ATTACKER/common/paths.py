"""
common/paths.py
----------------
Central place for the project's directory layout so every script works no
matter which folder you launch it from. ROOT is the project root (the folder
that contains common/, victim/, attacker/, models/, data/, results/).
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
