# Project 33 — Model Extraction (Stealing a Classifier)

### Real Flask HTTP prediction API + defense at the API layer

Implements and evaluates model-extraction attacks against a **black-box ML
classifier served over a real HTTP prediction API** (trained on real MNIST
digits), plus a bonus API-layer defense evaluation, based on the three
assigned papers:

- Tramèr, Zhang, Juels, Reiter, Ristenpart. *Stealing Machine Learning Models
  via Prediction APIs.* USENIX Security 2016. → **equation-solving attack**
- Orekondy, Schiele, Fritz. *Knockoff Nets: Stealing Functionality of
  Black-Box Models.* CVPR 2019. → **learning-based (knockoff) attack**
- Jagielski, Carlini, Berthelot, Kurakin, Papernot. *High Accuracy and High
  Fidelity Extraction of Neural Networks.* USENIX Security 2020. →
  **accuracy-vs-fidelity evaluation**

All attack/defense logic is hand-written. PyTorch/Flask are used only as the
*environment* — no ready-made attack library (Foolbox/ART/TextAttack) is used,
per the assignment rule for tools 26–38.

**What makes this build distinct:** the victim is a genuine **Flask HTTP
service**, the attacker is a **separate process** that can only see the API's
JSON output, and every **defense is applied server-side, at the API layer**
(exactly where a real cloud ML API would put it). The attacker code is
transport-agnostic: it talks to the victim only through `RemoteOracle`.

---

## 1. Folder layout (modular)

```
Project33-Model-Extraction/
├── common/                       # shared, non-secret utilities
│   ├── data_mnist.py             #   real MNIST loader (+ offline fallback)
│   ├── train_utils.py            #   plain PyTorch train / eval loop
│   └── paths.py                  #   project directory layout
├── victim/                       # the VICTIM side (runs the model + API)
│   ├── model_def.py              #   LinearSoftmax + SmallCNN (secret arch)
│   ├── train_victims.py          #   trains both victims -> models/*.pth
│   ├── defenses.py               #   round / top-k / labels-only (API layer)
│   ├── server_core.py            #   shared Flask app factory
│   ├── victim_api.py             #   UNDEFENDED API      (port 5000)
│   └── victim_api_defended.py    #   DEFENDED   API      (port 5001)
├── attacker/                     # the ATTACKER side (black-box only)
│   ├── substitute_def.py         #   attacker's OWN substitute CNN
│   ├── remote_oracle.py          #   HTTP client = the black box it sees
│   ├── attack_equation_solving.py#   Tramèr et al. attack
│   ├── attack_knockoff.py        #   Orekondy / Jagielski attack
│   ├── run_experiment.py         #   full sweep -> results.json + plots
│   ├── defense_analysis.py       #   bonus: why defenses (don't) work
│   └── demo_query.py             #   one query in/out (for screenshots)
├── models/                       # victim_linear.pth, victim_cnn.pth, meta
├── data/                         # real MNIST (shipped, no download needed)
├── results/                      # results.json + PNG plots (generated)
├── run_all.py                    # start servers + run everything
└── requirements.txt
```

The attacker package **never imports `victim/`** — the only channel between
them is HTTP. That is the black-box threat model of all three papers.

---

## 2. How to run

```bash
pip install -r requirements.txt
```

Pretrained victims (`models/victim_*.pth`) and real MNIST (`data/MNIST`) are
already included, so you can skip training.

### Option A — one command

```bash
python run_all.py
```

Starts both API servers, runs both attacks (with/without defenses), writes
`results/results.json` + 3 plots, runs the bonus analysis, then stops servers.

### Option B — manual (best for the live demo / screenshots)

Three terminals, all from the project root:

```bash
# terminal 1 — undefended production API
python victim/victim_api.py                    # http://localhost:5000

# terminal 2 — defended production API (pick a fixed policy)
python victim/victim_api_defended.py none      # or: round 2 | topk 3 | label_only | flip 0.3

# terminal 3 — the attacker
python attacker/demo_query.py                  # show one query/response (packet view)
python attacker/run_experiment.py              # full experiment -> results/ + plots
python attacker/defense_analysis.py            # bonus diagnostic
```

(Re-train the victims from scratch with `python victim/train_victims.py`.)

### The API contract ("packet / frame" details for the design report)

```
POST /predict/linear      POST /predict/cnn
request : {"images": [[28x28 floats in 0..1], ...]}      # batch
          {"image":  [28x28 floats in 0..1]}             # single
          optional (defended server only):
          {"defense": {"mode": "round"|"topk"|"label_only"|"flip",
                       "decimals":2, "k":3, "epsilon":0.3}}
response: {"probabilities": [[p0..p9], ...], "labels": [...],
           "count": N, "defense": "<active policy>"}
GET  /health  ->  liveness + which defense policy is active
```

---

## 3. Expected results (from the shipped pretrained victims)

Victim accuracy on real MNIST test images:
`victim_linear` ≈ **90.1%**, `victim_cnn` ≈ **96.5%**.

**Attack 1 — Equation solving (no defense):** perfect **100% fidelity** at
the theoretical minimum of `d+1 = 785` non-adaptive queries — a log-linear
victim is recovered exactly by linear algebra, no training.

**Attack 2 — Knockoff (closed-world, no defense):** accuracy climbs with
budget (~13% @100 → ~93% @4000). **Open-world** (pure-noise queries) stays at
chance (~10%), showing how much a realistic query distribution matters.

**Bonus — defenses (server-side, at the API layer):**

- *Confidence-hiding* (`round` / `topk` / `label_only`): equation-solving
  collapses (~90% → 6–33%) because it needs exact confidences; knockoff is
  essentially unaffected (still ~90%+) because it only uses the top-1 label,
  which these defenses preserve (`defense_analysis.py` Table 1: **~0% label
  change**).
- *Label-perturbation* (`flip`, epsilon): a randomized-response defense that
  returns a WRONG label with probability epsilon — deterministic per input so
  it can't be averaged away by re-querying. This is **the only defense that
  actually hurts knockoff** (label change ≈ epsilon), at a direct cost to
  honest-user accuracy (≈ epsilon). `defense_analysis.py` Table 2 +
  `plot_flip_tradeoff.png` quantify that utility/security trade-off — the open
  problem all three papers name but none implements.

---

## 4. Mapping to the report sections

**Design report**

- (a) *Topology:* `Attacker process ──HTTP POST image──> Victim Flask API ──JSON prediction──> Attacker`. Attacker never sees the weights.
- (b) *Timing/strategy:* equation-solving = one non-adaptive batch of 785 queries then one offline linear solve; knockoff = `B` queries to build a transfer set then one offline training run per budget.
- (c) *Packet/frame:* the API contract in §2 — show the request/response JSON from `demo_query.py`, including each defense variant.
- (d) *Justification:* the `d+1`-query linear-algebra argument (equation-solving docstring) + the budget-sweep curves (knockoff needs many more queries but assumes nothing about the victim architecture).

**Final report**

- (a) *Steps + snapshots:* run §2 Option B; capture the two server terminals + the attacker terminal + the 3 plots.
- (b) *Success & why:* §3 tables — equation-solving is exact because the victim truly is log-linear; knockoff improves with budget and is bounded by query-distribution realism (closed vs open world).
- (c) *Observed output at attacker/victim:* `results/results.json` is the full reproducible log; server terminals show the victim side.
- (d) *Countermeasure:* the API-layer defenses in `victim/defenses.py` + `defense_analysis.py`'s quantitative "why" (bonus 10%).

---
