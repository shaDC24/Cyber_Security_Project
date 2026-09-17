# Project 33 — Model Extraction (Stealing a Classifier)

### Real Flask HTTP prediction API + attacks + API-layer defenses

**Shatabdi Dutta Chowdhury (2105123) · Tasnim Zaman Tanmi (2105147)** — CSE 406

Implements and evaluates **model-extraction attacks** against a black-box ML
classifier served over a **real HTTP prediction API** (trained on real MNIST
digits), plus **API-layer defenses** and their utility/security trade-off, based
on the three assigned papers:

- Tramèr, Zhang, Juels, Reiter, Ristenpart. *Stealing Machine Learning Models
  via Prediction APIs.* USENIX Security 2016. → **equation-solving attack**
- Orekondy, Schiele, Fritz. *Knockoff Nets: Stealing Functionality of
  Black-Box Models.* CVPR 2019. → **learning-based (knockoff) attack**
- Jagielski, Carlini, Berthelot, Kurakin, Papernot. *High Accuracy and High
  Fidelity Extraction of Neural Networks.* USENIX Security 2020. →
  **accuracy-vs-fidelity evaluation**

> **Assignment rule (tools 26–38):** all attack/defense logic is hand-written.
> PyTorch / Flask / NumPy are used only as the *environment* — **no ready-made
> attack library** (Foolbox / ART / TextAttack) is used. Models, datasets and
> seeds are pinned so the demo is reproducible.

**What makes this build distinct:** the victim is a genuine **Flask HTTP
service**, the attacker is a **separate process** that can only see the API's
JSON output, and every **defense is applied server-side, at the API layer**
(exactly where a real cloud ML API would put it). The attacker package **never
imports `victim/`** — the only channel between the two sides is HTTP, which is
the black-box threat model of all three papers.

---

## 1. Folder layout

```
Cyber_Project/
├── common/                        # shared, non-secret utilities
│   ├── data_mnist.py              #   MNIST loader: offline local IDX parser first,
│   │                              #   then torchvision, then GitHub mirror;
│   │                              #   + Fashion-MNIST pool loader (open-world)
│   ├── train_utils.py             #   plain PyTorch train / eval loop
│   └── paths.py                   #   project directory layout
├── victim/                        # the VICTIM side (runs the models + API)
│   ├── model_def.py               #   LinearSoftmax + SmallCNN (secret arch)
│   ├── train_victims.py           #   trains both victims -> models/*.pth
│   ├── defenses.py                #   round / top-k / labels-only / flip (API layer)
│   ├── server_core.py             #   shared Flask app factory
│   ├── victim_api.py              #   UNDEFENDED API              (port 5000)
│   └── victim_api_defended.py     #   DEFENDED   API              (port 5001)
├── attacker/                      # the ATTACKER side (black-box only)
│   ├── remote_oracle.py           #   HTTP client = the black box it sees
│   ├── substitute_def.py          #   attacker's own SubstituteCNN + SubstituteMLP
│   ├── attack_equation_solving.py #   Tramèr et al. attack
│   ├── attack_knockoff.py         #   Orekondy / Jagielski attack (closed/open/open_real)
│   ├── run_experiment.py          #   full sweep -> results.json + 3 plots
│   ├── defense_analysis.py        #   BONUS: why defenses (don't) work + flip trade-off
│   ├── arch_transfer_demo.py      #   robustness: CNN vs MLP substitute
│   ├── open_world_demo.py         #   robustness: closed vs noise vs Fashion-MNIST
│   ├── flip_determinism_demo.py   #   proof: flip can't be averaged away by re-querying
│   └── demo_query.py              #   one query in/out (for packet screenshots)
├── models/                        # victim_linear.pth, victim_cnn.pth, meta
├── data/                          # real MNIST (shipped); Fashion-MNIST cached on first use
├── results/                       # results.json + PNG plots (generated)
├── run_all.py                     # start servers + run everything, then stop servers
├── FINAL_REPORT.md                # final report (results + analysis)
├── DEMO_GUIDE.md                  # step-by-step demo / screenshot runbook
└── requirements.txt
```

---

## 2. Setup

```bash
pip install -r requirements.txt          # torch, numpy, matplotlib, flask, requests
```

Real MNIST is **shipped** in `data/MNIST/raw` and is read **offline** by a
hand-written IDX parser — no download is needed to start. (Fashion-MNIST, used
only by the open-world demo, is fetched once from the official mirror and cached
under `data/FashionMNIST_mirror_raw/`.) Pretrained victims are shipped in
`models/`, so you can skip training. To retrain from scratch:

```bash
python victim/train_victims.py           # -> models/victim_linear.pth, victim_cnn.pth
```

---

## 3. How to run

### Option A — one command

```bash
python run_all.py
```

Starts both API servers, runs both attacks (with/without defenses), writes
`results/results.json` + plots, runs the bonus defense analysis and the three
robustness/defense demos, then stops the servers.

### Option B — manual, 3 terminals (best for the live demo / screenshots)

```bash
# terminal 1 — undefended production API
python victim/victim_api.py                    # http://localhost:5000

# terminal 2 — defended production API (pick a fixed policy)
python victim/victim_api_defended.py none      # or: round 2 | topk 3 | label_only | flip 0.3

# terminal 3 — the attacker
python attacker/demo_query.py                  # one query/response (packet view)
python attacker/run_experiment.py              # full experiment -> results/ + 3 plots
python attacker/defense_analysis.py            # BONUS: Table 1, Table 2 + flip plot
python attacker/arch_transfer_demo.py          # robustness: CNN vs MLP substitute
python attacker/open_world_demo.py             # robustness: closed vs noise vs Fashion
python attacker/flip_determinism_demo.py       # proof: flip is deterministic
```

Full runbook with the screenshots to capture: see **`DEMO_GUIDE.md`**.

### API contract ("packet / frame" details)

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

## 4. What the code does (attacks & defenses)

**Attack 1 — Equation solving (Tramèr, `attack_equation_solving.py`).**
Targets the log-linear `LinearSoftmax` victim. Sends `d+1 = 785` random images
in one non-adaptive batch, takes the log-odds of the returned probabilities to
get a linear system, and solves it with `numpy.linalg.lstsq` — **no training**.
Recovers the weights `(W, b)` to floating-point precision.

**Attack 2 — Knockoff (Orekondy / Jagielski, `attack_knockoff.py`).**
Targets `SmallCNN`. Queries a transfer set, keeps only the top-1 pseudo-label,
and trains a fresh substitute from scratch. Three transfer distributions:
`closed` (real MNIST, disjoint split), `open` (uniform noise), `open_real`
(Fashion-MNIST — real but out-of-distribution). Substitute architecture is
selectable: `cnn` (matches victim shape) or `mlp` (no convolutions).

**Evaluation (Jagielski framing).** Every result reports both **accuracy** (vs
ground truth) and **fidelity** (agreement with the victim).

**Defenses (server-side, `victim/defenses.py`).** `round`, `topk`, `label_only`
(confidence-hiding — preserve the top-1 label) and `flip` (label perturbation —
returns a wrong label with probability ε, made **deterministic per input** via a
keyed BLAKE2b hash of the image so it can't be averaged away by re-querying).

---

## 5. Expected results (from the shipped pretrained victims)

Victim clean accuracy on real MNIST: `victim_linear` ≈ **90.1 %**,
`victim_cnn` ≈ **96.5 %**. (Numbers vary by a few tenths of a percent across
machines due to CPU/library nondeterminism; the qualitative results are stable.)

**Attack 1 — Equation solving (no defense):** perfect **≈100 % fidelity** at the
theoretical minimum of `d+1 = 785` non-adaptive queries — a log-linear victim is
recovered exactly by linear algebra, no training. (At exactly 785 queries
fidelity ≈ 0.9995; over-determined batches reach 1.0000 — the exactly-determined
system is slightly more sensitive, as predicted in the design report.)

**Attack 2 — Knockoff (closed-world, no defense):** accuracy climbs with budget
(~19 % @100 → ~94 % @4000, i.e. ~0.97× the victim). **open-world (noise)** stays
at chance (~10.75 %). **open-world (Fashion-MNIST)** rises to ~50 % @4000 — a
real but out-of-distribution query set beats the noise floor, showing
query-distribution realism (not just query count) drives extraction.

**Robustness — architecture independence:** an **MLP** substitute (no
convolutions) still recovers ~91 % against the CNN victim → the attack does not
depend on matching the victim's architecture.

**Bonus — defenses (server-side):**

- *Confidence-hiding* (`round` / `topk` / `label_only`): equation-solving
  collapses (fidelity 1.0 → ~0.10–0.35) because it needs exact confidences;
  knockoff is essentially unaffected (still ~90 %+) because it only uses the
  top-1 label, which these defenses preserve — `defense_analysis.py` **Table 1**:
  **0 % label change** on the CNN.
- *Label-perturbation* (`flip`, ε): the only defense that hurts knockoff. It
  drops knockoff **fidelity** sharply (~94 % → ~62 % at ε=0.3) while its
  **accuracy** barely moves (~92 % → ~87 %), because a CNN tolerates minority
  label noise — so `flip` defeats a *fidelity-motivated* attacker far more than a
  *theft/accuracy-motivated* one (Jagielski's accuracy-vs-fidelity split). It
  also costs honest-user accuracy ~ε (96.5 % → 67.5 % at ε=0.3). **Table 2** and
  `plot_flip_tradeoff.png` quantify this utility/security trade-off.
- *flip determinism* (`flip_determinism_demo.py`): re-querying the same image
  15× changes **0/500** labels; majority vote gives **no gain** — while a naive
  per-query-random flip is trivially averaged away (→100 %). This is why the
  deterministic keying matters.

---

## 6. Generated artifacts (in `results/`)

```
results.json                     full reproducible sweep log (both attacks, all defenses)
arch_transfer.json               CNN vs MLP substitute accuracies
open_world.json                  closed / noise / Fashion-MNIST accuracies
plot_equation_solving.png        eq-solving vs defenses (LinearSoftmax victim)
plot_knockoff_closed.png         knockoff vs defenses (closed-world, CNN victim)
plot_knockoff_closed_vs_open.png closed vs open (noise)
plot_open_world.png              closed vs noise vs Fashion-MNIST
plot_arch_transfer.png           architecture-independent stealing (CNN vs MLP)
plot_flip_tradeoff.png           flip: honest acc vs knockoff acc vs knockoff fidelity vs ε
```

---

## 7. Mapping to the report sections

**Design report**

- *Topology:* `Attacker process ──HTTP POST image──> Victim Flask API ──JSON prediction──> Attacker`. Attacker never sees the weights.
- *Timing/strategy:* equation-solving = one non-adaptive batch of 785 queries then one offline linear solve; knockoff = `B` queries to build a transfer set then one offline training run per budget.
- *Packet/frame:* the API contract in §3 — show the request/response JSON from `demo_query.py`, including each defense variant.
- *Justification:* the `d+1`-query linear-algebra argument + the budget-sweep curves.

**Final report** (see `FINAL_REPORT.md` for the full write-up)

- *Steps + snapshots:* run §3 Option B; capture the two server terminals + the attacker terminal + the plots (see `DEMO_GUIDE.md`).
- *Success & why:* §5 numbers — equation-solving is exact because the victim truly is log-linear; knockoff improves with budget and is bounded by query-distribution realism (closed vs open vs open_real).
- *Observed output:* `results/results.json` is the full reproducible log; server terminals show the victim side.
- *Countermeasure (bonus):* the API-layer defenses in `victim/defenses.py` + `defense_analysis.py`'s quantitative "why" + the flip-determinism proof.
```
