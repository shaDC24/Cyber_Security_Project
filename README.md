# CSE406 — Project 33: Model Extraction (Stealing a Classifier)

Course project for **CSE406 (Computer Security), Jan 2026**. Assigned tool
**#33 — Model Extraction**: given only *black-box query access* to a trained
classifier's prediction API, reconstruct a copy of it, then design a
countermeasure. Full task statement: [CSE406ProjectJan2026-1.pdf](CSE406ProjectJan2026-1.pdf).

> **Rule followed:** we implement the core attacks and defenses ourselves. We
> use PyTorch / Flask only as the *environment* — no ready-made extraction or
> attack library (Foolbox / ART / TextAttack) is used anywhere.

---

## Assigned papers (in [Papers/](Papers/))

| Paper | File | What we implement from it |
|---|---|---|
| Tramèr et al., *Stealing ML Models via Prediction APIs* (USENIX'16) | [stealing_ML_via_production_apis.pdf](Papers/stealing_ML_via_production_apis.pdf) | **Equation-solving** attack (exact recovery of a log-linear model) + confidence-hiding countermeasures |
| Orekondy et al., *Knockoff Nets* (CVPR'19) | [knock-off-nets.pdf](Papers/knock-off-nets.pdf) | **Learning-based (knockoff)** attack; closed- vs open-world |
| Jagielski et al., *High Accuracy & High Fidelity Extraction* (USENIX'20) | [high-acc-high-fidelity.pdf](Papers/high-acc-high-fidelity.pdf) | **Accuracy-vs-fidelity** evaluation framing |

---

## Folder map

```
Cyber_security_project/
├── CSE406ProjectJan2026-1.pdf     # the assignment specification
├── Papers/                        # the 3 assigned reference papers
├── Reports/                       # <-- put the two lab reports here (see below)
├── Cyber_Project/                 # THE IMPLEMENTATION (attacks + API + defenses)
│   └── README.md                  #     full run instructions & design notes
└── README.md                      # (this file)
```

**The working code lives in [Cyber_Project/](Cyber_Project/).** It is a
modular client/server project: the victim is served over a real **Flask HTTP
prediction API**, the attacker is a separate process that only ever sees the
API's JSON output, and every defense is applied **server-side, at the API
layer**. See [Cyber_Project/README.md](Cyber_Project/README.md) for the file-by-file
breakdown, the API contract, and the report-section mapping.

---

## What is implemented

**Attacks (both over the HTTP API):**
- **Equation-solving** (Tramèr) against a `LinearSoftmax` victim — exact weight
  recovery from `d+1 = 785` non-adaptive queries via linear algebra.
- **Knockoff** (Orekondy / Jagielski) against a `SmallCNN` victim — trains a
  substitute on (query image → predicted label) pairs; swept over query
  budget, in closed-world (real images) and open-world (noise) settings.

**Defenses (server-side, at the API layer) — the bonus:**
- *Confidence-hiding*: `round`, `top-k`, `labels-only`. Stop equation-solving,
  but **not** knockoff (they preserve the top-1 label).
- *Label-perturbation* (`flip`, our addition): randomized-response that returns
  a wrong label with probability ε, made **deterministic per input** so it
  can't be averaged away by re-querying. This is **the only defense that
  actually hurts knockoff** — at a matching cost to honest-user accuracy. It
  implements the utility/security trade-off all three papers name but none
  builds.

---

## Quick start

Full details in [Cyber_Project/README.md](Cyber_Project/README.md). In short, from
inside `Cyber_Project/`:

```bash
pip install -r requirements.txt      # torch, torchvision, flask, requests, numpy, matplotlib

# one command (starts servers, runs everything, stops servers):
python run_all.py
```

Or the 3-terminal manual flow (best for demo screenshots):

```bash
python victim/victim_api.py                 # undefended API, port 5000
python victim/victim_api_defended.py none   # defended  API, port 5001
python attacker/run_experiment.py           # attacks + defenses -> results/
python attacker/defense_analysis.py         # bonus tables + flip tradeoff plot
```

Pretrained victim models and the MNIST dataset are already included, so no
retraining or download is required.

---

## Results (see [Cyber_Project/results/](Cyber_Project/results/))

- **Equation-solving:** ~**100% fidelity** at 785 queries with no defense;
  collapses to **6–33%** under every confidence-hiding defense.
- **Knockoff:** climbs to ~**95%** accuracy by budget 4000 (closed-world);
  stuck at ~10% (chance) in open-world. **Unaffected** by confidence-hiding
  defenses (~0% label change).
- **Label-flip defense:** the knockoff curve finally bends **down** as ε rises,
  while honest-user accuracy drops by ~ε — quantified in
  `plot_flip_tradeoff.png`.

Generated artifacts: `results.json`, `plot_equation_solving.png`,
`plot_knockoff_closed.png`, `plot_knockoff_closed_vs_open.png`,
`plot_flip_tradeoff.png`.

---

## Reports (deliverables → [Reports/](Reports/))

Per the spec, two reports are submitted here:
1. **Design report** (topology diagram, protocol vs attack timing diagram,
   query/response "packet" details, design justification).
2. **Final report & demo** (attack steps + snapshots, success analysis,
   observed output on attacker/victim, the countermeasure).

The section-by-section mapping to the code and figures is in
[Cyber_Project/README.md](Cyber_Project/README.md) §4.

---

