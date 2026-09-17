# ATTACKER PC — Setup & Run

This folder runs the **attacker** side only: it steals the victim model over the
network, seeing ONLY the API's JSON output. There is **no victim code and no
victim model file here** — that is the black-box threat model. The attacker even
measures the victim's accuracy by querying the API, so it needs no victim files.

Contents: `common/`, `attacker/`, `data/` (MNIST for the attacker's own transfer
pool + test set), `requirements.txt`. No `victim/`, no `models/*.pth`.

---

## 1. One-time setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Point the attacker at the victim PC's IP

Ask the victim PC for its IPv4 (its `ipconfig`), then in THIS terminal:

```powershell
$env:VICTIM_UNDEFENDED_URL = "http://<VICTIM-IP>:5000"
$env:VICTIM_DEFENDED_URL   = "http://<VICTIM-IP>:5001"
```
Example: `$env:VICTIM_UNDEFENDED_URL = "http://192.168.43.5:5000"`
(Same machine instead of two? Skip this step — it defaults to localhost.)

## 3. Check you can reach the victim

```powershell
curl "$env:VICTIM_UNDEFENDED_URL/health" -UseBasicParsing
```
Expect `{"status":"victim API running", ...}`.
If it times out: the two PCs can't see each other — see **Troubleshooting** below.

## 4. Run the attacks (in this order)

```powershell
python attacker/demo_query.py            # 1 query in/out (packet view, for screenshots)
python attacker/run_experiment.py        # MAIN: both attacks x all defenses -> results/ + 3 plots
python attacker/defense_analysis.py      # BONUS: Table 1, Table 2 + flip trade-off plot
python attacker/arch_transfer_demo.py    # robustness: CNN vs MLP substitute
python attacker/open_world_demo.py       # robustness: closed vs noise vs Fashion-MNIST
python attacker/flip_determinism_demo.py # proof: flip can't be averaged away
```

Outputs are written to `results/` (JSON logs + PNG plots).

---

## What each script shows (expected)

| script | key result |
|---|---|
| `run_experiment.py` | equation-solving fidelity ≈ 1.0 at 785 queries; knockoff closed ≈ 0.94, open (noise) ≈ 0.10 |
| `defense_analysis.py` | Table 1: 0% label change for round/top-k/labels-only, 30.4% for flip; Table 2: fidelity trade-off |
| `arch_transfer_demo.py` | MLP substitute ≈ 0.91 (architecture-independent) |
| `open_world_demo.py` | Fashion-MNIST rises above the noise floor |
| `flip_determinism_demo.py` | 0/500 labels change on re-query; majority vote gives no gain |

---

## Troubleshooting (2-device connection)

Run these on the ATTACKER PC (replace with the victim's IP):

```powershell
ping <VICTIM-IP>
```
- **Reply received** → network OK. If HTTP still fails: the victim's firewall is
  blocking 5000/5001, or the victim server is bound to 127.0.0.1 (must be 0.0.0.0).
- **Request timed out** → the two PCs can't reach each other (very common on
  university/office Wi-Fi due to *client isolation*). Fix: connect **both PCs to a
  phone hotspot**, then re-check the victim's IP (`ipconfig` on victim) and retry.

First run downloads MNIST (and, for `open_world_demo`, Fashion-MNIST) automatically
if not already cached under `data/`.

Same-machine fallback: everything also runs on one PC (victim + attacker) over
localhost — the black-box property (separate processes, HTTP-only, no weight
access) holds there too. Just skip step 2.
