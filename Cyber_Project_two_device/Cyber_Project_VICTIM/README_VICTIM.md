# VICTIM PC — Setup & Run

This folder runs the **victim** side only: the two ML models behind a Flask HTTP
prediction API. The attacker (on the other PC) will query these over the network.

Contents: `common/`, `victim/`, `models/` (pretrained victims), `data/` (MNIST),
`requirements.txt`. There is **no attacker code here** — that is on the other PC.

---

## 1. One-time setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Pretrained victims are already in `models/`, so you can skip training. (To retrain:
`python victim/train_victims.py`.)

## 2. Find this PC's IP (give it to the attacker)

```powershell
ipconfig
```

Note the **IPv4 Address** (e.g. `192.168.43.5`). The attacker will use this.

## 3. Allow the ports through Windows Firewall (once)

Windows Firewall → Advanced settings → Inbound Rules → New Rule → Port → TCP →
`5000,5001` → Allow. (Or, on a trusted demo network only, temporarily turn the
firewall off to test.)

## 4. Start BOTH servers (two terminals, venv activated in each)

```powershell
# terminal 1 — undefended API (port 5000)
python victim/victim_api.py

# terminal 2 — defended API (port 5001), pick a fixed policy
python victim/victim_api_defended.py none      # or: round 2 | topk 3 | label_only | flip 0.3
```

You should see `Running on http://0.0.0.0:5000` (0.0.0.0 = reachable from other PCs).
Leave both running during the demo.

## 5. Verify locally

```powershell
curl http://localhost:5000/health -UseBasicParsing
```

Expect `{"status":"victim API running", ...}`.

---

### Notes

- The servers bind to `0.0.0.0` by default (all interfaces) so the attacker PC can
  reach them. To restrict to localhost only, set `HOST=127.0.0.1` before running.
- Both PCs must be on the **same network** and able to reach each other. On
  university/office Wi-Fi that often fails (client isolation) — use a **phone
  hotspot** (connect both PCs to it) if `ping` between the PCs times out.
- The victim's weights (`models/*.pth`) never leave this PC — that is the whole
  point (black-box threat model).
