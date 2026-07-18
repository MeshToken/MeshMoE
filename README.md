# MeshMoE Node Client

**Your computer. The world's AI. Your reward.**

The open-source node client for the [MeshMoE](https://meshmoe.com) network — a distributed network of compute experts. Run it on your home PC, share your idle GPU, earn credits every time your node is routed to.

> **Audit before running.** This client is MIT-licensed and open source. Before you run it, you should verify it does what it claims and nothing more:
> - ✓ Does not mine cryptocurrency
> - ✓ Does not exfiltrate your files or personal data
> - ✓ Only runs the model you explicitly selected
> - ✓ Only responds to router-dispatched inference tasks

---

## Quick start

### Linux / macOS

```bash
# 1. Clone
git clone https://github.com/OpenMeshMoE/MeshMoE.git
cd MeshMoE

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Get a token — sign in at https://meshmoe.com/app/ and create an API key

# 4. Pick your expert model and run
python edge_node.py --model glm-4-9b
```

### Windows

```cmd
git clone https://github.com/OpenMeshMoE/MeshMoE.git
cd MeshMoE
pip install -r requirements.txt
python edge_node.py --model glm-4-9b
```

A `.exe` build (no Python install needed) is coming.

---

## Choose your expert

The model you pick decides your tier, the work you get, and your earning rate.

| Tier | Model | VRAM | Earn rate | Good for |
|------|-------|------|-----------|----------|
| light | `glm-4-9b` | 12 GB | ×1 | general chat, fast |
| **standard ⭐** | **`deepseek-r1-distill-qwen-14b`** | 16 GB | ×1.5 | code, math, reasoning — main workhorse |
| standard | `qwen3-coder-14b` | 16 GB | ×1.5 | pure code completion |
| heavy | `deepseek-r1-distill-qwen-32b` | 24 GB+ | ×3 | heavy reasoning |

**Don't have a GPU?** You can still run in MOCK_MODE for development/testing (no real inference, just verifies the routing loop):

```bash
MOCK_MODE=1 python edge_node.py
```

---

## How earnings work

When your node serves an edge request:

```
user pays N credits (50% of full price — edge discount)
  ├─ 70% → your account (the node owner)
  └─ 30% → meshmoe (network ops, router, infra)
```

- **No idle payout** — only real edge-served requests earn.
- **Withdrawable** at $10 minimum (1000 credits). Currently manual; auto PayPal/USDT coming.
- **Cloud-fallback requests** (when no edge node serves) earn nothing.

Track your earnings at `https://meshmoe.com/app/earnings`.

---

## Configuration

All config via environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `MESHMOE_ROUTER_URL` | `http://127.0.0.1:4001` | Router endpoint (use `https://meshmoe.com` for production) |
| `MESHMOE_NODE_ID` | hostname | Your node's unique ID |
| `MESHMOE_MODEL` | auto-detect | Model name from catalog |
| `MESHMOE_PORT` | `4002` | Local health server port |
| `MOCK_MODE` | `0` | Set to `1` to skip real model load (dev/test) |
| `MOCK_MODEL` | `GLM-4-9B-Q4_K_M` | Mock mode: which model to pretend to run |

---

## Architecture (for contributors)

```
edge_node.py
  ├─ detect_hardware()      → CPU/RAM/GPU, pick tier
  ├─ download_model()       → HuggingFace / ModelScope
  ├─ EdgeNode.load_model()  → llama.cpp via llama-cpp-python
  ├─ register_with_router() → POST /register to router
  ├─ worker_loop()          → poll /task/poll → infer → submit /task/result
  └─ HealthHandler          → /health endpoint
```

The router (`meshmoe.com/router/`, closed-source) decides which node serves each request based on:
- reputation (40%) · load (30%) · latency (20%) · uptime (10%)

---

## Roadmap

- [x] Single-node inference + register + poll loop
- [x] MOCK_MODE for dev/test
- [x] Auto hardware detection + tier assignment
- [ ] Cross-platform `.exe` / `.app` builds (PyInstaller)
- [ ] One-click fine-tuning wizard (Personal Expert Models — bring your private data)
- [ ] Multi-expert collaboration (multiple models per node)

See the [full vision](https://meshmoe.com/#vision).

---

## Related

- **[meshmoe.com](https://meshmoe.com)** — main site, narrative
- **[Status](https://meshmoe.com/status/)** — live network state
- **[Dashboard](https://meshmoe.com/app/)** — your account, keys, nodes, earnings

The router and gateway (server-side) are closed-source — they handle scheduling, billing, anti-abuse. The node client is fully open.

---

## License

MIT — see [LICENSE](LICENSE).

The MeshMoE name and logo are trademarks of OpenMeshMoE. The MIT license covers code, not brand.
