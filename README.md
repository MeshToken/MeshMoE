# MeshMoE Node Client

**Your computer. The world's AI. Shared.**

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

# 3. Run — the client auto-detects your hardware, picks a model,
#    downloads it, registers with the public router, and starts working
python edge_node.py
```

### Windows

```cmd
git clone https://github.com/OpenMeshMoE/MeshMoE.git
cd MeshMoE
pip install -r requirements.txt
python edge_node.py
```

No configuration needed. The client talks to the public router at `https://meshmoe.com/node` — NAT-friendly (outbound polling only, no port forwarding, no public IP required).

---

## Choose your expert

The model you pick decides your tier. **The router only dispatches tasks at or below your tier** — a light node will never be asked to fake a heavy model (anti-impersonation, tier-matched dispatch).

| Tier | Model | VRAM/RAM | Good for |
|------|-------|----------|----------|
| light | `Qwen3-0.6B` (CPU ok) | 2 GB RAM | smoke test, weakest machines |
| light | `Qwen2.5-1.5B` | 4 GB RAM | general chat, fast |
| standard | `Qwen2.5-7B` / `R1-Distill-7B` | 6 GB+ VRAM | the workhorse |
| heavy | `Qwen2.5-14B` / `R1-Distill-14B` | 16 GB+ VRAM | deep reasoning |

Set explicitly: `MESHMOE_MODEL=DeepSeek-R1-Distill-Qwen-14B-Q4_K_M python edge_node.py`

**Dev/test without a GPU or model download** (honest self-declared mock — exempt from anti-impersonation probes, clearly labeled):

```bash
MOCK_MODE=1 python edge_node.py
```

---

## How earnings work

```
user pays N credits (edge = 70% of cloud price)
  ├─ 70% → you (the node owner), scaled by your reputation
  └─ 30% → meshmoe (network ops, router, infra)
```

- **1 credit = 1¢** — credits are spent on AI usage across the network.
- **Credits are not withdrawable for cash.** Running a node = free AI for yourself + helping the network + reputation. Consumer GPUs can't undercut cloud cost — this is mutual aid, not a job.
- **No idle payout** — only real edge-served requests earn.
- **Reputation matters** (50 → 100): fingerprint probes and random shadow checks keep nodes honest. High reputation → better dispatch priority and full 70% share. Failing nodes drop toward the quarantine line (<20 = no dispatch).

Track your node at `https://meshmoe.com/app/nodes` and earnings at `/app/earnings`.

---

## Anti-impersonation (how the network stays honest)

| Layer | Mechanism |
|-------|-----------|
| L1 | **Tier-matched dispatch** — nodes only receive tasks at or below their hardware tier |
| L2 | **Fingerprint probes** — router periodically sends canary questions with known answers; failures cost reputation |
| L3 | **Shadow checks** — a sample of edge answers is recomputed in the cloud and compared; big divergence costs reputation |
| L4 | **Reputation-linked rewards** — share scales 0.4–0.7 with reputation; <20 = quarantine |

---

## Configuration

All config via environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `MESHMOE_ROUTER_URL` | `https://meshmoe.com/node` | Router endpoint (public, NAT-friendly) |
| `MESHMOE_NODE_ID` | hostname | Your node's unique ID |
| `MESHMOE_MODEL` | auto-detect | Model name from the built-in catalog |
| `MESHMOE_PORT` | `4002` | Local health server port |
| `MOCK_MODE` | `0` | `1` = self-declared mock (dev/test, no real model) |

Streaming and non-streaming inference are both supported (llama.cpp via `llama-cpp-python`).

---

## Architecture (for contributors)

```
edge_node.py
  ├─ detect_hardware()      → CPU/RAM/GPU, pick tier
  ├─ download_model()       → ModelScope / HF-Mirror / HuggingFace
  ├─ EdgeNode.load_model()  → llama.cpp via llama-cpp-python
  ├─ register_with_router() → POST /node/register (honest mock flag if MOCK_MODE)
  ├─ worker_loop()          → poll /node/task/poll → infer
  │                            ├─ non-stream → POST /node/task/result
  │                            └─ stream     → POST /node/task/chunk (batched deltas)
  └─ HealthHandler          → 127.0.0.1:PORT/health
```

The router (server-side, closed) schedules by reputation (40%) · load (30%) · latency (20%) · uptime (10%).

---

## Roadmap

- [x] Single-node inference + register + poll loop
- [x] Streaming inference (chunk-batched, NAT-friendly)
- [x] Auto hardware detection + tier assignment
- [x] Anti-impersonation L1–L4 (tier dispatch, probes, shadow checks, reputation)
- [ ] Cross-platform `.exe` / `.app` builds (PyInstaller)
- [ ] One-click fine-tuning wizard (Personal Expert Models — bring your private data)
- [ ] Multi-expert collaboration (multiple models per node)

See the [full vision](https://meshmoe.com/#vision).

---

## Related

- **[meshmoe.com](https://meshmoe.com)** — main site, narrative
- **[Status](https://meshmoe.com/status/)** — live network state
- **[Dashboard](https://meshmoe.com/app/)** — your account, keys, nodes, earnings
- **[Models & pricing](https://meshmoe.com/models/)** — 1 credit = 1¢, edge = 70% of cloud

The router and gateway (server-side) are closed-source — they handle scheduling, billing, anti-abuse. The node client is fully open.

---

## License

MIT — see [LICENSE](LICENSE).

The MeshMoE name and logo are trademarks of OpenMeshMoE. The MIT license covers code, not brand.
