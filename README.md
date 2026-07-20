# MeshMoE

**The open compute network.** Home computers form a global AI grid. Edge-routed inference at **70% of cloud price** — every open-weight model, one API.

> Definitely not Skynet. But just in case — [donors get amnesty](https://meshmoe.com/skynet/).

## What is this?

MeshMoE routes your API calls to **community edge nodes** (home GPUs) when possible, falling back to cloud when not. Same models, same quality, 30% cheaper on edge. Node operators earn credits for serving requests.

- **101 open-weight models** — DeepSeek, Qwen, GLM (incl. GLM-5.2), Kimi, MiniMax, Ernie, MiMo, Hy… (no closed models — those go to our sister gateway [meshtok.com](https://meshtok.com))
- **100 credits = $1** (1 credit = 1¢). Cloud = official price. **Edge = 70% of cloud**
- **OpenAI-compatible API**: `https://api.meshmoe.com/v1`
- Models & live pricing: https://meshmoe.com/models/

## Quick start (user)

```bash
pip install openai
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.meshmoe.com/v1", api_key="moe-...")  # get a key: meshmoe.com/app/keys
r = client.chat.completions.create(model="moe-lite", messages=[{"role": "user", "content": "hi"}])
print(r.choices[0].message.content)
```

Every response tells you where it ran (`meshmoe_source.edge_hit`, `origin`, `node_id`) and what it cost (`_credits_consumed`). **No silent substitution — if the edge can't serve your model, you get the real cloud model at cloud price, never a smaller model pretending.**

## Run a node (earn credits)

```bash
curl -sSL https://raw.githubusercontent.com/OpenMeshMoE/MeshMoE/main/install.sh | bash
```

Node tiers (what you can serve depends on your GPU):

| tier | example models | hardware |
|---|---|---|
| light | GLM-4-9B, Qwen3-14B | 12GB VRAM |
| standard | DeepSeek-R1-Distill-14B | 16GB |
| heavy | R1-Distill-32B, GLM-5.2-class | 24GB+ / multi-GPU |

**Anti-impersonation (L1)**: nodes only receive requests at or below their tier — a light node can never serve (and fake) a GLM-5.2 request. L2 probe checks + L3 cloud shadow-sampling + L4 reputation-weighted payout are built into the router. Credits are spendable on the network (not withdrawable for cash) — running a node means **free AI for yourself** first.

Node earnings: **70% of the edge price** the user paid, credited on real served requests only (no idle payout, no self-report).

## Repo layout

- `edge_node.py` — the node client (poll mode for NAT + local HTTP/SSE inference endpoint)
- `install.sh` — one-line installer
- Network core (gateway/router/credits) lives server-side; protocol documented in [`/docs`](https://meshmoe.com/docs/)

## Status (2026-07-20)

- ✅ Full-stack live: meshmoe.com (network), /models (pricing), /app (dashboard), /skynet (the Pardon List)
- ✅ USDT top-up + donations, fully automatic on-chain
- ✅ Edge streaming (SSE) with 70% edge pricing end-to-end
- ✅ MeshMoE CLI + Desktop (Tauri) — see releases
- 🟡 Card payments (Creem) — pending provider review
- 🔜 Real model nodes (llama.cpp) — mock nodes today, bring your GPU tomorrow

## License

MIT
