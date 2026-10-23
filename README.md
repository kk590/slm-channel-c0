# slm-channel C0 kit

HF Spaces hub (control plane) + Modal T4 worker (muscle, cold-starts only when work exists) + CPU fallback, all $0, all sized to the thresholds we computed.

## File tree

```
slm-channel/
├── hub/                    → upload to an HF Space (Docker SDK)
│   ├── hub.py              ← control plane: links, queue, SSE, SQLite
│   ├── page.html           ← channel viewer
│   └── Dockerfile
└── worker/
    ├── agents.py           ← your 5 shell agents
    └── worker_modal.py     ← deploy with `modal deploy`
```

**Worker protocol (HTTP — works through any free host):**  
`POST /worker/claim` · `POST /worker/event` · `POST /worker/done` · `GET /worker/pending`

## Deploy — 4 commands

```bash
# 1. Hub: create HF Space (Docker SDK), upload hub/ files → it builds & goes live at
#      https://YOURNAME-slm-hub.hf.space
# 2. Worker: pip install modal && modal setup
modal secret create hub-url HUB_URL=https://YOURNAME-slm-hub.hf.space
modal deploy worker/worker_modal.py
# 3. Test:
curl -X POST https://YOURNAME-slm-hub.hf.space/api/ask -H "content-type: application/json" -d '{"prompt":"explain KV cache in one line"}'
#    → open the /c/{cid} link — watch the tick wake the T4 (~60–90s first time), tokens stream.
# 4. Optional Kaggle burst: a notebook running the same claim/report loop over HTTPS for 30h/week extra.
```

## Persistence trick ($0, HF-native)

Free Spaces wipe on rebuild → add a 10-line hourly snapshot in `hub.py`: `huggingface_hub.HfApi().upload_file(...)` of `channels.db` to a **private dataset repo**, and pull it on boot. Links now survive rebuilds — still $0.

## What you just got

| | |
|---|---|
| Live shareable `/c/{cid}` links with NLL heat-map | ✅ |
| GPU bills **only while generating** (tick → wake → drain → idle-out) | ✅ credits last the month |
| CPU fallback in Space when GPU cold | add `fallback.py` later (same claim loop, cpu model) |
| Capacity v1 (sequential) | ~350–600 DAU, ~2–3.5k registered |
| Batch-4 upgrade (claim `capacity:4`) | → the full ~600–1,200 DAU |

Next upgrades in order: **batch-4 claim** → **CPU fallback worker** → **per-user fair queue + quotas**. Say which one you want built next.
