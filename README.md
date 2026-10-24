# slm-channel C0 kit

HF Spaces hub (control plane) + Modal T4 worker (muscle, cold-starts only when work exists) + CPU fallback, all $0.

Includes a **sharding layer** from day one: `SHARD_ID`, `/healthz`, token-batch SSE fan-out, and a tiny router page.

## File tree

```
slm-channel-c0/
├── hub/                    → upload to an HF Space (Docker SDK)
│   ├── hub.py              ← control plane: links, queue, SSE, SQLite, sharding
│   ├── page.html           ← channel viewer (handles batched tokens)
│   ├── router.html         ← shard-aware landing / ask form
│   └── Dockerfile
└── worker/
    ├── agents.py           ← shell agents
    └── worker_modal.py     ← deploy with `modal deploy`
```

**Worker protocol (HTTP — works through any free host):**  
`POST /worker/claim` · `POST /worker/event` · `POST /worker/done` · `GET /worker/pending`

**Ops / health:** `GET /healthz` → `{ ok, shard_id, num_shards, pending, leased, channels, … }`

## Sharding

| Env var | Default | Purpose |
|---------|---------|--------|
| `SHARD_ID` | `0` | Identity of this hub instance |
| `NUM_SHARDS` | `1` | Total shards (informational / router) |
| `SSE_BATCH_SIZE` | `4` | Tokens coalesced per SSE frame |
| `SSE_BATCH_MS` | `40` | Max wait before flushing a partial batch |

- **Single Space:** leave defaults. Router at `/` posts to itself.
- **Multi-Space:** deploy N Spaces with `SHARD_ID=0…N-1`. Edit `SHARDS = […]` in `router.html` (or host the router on one of them / a static page). The router probes `/healthz` and sends the job to the least-loaded healthy shard, then redirects to that shard’s `/c/{cid}`.

Token batching reduces SSE message rate under load; `page.html` accepts both single events and `{ "type": "batch", "events": […] }`.

## Deploy — 4 commands

```bash
# 1. Hub: create HF Space (Docker SDK), upload hub/ files →
#      https://YOURNAME-slm-hub.hf.space
#    Optional Space secrets / variables:
#      SHARD_ID=0  NUM_SHARDS=1
# 2. Worker: pip install modal && modal setup
modal secret create hub-url HUB_URL=https://YOURNAME-slm-hub.hf.space
modal deploy worker/worker_modal.py
# 3. Test:
curl -s https://YOURNAME-slm-hub.hf.space/healthz
curl -X POST https://YOURNAME-slm-hub.hf.space/api/ask \
  -H "content-type: application/json" \
  -d '{"prompt":"explain KV cache in one line"}'
#    → open the /c/{cid} link (or use the router at /)
# 4. Optional Kaggle burst: same claim/report loop over HTTPS for 30h/week extra.
```

## Persistence trick ($0, HF-native)

Free Spaces wipe on rebuild → add a short hourly snapshot in `hub.py`: `huggingface_hub.HfApi().upload_file(...)` of `channels.db` to a **private dataset repo**, and pull it on boot. Links survive rebuilds — still $0.

## What you get

| | |
|---|---|
| Live shareable `/c/{cid}` links with NLL heat-map | ✅ |
| GPU bills **only while generating** (tick → wake → drain → idle-out) | ✅ |
| `/healthz` + `SHARD_ID` for horizontal scale | ✅ |
| Token-batch SSE fan-out | ✅ |
| Router page (least-loaded / multi-Space ready) | ✅ |
| Capacity v1 (sequential) | ~350–600 DAU |
| Batch-4 claim upgrade | → ~600–1,200 DAU |

Next upgrades in order: **batch-4 claim** → **CPU fallback worker** → **per-user fair queue + quotas**.
