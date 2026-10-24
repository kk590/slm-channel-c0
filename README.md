# slm-channel C0 kit

HF Spaces hub + Modal T4 worker + CPU / Kaggle fallbacks. **$0**, clone-ready, efficiency levers stacked.

## Efficiency levers (in this repo)

| Lever | Where | Gain |
|-------|--------|------|
| **200ms SSE token batching** | `hub.py` `SSE_BATCH_MS=200` | 5–10× viewer capacity per hub |
| **Prompt dedupe** | `hub.py` `DEDUPE_WINDOW=3600` | 30–50% GPU load deleted (replay) |
| **INT4 quantization** | `worker_modal.py` bitsandbytes 4-bit | ~2–4× more batch headroom |
| **Batch-4 + 15s collect** | claim `capacity:4` + `collect_window_ms:15000` | 7 → 20–28 gens/min |
| **max_new 192** | `MAX_NEW` | ~2× more asks/hour |
| **Quotas 3/day + 1 in-flight** | `QUOTA_PER_DAY` | fair-share, no GPU hog |
| **Peak-hour Modal tick** | `PEAK_START`/`PEAK_END` UTC | credits last the month |
| **Link-as-receipt** | queued `#{pos} · ETA ~{s}s` | delay becomes UX |
| **Replay-first SSE** | history free, zero GPU | unlimited passive viewers |

Stacked: the difference between ~15k and ~50k+ served generations/day on the same free inventory.

## File tree

```
slm-channel-c0/
├── hub/
│   ├── hub.py           # control plane: dedupe, quotas, batch SSE, sharding, ETA
│   ├── page.html        # NLL heat-map + receipt status
│   ├── router.html      # least-loaded multi-shard ask form
│   └── Dockerfile
└── worker/
    ├── agents.py
    ├── worker_modal.py  # primary T4, peak-hour, batch-4, INT4
    ├── fallback_cpu.py  # always-on Qwen-0.5B (Space / free VPS)
    └── kaggle_worker.py # evening T4×2 burst, 30h/week
```

**Worker protocol:** `POST /worker/claim` · `POST /worker/event` · `POST /worker/done` · `GET /worker/pending`  
**Health:** `GET /healthz`

## Env knobs

| Var | Default | Purpose |
|-----|---------|--------|
| `SHARD_ID` / `NUM_SHARDS` | `0` / `1` | horizontal hub scale |
| `SSE_BATCH_SIZE` / `SSE_BATCH_MS` | `8` / `200` | token coalescing |
| `DEDUPE_WINDOW` | `3600` | prompt replay window (s) |
| `MAX_NEW` | `192` | hard token budget |
| `QUOTA_PER_DAY` | `3` | asks per user per day |
| `SECS_PER_GEN` | `12` | ETA heuristic |
| `PEAK_START` / `PEAK_END` | `13` / `23` | Modal GPU wake window (UTC) |
| `CLAIM_CAPACITY` | `4` | jobs per claim |
| `COLLECT_WINDOW_MS` | `15000` | wait to fill a batch |

## Deploy order (ROI)

```bash
# 1) Hub (HF Space, Docker SDK) — upload hub/
#    https://YOURNAME-slm-hub.hf.space

# 2) Modal primary worker (peak hours only by default)
pip install modal && modal setup
modal secret create hub-url HUB_URL=https://YOURNAME-slm-hub.hf.space
modal deploy worker/worker_modal.py

# 3) Optional: CPU fallback on the Space or a free always-on host
HUB_URL=https://YOURNAME-slm-hub.hf.space python worker/fallback_cpu.py

# 4) Optional: Kaggle evening burst — paste kaggle_worker.py, set HUB_URL, run

# 5) Test
curl -s https://YOURNAME-slm-hub.hf.space/healthz
curl -X POST https://YOURNAME-slm-hub.hf.space/api/ask \
  -H 'content-type: application/json' \
  -d '{"prompt":"explain KV cache in one line"}'
# → open /c/{cid}  (or use router at /)
```

**Shard #2 only when hub-1 actually hits ~1k concurrent viewers.** 3–4 free Spaces + least-loaded router → ~3–8k concurrent viewers; replay is free forever.

## Free GPU inventory

| Source | Free tier | Role |
|--------|-----------|------|
| Modal | ~$30/mo ≈ 45–60 GPU-hrs | primary — peak hours so credits last |
| Kaggle | T4×2, 30h/week | evening burst |
| HF Spaces CPU | always-on 2 vCPU | Qwen-0.5B fallback |
| Lightning / Saturn | ~10–15 hrs/mo | extra burst |

## $0 ceiling (honest)

```
Viewers:  3–4 shards × 1–2k     →  ~3–8k concurrent
Asks:     Modal + Kaggle + CPU  →  ~3–6k gens/day base
          + dedupe + INT4 + quotas → up to ~8–10k/day
Result:   ~1,500–2,500 DAU / ~8–12k registered   ← hard $0 asymptote
```

Beyond that: asks need real GPU-hours. Ladder is **$0 → ~2k DAU → C1 at $25–30/mo → ~2k more**.

## Persistence ($0, HF-native)

Spaces wipe on rebuild → snapshot `channels.db` hourly to a private dataset via `huggingface_hub`, restore on boot. Links survive — still free.
