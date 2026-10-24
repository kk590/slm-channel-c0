"""
Modal T4 worker — bills only while generating.
Peak-hour aware tick (default 13:00–23:00 UTC ≈ evenings in many regions).
Batch-4 claim + optional 15s collect window.
max_new hard cap from hub (default 192).
"""
import modal, os
from datetime import datetime, timezone

app = modal.App("slm-c0")
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime", add_python="3.10"
    ).pip_install("transformers", "accelerate", "requests", "bitsandbytes")
)

# set via: modal secret create hub-url HUB_URL=https://YOU-slm-hub.hf.space
HUB = os.environ.get("HUB_URL", "")

# peak window (UTC). Outside this, tick still checks but skips GPU wake
# so credits stretch to month-end. Override with PEAK_START / PEAK_END.
PEAK_START = int(os.environ.get("PEAK_START", "13"))  # 13:00 UTC
PEAK_END   = int(os.environ.get("PEAK_END", "23"))    # 23:00 UTC

# claim batching
CLAIM_CAPACITY = int(os.environ.get("CLAIM_CAPACITY", "4"))
COLLECT_MS     = int(os.environ.get("COLLECT_WINDOW_MS", "15000"))  # 15s
MAX_NEW_FALLBACK = int(os.environ.get("MAX_NEW", "192"))

with image.imports():
    import torch, time, requests
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.cache_utils import DynamicCache
    from agents import AGENTS, Runtime

# Qwen 1.5B — INT4 when bitsandbytes is available (~2–4× more batch headroom)
M = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
_cache = {}

def get_model():
    if "m" not in _cache:
        t = AutoTokenizer.from_pretrained(M)
        quant = None
        try:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            m = AutoModelForCausalLM.from_pretrained(
                M, quantization_config=quant, device_map="auto"
            ).eval()
        except Exception:
            m = AutoModelForCausalLM.from_pretrained(
                M, torch_dtype=torch.float16, device_map="cuda"
            ).eval()
        _cache["m"], _cache["t"] = m, t
    return _cache["m"], _cache["t"]

def _in_peak() -> bool:
    h = datetime.now(timezone.utc).hour
    if PEAK_START <= PEAK_END:
        return PEAK_START <= h < PEAK_END
    # window crosses midnight
    return h >= PEAK_START or h < PEAK_END


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=180,
    timeout=900,
    secrets=[modal.Secret.from_name("hub-url")],
)
def work():
    m, tok = get_model()
    hub = os.environ["HUB_URL"]
    for _ in range(60):  # one container wake ≈ up to ~12 min of jobs
        try:
            jobs = requests.post(
                f"{hub}/worker/claim",
                json={"capacity": CLAIM_CAPACITY, "collect_window_ms": COLLECT_MS},
                timeout=30,
            ).json()["jobs"]
        except Exception:
            time.sleep(5)
            continue
        if not jobs:
            try:
                if requests.get(f"{hub}/worker/pending", timeout=20).json()["pending"] == 0:
                    return
            except Exception:
                pass
            time.sleep(5)
            continue
        for job in jobs:
            run_one(m, tok, job, hub)


def run_one(m, tok, job, hub):
    rt = Runtime()
    max_new = int(job.get("max_new") or MAX_NEW_FALLBACK)
    # small local buffer → fewer HTTP round-trips (pairs with hub 200ms SSE batch)
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        try:
            requests.post(
                f"{hub}/worker/event",
                json={"cid": job["cid"], "events": buf},
                timeout=20,
            )
        except Exception:
            pass
        buf = []

    def emit(ev):
        buf.append(ev)
        if len(buf) >= 8 or ev.get("type") != "tok":
            flush()

    try:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(m.device)
        cache = DynamicCache()
        logits = m(ids, past_key_values=cache, use_cache=True).logits[:, -1]
        for i in range(max_new):
            rt.step, rt.backtrack_to = i, None
            for a in AGENTS:
                logits = a.on_logits(logits, rt, tok)
            probs = torch.softmax(logits[0] / 0.8, -1)
            topv, topi = torch.topk(probs, 50)
            nxt = topi[torch.multinomial(topv, 1)].item()
            nll = -torch.log2(probs[nxt].clamp_min(1e-12)).item()
            rt.last_nll = nll
            rt.gen_ids.append(nxt)
            emit({"i": i, "type": "tok", "text": tok.decode([nxt]), "nll": nll})
            # stop on EOS
            if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                break
            logits = m(
                torch.tensor([[nxt]], device=m.device),
                past_key_values=cache,
                use_cache=True,
            ).logits[:, -1]
        emit({"i": i + 1 if 'i' in dir() else max_new, "type": "done", "text": "", "nll": 0})
        flush()
        requests.post(
            f"{hub}/worker/done",
            json={"job_id": job["job_id"], "cid": job["cid"], "status": "done"},
            timeout=20,
        )
    except Exception:
        flush()
        try:
            requests.post(
                f"{hub}/worker/done",
                json={"job_id": job["job_id"], "cid": job["cid"], "status": "error"},
                timeout=20,
            )
        except Exception:
            pass


@app.function(
    image=image,
    schedule=modal.Period(minutes=1),
    secrets=[modal.Secret.from_name("hub-url")],
)
def tick():
    """
    Free CPU tick. Wakes the T4 only when:
      1) there is pending work, AND
      2) we are inside the peak-hour window (credits last the month).
    Set PEAK_START=0 PEAK_END=24 to always-on.
    """
    import requests
    hub = os.environ["HUB_URL"]
    try:
        pending = requests.get(f"{hub}/worker/pending", timeout=20).json()["pending"]
    except Exception:
        return
    if pending <= 0:
        return
    if not _in_peak():
        # off-peak: leave work queued for CPU fallback / next peak
        return
    work.remote()
