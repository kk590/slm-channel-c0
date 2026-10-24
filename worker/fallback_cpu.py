"""
CPU fallback worker — run inside the HF Space (or any always-on 2 vCPU host).
Uses a tiny model so it never needs a GPU. Same claim/event/done protocol.
Slow but never off — carries off-peak + cold-start gaps.

Usage (in Space or a free always-on box):
  HUB_URL=http://127.0.0.1:7860 python fallback_cpu.py
"""
import os, time, requests

HUB = os.environ.get("HUB_URL", "http://127.0.0.1:7860")
MODEL = os.environ.get("CPU_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
POLL = float(os.environ.get("POLL_SECS", "8"))

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    print(f"[cpu-fallback] loading {MODEL}…")
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    print("[cpu-fallback] ready")

    while True:
        try:
            jobs = requests.post(
                f"{HUB}/worker/claim",
                json={"capacity": 1, "collect_window_ms": 0},
                timeout=20,
            ).json().get("jobs", [])
        except Exception as e:
            print("claim error", e)
            time.sleep(POLL)
            continue
        if not jobs:
            time.sleep(POLL)
            continue
        for job in jobs:
            run_one(m, tok, job)

def run_one(m, tok, job):
    max_new = int(job.get("max_new") or MAX_NEW)
    def emit(ev):
        try:
            requests.post(
                f"{HUB}/worker/event",
                json={"cid": job["cid"], "events": [ev]},
                timeout=20,
            )
        except Exception:
            pass
    try:
        from transformers.cache_utils import DynamicCache
        import torch
        ids = tok.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        cache = DynamicCache()
        logits = m(ids, past_key_values=cache, use_cache=True).logits[:, -1]
        for i in range(max_new):
            probs = torch.softmax(logits[0] / 0.8, -1)
            topv, topi = torch.topk(probs, 40)
            nxt = topi[torch.multinomial(topv, 1)].item()
            nll = -torch.log2(probs[nxt].clamp_min(1e-12)).item()
            emit({"i": i, "type": "tok", "text": tok.decode([nxt]), "nll": nll})
            if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                break
            logits = m(
                torch.tensor([[nxt]]),
                past_key_values=cache,
                use_cache=True,
            ).logits[:, -1]
        emit({"i": i + 1, "type": "done", "text": "", "nll": 0})
        requests.post(
            f"{HUB}/worker/done",
            json={"job_id": job["job_id"], "cid": job["cid"], "status": "done"},
            timeout=20,
        )
    except Exception as e:
        print("gen error", e)
        try:
            requests.post(
                f"{HUB}/worker/done",
                json={"job_id": job["job_id"], "cid": job["cid"], "status": "error"},
                timeout=20,
            )
        except Exception:
            pass

if __name__ == "__main__":
    main()
