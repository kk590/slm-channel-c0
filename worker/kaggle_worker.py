"""
Kaggle evening burst worker (T4×2, 30h/week, 12h background runs).
Same HTTPS claim/event/done loop — no tunnel required.

In a Kaggle notebook:
  !pip install -q transformers accelerate bitsandbytes requests
  import os; os.environ["HUB_URL"] = "https://YOURNAME-slm-hub.hf.space"
  %run kaggle_worker.py
"""
import os, time, requests

HUB = os.environ["HUB_URL"]
MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
CAPACITY = int(os.environ.get("CLAIM_CAPACITY", "2"))
MAX_NEW = int(os.environ.get("MAX_NEW", "192"))

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.cache_utils import DynamicCache

    print("[kaggle] loading", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map="auto").eval()
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
    print("[kaggle] ready — dialing", HUB)

    idle = 0
    while idle < 40:  # ~exit after prolonged idle so the session can recycle
        try:
            jobs = requests.post(
                f"{HUB}/worker/claim",
                json={"capacity": CAPACITY, "collect_window_ms": 10000},
                timeout=30,
            ).json().get("jobs", [])
        except Exception as e:
            print("claim", e)
            time.sleep(10)
            idle += 1
            continue
        if not jobs:
            idle += 1
            time.sleep(8)
            continue
        idle = 0
        for job in jobs:
            run_one(m, tok, job)

def run_one(m, tok, job):
    import torch
    from transformers.cache_utils import DynamicCache
    max_new = int(job.get("max_new") or MAX_NEW)
    buf = []
    def flush():
        nonlocal buf
        if not buf:
            return
        try:
            requests.post(f"{HUB}/worker/event", json={"cid": job["cid"], "events": buf}, timeout=20)
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
            add_generation_prompt=True, return_tensors="pt",
        ).to(m.device)
        cache = DynamicCache()
        logits = m(ids, past_key_values=cache, use_cache=True).logits[:, -1]
        for i in range(max_new):
            probs = torch.softmax(logits[0] / 0.8, -1)
            topv, topi = torch.topk(probs, 50)
            nxt = topi[torch.multinomial(topv, 1)].item()
            nll = -torch.log2(probs[nxt].clamp_min(1e-12)).item()
            emit({"i": i, "type": "tok", "text": tok.decode([nxt]), "nll": nll})
            if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                break
            logits = m(torch.tensor([[nxt]], device=m.device), past_key_values=cache, use_cache=True).logits[:, -1]
        emit({"i": i + 1, "type": "done", "text": "", "nll": 0})
        flush()
        requests.post(f"{HUB}/worker/done", json={"job_id": job["job_id"], "cid": job["cid"], "status": "done"}, timeout=20)
    except Exception as e:
        print("gen", e)
        flush()
        try:
            requests.post(f"{HUB}/worker/done", json={"job_id": job["job_id"], "cid": job["cid"], "status": "error"}, timeout=20)
        except Exception:
            pass

if __name__ == "__main__":
    main()
