import modal, os
app = modal.App("slm-c0")
image = (modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime", add_python="3.10")
         .pip_install("transformers", "accelerate", "requests"))
HUB = os.environ["HUB_URL"]                    # set via: modal secret create hub-url HUB_URL=https://YOU-slm-hub.hf.space

with image.imports():
    import torch, time, requests
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from agents import AGENTS, Runtime

M = "Qwen/Qwen2.5-1.5B-Instruct"
_cache = {}
def get_model():
    if "m" not in _cache:
        t = AutoTokenizer.from_pretrained(M)
        m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16, device_map="cuda").eval()
        _cache["m"], _cache["t"] = m, t
    return _cache["m"], _cache["t"]

@app.function(image=image, gpu="T4", scaledown_window=180, timeout=900,
              secrets=[modal.Secret.from_name("hub-url")])
def work():
    m, tok = get_model()
    for _ in range(60):                                    # one container wake ≈ up to 12 min of jobs
        jobs = requests.post(f"{HUB}/worker/claim", json={"capacity": 1}, timeout=20).json()["jobs"]
        if not jobs:
            if requests.get(f"{HUB}/worker/pending", timeout=20).json()["pending"] == 0: return
            time.sleep(5); continue
        for job in jobs: run_one(m, tok, job)

def run_one(m, tok, job):
    rt = Runtime()
    def emit(ev): requests.post(f"{HUB}/worker/event", json={"cid": job["cid"], "events": [ev]}, timeout=20)
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": job["prompt"]}],
                                      add_generation_prompt=True, return_tensors="pt").to(m.device)
        cache = DynamicCache()
        logits = m(ids, past_key_values=cache, use_cache=True).logits[:, -1]
        for i in range(128):
            rt.step, rt.backtrack_to = i, None
            for a in AGENTS: logits = a.on_logits(logits, rt, tok)
            probs = torch.softmax(logits[0] / 0.8, -1)
            topv, topi = torch.topk(probs, 50)
            nxt = topi[torch.multinomial(topv, 1)].item()
            nll = -torch.log2(probs[nxt]).item(); rt.last_nll = nll; rt.gen_ids.append(nxt)
            emit({"i": i, "type": "tok", "text": tok.decode([nxt]), "nll": nll})
            logits = m(torch.tensor([[nxt]], device=m.device), past_key_values=cache, use_cache=True).logits[:, -1]
        emit({"i": 128, "type": "done", "text": "", "nll": 0})
        requests.post(f"{HUB}/worker/done", json={"job_id": job["job_id"], "status": "done"}, timeout=20)
    except Exception:
        requests.post(f"{HUB}/worker/done", json={"job_id": job["job_id"], "status": "error"}, timeout=20)

@app.function(image=image, schedule=modal.Period(minutes=1),
              secrets=[modal.Secret.from_name("hub-url")])
def tick():                                                # free CPU tick: wake GPU only if work exists
    if requests.get(f"{HUB}/worker/pending", timeout=20).json()["pending"] > 0:
        work.remote()
