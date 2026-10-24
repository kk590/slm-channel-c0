import os, sqlite3, time, uuid, json, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

# ── sharding ──────────────────────────────────────────────────────────────
SHARD_ID   = os.environ.get("SHARD_ID", "0")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
BATCH_SIZE = int(os.environ.get("SSE_BATCH_SIZE", "4"))   # tokens per SSE frame
BATCH_MS   = float(os.environ.get("SSE_BATCH_MS", "40"))  # max wait before flush

DB = "channels.db"
db = sqlite3.connect(DB, check_same_thread=False)
db.execute("PRAGMA journal_mode=WAL")
db.executescript("""
CREATE TABLE IF NOT EXISTS channels(
    cid TEXT PRIMARY KEY, prompt TEXT, status TEXT, created REAL, shard TEXT
);
CREATE TABLE IF NOT EXISTS events(
    cid TEXT, i INT, type TEXT, text TEXT, nll REAL, ts REAL
);
CREATE TABLE IF NOT EXISTS jobs(
    job_id TEXT PRIMARY KEY, cid TEXT, prompt TEXT, status TEXT, leased REAL
);
""")
# migrate older DBs that lack the shard column
try:
    db.execute("ALTER TABLE channels ADD COLUMN shard TEXT")
    db.commit()
except Exception:
    pass
db.commit()

app = FastAPI(title=f"slm-hub shard={SHARD_ID}")
SUBS: dict[str, list[asyncio.Queue]] = {}

# ── batched fan-out ────────────────────────────────────────────────────────
# Each channel has a buffer; tokens are coalesced into {"type":"batch","events":[...]}
_buffers: dict[str, list] = {}
_flush_tasks: dict[str, asyncio.Task] = {}

def _raw_push(cid: str, ev: dict):
    for q in SUBS.get(cid, []):
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            pass

async def _flush_later(cid: str):
    await asyncio.sleep(BATCH_MS / 1000.0)
    buf = _buffers.pop(cid, [])
    _flush_tasks.pop(cid, None)
    if not buf:
        return
    if len(buf) == 1:
        _raw_push(cid, buf[0])
    else:
        _raw_push(cid, {"type": "batch", "events": buf})

def push(cid: str, ev: dict):
    """Push with token batching. Control events (queued/done/error) flush immediately."""
    if ev.get("type") != "tok":
        # flush any pending tokens first, then the control event
        buf = _buffers.pop(cid, [])
        task = _flush_tasks.pop(cid, None)
        if task and not task.done():
            task.cancel()
        if buf:
            if len(buf) == 1:
                _raw_push(cid, buf[0])
            else:
                _raw_push(cid, {"type": "batch", "events": buf})
        _raw_push(cid, ev)
        return

    buf = _buffers.setdefault(cid, [])
    buf.append(ev)
    if len(buf) >= BATCH_SIZE:
        batch = _buffers.pop(cid, [])
        task = _flush_tasks.pop(cid, None)
        if task and not task.done():
            task.cancel()
        _raw_push(cid, {"type": "batch", "events": batch} if len(batch) > 1 else batch[0])
    elif cid not in _flush_tasks:
        try:
            loop = asyncio.get_event_loop()
            _flush_tasks[cid] = loop.create_task(_flush_later(cid))
        except RuntimeError:
            # no running loop (sync context) — just push immediately
            _raw_push(cid, ev)
            _buffers.pop(cid, None)


# ── health ────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    pending = db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0]
    leased  = db.execute("SELECT COUNT(*) FROM jobs WHERE status='leased'").fetchone()[0]
    channels = db.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
    return {
        "ok": True,
        "shard_id": SHARD_ID,
        "num_shards": NUM_SHARDS,
        "pending": pending,
        "leased": leased,
        "channels": channels,
        "subs": {k: len(v) for k, v in SUBS.items()},
        "ts": time.time(),
    }


# ── public API ─────────────────────────────────────────────────────────────
@app.post("/api/ask")
async def ask(r: Request):
    body = await r.json()
    p = (body.get("prompt") or "").strip()[:2000]
    if not p:
        return JSONResponse({"error": "empty prompt"}, status_code=400)

    # optional sticky routing: client may send a preferred cid prefix
    cid = body.get("cid") or uuid.uuid4().hex[:10]
    jid = uuid.uuid4().hex[:10]

    db.execute(
        "INSERT OR REPLACE INTO channels VALUES (?,?,?,?,?)",
        (cid, p, "queued", time.time(), SHARD_ID),
    )
    db.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?)",
        (jid, cid, p, "pending", 0),
    )
    db.commit()
    push(cid, {"i": 0, "type": "queued", "text": "", "nll": 0})
    return {
        "channel": cid,
        "link": f"/c/{cid}",
        "shard": SHARD_ID,
    }


# ── worker protocol ────────────────────────────────────────────────────────
@app.get("/worker/pending")
def pending():
    return {
        "pending": db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0],
        "shard": SHARD_ID,
    }

@app.post("/worker/claim")
async def claim(r: Request):
    body = await r.json()
    cap = min(int(body.get("capacity", 1)), 4)
    rows = db.execute(
        "SELECT job_id,cid,prompt FROM jobs WHERE status='pending' ORDER BY rowid LIMIT ?",
        (cap,),
    ).fetchall()
    now = time.time()
    for j, _, _ in rows:
        db.execute("UPDATE jobs SET status='leased', leased=? WHERE job_id=?", (now, j))
    db.commit()
    return {
        "jobs": [{"job_id": a, "cid": b, "prompt": c} for a, b, c in rows],
        "shard": SHARD_ID,
    }

@app.post("/worker/event")
async def event(r: Request):
    b = await r.json()
    cid = b["cid"]
    for ev in b["events"]:
        db.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (cid, ev["i"], ev["type"], ev["text"], ev.get("nll", 0), time.time()),
        )
        push(cid, ev)
    db.commit()
    return {"ok": True, "shard": SHARD_ID}

@app.post("/worker/done")
async def done(r: Request):
    b = await r.json()
    db.execute("UPDATE jobs SET status=? WHERE job_id=?", (b["status"], b["job_id"]))
    db.execute("UPDATE channels SET status=? WHERE cid=?", (b["status"], b["cid"]))
    db.commit()
    # surface terminal status to any live viewers
    push(b["cid"], {"i": -1, "type": b.get("status", "done"), "text": "", "nll": 0})
    return {"ok": True, "shard": SHARD_ID}


# ── SSE ────────────────────────────────────────────────────────────────────
@app.get("/channel/{cid}/events")
async def sse(cid: str):
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    SUBS.setdefault(cid, []).append(q)

    rows = db.execute(
        "SELECT i,type,text,nll FROM events WHERE cid=? ORDER BY rowid", (cid,)
    ).fetchall()

    async def gen():
        try:
            # replay history (already stored one-by-one; client handles both shapes)
            for i, t, x, n in rows:
                yield f"data: {json.dumps({'i': i, 'type': t, 'text': x, 'nll': n})}\n\n"
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            # clean up subscriber on disconnect
            lst = SUBS.get(cid, [])
            if q in lst:
                lst.remove(q)
            if not lst:
                SUBS.pop(cid, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Shard-Id": SHARD_ID,
        },
    )


# ── pages ──────────────────────────────────────────────────────────────────
@app.get("/c/{cid}", response_class=HTMLResponse)
def page(cid: str):
    return open("page.html").read()

@app.get("/", response_class=HTMLResponse)
def root():
    """Tiny router / landing — works standalone or behind a multi-shard front."""
    try:
        return open("router.html").read()
    except FileNotFoundError:
        return HTMLResponse(
            f"<pre>slm-hub shard={SHARD_ID} · <a href='/healthz'>/healthz</a></pre>"
        )
