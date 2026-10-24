import os, sqlite3, time, uuid, json, asyncio, hashlib
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

# ── sharding + efficiency knobs ───────────────────────────────────────────
SHARD_ID   = os.environ.get("SHARD_ID", "0")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))

# 200ms SSE token batching → 5–10× viewer capacity per hub
BATCH_SIZE = int(os.environ.get("SSE_BATCH_SIZE", "8"))
BATCH_MS   = float(os.environ.get("SSE_BATCH_MS", "200"))

# prompt dedupe window (seconds) — same prompt → replay cached stream
DEDUPE_WINDOW = int(os.environ.get("DEDUPE_WINDOW", "3600"))  # 1h

# hard token budget per ask
MAX_NEW = int(os.environ.get("MAX_NEW", "192"))

# quotas: 3 asks/day/user, 1 in-flight
QUOTA_PER_DAY = int(os.environ.get("QUOTA_PER_DAY", "3"))
SECS_PER_GEN  = float(os.environ.get("SECS_PER_GEN", "12"))  # ETA heuristic

DB = "channels.db"
db = sqlite3.connect(DB, check_same_thread=False)
db.execute("PRAGMA journal_mode=WAL")
db.executescript("""
CREATE TABLE IF NOT EXISTS channels(
    cid TEXT PRIMARY KEY, prompt TEXT, status TEXT, created REAL, shard TEXT,
    prompt_hash TEXT, user_key TEXT
);
CREATE TABLE IF NOT EXISTS events(
    cid TEXT, i INT, type TEXT, text TEXT, nll REAL, ts REAL
);
CREATE TABLE IF NOT EXISTS jobs(
    job_id TEXT PRIMARY KEY, cid TEXT, prompt TEXT, status TEXT, leased REAL
);
CREATE TABLE IF NOT EXISTS users(
    user_key TEXT PRIMARY KEY, asks_day INT, day_epoch INT, in_flight INT
);
CREATE INDEX IF NOT EXISTS idx_channels_phash ON channels(prompt_hash, status, created);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, rowid);
""")
for col, typ in [("shard", "TEXT"), ("prompt_hash", "TEXT"), ("user_key", "TEXT")]:
    try:
        db.execute(f"ALTER TABLE channels ADD COLUMN {col} {typ}")
        db.commit()
    except Exception:
        pass
db.commit()

app = FastAPI(title=f"slm-hub shard={SHARD_ID}")
SUBS: dict[str, list[asyncio.Queue]] = {}

# ── helpers ───────────────────────────────────────────────────────────────
def _phash(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:16]

def _day_epoch() -> int:
    return int(time.time()) // 86400

def _user_key(r: Request, body: dict) -> str:
    """Stable-ish identity: explicit user_id > X-User-Key > cookie > IP."""
    if body.get("user_id"):
        return "uid:" + str(body["user_id"])[:64]
    hk = r.headers.get("x-user-key") or r.headers.get("x-forwarded-user")
    if hk:
        return "hdr:" + hk[:64]
    # fallback: client IP (fine for free-tier fair-share)
    ip = r.headers.get("x-forwarded-for", "").split(",")[0].strip() or (r.client.host if r.client else "anon")
    return "ip:" + ip

def _quota_check(user_key: str) -> tuple[bool, str, dict]:
    """Returns (ok, error_msg, info). Enforces QUOTA_PER_DAY + 1 in-flight."""
    day = _day_epoch()
    row = db.execute("SELECT asks_day, day_epoch, in_flight FROM users WHERE user_key=?", (user_key,)).fetchone()
    if not row:
        db.execute("INSERT INTO users VALUES (?,?,?,?)", (user_key, 0, day, 0))
        db.commit()
        return True, "", {"asks_left": QUOTA_PER_DAY, "in_flight": 0}
    asks, d, inflight = row
    if d != day:
        asks, inflight = 0, 0
        db.execute("UPDATE users SET asks_day=0, day_epoch=?, in_flight=0 WHERE user_key=?", (day, user_key))
        db.commit()
    if inflight >= 1:
        return False, "1-in-flight limit: wait for your current ask to finish", {"asks_left": max(0, QUOTA_PER_DAY - asks), "in_flight": inflight}
    if asks >= QUOTA_PER_DAY:
        return False, f"daily quota ({QUOTA_PER_DAY}/day) exhausted — try tomorrow or watch a replay", {"asks_left": 0, "in_flight": inflight}
    return True, "", {"asks_left": QUOTA_PER_DAY - asks, "in_flight": inflight}

def _quota_consume(user_key: str):
    day = _day_epoch()
    db.execute(
        "UPDATE users SET asks_day = asks_day + 1, day_epoch=?, in_flight = in_flight + 1 WHERE user_key=?",
        (day, user_key),
    )
    db.commit()

def _quota_release(user_key: str):
    if not user_key:
        return
    db.execute("UPDATE users SET in_flight = MAX(0, in_flight - 1) WHERE user_key=?", (user_key,))
    db.commit()

def _queue_position(cid: str) -> int:
    """1-based position among pending jobs (by rowid order)."""
    row = db.execute(
        """SELECT 1 + COUNT(*) FROM jobs j
           WHERE j.status='pending'
             AND j.rowid < (SELECT rowid FROM jobs WHERE cid=? AND status='pending' LIMIT 1)""",
        (cid,),
    ).fetchone()
    return int(row[0]) if row else 0

def _eta_seconds(position: int) -> int:
    pending = db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0]
    leased  = db.execute("SELECT COUNT(*) FROM jobs WHERE status='leased'").fetchone()[0]
    # rough: position * secs_per_gen, floor at a few seconds
    return max(8, int((max(position, 1) + leased) * SECS_PER_GEN))


# ── batched fan-out (200ms) ───────────────────────────────────────────────
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
    if ev.get("type") != "tok":
        buf = _buffers.pop(cid, [])
        task = _flush_tasks.pop(cid, None)
        if task and not task.done():
            task.cancel()
        if buf:
            _raw_push(cid, {"type": "batch", "events": buf} if len(buf) > 1 else buf[0])
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
            loop = asyncio.get_running_loop()
            _flush_tasks[cid] = loop.create_task(_flush_later(cid))
        except RuntimeError:
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
        "max_new": MAX_NEW,
        "quota_per_day": QUOTA_PER_DAY,
        "dedupe_window": DEDUPE_WINDOW,
        "sse_batch_ms": BATCH_MS,
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

    user_key = _user_key(r, body)
    ok, err, qinfo = _quota_check(user_key)
    if not ok:
        return JSONResponse({"error": err, **qinfo}, status_code=429)

    ph = _phash(p)
    now = time.time()

    # ── prompt dedupe: replay a recent completed (or live) channel ────────
    hit = db.execute(
        """SELECT cid, status FROM channels
           WHERE prompt_hash=? AND created > ? AND status IN ('done','queued','leased','running')
           ORDER BY CASE status WHEN 'done' THEN 0 ELSE 1 END, created DESC LIMIT 1""",
        (ph, now - DEDUPE_WINDOW),
    ).fetchone()
    if hit:
        cid_hit, st = hit
        # replay costs zero GPU — still counts as "served", not as a new ask
        return {
            "channel": cid_hit,
            "link": f"/c/{cid_hit}",
            "shard": SHARD_ID,
            "replay": True,
            "status": st,
            "asks_left": qinfo["asks_left"],
        }

    cid = body.get("cid") or uuid.uuid4().hex[:10]
    jid = uuid.uuid4().hex[:10]

    db.execute(
        "INSERT OR REPLACE INTO channels VALUES (?,?,?,?,?,?,?)",
        (cid, p, "queued", now, SHARD_ID, ph, user_key),
    )
    db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (jid, cid, p, "pending", 0))
    db.commit()
    _quota_consume(user_key)

    pos = _queue_position(cid)
    eta = _eta_seconds(pos)
    queued_ev = {
        "i": 0,
        "type": "queued",
        "text": f"queued #{pos} · ETA ~{eta}s",
        "nll": 0,
        "position": pos,
        "eta": eta,
    }
    push(cid, queued_ev)

    return {
        "channel": cid,
        "link": f"/c/{cid}",
        "shard": SHARD_ID,
        "replay": False,
        "position": pos,
        "eta": eta,
        "max_new": MAX_NEW,
        "asks_left": qinfo["asks_left"] - 1,
    }


# ── worker protocol ────────────────────────────────────────────────────────
@app.get("/worker/pending")
def pending():
    return {
        "pending": db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0],
        "shard": SHARD_ID,
        "max_new": MAX_NEW,
    }

@app.post("/worker/claim")
async def claim(r: Request):
    """
    capacity up to 4. Optional collect_window_ms: if >0 and fewer than `capacity`
    jobs are ready, wait up to that many ms for more to accumulate (batch-4 + 15s window).
    """
    body = await r.json()
    cap = min(int(body.get("capacity", 1)), 4)
    collect_ms = int(body.get("collect_window_ms", 0))

    def _take(n):
        rows = db.execute(
            "SELECT job_id,cid,prompt FROM jobs WHERE status='pending' ORDER BY rowid LIMIT ?",
            (n,),
        ).fetchall()
        now = time.time()
        for j, _, _ in rows:
            db.execute("UPDATE jobs SET status='leased', leased=? WHERE job_id=?", (now, j))
            db.execute("UPDATE channels SET status='leased' WHERE cid=(SELECT cid FROM jobs WHERE job_id=?)", (j,))
        db.commit()
        return rows

    rows = _take(cap)
    if collect_ms > 0 and len(rows) < cap:
        # brief wait for more asks to pile up (worker-side collect window)
        deadline = time.time() + collect_ms / 1000.0
        while len(rows) < cap and time.time() < deadline:
            await asyncio.sleep(0.4)
            extra = _take(cap - len(rows))
            rows = list(rows) + list(extra)

    return {
        "jobs": [
            {"job_id": a, "cid": b, "prompt": c, "max_new": MAX_NEW}
            for a, b, c in rows
        ],
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
    status = b.get("status", "done")
    db.execute("UPDATE jobs SET status=? WHERE job_id=?", (status, b["job_id"]))
    db.execute("UPDATE channels SET status=? WHERE cid=?", (status, b["cid"]))
    # release 1-in-flight for the user who owns this channel
    row = db.execute("SELECT user_key FROM channels WHERE cid=?", (b["cid"],)).fetchone()
    if row and row[0]:
        _quota_release(row[0])
    db.commit()
    push(b["cid"], {"i": -1, "type": status, "text": "", "nll": 0})
    return {"ok": True, "shard": SHARD_ID}


# ── SSE (replay-first: history is free) ───────────────────────────────────
@app.get("/channel/{cid}/events")
async def sse(cid: str):
    q: asyncio.Queue = asyncio.Queue(maxsize=512)
    SUBS.setdefault(cid, []).append(q)

    rows = db.execute(
        "SELECT i,type,text,nll FROM events WHERE cid=? ORDER BY rowid", (cid,)
    ).fetchall()
    meta = db.execute("SELECT status, prompt FROM channels WHERE cid=?", (cid,)).fetchone()

    async def gen():
        try:
            # announce channel meta so the viewer can show receipt-style status
            if meta:
                st, prompt = meta
                yield f"data: {json.dumps({'type': 'meta', 'status': st, 'prompt': prompt, 'shard': SHARD_ID})}\n\n"
            for i, t, x, n in rows:
                yield f"data: {json.dumps({'i': i, 'type': t, 'text': x, 'nll': n})}\n\n"
            # if already terminal, don't hang the stream forever
            if meta and meta[0] in ("done", "error"):
                yield f"data: {json.dumps({'i': -1, 'type': meta[0], 'text': '', 'nll': 0})}\n\n"
                return
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") in ("done", "error"):
                    break
        finally:
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
    try:
        return open("router.html").read()
    except FileNotFoundError:
        return HTMLResponse(
            f"<pre>slm-hub shard={SHARD_ID} · <a href='/healthz'>/healthz</a></pre>"
        )
