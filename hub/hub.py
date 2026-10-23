import os, sqlite3, time, uuid, json, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse

DB = "channels.db"
db = sqlite3.connect(DB, check_same_thread=False); db.execute("PRAGMA journal_mode=WAL")
db.executescript("""CREATE TABLE IF NOT EXISTS channels(cid TEXT PRIMARY KEY, prompt TEXT, status TEXT, created REAL);
CREATE TABLE IF NOT EXISTS events(cid TEXT, i INT, type TEXT, text TEXT, nll REAL, ts REAL);
CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, cid TEXT, prompt TEXT, status TEXT, leased REAL);""")
db.commit()

app = FastAPI()
SUBS: dict[str, list[asyncio.Queue]] = {}
push = lambda cid, ev: [q.put_nowait(ev) for q in SUBS.get(cid, [])]

@app.post("/api/ask")
async def ask(r: Request):
    p = (await r.json()).get("prompt", "").strip()[:2000]
    cid, jid = uuid.uuid4().hex[:10], uuid.uuid4().hex[:10]
    db.execute("INSERT INTO channels VALUES (?,?,?,?)", (cid, p, "queued", time.time()))
    db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (jid, cid, p, "pending", 0)); db.commit()
    push(cid, {"i": 0, "type": "queued", "text": "", "nll": 0})
    return {"channel": cid, "link": f"/c/{cid}"}

@app.get("/worker/pending")
def pending(): return {"pending": db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0]}

@app.post("/worker/claim")
async def claim(r: Request):
    cap = min(int((await r.json()).get("capacity", 1)), 4)
    rows = db.execute("SELECT job_id,cid,prompt FROM jobs WHERE status='pending' ORDER BY rowid LIMIT ?", (cap,)).fetchall()
    for j, _, _ in rows: db.execute("UPDATE jobs SET status='leased',leased=? WHERE job_id=?", (time.time(), j))
    db.commit(); return {"jobs": [{"job_id": a, "cid": b, "prompt": c} for a, b, c in rows]}

@app.post("/worker/event")
async def event(r: Request):
    b = await r.json()
    for ev in b["events"]:
        db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)", (b["cid"], ev["i"], ev["type"], ev["text"], ev.get("nll", 0), time.time()))
        push(b["cid"], ev)
    db.commit(); return {"ok": True}

@app.post("/worker/done")
async def done(r: Request):
    b = await r.json()
    db.execute("UPDATE jobs SET status=? WHERE job_id=?", (b["status"], b["job_id"]))
    db.execute("UPDATE channels SET status=? WHERE cid=?", (b["status"], b["cid"])); db.commit()
    return {"ok": True}

@app.get("/channel/{cid}/events")
async def sse(cid: str):
    q = asyncio.Queue(); SUBS.setdefault(cid, []).append(q)
    rows = db.execute("SELECT i,type,text,nll FROM events WHERE cid=? ORDER BY rowid", (cid,)).fetchall()
    async def gen():
        for i, t, x, n in rows: yield f"data: {json.dumps({'i':i,'type':t,'text':x,'nll':n})}\n\n"
        while True: yield f"data: {json.dumps(await q.get())}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/c/{cid}", response_class=HTMLResponse)
def page(cid: str): return open("page.html").read()
