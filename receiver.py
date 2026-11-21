"""MIMESIS receiver + dashboard backend.

Ingests Sensor Logger HTTP Push batches (POST /data), keeps a live ring buffer, streams it to the
dashboard over a WebSocket (/ws), writes per-episode parquet/CSV, and runs pipeline jobs
(process / multiply / metrics / export) in subprocesses.

    python receiver.py --host 0.0.0.0 --port 8000 --out-dir data/raw_imu
    python receiver.py --import-csv-dir ~/Downloads/ep_001 --episode-id ep_001 --task pick_place_mug
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_DIR = Path(os.environ.get("RIG_IMU_DIR", DATA / "raw_imu"))
WEB_DIR = ROOT / "web"
PUB_DIR = DATA / "web"  # files the dashboard reads under /files
for d in (OUT_DIR, PUB_DIR, DATA / "videos"):
    d.mkdir(parents=True, exist_ok=True)

SENSOR_ALIASES = {
    "accelerometer": "accel", "linear_acceleration": "accel", "linearacceleration": "accel",
    "totalacceleration": "total_accel", "accelerometeruncalibrated": "accel_raw",
    "gyroscope": "gyro", "gyroscopeuncalibrated": "gyro_raw",
    "orientation": "orient", "attitude": "orient", "gravity": "gravity",
}
LIVE_MAX = 2500

SESSIONS: dict[str, dict] = {}
ACTIVE_EPISODE: dict | None = None
LAST_SESSION: str | None = None
LATEST_STATE: dict = {k: 0.0 for k in ["ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw",
                                       "qx", "qy", "qz", "grav_x", "grav_y", "grav_z"]}
LATEST_STATE["qw"] = 1.0
LIVE: deque = deque(maxlen=LIVE_MAX)
LIVE_SEQ = 0  # total samples ever appended (for WS cursors)
LAST_PUSH_WALL = 0.0
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}


def canon(name: str) -> str:
    n = str(name).strip().lower()
    return SENSOR_ALIASES.get(n, n)


def item_values(item: dict) -> dict:
    v = item.get("values")
    if isinstance(v, dict):
        return v
    return {k: item[k] for k in item if k not in ("name", "sensor", "time", "accuracy")}


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def update_state(key: str, vals: dict) -> None:
    if key in ("accel", "accel_raw"):
        LATEST_STATE.update(ax=_f(vals.get("x")), ay=_f(vals.get("y")), az=_f(vals.get("z")))
    elif key in ("gyro", "gyro_raw"):
        LATEST_STATE.update(gx=_f(vals.get("x")), gy=_f(vals.get("y")), gz=_f(vals.get("z")))
    elif key == "orient":
        for k in ("qx", "qy", "qz", "qw", "roll", "pitch", "yaw"):
            if k in vals:
                LATEST_STATE[k] = _f(vals[k])
    elif key == "gravity":
        LATEST_STATE.update(grav_x=_f(vals.get("x")), grav_y=_f(vals.get("y")), grav_z=_f(vals.get("z")))


def ingest(msg: dict) -> tuple[str, int]:
    global LAST_SESSION, LIVE_SEQ, LAST_PUSH_WALL
    payload = msg.get("payload")
    if not isinstance(payload, list):
        raise ValueError("payload must be a list")
    sid = str(msg.get("sessionId") or "default")
    with LOCK:
        s = SESSIONS.get(sid)
        if s is None:
            ep = ACTIVE_EPISODE or {}
            s = SESSIONS[sid] = {
                "episode_id": ep.get("episode_id") or datetime.now(timezone.utc).strftime("ep_%H%M%S"),
                "task": ep.get("task") or "manipulation_task", "device_id": msg.get("deviceId"),
                "messages": 0, "streams": {}, "flushed_messages": 0,
            }
            if ACTIVE_EPISODE is not None:
                ACTIVE_EPISODE["session_id"] = sid
        s["messages"] += 1
        LAST_SESSION = sid
        items = []
        for it in payload:
            if not isinstance(it, dict):
                continue
            key = canon(it.get("name") or it.get("sensor") or "")
            try:
                tns = int(float(it.get("time")))
            except (TypeError, ValueError):
                continue
            items.append((tns, key, item_values(it)))
        items.sort(key=lambda x: x[0])
        n = 0
        for tns, key, vals in items:
            s["streams"].setdefault(key, []).append((tns, vals))
            update_state(key, vals)
            n += 1
            if key == "accel" or (key == "accel_raw" and "accel" not in s["streams"]):
                LIVE.append(dict(LATEST_STATE, time_ns=tns))
                LIVE_SEQ += 1
        LAST_PUSH_WALL = time.time()
    if s["messages"] % 5 == 0 and ACTIVE_EPISODE is not None:
        try:
            flush_session(sid)
        except Exception as e:  # never break ingest on a flush error
            print("flush error:", e)
    return sid, n


def stream_frame(rows: list, cols: dict) -> pd.DataFrame:
    """rows: [(t_ns, values)] -> DataFrame(time_ns, *cols) deduped + sorted."""
    t = np.array([r[0] for r in rows], dtype=np.int64)
    data = {"time_ns": t}
    for out, src in cols.items():
        data[out] = np.array([_f(r[1].get(src), np.nan) for r in rows], dtype=float)
    return pd.DataFrame(data).drop_duplicates("time_ns").sort_values("time_ns")


def align_session_streams(streams: dict) -> pd.DataFrame:
    prim = "accel" if streams.get("accel") else "accel_raw" if streams.get("accel_raw") else None
    if prim is None:
        if not streams:
            return pd.DataFrame()
        prim = max(streams, key=lambda k: len(streams[k]))
    base = stream_frame(streams[prim], {"ax": "x", "ay": "y", "az": "z"})
    t0 = int(base["time_ns"].iloc[0])
    tb = (base["time_ns"].to_numpy() - t0) / 1e9
    out = pd.DataFrame({"timestamp_ns": base["time_ns"].to_numpy(), "t_rel_s": tb})
    for c in ("ax", "ay", "az"):
        out[c] = base[c].to_numpy()
    spec = {
        "gyro": {"gx": "x", "gy": "y", "gz": "z"},
        "gravity": {"grav_x": "x", "grav_y": "y", "grav_z": "z"},
        "orient": {"qx": "qx", "qy": "qy", "qz": "qz", "qw": "qw", "roll": "roll", "pitch": "pitch", "yaw": "yaw"},
    }
    for key, cols in spec.items():
        rows = streams.get(key) or (streams.get("gyro_raw") if key == "gyro" else None)
        if rows:
            f = stream_frame(rows, cols)
            ts = (f["time_ns"].to_numpy() - t0) / 1e9
            for c in cols:
                y = f[c].to_numpy()
                ok = np.isfinite(y)
                out[c] = np.interp(tb, ts[ok], y[ok]) if ok.sum() >= 2 else (y[ok][0] if ok.any() else 0.0)
        else:
            for c in cols:
                out[c] = 1.0 if c == "qw" else 0.0
    out["accel_mag"] = np.linalg.norm(out[["ax", "ay", "az"]].to_numpy(), axis=1)
    out["gyro_mag"] = np.linalg.norm(out[["gx", "gy", "gz"]].to_numpy(), axis=1)
    return out


def write_episode(df: pd.DataFrame, episode_id: str, task: str, extra: dict | None = None) -> dict:
    pq = OUT_DIR / f"{episode_id}.parquet"
    csv = OUT_DIR / f"{episode_id}.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    dur = float(df["t_rel_s"].iloc[-1]) if len(df) else 0.0
    meta = {"episode_id": episode_id, "task": task, "num_samples": int(len(df)), "duration_s": dur,
            "effective_hz": (len(df) - 1) / dur if dur > 0 else 0.0,
            "parquet_path": str(pq), "csv_path": str(csv)}
    meta.update(extra or {})
    (OUT_DIR / f"{episode_id}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def flush_session(sid: str) -> dict | None:
    with LOCK:
        s = SESSIONS.get(sid)
        if not s:
            return None
        streams = {k: list(v) for k, v in s["streams"].items()}
    df = align_session_streams(streams)
    if df.empty:
        return None
    return write_episode(df, s["episode_id"], s["task"], {
        "session_id": sid, "device_id": s["device_id"], "message_count": s["messages"],
        "sensor_counts": {k: len(v) for k, v in streams.items()}})


def import_csv_dir(folder: str, episode_id: str, task: str) -> dict:
    p = Path(folder).expanduser()
    files = {"accel": ["Accelerometer.csv"], "accel_raw": ["AccelerometerUncalibrated.csv"],
             "gyro": ["Gyroscope.csv"], "orient": ["Orientation.csv"], "gravity": ["Gravity.csv"]}
    streams = {}
    for key, names in files.items():
        for nm in names:
            f = p / nm
            if f.exists():
                df = pd.read_csv(f)
                tcol = "time" if "time" in df else df.columns[0]
                streams[key] = [(int(r[tcol]), {k: r[k] for k in df.columns if k != tcol}) for _, r in df.iterrows()]
    if not streams:
        raise SystemExit(f"no Sensor Logger CSVs in {p}")
    meta = write_episode(align_session_streams(streams), episode_id, task, {"source": str(p)})
    print(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- live contact detection
class LiveContacts:
    """Runs contact_detect on the last ~3 s of the live buffer and emits new events once."""

    def __init__(self):
        self.last_emit_ns = 0
        self.last_run = 0.0
        self.events: deque = deque(maxlen=200)
        self.seq = 0

    def update(self):
        now = time.time()
        if now - self.last_run < 0.25:
            return
        self.last_run = now
        with LOCK:
            samples = list(LIVE)[-300:]
        if len(samples) < 40:
            return
        from contact_detect import UI_KIND, detect_contacts
        df = pd.DataFrame(samples)
        df["timestamp_ns"] = df["time_ns"].astype(np.int64)
        try:
            _, evs = detect_contacts(df)
        except Exception as e:
            print("live contact error:", e)
            return
        for ev in evs:
            tns = int(df["timestamp_ns"].iloc[ev["sample_idx"]])
            # ignore events in the trailing 0.6 s (classification needs context after the peak)
            if tns <= self.last_emit_ns or tns > int(df["timestamp_ns"].iloc[-1]) - 600_000_000:
                continue
            self.last_emit_ns = tns
            self.seq += 1
            self.events.append({"seq": self.seq, "time_ns": tns, "kind": UI_KIND.get(ev["event"], "TAP"),
                                "event": ev["event"], "jerk": round(ev["jerk_peak_ms3"], 1)})


LIVE_CONTACTS = LiveContacts()
T0_NS: int | None = None  # dashboard time origin


def live_hz() -> float:
    with LOCK:
        if len(LIVE) < 10 or time.time() - LAST_PUSH_WALL > 3.0:
            return 0.0
        a = LIVE[-1]["time_ns"]
        tail = [s["time_ns"] for s in list(LIVE)[-400:] if a - s["time_ns"] <= 2e9]
    span = (tail[-1] - tail[0]) / 1e9 if len(tail) > 1 else 0
    return (len(tail) - 1) / span if span > 0 else 0.0


def list_episodes() -> list[dict]:
    eps = {}
    for m in sorted(OUT_DIR.glob("*_meta.json")):
        try:
            meta = json.loads(m.read_text())
        except Exception:
            continue
        eid = meta.get("episode_id", m.stem.replace("_meta", ""))
        eps[eid] = {"episode_id": eid, "task": meta.get("task"), "duration_s": meta.get("duration_s"),
                    "num_samples": meta.get("num_samples"), "effective_hz": meta.get("effective_hz"),
                    "raw": True}
    for eid, e in eps.items():
        vids = [p for p in (DATA / "videos").glob(f"{eid}.*") if p.suffix.lower() in (".mov", ".mp4", ".webm", ".m4v")]
        e["video"] = vids[0].name if vids else None
        e["aligned"] = (DATA / "aligned" / f"{eid}.parquet").exists()
        e["processed"] = (PUB_DIR / f"ep_{eid}.json").exists()
    return list(eps.values())


def status_dict() -> dict:
    return {
        "active_episode": ACTIVE_EPISODE,
        "sessions": {sid: {"episode_id": s["episode_id"], "task": s["task"], "messages": s["messages"],
                           "counts": {k: len(v) for k, v in s["streams"].items()}} for sid, s in SESSIONS.items()},
        "live_buffer_len": len(LIVE), "hz": round(live_hz(), 1),
        "episodes": len(list(OUT_DIR.glob("*_meta.json"))),
        "jobs": {k: {kk: vv for kk, vv in j.items() if kk != "log"} for k, j in JOBS.items()},
    }


# --------------------------------------------------------------------------- jobs (subprocesses)
def start_job(name: str, args: list[str]) -> dict:
    j = JOBS.get(name)
    if j and j["state"] == "running":
        return j
    j = JOBS[name] = {"name": name, "state": "running", "started": time.time(), "progress": None,
                      "log": deque(maxlen=200), "returncode": None, "cmd": " ".join(args)}

    def run():
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        j["pid"] = proc.pid
        for line in proc.stdout:
            line = line.rstrip()
            j["log"].append(line)
            if line.startswith("PROGRESS "):
                try:
                    j["progress"] = json.loads(line[9:])
                except json.JSONDecodeError:
                    pass
            elif line.startswith("STEP "):
                j["step"] = line[5:]
        j["returncode"] = proc.wait()
        j["state"] = "done" if j["returncode"] == 0 else "error"
        j["ended"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return j


def job_view(name: str) -> dict:
    j = JOBS.get(name)
    if not j:
        return {"name": name, "state": "idle"}
    v = {k: vv for k, vv in j.items() if k != "log"}
    v["log_tail"] = list(j["log"])[-15:]
    return v


SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")

# --------------------------------------------------------------------------- app
app = FastAPI(title="MIMESIS receiver")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/data")
async def data(request: Request):
    try:
        msg = json.loads(await request.body())
        sid, n = ingest(msg)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)
    return {"status": "ok", "sessionId": sid, "ingested": n}


@app.post("/episode/start")
async def episode_start(request: Request):
    global ACTIVE_EPISODE
    try:
        body = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError:
        body = {}
    eid = body.get("episode_id") or datetime.now().strftime("ep_%H%M%S")
    if not SAFE.match(eid):
        return JSONResponse({"status": "error", "detail": "bad episode_id"}, status_code=400)
    task = body.get("task") or "manipulation_task"
    ACTIVE_EPISODE = {"episode_id": eid, "task": task, "session_id": None,
                      "started_at": datetime.now(timezone.utc).isoformat()}
    # A Sensor Logger recording that is already streaming keeps its sessionId: start a fresh
    # episode on it by clearing its buffers.
    with LOCK:
        if LAST_SESSION and LAST_SESSION in SESSIONS and time.time() - LAST_PUSH_WALL < 3:
            s = SESSIONS[LAST_SESSION]
            s.update(episode_id=eid, task=task, streams={}, messages=0)
            ACTIVE_EPISODE["session_id"] = LAST_SESSION
    return {"status": "started", "episode_id": eid, "task": task}


@app.post("/episode/stop")
async def episode_stop(request: Request):
    global ACTIVE_EPISODE
    try:
        body = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError:
        body = {}
    sid = body.get("session_id") or (ACTIVE_EPISODE or {}).get("session_id") or LAST_SESSION
    ep = ACTIVE_EPISODE
    ACTIVE_EPISODE = None
    if not sid or sid not in SESSIONS:
        return {"status": "no_active_session", "episode": ep}
    meta = flush_session(sid)
    with LOCK:  # detach so the next recording on the same sessionId starts clean
        SESSIONS.pop(sid, None)
    return {"status": "flushed", "meta": meta}


@app.post("/upload/{episode}")
async def upload(episode: str, file: UploadFile = File(...)):
    """Browser-recorded chest-cam clip (MediaRecorder webm/mp4) or any video file."""
    if not SAFE.match(episode):
        return JSONResponse({"status": "error"}, status_code=400)
    ext = Path(file.filename or "x.webm").suffix.lower() or ".webm"
    dst = DATA / "videos" / f"{episode}{ext}"
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"status": "ok", "path": str(dst.relative_to(ROOT)), "bytes": dst.stat().st_size}


@app.get("/status")
def status():
    return status_dict()


@app.get("/latest")
def latest(n: int = 1000):
    with LOCK:
        s = list(LIVE)[-max(1, min(n, LIVE_MAX)):]
    return {"count": len(s), "samples": s}


@app.get("/episodes")
def episodes():
    return {"episodes": list_episodes()}


@app.post("/process/{episode}")
def process(episode: str):
    if not SAFE.match(episode):
        return JSONResponse({"status": "error"}, status_code=400)
    return job_view(start_job(f"process:{episode}", ["jobs.py", "process", episode])["name"])


@app.post("/process_all")
def process_all():
    return job_view(start_job("process_all", ["jobs.py", "process_all"])["name"])


@app.post("/multiply/{episode}")
def multiply(episode: str, n: int = 100, render: int = 9):
    if not SAFE.match(episode):
        return JSONResponse({"status": "error"}, status_code=400)
    n = max(1, min(n, 2000))
    return job_view(start_job("multiply", ["jobs.py", "multiply", episode, "--n", str(n),
                                           "--render-n", str(max(0, min(render, 25)))])["name"])


@app.post("/multiply")
def multiply_latest(n: int = 100, render: int = 9):
    eps = [e for e in list_episodes() if e["aligned"]] or list_episodes()
    ep = eps[-1]["episode_id"] if eps else "latest"
    return multiply(ep, n, render)


@app.post("/metrics")
def metrics_run():
    return job_view(start_job("metrics", ["jobs.py", "metrics"])["name"])


@app.get("/metrics")
def metrics():
    out = {}
    for name in ("metrics.json", "dataset_stats.json"):
        p = PUB_DIR / name
        if p.exists():
            out[name.split(".")[0]] = json.loads(p.read_text())
    rep = DATA / "sim_aug" / "sim_to_real_report.json"
    if rep.exists():
        out["sim_to_real_report"] = json.loads(rep.read_text())
    return out


@app.post("/export")
def export():
    return job_view(start_job("export", ["jobs.py", "export"])["name"])


@app.post("/export/open")
def export_open():
    p = DATA / "mimesis_lerobot"
    if p.exists() and sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    return {"status": "ok" if p.exists() else "missing", "path": str(p)}


@app.post("/export/push")
def export_push():
    return JSONResponse({"status": "not_configured", "detail": "push_to_hub needs HF credentials"}, status_code=501)


@app.get("/jobs/{name}")
def job(name: str):
    return job_view(name)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """~20 Hz: new IMU samples (paced out so 1 s Sensor Logger batches scroll smoothly), new contact
    events, and a status block every second."""
    global T0_NS
    await sock.accept()
    cursor = LIVE_SEQ
    ev_seq = LIVE_CONTACTS.seq
    pending: deque = deque()
    last_status = 0.0
    try:
        while True:
            with LOCK:
                new = LIVE_SEQ - cursor
                if new > 0:
                    buf = list(LIVE)
                    pending.extend(buf[-min(new, len(buf)):])
                    cursor = LIVE_SEQ
            if pending and T0_NS is None:
                T0_NS = pending[0]["time_ns"]
            # pacing: drain the backlog over ~1 s, at least 5 samples/tick (100 Hz at 20 Hz ticks)
            k = min(len(pending), max(5, len(pending) // 10)) if len(pending) < 400 else len(pending)
            out = []
            for _ in range(k):
                s = pending.popleft()
                out.append({"t": round((s["time_ns"] - T0_NS) / 1e9, 3),
                            **{c: round(s[c], 4) for c in ("ax", "ay", "az", "gx", "gy", "gz")},
                            "q": [round(s["qw"], 4), round(s["qx"], 4), round(s["qy"], 4), round(s["qz"], 4)]})
            await asyncio.to_thread(LIVE_CONTACTS.update)
            contacts = [{"t": round((e["time_ns"] - (T0_NS or e["time_ns"])) / 1e9, 3), "kind": e["kind"],
                         "event": e["event"]} for e in LIVE_CONTACTS.events if e["seq"] > ev_seq]
            ev_seq = LIVE_CONTACTS.seq
            msg = {"type": "tick", "imu": out, "contacts": contacts}
            if time.time() - last_status > 1.0:
                last_status = time.time()
                hz = live_hz()
                msg["status"] = {"wrist": hz > 1, "hz": round(hz, 1), "episodes": len(list(OUT_DIR.glob("*_meta.json"))),
                                 "recording": ACTIVE_EPISODE is not None,
                                 "episode": (ACTIVE_EPISODE or {}).get("episode_id"),
                                 "jobs": {k: {"state": j["state"], "progress": j.get("progress"), "step": j.get("step")}
                                          for k, j in JOBS.items()}}
            if out or contacts or "status" in msg:
                await sock.send_text(json.dumps(msg))
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/files", StaticFiles(directory=PUB_DIR), name="files")
app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")


def usb_ip() -> tuple[str, str]:
    """(ip, interface): prefer the iPhone USB / hotspot network 172.20.10.x, else en0."""
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    except Exception:
        out = ""
    iface, found = None, {}
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":")[0]
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
        if m and iface:
            found.setdefault(iface, m.group(1))
    for i, ip in found.items():
        if ip.startswith("172.20.10."):
            return ip, i
    if "en0" in found:
        return found["en0"], "en0"
    for i, ip in found.items():
        if not ip.startswith("127."):
            return ip, i
    return "127.0.0.1", "lo0"


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--import-csv-dir")
    ap.add_argument("--episode-id", default="ep_offline_001")
    ap.add_argument("--task", default="unlabeled_task")
    a = ap.parse_args()
    OUT_DIR = Path(a.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.import_csv_dir:
        import_csv_dir(a.import_csv_dir, a.episode_id, a.task)
        return
    ip, iface = usb_ip()
    print("=" * 64)
    print(f"  MIMESIS receiver on {a.host}:{a.port}  (interface {iface})")
    print(f"  Sensor Logger HTTP Push URL:  http://{ip}:{a.port}/data")
    print(f"  Dashboard:                    http://localhost:{a.port}/?live")
    print("=" * 64, flush=True)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
