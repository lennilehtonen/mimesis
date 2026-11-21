"""End-to-end synthetic test (no phones): HTTP ingest, contacts, hands, clap-sync, retarget,
MuJoCo multiplier, sim-to-real, LeRobot export, demo grid, dashboard backend endpoints.

    python test_pipeline.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OK = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        raise SystemExit(1)
    OK.append(name)


def run(*args):
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


def main():
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="mimesis_test_"))
    os.environ["RIG_IMU_DIR"] = str(tmp / "raw_imu")
    import pandas as pd

    # 1. HTTP ingest with the verified Sensor Logger payload shape (real uvicorn server on a spare port)
    import urllib.request
    import socket
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    env = dict(os.environ, RIG_IMU_DIR=str(tmp / "raw_imu"))
    srv = subprocess.Popen([PY, "receiver.py", "--port", str(port), "--out-dir", str(tmp / "raw_imu")], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=10).read())

    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(base + "/status", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        post("/episode/start", {"episode_id": "ep_http", "task": "pick_place_mug"})
        t_ns = time.time_ns()
        for m in range(3):
            pl = []
            for i in range(100):
                k, t = m * 100 + i, t_ns + (m * 100 + i) * 10_000_000
                pl += [{"name": "accelerometer", "time": t, "values": {"x": 0.1 * math.sin(k / 9), "y": 0.2, "z": -0.05}},
                       {"name": "gyroscope", "time": t, "values": {"x": 0.01, "y": -0.02, "z": 0.03}},
                       {"name": "orientation", "time": t, "values": {"qx": 0, "qy": 0, "qz": 0, "qw": 1, "roll": 0, "pitch": 0, "yaw": 0}},
                       {"name": "gravity", "time": t, "values": {"x": 0, "y": 0, "z": 9.81}}]
            assert post("/data", {"messageId": m, "sessionId": "s", "deviceId": "d", "payload": pl})["status"] == "ok"
        import asyncio
        import websockets

        async def ws_first():
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as w:
                return json.loads(await asyncio.wait_for(w.recv(), 5))
        msg = asyncio.run(ws_first())
        meta = post("/episode/stop", {})["meta"]
        latest = json.loads(urllib.request.urlopen(base + "/latest?n=5").read())
    finally:
        srv.terminate()
    check("http ingest + /ws", meta["num_samples"] == 300 and abs(meta["effective_hz"] - 100) < 1 and "status" in msg,
          f"{meta['num_samples']} samples @ {meta['effective_hz']:.0f} Hz")
    check("/latest", latest["count"] == 5)

    # 2. synthetic chest-cam + wrist IMU with a planted 0.8 s offset
    syn = tmp / "syn"
    assert run("scripts/make_synthetic.py", "--out-dir", str(syn), "--offset", "0.8", "--duration", "10").returncode == 0
    imu, vid = syn / "ep_synth.parquet", syn / "ep_synth.mp4"

    from contact_detect import detect_contacts
    _, events = detect_contacts(pd.read_parquet(imu))
    kinds = {e["event"] for e in events}
    check("contact detection", {"grab_onset", "impact_or_tap", "slip_or_scrape", "set_down"} <= kinds, f"{len(events)} events")

    from hands import run_hands
    run_hands(str(vid), str(tmp / "hands.parquet"), str(tmp / "overlay.mp4"))
    check("hands.py", (tmp / "hands.parquet").exists() and (tmp / "overlay.mp4").exists())

    from sync import run_sync
    s = run_sync(str(vid), str(imu), str(tmp / "hands.parquet"), str(tmp / "aligned" / "ep_synth.parquet"), task="pick_place_mug")
    err = abs(s["offset_s"] - 0.8) * 1000
    check("clap sync", err <= 30, f"offset {s['offset_s']:+.3f} s, error {err:.0f} ms")

    # 3. multiply in a subprocess (MuJoCo renderer on that process's main thread)
    sim = tmp / "sim_aug"
    r = run("-m", "sim.augment", str(tmp / "aligned" / "ep_synth.parquet"), "--n", "12", "--render-n", "9",
            "--out-dir", str(sim), "--lerobot-out", str(tmp / "mimesis_lerobot"), "--task", "pick_place_mug")
    man = json.loads((sim / "manifest.json").read_text())
    vs = man["variants"] if isinstance(man, dict) else man
    n_ok, n_slip = sum(v["success"] for v in vs), sum(v["slipped"] for v in vs)
    n_vid = len(list((sim / "videos").glob("sim_v*.mp4")))
    check("sim.augment", r.returncode == 0 and len(vs) == 12 and n_vid == 9 and n_slip > 0 and n_ok > 0,
          f"12 variants: {n_ok} success, {n_slip} slip, {n_vid} rendered")
    check("sim LeRobot export", (tmp / "mimesis_lerobot" / "meta" / "info.json").exists())

    r = run("-m", "sim.sim_to_real", "--sim-dir", str(sim / "episodes"), "--real", str(imu),
            "--out-json", str(sim / "sim_to_real_report.json"))
    rep = json.loads((sim / "sim_to_real_report.json").read_text())
    check("sim_to_real", r.returncode == 0, f"AUC {rep.get('zero_shot_roc_auc')}, "
          f"bal.acc {rep.get('zero_shot_balanced_accuracy'):.3f} (synthetic real clip)")

    r = run("export.py", "--aligned-dir", str(tmp / "aligned"), "--video-dir", str(syn), "--out-dir", str(tmp / "real_lerobot"))
    check("real LeRobot export", r.returncode == 0 and (tmp / "real_lerobot" / "data" / "chunk-000" / "file-000.parquet").exists())

    r = run("-m", "sim.demo_grid", "--real-video", str(tmp / "overlay.mp4"), "--sim-dir", str(sim),
            "--out", str(tmp / "demo.mp4"), "--out-png", str(tmp / "demo.png"))
    check("demo_grid", r.returncode == 0 and (tmp / "demo.mp4").stat().st_size > 10_000)

    print(f"\nALL {len(OK)} CHECKS PASSED in {time.time() - t0:.0f}s  (outputs in {tmp})")


if __name__ == "__main__":
    main()
