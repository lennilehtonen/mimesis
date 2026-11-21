"""Pipeline jobs run by receiver.py in subprocesses (MuJoCo rendering stays on this process's main thread).

    python jobs.py process ep_001          # contacts, hands, clap-sync, sim twin, publish for the UI
    python jobs.py process_all
    python jobs.py multiply ep_001 --n 300 --render-n 9
    python jobs.py metrics                  # sim-to-real + dataset stats + drift plot data
    python jobs.py export                   # LeRobot v3 folders (sim + real)

Every job prints `STEP <text>` lines and, for multiply, `PROGRESS {json}` lines (parsed by receiver.py).
Files the dashboard reads are published to data/web/ (served at /files).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW, VID, HANDS, ALIGNED, SIM = (DATA / d for d in ("raw_imu", "videos", "hands", "aligned", "sim_aug"))
PUB = DATA / "web"
for d in (RAW, VID, HANDS, ALIGNED, SIM, PUB):
    d.mkdir(parents=True, exist_ok=True)
VIDEO_EXT = (".mov", ".mp4", ".m4v", ".webm")
MIN_REAL_S = 3.0


def step(msg: str) -> None:
    print("STEP " + msg, flush=True)


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def raw_episodes() -> list[str]:
    out = []
    for p in sorted(RAW.glob("*.parquet")):
        if p.stem.endswith("_contacts"):
            continue
        meta = RAW / f"{p.stem}_meta.json"
        try:
            dur = json.loads(meta.read_text()).get("duration_s", 0) if meta.exists() else 0
        except Exception:
            dur = 0
        if dur >= MIN_REAL_S:
            out.append(p.stem)
    return out


def episode_meta(eid: str) -> dict:
    m = RAW / f"{eid}_meta.json"
    return json.loads(m.read_text()) if m.exists() else {"episode_id": eid}


def find_video(eid: str) -> Path | None:
    """Chest-cam clip for an episode. Browser uploads (webm / VFR mp4) are transcoded to 30 fps CFR."""
    cands = [p for p in VID.glob(f"{eid}.*") if p.suffix.lower() in VIDEO_EXT and p.stat().st_size > 10_000]
    if not cands:
        return None
    src = sorted(cands, key=lambda p: p.suffix.lower() != ".mov")[0]
    if src.suffix.lower() == ".mov":
        return src
    cfr = VID / f"{eid}_cfr.mp4"
    if not cfr.exists() or cfr.stat().st_mtime < src.stat().st_mtime:
        step(f"transcode {src.name} -> 30 fps CFR")
        ffmpeg("-i", str(src), "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", str(cfr))
    return cfr


def h264(src: Path, dst: Path) -> None:
    ffmpeg("-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(dst))


# --------------------------------------------------------------------------- process
def hand_events(eid: str, video: Path, vdur: float):
    """GRAB / RELEASE from the video: thumb-index opening of the metric 3D hand track (sim.hand3d).
    Returns (contacts, gripper segments, opening in cm at 100 Hz on the video clock)."""
    import cv2
    from sim import hand3d
    hp = HANDS / f"{eid}_hands.parquet"
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    t100 = np.arange(0, vdur, 0.01)
    try:
        tr = hand3d.track_to_table(pd.read_parquet(hp), w, h, table=hand3d.calibrate_table(str(video)))
    except Exception as e:
        print("hand track failed:", e, flush=True)
        return [], [], [0.0] * len(t100)
    v = tr[tr["valid"].astype(bool)]
    if len(v) < 10:
        return [], [{"t0": 0, "t1": vdur, "state": "open"}], [0.0] * len(t100)
    t, ap = v["t_s"].to_numpy(float), pd.Series(v["aperture_m"].to_numpy(float)).rolling(5, center=True, min_periods=1).median().to_numpy()
    lo, hi = np.percentile(ap, 10), np.percentile(ap, 90)
    close_th, open_th = lo + 0.35 * (hi - lo), lo + 0.65 * (hi - lo)   # hysteresis
    contacts, gripper, state, t_last = [], [], "open", 0.0
    if hi - lo > 0.005:
        for ti, a in zip(t, ap):
            if state == "open" and a < close_th:
                contacts.append({"kind": "GRAB", "t": round(float(ti), 2), "event": "grab_onset"})
                gripper.append({"t0": t_last, "t1": float(ti), "state": "open"}); state, t_last = "closed", float(ti)
            elif state == "closed" and a > open_th:
                contacts.append({"kind": "SET-DOWN", "t": round(float(ti), 2), "event": "release"})
                gripper.append({"t0": t_last, "t1": float(ti), "state": "closed"}); state, t_last = "open", float(ti)
    gripper.append({"t0": t_last, "t1": vdur, "state": state})
    ap100 = np.interp(t100, t, ap * 100, left=0, right=0)
    ap100[(t100 < t[0]) | (t100 > t[-1])] = 0
    return contacts, gripper, [round(float(x), 2) for x in ap100]


def process_video_only(eid: str) -> dict:
    """Take with chest video but no wrist IMU: hands -> exact 3D twin -> Episode tab (no IMU timeline)."""
    video = find_video(eid)
    if video is None:
        raise SystemExit(f"{eid}: no IMU and no video")
    hands_p, overlay = HANDS / f"{eid}_hands.parquet", HANDS / f"{eid}_overlay.mp4"
    step("hands (MediaPipe), video only")
    from hands import run_hands
    run_hands(str(video), str(hands_p), str(overlay), swap_handedness=True)
    step("exact 3D twin (MuJoCo)")
    twin = sim_twin(eid, ALIGNED / f"{eid}.parquet", "Pick up cup")
    vdur = video_duration(video)
    contacts, gripper, ap100 = hand_events(eid, video, vdur)
    pub = {"id": eid, "task": "pick up cup", "duration": round(vdur, 2), "fps": 30,
           "frames": int(round(vdur * 30)), "sync_offset_ms": 0, "contacts": contacts, "gripper": gripper,
           "imu": ap100, "imu_hz": 100, "signal": "finger opening (cm)"}
    shutil.copy(overlay, PUB / f"ep_{eid}_overlay.mp4")
    pub["overlay"] = f"/files/ep_{eid}_overlay.mp4?v={int(time.time())}"
    if twin:
        shutil.copy(twin, PUB / f"ep_{eid}_twin.mp4")
        pub["twin"] = f"/files/ep_{eid}_twin.mp4?v={int(time.time())}"
    (PUB / f"ep_{eid}.json").write_text(json.dumps(pub))
    publish_index()
    step("done (video only)")
    return pub


def process(eid: str) -> dict:
    return process_video_only(eid)


def video_duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def sim_twin(eid: str, aligned_p: Path, task: str) -> Path | None:
    """Digital twin = 1:1 copy of the tracked hand (sim.retarget.retarget_exact), one nominal cup variant,
    rendered at 30 fps for the full video length. No scripted fallback: if the hand cannot be copied, no twin."""
    import cv2
    from sim.augment import run_augment
    from sim.retarget import HZ, retarget_exact
    hands_p = HANDS / f"{eid}_hands.parquet"
    video = find_video(eid)
    if video is None or not hands_p.exists():
        print("twin skipped: needs the video and a hands parquet for an exact copy", flush=True)
        return None
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    imu = None
    try:
        imu = pd.read_parquet(aligned_p) if aligned_p.exists() else None
    except Exception:
        imu = None
    try:
        rt = retarget_exact(pd.read_parquet(hands_p), w, h, video_path=str(video), imu_df=imu)
    except Exception as e:
        print(f"twin skipped (no exact copy): {e}", flush=True)
        return None
    vdur = video_duration(video)
    if vdur > float(rt["t_s"].iloc[-1]) + 1.0 / HZ:  # hold the last pose to the end of the clip
        attrs = dict(rt.attrs)
        tail = np.arange(float(rt["t_s"].iloc[-1]) + 1.0 / HZ, vdur + 1e-9, 1.0 / HZ)
        pad = pd.DataFrame([rt.iloc[-1]] * len(tail)).reset_index(drop=True)
        pad["t_s"] = tail
        rt = pd.concat([rt, pad], ignore_index=True)
        rt.attrs.update(attrs)
    out = SIM / "twins" / eid
    if out.exists():
        shutil.rmtree(out)
    m = run_augment(str(aligned_p), n=1, render_n=1, out_dir=str(out), lerobot_out=None, task=task, seed=7,
                    exact=rt, nominal=True)
    v = m["variants"][0]
    print(f"twin exact copy ({rt.attrs.get('retarget_mode')}, valid {rt.attrs.get('valid_frac', 0):.0%}): "
          f"{'grasp OK' if v['success'] else 'no grasp'}, lift {v['max_lift_m'] * 100:.1f} cm", flush=True)
    vids = sorted((out / "videos").glob("sim_v*.mp4"))
    return vids[0] if vids else None


def publish_index() -> None:
    """episode_<i>.json (Episode screen rail order) + dataset_stats.json."""
    eps = []
    ids = sorted({p.stem[3:] for p in PUB.glob("ep_*.json")})
    for i, eid in enumerate(ids):
        d = json.loads((PUB / f"ep_{eid}.json").read_text())
        (PUB / f"episode_{i}.json").write_text(json.dumps(d))
        eps.append({"n": i, "id": eid, "task": d["task"], "duration": d["duration"]})
    for old in PUB.glob("episode_*.json"):
        try:
            if int(old.stem.split("_")[1]) >= len(eps):
                old.unlink()
        except ValueError:
            pass
    man = SIM / "manifest.json"
    sim = json.loads(man.read_text()) if man.exists() else []
    if isinstance(sim, dict):
        sim = sim.get("variants", [])
    real_s = sum(e["duration"] for e in eps)
    sim_s = 0.0
    for v in sim:
        sim_s += float(v.get("duration_s", 0) or 0)
    if sim and sim_s == 0:
        sim_s = len(sim) * (real_s / max(1, len(eps)) or 10.0)
    size_mb = 0.0
    for d in (DATA / "mimesis_lerobot", DATA / "airtruth_lerobot"):
        if d.exists():
            size_mb += sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
    stats = {"episodes_real": len(eps), "episodes_sim": len(sim), "frames": int(round((real_s + sim_s) * 30)),
             "hours": round((real_s + sim_s) / 3600, 3), "size_mb": round(size_mb, 1), "fps": 30, "episodes": eps,
             "sim_ok": sum(1 for v in sim if v.get("success")), "sim_slip": sum(1 for v in sim if v.get("slipped"))}
    (PUB / "dataset_stats.json").write_text(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- multiply
def multiply(eid: str, n: int, render_n: int, seed: int = 42) -> None:
    if not ((HANDS / f"{eid}_hands.parquet").exists() and find_video(eid)):
        raise SystemExit(f"{eid}: process the video first (python jobs.py process {eid})")
    multiply_video(eid, n, render_n, seed)


def multiply_video(eid: str, n: int, render_n: int, seed: int = 42) -> None:
    """Video-only take: every variant replays the exact 3D hand copy; physics decides OK / SLIP."""
    import cv2
    from sim.augment import run_augment
    video = find_video(eid)
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    for sub in ("episodes", "videos"):
        if (SIM / sub).exists():
            shutil.rmtree(SIM / sub)
    step(f"multiply {eid} (exact hand copy) x{n}")
    print(f'PROGRESS {{"done": 0, "total": {n}, "ok": 0, "slip": 0}}', flush=True)
    run_augment(str(HANDS / f"{eid}_hands.parquet"), n=n, render_n=render_n, out_dir=str(SIM), lerobot_out=None,
                task="pick up cup", seed=seed,
                exact={"hands_parquet": str(HANDS / f"{eid}_hands.parquet"), "video_path": str(video), "width": w, "height": h})
    publish_variants()
    publish_index()


def publish_variants() -> None:
    dst = PUB / "variants"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    lab_p = SIM / "videos" / "labels.json"
    labels = json.loads(lab_p.read_text()) if lab_p.exists() else []
    out = []
    for j, L in enumerate([L for L in labels if L.get("video")]):
        src = SIM / L["video"] if not Path(L["video"]).is_absolute() else Path(L["video"])
        if src.exists():
            shutil.copy(src, dst / f"{j}.mp4")
        out.append({**L, "n": j, "mu": round(float(L.get("mu", 0)), 2)})
    (dst / "labels.json").write_text(json.dumps(out))


# --------------------------------------------------------------------------- metrics
def export() -> dict:
    from export import export_lerobot_dataset
    out = {}
    man = SIM / "manifest.json"
    if man.exists():
        step("export sim variants")
        sim = json.loads(man.read_text())
        sim = sim.get("variants", sim) if isinstance(sim, dict) else sim
        eps = []
        for v in sim:
            pq = SIM / "episodes" / f"sim_v{int(v['variant_idx']):04d}_30fps.parquet"
            vp = v.get("video_path")
            eps.append({"episode_id": v.get("episode_id", pq.stem), "task": v.get("task", "sim"),
                        "aligned_parquet": str(pq), "video_path": vp if vp and Path(vp).exists() else None})
        out["sim"] = export_lerobot_dataset(eps, str(DATA / "mimesis_lerobot"), fps=30.0, repo_id="nullhacks/mimesis-sim-aug")
    real = []
    for eid in raw_episodes():
        al = ALIGNED / f"{eid}.parquet"
        if al.exists():
            v = find_video(eid)
            real.append({"episode_id": eid, "task": episode_meta(eid).get("task", "egocentric_manipulation"),
                         "aligned_parquet": str(al), "video_path": str(v) if v else None})
    if real:
        step("export real episodes")
        out["real"] = export_lerobot_dataset(real, str(DATA / "airtruth_lerobot"), fps=30.0)
    (PUB / "export.json").write_text(json.dumps(out, indent=2, default=str))
    publish_index()
    print(json.dumps(out, default=str), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["process", "process_all", "multiply", "export", "index"])
    ap.add_argument("episode", nargs="?")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--render-n", type=int, default=9)
    a = ap.parse_args()
    t0 = time.time()
    if a.job == "process":
        process(a.episode)
    elif a.job == "process_all":
        for eid in raw_episodes():
            step(f"process {eid}")
            try:
                process(eid)
            except Exception as e:
                print(f"{eid} failed: {e}", flush=True)
    elif a.job == "multiply":
        multiply(a.episode or "latest", a.n, a.render_n)
    elif a.job == "export":
        export()
    elif a.job == "index":
        publish_index()
    step(f"finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
