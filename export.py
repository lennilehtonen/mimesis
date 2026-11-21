"""Package aligned (real) or simulated episodes as a LeRobot v3-style dataset folder.

    python export.py --aligned-dir data/aligned --video-dir data/videos --out-dir data/airtruth_lerobot \
        --fps 30.0 --repo-id nullhacks/airtruth-ego-imu
    python export.py --manifest episodes.json --out-dir data/airtruth_lerobot

Manifest: JSON list of {episode_id, task, aligned_parquet, video_path}. Schema: docs/DATASET.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
CODEBASE_VERSION = "v3.0-airtruth-hack"
VIDEO_KEY = "observation.images.egocentric"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/ep_{episode_index:04d}.mp4"
STATE_NAMES = ["ax", "ay", "az", "gx", "gy", "gz", "qx", "qy", "qz", "qw", "grav_x", "grav_y", "grav_z", "jerk_mag"]
HAND_NAMES = []
for _s in ("left", "right"):
    HAND_NAMES += [f"{_s}_x0", f"{_s}_y0", f"{_s}_z0", f"{_s}_x4", f"{_s}_y4", f"{_s}_z4",
                   f"{_s}_x8", f"{_s}_y8", f"{_s}_z8", f"{_s}_wrist_speed_px_s"]
HAND_NAMES = ["left_present", "right_present"] + HAND_NAMES[:10] + HAND_NAMES[10:]
DEFAULT_TASK = "egocentric_manipulation"
VIDEO_EXTS = (".mp4", ".mov", ".MOV", ".MP4", ".m4v", ".avi", ".mkv")


def _time(df: pd.DataFrame, fps: float) -> np.ndarray:
    for c in ("t_s", "t_rel_s"):
        if c in df:
            return df[c].to_numpy(float)
    if "timestamp_ns" in df:
        ns = df["timestamp_ns"].to_numpy(np.int64)
        return (ns - ns[0]) / 1e9
    return np.arange(len(df)) / fps


def _extract_state(df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    cols = []
    for c in STATE_NAMES:
        if c in df:
            cols.append(df[c].to_numpy(float))
        elif c == "jerk_mag" and {"ax", "ay", "az"} <= set(df):
            a = df[["ax", "ay", "az"]].to_numpy(float)
            dt = np.diff(t, prepend=t[0] - 0.01) if len(t) else t
            dt = np.where(dt <= 1e-4, 0.01, dt)
            cols.append(np.linalg.norm(np.diff(a, axis=0, prepend=a[:1]), axis=1) / dt)
        elif c == "qw":
            cols.append(np.ones(len(df)))
        else:
            cols.append(np.zeros(len(df)))
    return np.nan_to_num(np.stack(cols, axis=1)).astype(np.float32)


def _extract_hand_summary(df: pd.DataFrame) -> np.ndarray:
    cols = [df[c].to_numpy(float) if c in df else np.zeros(len(df)) for c in HAND_NAMES]
    return np.nan_to_num(np.stack(cols, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _int_col(df: pd.DataFrame, c: str) -> np.ndarray:
    return np.nan_to_num(df[c].to_numpy(float)).astype(np.int32) if c in df else np.zeros(len(df), np.int32)


def _video_codec(p: Path) -> str:
    try:
        return subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                               "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _package_video(src: Path, dst: Path, fps: float):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".mp4" and _video_codec(src) == "h264":
        shutil.copy2(src, dst)
        return
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-r", f"{fps:g}", "-c:a", "aac", "-movflags", "+faststart", str(dst)],
                   check=True)


def _dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        z = [0.0] * x.shape[1]
        return {"mean": z, "std": [1e-6] * x.shape[1], "min": z, "max": z, "count": [0]}
    return {"mean": x.mean(0).astype(float).tolist(), "std": np.maximum(x.std(0), 1e-6).astype(float).tolist(),
            "min": x.min(0).astype(float).tolist(), "max": x.max(0).astype(float).tolist(), "count": [int(len(x))]}


def export_lerobot_dataset(episodes: list[dict], out_dir, fps: float = 30.0,
                           repo_id: str = "nullhacks/airtruth-ego-imu") -> dict:
    """episodes: [{episode_id, task, aligned_parquet, video_path?}]. Returns summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "meta", "videos"):  # fresh export: drop stale shards / videos from a previous run
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)

    tasks: dict[str, int] = {}
    frames, ep_rows = [], []
    all_state, all_hands = [], []
    gidx, n_videos = 0, 0
    for ep_i, ep in enumerate(episodes):
        df = pd.read_parquet(ep["aligned_parquet"]).reset_index(drop=True)
        t = _time(df, fps)
        if len(t) > 2:
            dt = float(np.median(np.diff(t)))
            step = int(round((1.0 / fps) / dt)) if dt > 0 else 1
            if step > 1:  # e.g. 100 Hz sim IMU -> ~33 Hz rows
                df = df.iloc[::step].reset_index(drop=True)
                t = t[::step]
        task = ep.get("task") or DEFAULT_TASK
        task_index = tasks.setdefault(task, len(tasks))
        n = len(df)
        state = _extract_state(df, t)
        hands = _extract_hand_summary(df)
        cid = _int_col(df, "contact_event_id")
        cflag = _int_col(df, "contact_flag") if "contact_flag" in df else (cid >= 2).astype(np.int32)
        slip = np.nan_to_num(df["slip_score"].to_numpy(float)).astype(np.float32) if "slip_score" in df \
            else np.zeros(n, np.float32)
        ts = (t - t[0]) if (len(t) and "t_s" not in df) else t
        frames.append(pd.DataFrame({
            "index": np.arange(gidx, gidx + n, dtype=np.int64),
            "episode_index": np.full(n, ep_i, np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "task_index": np.full(n, task_index, np.int64),
            "timestamp": np.asarray(ts, np.float32),
            "observation.state": [r.tolist() for r in state],
            "observation.hands": [r.tolist() for r in hands],
            "observation.contact_event_id": cid,
            "observation.contact_flag": cflag,
            "observation.slip_score": slip,
        }))
        all_state.append(state)
        all_hands.append(hands)
        vrel = None
        vp = ep.get("video_path")
        if vp and Path(vp).exists():
            vrel = VIDEO_PATH.format(video_key=VIDEO_KEY, chunk_index=0, episode_index=ep_i)
            _package_video(Path(vp), out / vrel, fps)
            n_videos += 1
        ep_rows.append({"episode_index": ep_i, "episode_id": str(ep.get("episode_id", f"ep_{ep_i:04d}")),
                        "task": task, "task_index": task_index, "length": n,
                        "duration_s": float(t[-1] - t[0]) if n > 1 else 0.0,
                        "contact_frames": int(cflag.sum()), "data_from_index": gidx, "data_to_index": gidx + n,
                        "video_path": vrel})
        gidx += n

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    data.to_parquet(out / DATA_PATH.format(chunk_index=0, file_index=0), index=False)
    pd.DataFrame(ep_rows).to_parquet(out / "meta" / "episodes.parquet", index=False)
    pd.DataFrame({"task_index": list(tasks.values()), "task": list(tasks.keys())}).to_parquet(
        out / "meta" / "tasks.parquet", index=False)
    st = np.concatenate(all_state) if all_state else np.zeros((0, 14), np.float32)
    hd = np.concatenate(all_hands) if all_hands else np.zeros((0, 22), np.float32)
    (out / "meta" / "stats.json").write_text(json.dumps(
        {"observation.state": _stats(st), "observation.hands": _stats(hd)}, indent=2))
    n_ep, n_fr = len(ep_rows), int(gidx)
    info = {
        "codebase_version": CODEBASE_VERSION, "repo_id": repo_id, "robot_type": "human_egocentric_wrist_imu",
        "fps": float(fps), "total_episodes": n_ep, "total_frames": n_fr, "total_tasks": len(tasks),
        "total_videos": n_videos, "total_chunks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500, "splits": {"train": f"0:{n_ep}"},
        "data_path": DATA_PATH, "video_path": VIDEO_PATH,
        "features": {
            VIDEO_KEY: {"dtype": "video", "shape": [1080, 1920, 3], "names": ["height", "width", "channels"],
                        "info": {"video.fps": float(fps), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                 "video.is_depth_map": False, "has_audio": False}},
            "observation.state": {"dtype": "float32", "shape": [14], "names": STATE_NAMES},
            "observation.hands": {"dtype": "float32", "shape": [22], "names": HAND_NAMES},
            "observation.contact_event_id": {"dtype": "int32", "shape": [1], "names": None,
                                             "labels": {"0": "idle", "1": "motion", "2": "grab_onset",
                                                        "3": "impact_or_tap", "4": "set_down", "5": "slip_or_scrape"}},
            "observation.contact_flag": {"dtype": "int32", "shape": [1], "names": None},
            "observation.slip_score": {"dtype": "float32", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out / "README.md").write_text(
        f"# {repo_id}\n\nLeRobot v3-style dataset (`{CODEBASE_VERSION}`), robot_type `human_egocentric_wrist_imu`.\n\n"
        f"- episodes: {n_ep}\n- frames: {n_fr}\n- fps: {fps:g}\n- tasks: {', '.join(tasks) or '-'}\n- videos: {n_videos}\n\n"
        "Features: `observation.state` (14: ax..az, gx..gz, qx..qw, grav_x..z, jerk_mag), `observation.hands` (22), "
        "`observation.contact_event_id`, `observation.contact_flag`, `observation.slip_score`, "
        f"`{VIDEO_KEY}`.\n\nGenerated by MIMESIS `export.py`. Schema: docs/DATASET.md.\n")
    summary = {"episodes": n_ep, "frames": n_fr, "videos": n_videos, "tasks": list(tasks),
               "out_dir": str(out), "size_mb": 0.0}
    summary["size_mb"] = round(_dir_size_mb(out), 3)
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def discover_episodes(aligned_dir, video_dir=None) -> list[dict]:
    eps = []
    for p in sorted(Path(aligned_dir).glob("*.parquet")):
        sj = p.with_name(p.stem + ".sync.json")
        meta = json.loads(sj.read_text()) if sj.exists() else {}
        video = None
        if video_dir:
            for ext in VIDEO_EXTS:
                c = Path(video_dir) / (p.stem + ext)
                if c.exists():
                    video = str(c)
                    break
        if video is None and meta.get("video") and Path(meta["video"]).exists():
            video = meta["video"]
        eps.append({"episode_id": p.stem, "task": meta.get("task", DEFAULT_TASK), "aligned_parquet": str(p),
                    "video_path": video})
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned-dir")
    ap.add_argument("--video-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--repo-id", default="nullhacks/airtruth-ego-imu")
    a = ap.parse_args()
    if a.manifest:
        m = json.loads(Path(a.manifest).read_text())
        episodes = m["episodes"] if isinstance(m, dict) else m
    elif a.aligned_dir:
        episodes = discover_episodes(a.aligned_dir, a.video_dir)
    else:
        ap.error("pass --aligned-dir or --manifest")
    if not episodes:
        ap.error("no episodes found")
    s = export_lerobot_dataset(episodes, a.out_dir, a.fps, a.repo_id)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
