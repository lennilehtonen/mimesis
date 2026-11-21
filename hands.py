"""MediaPipe Tasks HandLandmarker (VIDEO mode, 2 hands) -> per-frame landmarks parquet + amber overlay MP4.

    python hands.py data/videos/ep_001.mov --out data/hands/ep_001_hands.parquet \
        --out-video data/hands/ep_001_overlay.mp4
    python hands.py --live --camera 0

Handedness is MediaPipe's label as-is (it assumes a mirrored selfie image). For a non-mirrored
chest camera pass --swap-handedness so "left_*" is the wearer's left hand.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
AMBER_BGR = (71, 181, 255)  # #FFB547
INFER_MAX_W = 960
SIDES = ("left", "right")
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def ensure_hand_model(path: Path = MODEL_PATH) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {path}", file=sys.stderr)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(path)
    return path


def make_landmarker(num_hands: int = 2, live: bool = False):
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_hand_model())),
        running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(opts)


def columns() -> list[str]:
    cols = ["frame_idx", "t_s"]
    for s in SIDES:
        cols.append(f"{s}_present")
        for i in range(21):
            cols += [f"{s}_x{i}", f"{s}_y{i}", f"{s}_z{i}"]
        for i in range(21):
            cols += [f"{s}_wx{i}", f"{s}_wy{i}", f"{s}_wz{i}"]
        cols += [f"{s}_wrist_speed_px_s", f"{s}_index_speed_px_s"]
    return cols


def _detect(landmarker, frame_bgr: np.ndarray, ts_ms: int):
    import mediapipe as mp
    h, w = frame_bgr.shape[:2]
    small = frame_bgr
    if w > INFER_MAX_W:
        small = cv2.resize(frame_bgr, (INFER_MAX_W, int(round(h * INFER_MAX_W / w))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    return landmarker.detect_for_video(img, ts_ms)


def _result_to_row(res, frame_idx: int, t_s: float, swap: bool) -> dict:
    row = {"frame_idx": frame_idx, "t_s": t_s, "left_present": 0, "right_present": 0}
    for s in SIDES:
        for i in range(21):
            for k in ("x", "y", "z", "wx", "wy", "wz"):
                row[f"{s}_{k}{i}"] = np.nan
    lms = res.hand_landmarks or []
    wlms = res.hand_world_landmarks or []
    hd = res.handedness or []
    for h_i, lm in enumerate(lms):
        label = hd[h_i][0].category_name.lower() if h_i < len(hd) and hd[h_i] else "right"
        side = label if label in SIDES else "right"
        if swap:
            side = "left" if side == "right" else "right"
        if row[f"{side}_present"]:
            side = "left" if side == "right" else "right"
            if row[f"{side}_present"]:
                continue
        row[f"{side}_present"] = 1
        for i, p in enumerate(lm[:21]):
            row[f"{side}_x{i}"], row[f"{side}_y{i}"], row[f"{side}_z{i}"] = p.x, p.y, p.z
        if h_i < len(wlms):
            for i, p in enumerate(wlms[h_i][:21]):
                row[f"{side}_wx{i}"], row[f"{side}_wy{i}"], row[f"{side}_wz{i}"] = p.x, p.y, p.z
    return row


def draw_overlay(frame: np.ndarray, row: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    th = max(2, w // 480)
    for s in SIDES:
        if not row.get(f"{s}_present"):
            continue
        pts = [(int(row[f"{s}_x{i}"] * w), int(row[f"{s}_y{i}"] * h)) for i in range(21)]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], AMBER_BGR, th, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(frame, p, th + 2 if i in (0, 4, 8) else th + 1, (255, 255, 255) if i in (0, 4, 8) else AMBER_BGR,
                       -1, cv2.LINE_AA)
        cv2.putText(frame, s[0].upper(), (pts[0][0] + 6, pts[0][1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th,
                    AMBER_BGR, th, cv2.LINE_AA)
    return frame


def add_speeds(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    t = df["t_s"].to_numpy(float)
    dt = np.diff(t, prepend=np.nan)
    for s in SIDES:
        for name, i in (("wrist", 0), ("index", 8)):
            x = df[f"{s}_x{i}"].to_numpy(float) * width
            y = df[f"{s}_y{i}"].to_numpy(float) * height
            v = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) / dt
            df[f"{s}_{name}_speed_px_s"] = v
    return df


def transcode_h264(src: Path, dst: Path, fps: float | None = None):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart"]
    if fps:
        cmd += ["-r", f"{fps:.3f}"]
    subprocess.run(cmd + [str(dst)], check=True)


def run_hands(video_path, out_parquet=None, out_video=None, stride: int = 1, swap_handedness: bool = False,
              max_frames: int | None = None, verbose: bool = True) -> pd.DataFrame:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(stride))
    landmarker = make_landmarker()
    writer, tmp_video, W, H = None, None, 0, 0
    rows, idx, t0 = [], 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and idx >= max_frames:
                break
            if idx % stride == 0:
                H, W = frame.shape[:2]
                t_s = idx / fps
                res = _detect(landmarker, frame, int(round(t_s * 1000)))
                row = _result_to_row(res, idx, t_s, swap_handedness)
                rows.append(row)
                if out_video:
                    if writer is None:
                        tmp_video = Path(out_video).with_suffix(".tmp.mp4")
                        tmp_video.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))
                    writer.write(draw_overlay(frame, row))
                if verbose and len(rows) % 100 == 0:
                    print(f"  frame {idx}/{total} ({len(rows) / (time.time() - t0):.1f} fps)", file=sys.stderr)
            idx += 1
    finally:
        cap.release()
        landmarker.close()
        if writer is not None:
            writer.release()
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns())
    df = add_speeds(df, W or 1, H or 1)
    df = df[columns()]
    df["frame_idx"] = df["frame_idx"].astype(np.int64)
    for s in SIDES:
        df[f"{s}_present"] = df[f"{s}_present"].astype(np.int64)
    if out_parquet:
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
    if out_video and tmp_video is not None:
        transcode_h264(tmp_video, Path(out_video))
        tmp_video.unlink(missing_ok=True)
    if verbose:
        print(f"{len(df)} frames, left present {int(df['left_present'].sum())}, right present "
              f"{int(df['right_present'].sum())} ({time.time() - t0:.1f}s)", file=sys.stderr)
    return df


def run_live(camera: int = 0, swap_handedness: bool = False, out_parquet=None):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera}")
    landmarker = make_landmarker()
    use_cv = True
    plt = fig = im = None
    rows, t0, idx = [], time.time(), 0
    print("live hands: press q (or close the window / Ctrl-C) to quit", file=sys.stderr)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_s = time.time() - t0
            res = _detect(landmarker, frame, int(t_s * 1000))
            row = _result_to_row(res, idx, t_s, swap_handedness)
            rows.append(row)
            vis = draw_overlay(frame, row)
            cv2.putText(vis, f"{1.0 / max(1e-3, (time.time() - t0) / (idx + 1)):.1f} fps", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBER_BGR, 2)
            if use_cv:
                try:
                    cv2.imshow("MIMESIS hands", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    use_cv = False  # headless OpenCV build: fall back to matplotlib
                    import matplotlib.pyplot as plt
                    plt.ion()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.axis("off")
                    im = ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            else:
                if not plt.fignum_exists(fig.number):
                    break
                im.set_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                plt.pause(0.001)
            idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    if out_parquet and rows:
        h, w = frame.shape[:2]
        df = add_speeds(pd.DataFrame(rows), w, h)[columns()]
        df.to_parquet(out_parquet, index=False)
        print(f"wrote {len(df)} rows -> {out_parquet}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--out", help="output parquet (default data/hands/<stem>_hands.parquet)")
    ap.add_argument("--out-video", help="H.264 overlay MP4")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--swap-handedness", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    if a.live:
        run_live(a.camera, a.swap_handedness, a.out)
        return
    if not a.video:
        ap.error("video path required (or --live)")
    out = a.out or str(ROOT / "data" / "hands" / f"{Path(a.video).stem}_hands.parquet")
    run_hands(a.video, out, a.out_video, a.stride, a.swap_handedness, a.max_frames)
    print(f"-> {out}" + (f", {a.out_video}" if a.out_video else ""))


if __name__ == "__main__":
    main()
