"""Track the cup in the chest-cam video and place it on the calibrated table, frame by frame.

MediaPipe ObjectDetector (EfficientDet-Lite0, COCO "cup"/"bowl"/"vase"/"bottle"/"wine glass") per frame ->
bottom-centre of the box -> camera ray -> intersection with the table plane (A4 calibration from sim.hand3d).

    python -m sim.cup_track data/videos/ep_173510_cfr.mp4 --out cup.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "efficientdet_lite0.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/"
             "float16/latest/efficientdet_lite0.tflite")
CLASSES = ["cup", "bowl", "vase", "bottle", "wine glass"]


def ensure_model() -> Path:
    if not MODEL.exists():
        import urllib.request
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    return MODEL


def detect_cups(video_path: str, stride: int = 2, max_width: int = 960) -> pd.DataFrame:
    """Per processed frame: t_s, u, v (bottom-centre px in full-res coords), w_px, h_px, score; NaN if none."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.ObjectDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO, score_threshold=0.25, max_results=3,
        category_allowlist=CLASSES)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, idx, prev = [], 0, None
    with vision.ObjectDetector.create_from_options(opts) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                h, w = frame.shape[:2]
                sc = min(1.0, max_width / w)
                small = cv2.resize(frame, (int(w * sc), int(h * sc))) if sc < 1 else frame
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                res = det.detect_for_video(img, int(idx * 1000 / fps))
                best = None
                for d in res.detections:
                    b = d.bounding_box
                    u, v = (b.origin_x + b.width / 2) / sc, (b.origin_y + b.height) / sc
                    s = d.categories[0].score
                    # keep identity: prefer the detection closest to the previous one
                    cost = -s + (0.002 * np.hypot(u - prev[0], v - prev[1]) if prev is not None else 0.0)
                    if best is None or cost < best[0]:
                        best = (cost, u, v, b.width / sc, b.height / sc, s)
                t = idx / fps
                if best:
                    rows.append((t, best[1], best[2], best[3], best[4], best[5]))
                    prev = (best[1], best[2])
                else:
                    rows.append((t, np.nan, np.nan, np.nan, np.nan, 0.0))
            idx += 1
    cap.release()
    return pd.DataFrame(rows, columns=["t_s", "u", "v", "w_px", "h_px", "score"])


def to_table(det: pd.DataFrame, table: dict) -> pd.DataFrame:
    """Ray through the box bottom-centre intersected with the table plane z=0 (table frame, metres)."""
    K = np.asarray(table["K"], float)
    R, t = np.asarray(table["R"], float), np.asarray(table["t"], float).ravel()
    C = -R.T @ t                                    # camera centre in the table frame
    Ki = np.linalg.inv(K)
    xs, ys = np.full(len(det), np.nan), np.full(len(det), np.nan)
    for i, (u, v) in enumerate(zip(det["u"].to_numpy(float), det["v"].to_numpy(float))):
        if not np.isfinite(u):
            continue
        r = R.T @ (Ki @ np.array([u, v, 1.0]))
        if abs(r[2]) < 1e-6:
            continue
        s = -C[2] / r[2]
        if s <= 0:
            continue
        p = C + s * r
        xs[i], ys[i] = p[0], p[1]
    out = det[["t_s", "score", "h_px"]].copy()
    out["cup_x"], out["cup_y"] = xs, ys
    # light smoothing over detected frames; short gaps (<0.4 s) interpolated
    for c in ("cup_x", "cup_y"):
        s = pd.Series(out[c].to_numpy(float))
        s = s.rolling(5, center=True, min_periods=1).median()
        out[c] = s.interpolate(limit=6, limit_area="inside").to_numpy()
    return out


def track_cup(video_path: str, table: dict | None = None, width: int | None = None, height: int | None = None):
    from sim import hand3d
    if table is None:
        table = hand3d.calibrate_table(str(video_path))
        if table is None:
            cap = cv2.VideoCapture(str(video_path))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            table = hand3d.default_table_pose(hand3d.camera_matrix(width or w, height or h))
    return to_table(detect_cups(video_path), table)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out")
    a = ap.parse_args()
    tr = track_cup(a.video)
    ok = np.isfinite(tr["cup_x"])
    print(f"cup found in {ok.mean():.0%} of frames; x {np.nanmin(tr.cup_x):.3f}..{np.nanmax(tr.cup_x):.3f} m, "
          f"y {np.nanmin(tr.cup_y):.3f}..{np.nanmax(tr.cup_y):.3f} m")
    if a.out:
        tr.to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
