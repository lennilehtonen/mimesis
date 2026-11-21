"""Compose the stage video: [real | MuJoCo twin | 3x3 variant grid] with a footer.

    python -m sim.demo_grid --real-video data/hands/ep_001_overlay.mp4 --sim-dir data/sim_aug \
        --out data/mimesis_demo.mp4 --out-png data/mimesis_demo.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sim.augment import transcode_h264  # noqa: E402

PANE_W, PANE_H = 480, 360
CELL_W, CELL_H = 160, 120
FOOTER = 36
FPS = 30.0
GREEN, RED, WHITE = (60, 200, 80), (50, 50, 230), (240, 240, 240)


def read_frames(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    cap.release()
    return frames, fps


def _label(img, text, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def placeholder(text="REAL DEMO (no video)"):
    img = np.full((PANE_H, PANE_W, 3), 28, np.uint8)
    _label(img, text, (40, PANE_H // 2), 0.8)
    return img


def letterbox(frame, w, h):
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def at(frames, k):
    return frames[min(k, len(frames) - 1)] if frames else None


def build(real_video, sim_dir, out, out_png):
    sim_dir = Path(sim_dir)
    man = json.loads((sim_dir / "manifest.json").read_text())
    variants = man["variants"] if isinstance(man, dict) else man
    rendered = []
    for v in variants:
        vp = v.get("video_path")
        if vp and not Path(vp).is_absolute() and not Path(vp).exists():
            vp = str(sim_dir / vp)
        if vp and Path(vp).exists():
            rendered.append((v, vp))
    if not rendered:
        raise SystemExit("manifest has no rendered video_path; run sim.augment with --render-n >= 1")

    twin = next(((v, p) for v, p in rendered if not v.get("slipped")), None)
    if twin is None:
        twin = next(((v, p) for v, p in rendered if v.get("variant_idx") == 0), rendered[0])
    grid = rendered[:9]
    sim_frames = {p: read_frames(p)[0] for _, p in rendered[:9] + [twin]}
    n_frames = max(len(f) for f in sim_frames.values())

    real_frames, real_fps = [], FPS
    if real_video and Path(real_video).exists():
        real_frames, real_fps = read_frames(real_video, max_frames=int(n_frames * 60 / FPS) + 10)
        real_frames = [letterbox(f, PANE_W, PANE_H) for f in real_frames]
    ph = placeholder()

    W, H = PANE_W * 3, PANE_H + FOOTER
    n_ok = sum(1 for v in variants if v.get("success"))
    n_slip = sum(1 for v in variants if v.get("slipped"))
    footer_txt = (f"MIMESIS  |  1 real demo -> {len(variants)} physics variants  |  "
                  f"{n_ok} success, {n_slip} slip  |  task: {man.get('task', '') if isinstance(man, dict) else ''}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    mid = n_frames // 2
    for k in range(n_frames):
        canvas = np.zeros((H, W, 3), np.uint8)
        if real_frames:
            rf = at(real_frames, int(k / FPS * real_fps))
            canvas[:PANE_H, :PANE_W] = rf
        else:
            canvas[:PANE_H, :PANE_W] = ph
        _label(canvas, "REAL", (10, 24), 0.7)
        tv, tp = twin
        canvas[:PANE_H, PANE_W:2 * PANE_W] = cv2.resize(at(sim_frames[tp], k), (PANE_W, PANE_H))
        _label(canvas, f"MUJOCO TWIN  v{tv['variant_idx']:02d} {tv['obj_type']}", (PANE_W + 10, 24), 0.7)
        for gi in range(9):
            r, c = divmod(gi, 3)
            x0, y0 = 2 * PANE_W + c * CELL_W, r * CELL_H
            if gi < len(grid):
                v, p = grid[gi]
                cell = cv2.resize(at(sim_frames[p], k), (CELL_W, CELL_H))
                ok = not v.get("slipped")
                col = GREEN if ok else RED
                cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), col, 3)
                _label(cell, "OK" if ok else "SLIP!", (8, 20), 0.55, col, 2)
                _label(cell, f"mu={v['obj_friction']:.2f} {int(v['obj_mass_kg'] * 1000)}g", (8, CELL_H - 10), 0.4)
            else:
                cell = np.full((CELL_H, CELL_W, 3), 20, np.uint8)
            canvas[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell
        _label(canvas, footer_txt, (12, PANE_H + 24), 0.55)
        wr.write(canvas)
        if k == mid and out_png:
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_png), canvas)
    wr.release()
    h264 = transcode_h264(tmp, out)
    print(f"{n_frames} frames {W}x{H} ({'H.264' if h264 else 'mp4v'}) -> {out}" + (f", {out_png}" if out_png else ""))
    return str(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-video")
    ap.add_argument("--sim-dir", default="data/sim_aug")
    ap.add_argument("--out", default="data/mimesis_demo.mp4")
    ap.add_argument("--out-png", default="data/mimesis_demo.png")
    a = ap.parse_args()
    build(a.real_video, a.sim_dir, a.out, a.out_png)


if __name__ == "__main__":
    main()
