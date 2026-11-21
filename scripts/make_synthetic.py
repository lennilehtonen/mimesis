"""Synthetic episode generator for regression tests.

Writes <out-dir>/ep_synth.parquet (raw IMU, 100 Hz, receiver.py format) and
<out-dir>/ep_synth.mp4 (640x360, 30 fps, AAC audio with 3 clap bursts).
IMU time = video time + offset.

    python scripts/make_synthetic.py --out-dir /tmp/synth --offset 0.8 --duration 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
CLAPS_VIDEO_S = (0.6, 1.3, 2.1)


def _impulse(a: np.ndarray, t: np.ndarray, t0: float, amp: float, axis_w=(1.0, 0.4, 0.2), tau=0.03):
    k = int(round(t0 / (t[1] - t[0])))  # avoid float searchsorted off-by-one
    w = np.asarray(axis_w) / np.linalg.norm(axis_w)
    dt = t[k:] - t[k]
    env = amp * np.exp(-dt / tau) * np.cos(2 * np.pi * 15 * dt)
    n = min(len(env), 40)
    a[k:k + n] += env[:n, None] * w[None, :]


def make_imu(duration: float, offset: float, hz: float = 100.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    imu_dur = duration + offset + 1.0
    n = int(imu_dur * hz)
    t = np.arange(n) / hz
    a = rng.normal(0, 0.03, (n, 3))
    g = rng.normal(0, 0.01, (n, 3))
    # claps (IMU clock)
    for tc in CLAPS_VIDEO_S:
        _impulse(a, t, tc + offset, 14.0)
        _impulse(g, t, tc + offset, 2.0, (0.2, 1.0, 0.3), tau=0.05)
    # reach motion before grab (video 3.4..4.0)
    base = offset
    def bump(t0, t1, amp, axis):
        m = (t >= t0 + base) & (t < t1 + base)
        ph = (t[m] - t0 - base) / (t1 - t0)
        a[m, axis] += amp * np.sin(2 * np.pi * ph)
        g[m, (axis + 1) % 3] += 0.6 * amp / 2 * np.sin(np.pi * ph)
    # still hand before the grab so the detector labels it grab_onset
    _impulse(a, t, 4.0 + base, 6.0, (0.3, 0.3, 1.0))  # grab
    bump(4.2, 6.2, 0.9, 1)
    _impulse(a, t, 5.5 + base, 9.0, (0.1, 0.2, 1.0))  # tap
    m = (t >= 6.3 + base) & (t < 7.1 + base)            # scrape / slip vibration
    # |a| stays < 3.5 m/s^2 so it is not an impact, but high-pass RMS > 0.75 -> slip_or_scrape
    a[m] += rng.normal(0, 0.25, (m.sum(), 3)) + np.array([1.4, 1.0, 0.8])[None, :] * np.sin(2 * np.pi * 13 * t[m])[:, None]
    bump(7.3, 8.3, 1.0, 2)
    _impulse(a, t, 8.5 + base, 8.0, (0.2, 0.1, 1.0))  # set-down, then still
    roll = 0.1 * np.sin(0.3 * t)
    pitch = 0.05 * np.cos(0.2 * t)
    yaw = 0.02 * t
    cr, sr, cp, sp, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    grav = np.stack([-9.81 * np.sin(pitch), 9.81 * np.sin(roll) * np.cos(pitch), 9.81 * np.cos(roll) * np.cos(pitch)], 1)
    t0_ns = 1_763_700_000_000_000_000
    df = pd.DataFrame({
        "timestamp_ns": (t0_ns + np.round(t * 1e9)).astype(np.int64), "t_rel_s": t,
        "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2], "gx": g[:, 0], "gy": g[:, 1], "gz": g[:, 2],
        "grav_x": grav[:, 0], "grav_y": grav[:, 1], "grav_z": grav[:, 2],
        "qx": qx, "qy": qy, "qz": qz, "qw": qw, "roll": roll, "pitch": pitch, "yaw": yaw,
    })
    df["accel_mag"] = np.linalg.norm(a, axis=1)
    df["gyro_mag"] = np.linalg.norm(g, axis=1)
    return df


def make_audio(path: Path, duration: float, sr: int = 44100, seed: int = 1):
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    x = rng.normal(0, 0.005, n)
    for tc in CLAPS_VIDEO_S:
        k = int(tc * sr)
        m = int(0.04 * sr)
        env = np.exp(-np.arange(m) / (0.008 * sr))
        x[k:k + m] += 0.8 * env * rng.normal(0, 1, m)
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def make_video(path: Path, wav: Path, duration: float, fps: float = 30.0, W: int = 640, H: int = 360):
    tmp = path.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    nf = int(round(duration * fps))
    for i in range(nf):
        ts = i / fps
        img = np.full((H, W, 3), (40, 32, 28), np.uint8)
        cv2.rectangle(img, (0, int(H * 0.7)), (W, H), (70, 90, 110), -1)  # table
        cx = int(W / 2 + 200 * np.sin(2 * np.pi * ts / 6.0))
        cy = int(H / 2 + 60 * np.cos(2 * np.pi * ts / 3.0))
        if any(abs(ts - tc) < 0.05 for tc in CLAPS_VIDEO_S):
            cv2.circle(img, (W // 2, H // 3), 60, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), 28, (150, 190, 230), -1)
        cv2.putText(img, f"t={ts:5.2f}s f={i}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        vw.write(img)
    vw.release()
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(wav), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(path)], check=True)
    tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--offset", type=float, default=0.8, help="IMU time = video time + offset (s)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if min(CLAPS_VIDEO_S) + a.offset < 0.1:
        ap.error(f"offset too negative: first clap would land before IMU start (need offset >= {0.1 - min(CLAPS_VIDEO_S):.1f})")
    df = make_imu(a.duration, a.offset, seed=a.seed)
    df.to_parquet(out / "ep_synth.parquet", index=False)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        make_audio(wav, a.duration)
        make_video(out / "ep_synth.mp4", wav, a.duration)
    meta = {"episode_id": "ep_synth", "task": "synthetic_pick_place", "offset_s": a.offset,
            "claps_video_s": list(CLAPS_VIDEO_S), "duration_s": a.duration}
    (out / "ep_synth_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out/'ep_synth.parquet'} ({len(df)} rows) and {out/'ep_synth.mp4'}; planted offset {a.offset:+.3f}s")


if __name__ == "__main__":
    main()
