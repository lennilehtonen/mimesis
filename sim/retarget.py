"""Retarget a real (aligned or raw IMU) episode into a 100 Hz mocap + gripper trajectory.

    python -m sim.retarget data/aligned/ep_001.parquet --out data/sim_aug/ep_001_retarget.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from contact_detect import detect_contacts, time_axis  # noqa: E402

GRIP_OPEN = -0.002
GRIP_CLOSED = 0.032
HZ = 100.0
IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def read_any(path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _episode_time(df: pd.DataFrame) -> np.ndarray:
    # aligned files carry the video clock t_s; raw IMU has t_rel_s / timestamp_ns
    if "t_s" in df and "t_rel_s" not in df:
        t = df["t_s"].to_numpy(float)
        return t - t[0]
    return time_axis(df)


def _interp(t_src, y, t_dst):
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(t_src)
    if ok.sum() == 0:
        return np.full(len(t_dst), np.nan)
    if ok.sum() == 1:
        return np.full(len(t_dst), y[ok][0])
    return np.interp(t_dst, t_src[ok], y[ok])


def _nearest(t_src, y, t_dst):
    idx = np.clip(np.searchsorted(t_src, t_dst), 0, len(t_src) - 1)
    return np.asarray(y)[idx]


def _cos_blend(t, knots):
    """knots: list of (time, value-array). Cosine-eased piecewise interpolation."""
    ts = np.array([k[0] for k in knots], float)
    vs = np.array([k[1] for k in knots], float)
    out = np.empty((len(t), vs.shape[1]))
    for i, tt in enumerate(t):
        if tt <= ts[0]:
            out[i] = vs[0]
            continue
        if tt >= ts[-1]:
            out[i] = vs[-1]
            continue
        j = int(np.searchsorted(ts, tt, side="right") - 1)
        span = max(ts[j + 1] - ts[j], 1e-6)
        u = (tt - ts[j]) / span
        w = 0.5 - 0.5 * np.cos(np.pi * u)
        out[i] = vs[j] * (1 - w) + vs[j + 1] * w
    return out


def _mediapipe_side(df: pd.DataFrame):
    n = len(df)
    need = max(5, int(0.1 * n))
    best, best_cnt = None, 0
    for side in ("right", "left"):
        c = f"{side}_x0"
        if c in df:
            cnt = int(np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(float)).sum())
            if cnt > need and cnt > best_cnt:
                best, best_cnt = side, cnt
    return best


def _orientation(df, t_src, t):
    n = len(t)
    if not {"roll", "pitch", "yaw"} <= set(df):
        return np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    eul = []
    for c, lim in (("roll", 0.45), ("pitch", 0.45), ("yaw", 0.65)):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        v = np.unwrap(np.nan_to_num(v, nan=np.nanmean(v) if np.isfinite(v).any() else 0.0))
        v = _interp(t_src, v, t)
        eul.append(np.clip(v - v[0], -lim, lim))
    q = Rotation.from_euler("xyz", np.stack(eul, axis=1)).as_quat()  # x y z w
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def _mediapipe_path(df, side, t_src, t):
    def col(name):
        c = f"{side}_{name}"
        if c not in df:
            return np.full(len(t), np.nan)
        return _interp(t_src, pd.to_numeric(df[c], errors="coerce").to_numpy(float), t)

    wx = pd.Series(col("x0")).rolling(15, center=True, min_periods=1).mean().to_numpy()
    wy = pd.Series(col("y0")).rolling(15, center=True, min_periods=1).mean().to_numpy()
    wx = pd.Series(wx).ffill().bfill().to_numpy()
    wy = pd.Series(wy).ffill().bfill().to_numpy()
    x = np.clip((wx - wx[0]) * 0.40, -0.16, 0.16)
    y = np.clip((0.65 - wy) * 0.35 - 0.10, -0.22, 0.14)
    z = np.clip(0.24 - (wy - np.nanmin(wy)) * 0.25, 0.085, 0.24)
    d = np.hypot(col("x4") - col("x8"), col("y4") - col("y8"))
    d = pd.Series(d).rolling(9, center=True, min_periods=1).mean().ffill().bfill().to_numpy()
    if np.isfinite(d).any():
        lo, hi = np.nanpercentile(d, 5), np.nanpercentile(d, 95)
        u = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)  # 0 = pinched
        grip = GRIP_CLOSED - u * (GRIP_CLOSED - GRIP_OPEN)
    else:
        grip = np.full(len(t), GRIP_OPEN)
    return np.column_stack([x, y, z]), np.nan_to_num(grip, nan=GRIP_OPEN)


def _anchor_to_object(t, pos, grip):
    """Contact-lock the coarse camera path onto the simulated object.

    The camera->table calibration is rough, so the raw MediaPipe path rarely meets the cup. Find the pinch
    interval [t_g, t_r] from the thumb-index command, translate the whole path so the wrist sits on the grasp
    pose (0, 0, 0.090) at t_g, hold still while the jaw closes, and enforce a z envelope (approach from above,
    lift >= ~11.5 cm while closed, place, retract). x/y keep the camera motion. Returns (None, None) if no pinch.
    """
    mid = 0.5 * (GRIP_OPEN + GRIP_CLOSED)
    closed = grip > mid
    dt = t[1] - t[0] if len(t) > 1 else 0.01
    min_len = int(1.2 / dt)
    best = None
    i, n = 0, len(t)
    while i < n:
        if closed[i]:
            j = i
            while j + 1 < n and closed[j + 1]:
                j += 1
            if j - i >= min_len and (best is None or j - i > best[1] - best[0]):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    if best is None:
        return None, None
    tg, tr = float(t[best[0]]), float(t[best[1]])
    if tg < 1.0 or tr - tg < 1.6:
        return None, None
    zg, zl = 0.090, 0.205
    pos = pos - pos[best[0]] + np.array([0.0, 0.0, zg])
    # hold xy during the close
    hold = (t >= tg - 0.15) & (t <= tg + 0.5)
    w_in = np.clip((t - (tg - 0.6)) / 0.45, 0, 1)
    w_out = np.clip(((tg + 1.0) - t) / 0.5, 0, 1)
    w = np.minimum(w_in, w_out)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    pos[:, :2] = pos[:, :2] * (1 - w[:, None])
    pos[hold, :2] = 0.0
    zf = _cos_blend(t, [(0.0, [0.20]), (tg - 0.7, [0.20]), (tg - 0.15, [zg]), (tg + 0.5, [zg]),
                        (tg + 1.3, [zl]), (tr - 0.85, [zl]), (tr, [zg + 0.002]), (tr + 0.4, [zg + 0.002]),
                        (tr + 1.1, [0.18])])[:, 0]
    closed_win = (t >= tg - 0.15) & (t <= tr + 0.4)
    pos[:, 2] = np.where(closed_win, zf, np.maximum(pos[:, 2], zf))
    pos[:, 2] = np.clip(pos[:, 2], 0.085, 0.26)
    O, C = GRIP_OPEN, GRIP_CLOSED
    grip = _cos_blend(t, [(0.0, [O]), (tg - 0.15, [O]), (tg + 0.35, [C]), (tr, [C]), (tr + 0.4, [O])])[:, 0]
    return pos, grip


def _imu_fallback(df, t_src, t, T):
    t_grab, t_set = 0.30 * T, 0.78 * T
    try:
        if {"ax", "ay", "az"} <= set(df):
            tmp = df.copy()
            tmp["t_rel_s"] = t_src
            _, events = detect_contacts(tmp)
            grabs = [e["t_s"] for e in events if e["event"] == "grab_onset"]
            sets = [e["t_s"] for e in events if e["event"] == "set_down"]
            g = grabs[0] if grabs else None
            s = sets[-1] if sets else None
            if g is not None and s is not None and s > g + 1.0:
                t_grab, t_set = g, s
            elif g is not None and g + 1.5 < 0.78 * T:
                t_grab = g
    except Exception as exc:  # pragma: no cover
        print(f"[retarget] detect_contacts failed ({exc}); using defaults", file=sys.stderr)
    t_grab = max(t_grab, 1.3)
    t_set = max(t_set, t_grab + 2.6)
    O, C = GRIP_OPEN, GRIP_CLOSED
    zg, zl = 0.090, 0.205
    knots = [
        (0.0, [0.0, -0.18, 0.18, O]),
        (t_grab - 0.75, [0.0, 0.0, 0.18, O]),
        (t_grab - 0.15, [0.0, 0.0, zg, O]),
        (t_grab + 0.35, [0.0, 0.0, zg, C]),
        (t_grab + 0.55, [0.0, 0.0, zg, C]),
        (t_grab + 1.35, [0.0, 0.0, zl, C]),
        (t_set - 0.85, [0.13, 0.0, zl, C]),
        (t_set, [0.13, 0.0, zg + 0.002, C]),
        (t_set + 0.40, [0.13, 0.0, zg + 0.002, O]),
        (t_set + 1.10, [0.13, 0.0, 0.18, O]),
    ]
    v = _cos_blend(t, knots)
    pos, grip = v[:, :3], v[:, 3]
    if {"ax", "ay"} <= set(df):
        for k, c in enumerate(("ax", "ay")):
            a = np.nan_to_num(_interp(t_src, pd.to_numeric(df[c], errors="coerce").to_numpy(float), t))
            pos[:, k] += np.clip(a * 0.0012, -0.008, 0.008)
    return pos, grip


def retarget_episode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    t_src = _episode_time(df)
    order = np.argsort(t_src, kind="stable")
    df, t_src = df.iloc[order].reset_index(drop=True), t_src[order]
    T = float(t_src[-1]) if len(t_src) else 0.0
    side = _mediapipe_side(df)
    T_out = T
    if side is None:
        T_out = max(T, 4.0)
    t = np.arange(0.0, T_out + 1e-9, 1.0 / HZ)
    pos = None
    if side is not None:
        pos, grip = _mediapipe_path(df, side, t_src, t)
        pos, grip = _anchor_to_object(t, pos, grip)
        mode = f"mediapipe_{side}"
    if pos is None:
        if side is not None:
            print("[retarget] no usable pinch interval in MediaPipe track; IMU-only fallback", file=sys.stderr)
        pos, grip = _imu_fallback(df, t_src, t, T_out)
        mode = "imu_fallback"
    quat = _orientation(df, t_src, t)
    out = pd.DataFrame({
        "t_s": t,
        "mocap_x": pos[:, 0], "mocap_y": pos[:, 1], "mocap_z": pos[:, 2],
        "mocap_qw": quat[:, 0], "mocap_qx": quat[:, 1], "mocap_qy": quat[:, 2], "mocap_qz": quat[:, 3],
        "grip_ctrl": grip,
    })
    for c in IMU_COLS:
        if c in df:
            out[f"real_{c}"] = _interp(t_src, pd.to_numeric(df[c], errors="coerce").to_numpy(float), t)
    for c in ("contact_flag", "contact_event_id"):
        if c in df:
            out[f"real_{c}"] = _nearest(t_src, pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(), t).astype(int)
    out.attrs["retarget_mode"] = mode
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out")
    a = ap.parse_args()
    p = Path(a.input)
    rt = retarget_episode(read_any(p))
    op = Path(a.out) if a.out else p.with_name(p.stem + "_retarget.parquet")
    op.parent.mkdir(parents=True, exist_ok=True)
    rt.to_parquet(op, index=False)
    print(f"{len(rt)} rows @100 Hz ({rt.attrs.get('retarget_mode')}) -> {op}")


if __name__ == "__main__":
    main()
