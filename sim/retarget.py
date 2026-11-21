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
    palm = np.hypot(col("x0") - col("x9"), col("y0") - col("y9"))  # wrist->middle MCP: scale-free pinch
    d = d / np.where(np.isfinite(palm) & (palm > 1e-3), palm, np.nan)
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
            ok = t[i] >= 1.0 and t[j] - t[i] >= 1.6  # skip runs already closed at clip start
            if ok and j - i >= min_len and (best is None or j - i > best[1] - best[0]):
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
    w_in = np.clip((t - (tg - 1.2)) / 0.45, 0, 1)  # xy centred on the object before the descent starts
    w_out = np.clip(((tg + 1.0) - t) / 0.5, 0, 1)
    w = np.minimum(w_in, w_out)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    pos[:, :2] = pos[:, :2] * (1 - w[:, None])
    pos[hold, :2] = 0.0
    zf = _cos_blend(t, [(0.0, [0.20]), (tg - 0.7, [0.20]), (tg - 0.15, [zg]), (tg + 0.5, [zg]),
                        (tg + 1.3, [zl]), (tr - 0.85, [zl]), (tr, [zg + 0.002]), (tr + 0.4, [zg + 0.002]),
                        (tr + 1.1, [0.18])])[:, 0]
    closed_win = (t >= tg - 0.15) & (t <= tr + 0.4)
    # while the object is held, follow only slow (~1 s) camera xy drift and freeze xy for the set-down, so hand
    # jitter / fast lateral swings don't swing the object out of the jaw or knock it over on release
    win = max(1, int(round(1.0 / dt)))
    sm = pd.DataFrame(pos[:, :2]).rolling(win, center=True, min_periods=1).mean().to_numpy()
    pos[closed_win, :2] = sm[closed_win]
    rel = t >= tr - 0.6  # set down, release and retract straight up (no sweep back into the placed object)
    if rel.any():
        pos[rel, :2] = pos[np.argmax(rel), :2]
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
            # first complete pick-and-place: a grab followed by a set-down 1-8 s later
            g, s = None, None
            for gg in grabs:
                nxt = [x for x in sets if gg + 1.0 < x < gg + 8.0]
                if nxt:
                    g, s = gg, nxt[0]
                    break
            if g is None:
                g = grabs[0] if grabs else None
            if g is not None and s is not None and s > g + 1.0:
                t_grab, t_set = g, s
            elif g is not None and g + 1.5 < 0.78 * T:
                t_grab = g
    except Exception as exc:  # pragma: no cover
        print(f"[retarget] detect_contacts failed ({exc}); using defaults", file=sys.stderr)
    t_grab = max(t_grab, 1.3)
    t_set = max(t_set, t_grab + 4.0)  # a real pick-and-place: lift, carry, place
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


EXACT_GRASP_Z = 0.090
EXACT_MIN_Z = 0.03
EXACT_MAX_TILT = np.deg2rad(0.0)  # two-finger gripper stays vertical; yaw follows the hand


def _limit_tilt(R: Rotation, max_tilt: float) -> Rotation:
    """Swing-twist split about world Z: keep the twist (yaw), clamp the swing (tilt of the tool axis)."""
    zt = R.apply([0.0, 0.0, 1.0])
    zt /= np.linalg.norm(zt, axis=1, keepdims=True) + 1e-12
    ang = np.arccos(np.clip(zt[:, 2], -1.0, 1.0))
    axis = np.cross(np.tile([0.0, 0.0, 1.0], (len(zt), 1)), zt)
    n = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(n > 1e-9, axis / np.maximum(n, 1e-12), [1.0, 0.0, 0.0])
    swing = Rotation.from_rotvec(axis * ang[:, None])
    twist = swing.inv() * R
    return Rotation.from_rotvec(axis * np.minimum(ang, max_tilt)[:, None]) * twist


def retarget_exact(hands_df: pd.DataFrame, width: int, height: int, video_path=None,
                   imu_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """1:1 copy of the tracked human hand (no scripted grasp) on the video clock.

    Raises ValueError if the hand is visible in < 30 % of frames or never closes.
    """
    from sim import hand3d

    table, src = None, "default"
    if video_path is not None:
        try:
            table = hand3d.calibrate_table(str(video_path))
        except Exception as exc:  # pragma: no cover
            print(f"[retarget] calibrate_table failed ({exc!r}); default table pose", file=sys.stderr)
            table = None
    K = hand3d.camera_matrix(width, height)
    if table is None:
        table = hand3d.default_table_pose(K)
    src = str(table.get("source") or src) if isinstance(table, dict) else src
    tr = hand3d.track_to_table(hands_df, width, height, table=table, K=K).reset_index(drop=True)
    if len(tr) == 0:
        raise ValueError("hand not visible enough for an exact copy")
    tf = pd.to_numeric(tr["t_s"], errors="coerce").to_numpy(float)
    tf = tf - tf[0] if np.isfinite(tf[0]) else tf
    valid = tr["valid"].fillna(False).astype(bool).to_numpy().copy()
    for c in ("x", "y", "z"):
        valid &= np.isfinite(pd.to_numeric(tr[c], errors="coerce").to_numpy(float))
    valid_frac = float(valid.mean())
    # judge visibility within the span where the hand is in view (first..last valid frame), not the whole clip
    iv = np.flatnonzero(valid)
    span_frac = float(valid[iv[0]:iv[-1] + 1].mean()) if len(iv) else 0.0
    if len(iv) < 10 or span_frac < 0.30:
        raise ValueError(f"hand not visible enough for an exact copy (valid {valid_frac:.0%}, "
                         f"{span_frac:.0%} within the hand span)")

    def held(col, smooth=3):
        v = pd.to_numeric(tr[col], errors="coerce").to_numpy(float).copy()
        v[~valid] = np.nan
        s = pd.Series(v).ffill().bfill()  # invalid stretches: hold last valid value
        if smooth > 1:
            s = s.rolling(smooth, center=True, min_periods=1).median()
        return s.to_numpy(float)

    T = float(np.nanmax(tf)) if len(tf) else 0.0
    t = np.arange(0.0, T + 1e-9, 1.0 / HZ)
    pc = ("gx", "gy", "gz") if {"gx", "gy", "gz"} <= set(tr) else ("x", "y", "z")  # follow the grip centre
    P = np.column_stack([np.interp(t, tf, held(c)) for c in pc])
    # outside the span where the hand is in view the gripper waits high above (no hand = nothing to copy)
    t_in, t_out = float(tf[iv[0]]), float(tf[iv[-1]])

    ap = pd.to_numeric(tr.get("aperture_m", pd.Series(np.nan, index=tr.index)), errors="coerce").to_numpy(float)
    ap_ok = valid & np.isfinite(ap)
    if ap_ok.sum() < 5:
        raise ValueError("hand not visible enough for an exact copy (no aperture)")
    apf = ap.copy()
    apf[~ap_ok] = np.nan
    apf = pd.Series(apf).ffill().bfill().rolling(5, center=True, min_periods=1).median().to_numpy(float)
    a_open = float(np.nanpercentile(ap[ap_ok], 90))
    a_lo = float(np.nanpercentile(ap[ap_ok], 10))
    if a_open - a_lo < 0.005:
        raise ValueError("hand not visible enough for an exact copy (hand never closes)")
    thr = a_lo + 0.4 * (a_open - a_lo)
    # first closure: aperture drops below thr after having been open (above the midpoint) on a valid frame
    was_open, ig = False, None
    for i in range(len(apf)):
        if apf[i] > a_lo + 0.6 * (a_open - a_lo):
            was_open = True
        elif was_open and apf[i] < thr and ap_ok[i]:
            ig = i
            break
    if ig is None:
        raise ValueError("hand not visible enough for an exact copy (no grip closure)")
    t_g = float(tf[ig])
    # closed reference = the hand's aperture while it holds the object (next ~0.5 s after closure)
    win = (tf >= t_g) & (tf <= t_g + 0.5) & ap_ok
    a_closed = float(np.nanmedian(ap[win])) if win.any() else float(apf[ig])
    a_closed = min(a_closed, thr)
    u = np.clip((np.interp(t, tf, apf) - a_closed) / max(a_open - a_closed, 1e-6), 0.0, 1.0)
    grip = GRIP_CLOSED - u * (GRIP_CLOSED - GRIP_OPEN)

    # place the cup under the hand at the first closure; shift z so the wrist sits at grasp height there
    kg = int(np.clip(np.searchsorted(t, t_g), 0, len(t) - 1))
    obj_xy = [float(P[kg, 0]), float(P[kg, 1])]
    P[:, 2] = np.maximum(P[:, 2] + (EXACT_GRASP_Z - P[kg, 2]), EXACT_MIN_Z)
    park_z = 0.25
    for k in range(len(t)):
        d = max(t_in - t[k], t[k] - t_out, 0.0)
        if d > 0:  # blend over 0.5 s from the first/last seen pose up to the parking height
            w = min(1.0, d / 0.5)
            P[k, 2] = (1 - w) * P[k, 2] + w * park_z
            grip[k] = (1 - w) * grip[k] + w * GRIP_OPEN

    # orientation relative to the grasp pose (gripper points straight down at the closure), tilt <= 40 deg
    q = tr[["qw", "qx", "qy", "qz"]].apply(pd.to_numeric, errors="coerce").to_numpy(float) \
        if {"qw", "qx", "qy", "qz"} <= set(tr) else np.full((len(tr), 4), np.nan)
    q_ok = valid & np.isfinite(q).all(axis=1) & (np.linalg.norm(np.nan_to_num(q), axis=1) > 0.5)
    if q_ok.mean() >= 0.5 * max(valid_frac, 1e-6):
        qq = q.copy()
        qq[~q_ok] = np.nan
        qq = pd.DataFrame(qq).ffill().bfill().to_numpy(float)
        # hemisphere-consistent before resampling
        for i in range(1, len(qq)):
            if np.dot(qq[i], qq[i - 1]) < 0:
                qq[i] = -qq[i]
        Qr = np.column_stack([np.interp(t, tf, qq[:, k]) for k in range(4)])
        Qr /= np.linalg.norm(Qr, axis=1, keepdims=True) + 1e-12
        R = Rotation.from_quat(Qr[:, [1, 2, 3, 0]])
        R = R * R[kg].inv()  # rotation since the grasp, applied to the straight-down gripper
        rot_src = "track"
    elif imu_df is not None and {"roll", "pitch", "yaw"} <= set(imu_df):
        it = _episode_time(imu_df)
        quat = _orientation(imu_df, it, t)
        R = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
        R = R * R[kg].inv()
        rot_src = "imu"
    else:
        R = Rotation.identity(len(t))
        rot_src = "none"
    R = _limit_tilt(R, EXACT_MAX_TILT)
    qx = R.as_quat()
    quat = np.column_stack([qx[:, 3], qx[:, 0], qx[:, 1], qx[:, 2]])

    # shift hand path and cup together so the grasp happens at the scene centre (in view of the cameras);
    # the motion relative to the cup is unchanged
    P = P.copy()
    P[:, 0] -= obj_xy[0]
    P[:, 1] -= obj_xy[1]
    table_xy = obj_xy
    obj_xy = [0.0, 0.0]
    out = pd.DataFrame({
        "t_s": t,
        "mocap_x": P[:, 0], "mocap_y": P[:, 1], "mocap_z": P[:, 2],
        "mocap_qw": quat[:, 0], "mocap_qx": quat[:, 1], "mocap_qy": quat[:, 2], "mocap_qz": quat[:, 3],
        "grip_ctrl": grip,
    })
    out.attrs["retarget_mode"] = f"exact_{src}"
    out.attrs["object_xy"] = obj_xy
    out.attrs["grasp_xy_table"] = [float(v) for v in table_xy]
    out.attrs["valid_frac"] = valid_frac
    out.attrs["valid_span_frac"] = span_frac
    out.attrs["hand_span_s"] = [float(tf[iv[0]]), float(tf[iv[-1]])]
    out.attrs["t_grasp_s"] = t_g
    out.attrs["orientation_src"] = rot_src
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
