"""Metric 3D hand trajectory from ONE camera, expressed in a table frame (metres, Z up).

Pipeline per frame: MediaPipe world landmarks (metric, hand-centred) + image landmarks (pixels)
-> solvePnP with the camera intrinsics K -> wrist position/orientation in the camera frame
-> table frame via an A4-sheet calibration (calibrate_table) or an assumed chest-camera pose.

    python -m sim.hand3d hands.parquet --video ep.mp4 --out traj.parquet --plot traj.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

A4_W, A4_H = 0.297, 0.210            # long side along table X, short side along table Y
DEFAULT_HFOV_DEG = 69.0              # iPhone main / Continuity Camera, 16:9 landscape
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
MAX_GAP_S = 0.3
DEPTH_MIN, DEPTH_MAX, MAX_DEPTH_JUMP = 0.10, 1.5, 0.15  # 0.10: hands right under a chest cam measure 0.13-0.15 m


# ---------------------------------------------------------------- intrinsics / table pose
def camera_matrix(width: int, height: int, hfov_deg: float | None = None) -> np.ndarray:
    """Pinhole K. hfov_deg is the FOV across the LONG image side (landscape: horizontal; portrait: vertical)."""
    fov = math.radians(hfov_deg if hfov_deg is not None else DEFAULT_HFOV_DEG)
    long_side = max(width, height)
    f = (long_side / 2.0) / math.tan(fov / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def default_table_pose(K: np.ndarray, height_m: float = 0.45, pitch_deg: float = 35.0) -> dict:
    """Assumed chest camera: height_m above the table, pitched pitch_deg below horizontal.
    Table origin = where the optical axis hits the table; X = camera right, Y = forward (away), Z up.
    Returns cam = R @ p_table + t."""
    p = math.radians(pitch_deg)
    C = np.array([0.0, -height_m / math.tan(p), height_m])         # camera centre in table frame
    x_c = np.array([1.0, 0.0, 0.0])
    z_c = np.array([0.0, math.cos(p), -math.sin(p)])
    y_c = np.cross(z_c, x_c)
    R = np.vstack([x_c, y_c, z_c])
    t = -R @ C
    return {"R": R, "t": t, "K": np.asarray(K, float), "source": "assumed", "frame": None, "corners_px": None,
            "height_m": height_m, "pitch_deg": pitch_deg}


def _order_quad(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]                                    # consistent (clockwise in image) order


def _find_sheet(gray: np.ndarray):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best = None
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [otsu, cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)[1]]
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.005 * w * h or area > 0.6 * w * h:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            q = approx.reshape(4, 2).astype(np.float64)
            if np.any(q[:, 0] <= 2) or np.any(q[:, 1] <= 2) or np.any(q[:, 0] >= w - 3) or np.any(q[:, 1] >= h - 3):
                continue                                           # touches border -> not a full sheet
            sides = np.linalg.norm(q - np.roll(q, 1, 0), axis=1)
            ratio = max(sides[0] + sides[2], sides[1] + sides[3]) / max(1e-6, min(sides[0] + sides[2], sides[1] + sides[3]))
            if not (1.0 <= ratio <= 3.5):
                continue
            # must be brighter inside than its surroundings
            if area / cv2.contourArea(cv2.convexHull(c)) < 0.9:
                continue
            if best is None or area > best[0]:
                best = (area, q)
    if best is None:
        return None
    q = best[1].astype(np.float32).reshape(-1, 1, 2)
    side = float(np.min(np.linalg.norm(best[1] - np.roll(best[1], 1, 0), axis=1)))
    win = int(max(2, min(11, side / 8)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    q = cv2.cornerSubPix(blur, q, (win, win), (-1, -1), crit)
    return q.reshape(4, 2).astype(np.float64)


def _pnp_sheet(q: np.ndarray, K: np.ndarray):
    """Try corner assignments; keep camera above the sheet (+Z) and the lowest reprojection error."""
    q = _order_quad(q)
    obj = np.array([[-A4_W / 2, A4_H / 2, 0], [A4_W / 2, A4_H / 2, 0],
                    [A4_W / 2, -A4_H / 2, 0], [-A4_W / 2, -A4_H / 2, 0]], np.float64)
    best = None
    for rev in (False, True):
        qq = q[::-1] if rev else q
        for k in range(4):
            img = np.roll(qq, k, 0)
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.ravel()
            cam_in_table = -R.T @ t
            if cam_in_table[2] <= 0:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img) ** 2, 1))))
            # 180 deg ambiguity: prefer table X pointing to camera right
            score = err + (0.0 if R[0, 0] >= 0 else 1e-3)
            if best is None or score < best[0]:
                best = (score, R, t, img, err)
    return best


def calibrate_table(video_path, max_frames: int = 90, K: np.ndarray | None = None) -> dict | None:
    """Find a white A4 sheet on a darker table in the first max_frames frames; table frame via IPPE."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cands = []
    idx = 0
    try:
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % 3 == 0:
                if K is None:
                    K = camera_matrix(frame.shape[1], frame.shape[0])
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                q = _find_sheet(gray)
                if q is not None:
                    r = _pnp_sheet(q, K)
                    if r is not None and r[4] < 0.02 * max(gray.shape):
                        cands.append((cv2.contourArea(q.astype(np.float32)), idx, r))
            idx += 1
    finally:
        cap.release()
    if not cands:
        return None
    # take the median-depth candidate among the largest ones (robust to a single bad frame)
    cands.sort(key=lambda c: -c[0])
    top = cands[:max(1, len(cands) // 2)]
    depths = np.array([c[2][2][2] for c in top])
    _, fidx, (_, R, t, img, err) = top[int(np.argsort(depths)[len(depths) // 2])]
    return {"R": R, "t": t, "K": np.asarray(K, float), "source": "a4", "frame": int(fidx),
            "corners_px": img.tolist(), "reproj_px": err}


def _calibrate_image(img: np.ndarray, K: np.ndarray) -> dict | None:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    q = _find_sheet(gray)
    if q is None:
        return None
    r = _pnp_sheet(q, K)
    if r is None:
        return None
    return {"R": r[1], "t": r[2], "K": K, "source": "a4", "frame": 0, "corners_px": r[3].tolist(), "reproj_px": r[4]}


# ---------------------------------------------------------------- hand tracking
def _pick_side(df: pd.DataFrame) -> str:
    counts = {}
    for s in ("left", "right"):
        col = f"{s}_present"
        counts[s] = int(df[col].fillna(False).astype(bool).sum()) if col in df else 0
    return max(counts, key=counts.get)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _palm_frame(P: np.ndarray) -> np.ndarray | None:
    """Columns: x = wrist->middle MCP, y = index MCP->pinky MCP (orthogonalised), z = x cross y."""
    x = P[MIDDLE_MCP] - P[WRIST]
    y = P[PINKY_MCP] - P[INDEX_MCP]
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x = x / nx
    y = y - x * (y @ x)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y = y / ny
    return np.column_stack([x, y, np.cross(x, y)])


def _hand_pnp(world: np.ndarray, img: np.ndarray, K: np.ndarray):
    flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    ok, rvec, tvec = cv2.solvePnP(world, img, K, None, flags=flag)
    if not ok:
        return None
    ok2, rvec2, tvec2 = cv2.solvePnP(world, img, K, None, rvec.copy(), tvec.copy(), True, cv2.SOLVEPNP_ITERATIVE)
    if ok2 and np.all(np.isfinite(tvec2)):
        rvec, tvec = rvec2, tvec2
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def _segments(mask: np.ndarray):
    """Yield (start, end_exclusive) of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def track_to_table(hands_df: pd.DataFrame, width: int, height: int, table: dict | None = None,
                   side: str | None = None, K: np.ndarray | None = None) -> pd.DataFrame:
    """Per-frame wrist pose in the table frame. See module docstring."""
    if K is None:
        K = table["K"] if table is not None and table.get("K") is not None else camera_matrix(width, height)
    K = np.asarray(K, float)
    if table is None:
        table = default_table_pose(K)
    side = side or _pick_side(hands_df)
    n = len(hands_df)
    t_s = hands_df["t_s"].to_numpy(float) if "t_s" in hands_df else np.arange(n) / 30.0
    present = hands_df.get(f"{side}_present", pd.Series(False, index=hands_df.index)).fillna(False).astype(bool).to_numpy()
    W = np.stack([hands_df[[f"{side}_w{a}{i}" for a in "xyz"]].to_numpy(float) for i in range(21)], 1)  # n,21,3
    I = np.stack([hands_df[[f"{side}_{a}{i}" for a in "xy"]].to_numpy(float) for i in range(21)], 1)    # n,21,2
    I = I * np.array([width, height])

    cam = np.full((n, 3), np.nan)
    quat_cam = np.full((n, 4), np.nan)
    aper = np.full(n, np.nan)
    grasp = np.full((n, 3), np.nan)   # grip centre: midpoint of thumb tip (4) and middle fingertip (12)
    last_z = None
    last_i = -10**9
    for k in range(n):
        if not present[k] or not (np.all(np.isfinite(W[k])) and np.all(np.isfinite(I[k]))):
            continue
        r = _hand_pnp(W[k].astype(np.float64), I[k].astype(np.float64), K)
        if r is None:
            continue
        Rh, th = r
        P = W[k] @ Rh.T + th
        z = P[WRIST, 2]
        if not (DEPTH_MIN <= z <= DEPTH_MAX):
            continue
        # jump test only against a recent accepted frame (within the interpolation window)
        if last_z is not None and (t_s[k] - t_s[last_i]) <= MAX_GAP_S and abs(z - last_z) > MAX_DEPTH_JUMP:
            continue
        F = _palm_frame(P)
        if F is None:
            continue
        cam[k] = P[WRIST]
        quat_cam[k] = _rot_to_quat(F)
        aper[k] = float(np.linalg.norm(W[k, THUMB_TIP] - W[k, INDEX_TIP]))
        grasp[k] = 0.5 * (P[4] + P[12])
        last_z, last_i = z, k

    raw_valid = np.isfinite(cam[:, 0])
    # quaternion sign continuity
    prev = None
    for k in np.flatnonzero(raw_valid):
        if prev is not None and quat_cam[k] @ prev < 0:
            quat_cam[k] = -quat_cam[k]
        prev = quat_cam[k]

    # fill short gaps (time-based), mark long gaps invalid
    valid = raw_valid.copy()
    idx = np.flatnonzero(raw_valid)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and (t_s[b] - t_s[a]) <= MAX_GAP_S:
            valid[a + 1:b] = True
    cols = np.column_stack([cam, quat_cam, aper, grasp])
    filled = np.full_like(cols, np.nan)
    if len(idx) >= 1:
        for j in range(cols.shape[1]):
            filled[valid, j] = np.interp(np.flatnonzero(valid), idx, cols[idx, j]) if len(idx) > 1 else cols[idx[0], j]

    # smooth positions + aperture per contiguous valid segment: 5-median then 5-mean
    for a, b in _segments(valid):
        for j in (0, 1, 2, 7, 8, 9, 10):
            s = pd.Series(filled[a:b, j])
            s = s.rolling(5, center=True, min_periods=1).median().rolling(5, center=True, min_periods=1).mean()
            filled[a:b, j] = s.to_numpy()
        q = pd.DataFrame(filled[a:b, 3:7]).rolling(3, center=True, min_periods=1).mean().to_numpy()
        filled[a:b, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)

    Rt, tt = np.asarray(table["R"], float), np.asarray(table["t"], float).ravel()
    cam_f = filled[:, :3]
    tab = (cam_f - tt) @ Rt                                     # R^T (cam - t) per row
    gtab = (filled[:, 8:11] - tt) @ Rt
    q_tab = np.full((n, 4), np.nan)
    for k in np.flatnonzero(valid):
        qw, qx, qy, qz = filled[k, 3:7]
        Rc = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                       [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                       [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        q_tab[k] = _rot_to_quat(Rt.T @ Rc)
    prev = None
    for k in np.flatnonzero(valid):
        if prev is not None and q_tab[k] @ prev < 0:
            q_tab[k] = -q_tab[k]
        prev = q_tab[k]

    out = pd.DataFrame({
        "t_s": t_s, "valid": valid,
        "x": tab[:, 0], "y": tab[:, 1], "z": tab[:, 2],
        "qw": q_tab[:, 0], "qx": q_tab[:, 1], "qy": q_tab[:, 2], "qz": q_tab[:, 3],
        "aperture_m": filled[:, 7],
        "gx": gtab[:, 0], "gy": gtab[:, 1], "gz": gtab[:, 2],
        "cam_x": cam_f[:, 0], "cam_y": cam_f[:, 1], "cam_z": cam_f[:, 2],
    })
    if "frame_idx" in hands_df:
        out.insert(0, "frame_idx", hands_df["frame_idx"].to_numpy())
    out.attrs.update({"side": side, "table_source": table.get("source")})
    return out


# ---------------------------------------------------------------- CLI
def _video_size(path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def plot_track(tr: pd.DataFrame, out_png, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = tr[tr.valid]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    sc = ax[0].scatter(v.x, v.y, c=v.t_s, s=6, cmap="viridis")
    ax[0].plot(v.x, v.y, lw=0.5, color="0.6")
    ax[0].set_xlabel("x [m] (table)"); ax[0].set_ylabel("y [m] (table)"); ax[0].set_title("top view (colour = t)")
    ax[0].set_aspect("equal", adjustable="datalim"); fig.colorbar(sc, ax=ax[0], label="t [s]")
    z = tr.z.where(tr.valid)
    ax[1].plot(tr.t_s, z); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("z [m] above table"); ax[1].set_title("wrist height")
    ax[2].plot(tr.t_s, tr.aperture_m.where(tr.valid) * 100)
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("thumb-index [cm]"); ax[2].set_title("aperture")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hands")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out")
    ap.add_argument("--plot")
    ap.add_argument("--side", choices=["left", "right"])
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--cam-height", type=float, default=0.45)
    ap.add_argument("--pitch", type=float, default=35.0)
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.hands)
    w, h = _video_size(a.video)
    K = camera_matrix(w, h, a.hfov)
    table = calibrate_table(a.video, K=K)
    if table is None:
        table = default_table_pose(K, a.cam_height, a.pitch)
        print(f"table pose: ASSUMED (no A4 found) height={a.cam_height} m pitch={a.pitch} deg")
    else:
        cam_c = -table["R"].T @ table["t"]
        print(f"table pose: A4 sheet at frame {table['frame']} reproj={table['reproj_px']:.2f}px "
              f"camera at {np.round(cam_c, 3).tolist()} m (table frame)")
    tr = track_to_table(df, w, h, table=table, side=a.side, K=K)
    v = tr[tr.valid]
    print(f"side={tr.attrs['side']} frames={len(tr)} valid={100 * tr.valid.mean():.1f}%  fx={K[0, 0]:.1f}px")
    if len(v):
        print(f"depth cam_z {v.cam_z.min():.3f}..{v.cam_z.max():.3f} m")
        for c in "xyz":
            print(f"{c} {v[c].min():+.3f}..{v[c].max():+.3f} m")
        print(f"aperture {v.aperture_m.min():.3f}..{v.aperture_m.max():.3f} m")
    if a.out:
        tr.to_parquet(a.out, index=False)
        print(f"wrote {a.out}")
    if a.plot:
        plot_track(tr, a.plot, f"{Path(a.hands).name}  side={tr.attrs['side']}  table={table['source']}")
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
