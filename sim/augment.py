"""Multiply one retargeted demonstration into N domain-randomized MuJoCo variants.

    python -m sim.augment data/aligned/ep_001.parquet --n 20 --render-n 9 \
        --out-dir data/sim_aug --lerobot-out data/mimesis_lerobot --task pick_place_mug --seed 42

stdout carries one line per finished variant (for the web UI):
    PROGRESS {"done": 3, "total": 20, "ok": 2, "slip": 1}
Everything else is logged to stderr.
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
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contact_detect import add_features  # noqa: E402
from sim.retarget import GRIP_CLOSED, GRIP_OPEN, read_any, retarget_episode, retarget_exact  # noqa: E402
from sim.scene import PALETTE, configure_gl_backend, load_model_and_data  # noqa: E402

PHYS_DT = 0.002
SUBSTEPS = 5          # 500 Hz physics -> 100 Hz IMU
RENDER_FPS = 30.0
RENDER_W, RENDER_H = 320, 240
MAX_EPISODE_S = 12.0
FFMPEG_CANDIDATES = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]
G = 9.81


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def find_ffmpeg():
    for c in FFMPEG_CANDIDATES:
        p = shutil.which(c) if not c.startswith("/") else (c if Path(c).exists() else None)
        if p:
            return p
    return None


def transcode_h264(src: Path, dst: Path) -> bool:
    ff = find_ffmpeg()
    if ff is None:
        if src != dst:
            shutil.move(str(src), str(dst))
        return False
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], capture_output=True)
    if r.returncode != 0:
        log(f"[augment] ffmpeg failed: {r.stderr.decode()[-300:]}")
        shutil.move(str(src), str(dst))
        return False
    src.unlink(missing_ok=True)
    return True


def sample_params(i: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed + i * 101)
    hard = i % 3 == 2
    if hard:
        mu, mass = rng.uniform(0.10, 0.24), rng.uniform(0.45, 0.85)
    else:
        mu, mass = rng.uniform(0.50, 1.05), rng.uniform(0.12, 0.42)
    return {
        "variant_idx": i,
        "obj_type": "cup" if i % 2 == 0 else "box",
        "obj_friction": float(mu),
        "obj_mass_kg": float(mass),
        "obj_scale": float(rng.uniform(0.82, 1.18)),
        "obj_xy": [float(rng.uniform(-0.015, 0.015)), float(rng.uniform(-0.015, 0.015))],
        "time_warp": float(rng.uniform(0.88, 1.15)),
        "rgba": list(PALETTE[i % len(PALETTE)]),
        "hard": bool(hard),
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
    }


def _grip_to_joint(g):
    """Map retarget grip command [-0.002, 0.032] onto the full finger stroke [-0.002, 0.042]."""
    u = (np.asarray(g, float) - GRIP_OPEN) / (GRIP_CLOSED - GRIP_OPEN)
    return np.clip(GRIP_OPEN + u * (0.042 - GRIP_OPEN), -0.005, 0.042)


def crop_retarget(rt: pd.DataFrame, max_len_s: float) -> pd.DataFrame:
    """Keep long episodes fast: if longer than max_len_s, start ~2.5 s before the first grip closure."""
    t = rt["t_s"].to_numpy(float)
    if len(t) == 0 or t[-1] - t[0] <= max_len_s:
        return rt
    closed = np.where(rt["grip_ctrl"].to_numpy(float) > 0.5 * (GRIP_OPEN + GRIP_CLOSED))[0]
    t0 = max(t[0], t[closed[0]] - 2.5) if len(closed) else t[0]
    t0 = min(t0, max(t[0], t[-1] - max_len_s))
    out = rt[(t >= t0) & (t <= t0 + max_len_s)].reset_index(drop=True)
    out["t_s"] = out["t_s"] - out["t_s"].iloc[0]
    return out


def simulate_variant(rt: pd.DataFrame, params: dict, render: bool = False, video_path: Path | None = None,
                     max_episode_s: float = MAX_EPISODE_S):
    """Run one variant. Returns (episode DataFrame @100 Hz, summary dict). Rendering on the calling thread."""
    import mujoco

    exact = str(rt.attrs.get("retarget_mode", "")).startswith("exact")
    base_xy = rt.attrs.get("object_xy", (0.0, 0.0)) if exact else (0.0, 0.0)
    obj_xy = [float(base_xy[0]) + float(params["obj_xy"][0]), float(base_xy[1]) + float(params["obj_xy"][1])]
    m, d = load_model_and_data(params["obj_type"], params["obj_mass_kg"], params["obj_friction"],
                               params["obj_scale"], obj_xy, params["rgba"])
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_obj")
    wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_body")
    mocap_id = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_mocap")])
    s_acc = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_accel")]
    s_gyr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_gyro")]
    s_quat = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_quat")]
    wrist_jadr = m.jnt_qposadr[m.body_jntadr[wrist_bid]]
    obj_jadr = m.jnt_qposadr[m.body_jntadr[obj_bid]]

    warp = params["time_warp"]
    if exact:  # replay the human timing 1:1: no time warp, no crop, full length
        warp = 1.0
        max_episode_s = max(max_episode_s, float(rt["t_s"].iloc[-1]))
    else:
        rt = crop_retarget(rt, max_episode_s / warp)
    t_traj = rt["t_s"].to_numpy(float) * warp
    T = min(float(t_traj[-1]), max_episode_s)
    P = rt[["mocap_x", "mocap_y", "mocap_z"]].to_numpy(float)
    Q = rt[["mocap_qw", "mocap_qx", "mocap_qy", "mocap_qz"]].to_numpy(float)
    GR = _grip_to_joint(rt["grip_ctrl"].to_numpy(float))

    def traj(tt):
        p = np.array([np.interp(tt, t_traj, P[:, k]) for k in range(3)])
        q = np.array([np.interp(tt, t_traj, Q[:, k]) for k in range(4)])
        q /= np.linalg.norm(q) + 1e-12
        return p, q, float(np.interp(tt, t_traj, GR))

    # initial state: wrist at first pose, object resting
    p0, q0, g0 = traj(0.0)
    d.mocap_pos[mocap_id], d.mocap_quat[mocap_id] = p0, q0
    d.qpos[wrist_jadr:wrist_jadr + 3], d.qpos[wrist_jadr + 3:wrist_jadr + 7] = p0, q0
    d.ctrl[:] = g0
    mujoco.mj_forward(m, d)
    obj_z0 = float(d.qpos[obj_jadr + 2])

    rng = np.random.default_rng(params.get("noise_seed", 0))
    b_a, b_w = rng.normal(0, 0.04, 3), rng.normal(0, 0.008, 3)

    renderer, writer, tmp_video = None, None, None
    if render and video_path is not None:
        import cv2
        renderer = mujoco.Renderer(m, RENDER_H, RENDER_W)
        tmp_video = Path(video_path).with_suffix(".tmp.mp4")
        writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), RENDER_FPS, (RENDER_W, RENDER_H))
    next_frame_t = 0.0

    n_samples = int(np.floor(T * 100.0)) + 1
    rows = np.zeros((n_samples, 23))
    cf = np.zeros(6)
    R = np.zeros(9)
    step = 0
    try:
        for k in range(n_samples):
            for _ in range(SUBSTEPS if k > 0 else 0):
                tt = step * PHYS_DT
                p, q, g = traj(tt)
                d.mocap_pos[mocap_id], d.mocap_quat[mocap_id] = p, q
                d.ctrl[:] = g
                mujoco.mj_step(m, d)
                step += 1
                if renderer is not None and d.time >= next_frame_t - 1e-9:
                    renderer.update_scene(d, camera="side_cam")
                    writer.write(renderer.render()[:, :, ::-1])
                    next_frame_t += 1.0 / RENDER_FPS
            if k == 0:
                mujoco.mj_forward(m, d)
                if renderer is not None:
                    renderer.update_scene(d, camera="side_cam")
                    writer.write(renderer.render()[:, :, ::-1])
                    next_frame_t = 1.0 / RENDER_FPS
            f = d.sensordata[s_acc:s_acc + 3].copy()
            w = d.sensordata[s_gyr:s_gyr + 3].copy()
            qs = d.sensordata[s_quat:s_quat + 4].copy()
            qs /= np.linalg.norm(qs) + 1e-12
            mujoco.mju_quat2Mat(R, qs)
            g_body = R.reshape(3, 3).T @ np.array([0.0, 0.0, G])
            a_lin = f - g_body + b_a + rng.normal(0, 0.06, 3)
            w_meas = w + b_w + rng.normal(0, 0.015, 3)
            force = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if b1 == obj_bid:
                    other = b2
                elif b2 == obj_bid:
                    other = b1
                else:
                    continue
                if other == 0:
                    continue
                mujoco.mj_contactForce(m, d, ci, cf)
                force += float(np.linalg.norm(cf[:3]))
            rows[k, 0] = d.time
            rows[k, 1:4], rows[k, 4:7], rows[k, 7:11] = a_lin, w_meas, qs
            rows[k, 11:14] = g_body
            rows[k, 14] = force
            rows[k, 15:18] = d.qpos[obj_jadr:obj_jadr + 3]
            rows[k, 18] = float(np.interp(d.time, t_traj, rt["grip_ctrl"].to_numpy(float)))
    finally:
        if renderer is not None:
            writer.release()
            renderer.close()
    qo = d.qpos[obj_jadr + 3:obj_jadr + 7]  # final object quat (w x y z)
    obj_up_z = float(1.0 - 2.0 * (qo[1] ** 2 + qo[2] ** 2))  # cos(tilt from vertical)
    video_ok = None
    if tmp_video is not None and tmp_video.exists():
        transcode_h264(tmp_video, Path(video_path))
        video_ok = Path(video_path).exists()

    t = rows[:, 0] - rows[0, 0]
    df = pd.DataFrame({"t_rel_s": t, "t_s": t})
    for j, c in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
        df[c] = rows[:, 1 + j]
    df["qw"], df["qx"], df["qy"], df["qz"] = rows[:, 7], rows[:, 8], rows[:, 9], rows[:, 10]
    eul = Rotation.from_quat(rows[:, [8, 9, 10, 7]]).as_euler("xyz")
    df["roll"], df["pitch"], df["yaw"] = eul[:, 0], eul[:, 1], eul[:, 2]
    df["grav_x"], df["grav_y"], df["grav_z"] = rows[:, 11], rows[:, 12], rows[:, 13]
    df["sim_contact_force_n"] = rows[:, 14]
    df["sim_contact_flag"] = (rows[:, 14] > 0.25).astype(int)
    df["obj_x"], df["obj_y"], df["obj_z"] = rows[:, 15], rows[:, 16], rows[:, 17]
    df["grip_ctrl"] = rows[:, 18]
    feats = add_features(df)
    df["jerk_mag"] = feats["jerk_mag"].to_numpy()
    df["slip_score"] = feats["slip_score"].to_numpy()
    df["contact_flag"] = df["sim_contact_flag"].astype(int)
    df["contact_event_id"] = np.where(df["contact_flag"] > 0, 3, 1).astype(int)

    max_lift = float(np.max(rows[:, 17]) - obj_z0)
    max_force = float(np.max(rows[:, 14])) if len(rows) else 0.0
    lifted = max_lift > 0.025
    mu = params["obj_friction"]
    slipped = bool((not lifted and max_force > 0.5) or (mu < 0.25 and max_lift < 0.05 and max_force > 0.5))
    tipped = obj_up_z < 0.707  # object ended > 45 deg off upright (knocked over / dropped on its side)
    summary = {"max_lift_m": max_lift, "max_contact_force_n": max_force, "lifted": bool(lifted),
               "slipped": slipped, "tipped": bool(tipped), "success": bool(lifted and not slipped and not tipped),
               "duration_s": float(t[-1]),
               "video_ok": video_ok}
    return df, summary


def _call_export(lerobot_out, variants, task, out_dir):
    try:
        import export  # repo-root export.py (owned separately)
        fn = export.export_lerobot_dataset
    except Exception as exc:
        log(f"[augment] export.export_lerobot_dataset unavailable ({exc!r}); skipping LeRobot export")
        return None
    episodes = [{"episode_id": v["episode_id"], "task": v["task"], "aligned_parquet": v["aligned_parquet"],
                 "video_path": v["video_path"]} for v in variants]
    try:
        res = fn(episodes, str(lerobot_out), fps=30.0, repo_id="nullhacks/mimesis-sim-aug")
        log(f"[augment] LeRobot export ({len(episodes)} episodes) -> {lerobot_out}")
        return res if isinstance(res, (dict, list, str)) else str(lerobot_out)
    except Exception as exc:
        log(f"[augment] LeRobot export failed: {exc!r}")
        return None


def nominal_params(i: int = 0) -> dict:
    """Nominal object: cup, 0.25 kg, mu 0.8, scale 1, no offset, no time warp."""
    return {"variant_idx": i, "obj_type": "cup", "obj_friction": 0.8, "obj_mass_kg": 0.25, "obj_scale": 1.0,
            "obj_xy": [0.0, 0.0], "time_warp": 1.0, "rgba": list(PALETTE[0]), "hard": False, "noise_seed": 7}


def run_augment(input_parquet, n: int = 20, render_n: int = 9, out_dir="data/sim_aug", lerobot_out=None,
                task: str = "pick_place_mug", seed: int = 42, progress_cb=None, exact=None,
                nominal: bool = False) -> dict:
    """exact: None (scripted/anchored retarget of input_parquet), a dict {hands_parquet, video_path, width,
    height[, imu_df]} for a 1:1 hand copy, or a precomputed retarget_exact DataFrame. In exact mode every
    variant replays the human timing 1:1 (time_warp 1.0, no crop) with the cup at attrs["object_xy"]
    (+ the variant's +-1.5 cm offset). nominal=True uses nominal_params() for every variant."""
    configure_gl_backend()
    t_start = time.time()
    out_dir = Path(out_dir)
    ep_dir, vid_dir = out_dir / "episodes", out_dir / "videos"
    ep_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    if exact is None:
        inp = Path(input_parquet)
        rt = retarget_episode(read_any(inp))
    elif isinstance(exact, pd.DataFrame):
        inp = Path(input_parquet or "exact")
        rt = exact
    else:
        hp = Path(exact["hands_parquet"])
        inp = Path(input_parquet) if input_parquet else hp
        rt = retarget_exact(read_any(hp), int(exact["width"]), int(exact["height"]),
                            video_path=exact.get("video_path"), imu_df=exact.get("imu_df"))
    attrs = dict(rt.attrs)
    mode = rt.attrs.get("retarget_mode", "?")
    rt_path = out_dir / f"{inp.stem}_retarget.parquet"
    rt.to_parquet(rt_path, index=False)
    rt.attrs.update(attrs)
    log(f"[augment] retarget {mode}: {len(rt)} rows ({rt['t_s'].iloc[-1]:.1f}s) -> {rt_path}")

    variants, labels = [], []
    n_ok = n_slip = 0
    for i in range(n):
        p = nominal_params(i) if nominal else sample_params(i, seed)
        if exact is not None:
            p["time_warp"] = 1.0
        ep_id = f"sim_v{i:04d}"
        render = i < render_n
        vpath = vid_dir / f"{ep_id}.mp4"
        t0 = time.time()
        df, s = simulate_variant(rt, p, render=render, video_path=vpath if render else None)
        pq = ep_dir / f"{ep_id}.parquet"
        pq30 = ep_dir / f"{ep_id}_30fps.parquet"
        df["slip_score"] = 1.0 if s["slipped"] else 0.0  # episode-level label (DATASET.md: sim slip_score)
        df.to_parquet(pq, index=False)
        df.iloc[::3].reset_index(drop=True).to_parquet(pq30, index=False)
        n_ok += int(s["success"])
        n_slip += int(s["slipped"])
        video_rel = f"videos/{ep_id}.mp4" if (render and vpath.exists()) else None
        rec = {
            "variant_idx": i, "episode_id": ep_id, "obj_type": p["obj_type"],
            "obj_mass_kg": p["obj_mass_kg"], "obj_friction": p["obj_friction"], "obj_scale": p["obj_scale"],
            "obj_xy": p["obj_xy"], "time_warp": p["time_warp"], "hard": p["hard"],
            "max_lift_m": s["max_lift_m"], "max_contact_force_n": s["max_contact_force_n"],
            "lifted": s["lifted"], "slipped": s["slipped"], "success": s["success"],
            "duration_s": s["duration_s"],
            "video_path": str(vpath) if video_rel else None,
            "parquet": str(pq),
            "aligned_parquet": str(pq30),
            "task": f"{task}_{'success' if s['success'] else 'slip'}",
        }
        variants.append(rec)
        if render:
            labels.append({"n": i, "object": p["obj_type"], "mass_g": int(round(p["obj_mass_kg"] * 1000)),
                           "mu": round(p["obj_friction"], 3), "slip": bool(s["slipped"]), "video": video_rel})
        log(f"[augment] {ep_id} {p['obj_type']:3s} mu={p['obj_friction']:.2f} m={p['obj_mass_kg']:.2f}kg "
            f"lift={s['max_lift_m'] * 100:.1f}cm F={s['max_contact_force_n']:.1f}N "
            f"{'OK' if s['success'] else ('SLIP' if s['slipped'] else 'FAIL')} ({time.time() - t0:.2f}s)")
        prog = {"done": i + 1, "total": n, "ok": n_ok, "slip": n_slip}
        print("PROGRESS " + json.dumps(prog), flush=True)
        if progress_cb is not None:
            try:
                progress_cb(prog)
            except Exception as exc:
                log(f"[augment] progress_cb error: {exc!r}")

    (vid_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    manifest = {
        "input": str(inp), "retarget_parquet": str(rt_path), "retarget_mode": mode, "task": task, "seed": seed,
        "object_xy": attrs.get("object_xy"), "valid_frac": attrs.get("valid_frac"),
        "n": n, "render_n": min(render_n, n), "n_success": n_ok, "n_slipped": n_slip,
        "elapsed_s": round(time.time() - t_start, 2), "variants": variants,
    }
    if lerobot_out:
        manifest["lerobot_export"] = _call_export(lerobot_out, variants, task, out_dir)
    manifest["elapsed_s"] = round(time.time() - t_start, 2)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"[augment] {n} variants: {n_ok} success, {n_slip} slipped, {manifest['elapsed_s']}s -> {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--render-n", type=int, default=9)
    ap.add_argument("--out-dir", default="data/sim_aug")
    ap.add_argument("--lerobot-out", default="data/mimesis_lerobot")
    ap.add_argument("--task", default="pick_place_mug")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_augment(a.input, a.n, a.render_n, a.out_dir, a.lerobot_out or None, a.task, a.seed)


if __name__ == "__main__":
    main()
