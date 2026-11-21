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
    cands = [p for p in VID.glob(f"{eid}.*") if p.suffix.lower() in VIDEO_EXT]
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
def process(eid: str) -> dict:
    from contact_detect import UI_KIND, detect_contacts

    meta = episode_meta(eid)
    task = meta.get("task") or "manipulation_task"
    raw_p = RAW / f"{eid}.parquet"
    raw = pd.read_parquet(raw_p)
    step("contacts")
    lab, events = detect_contacts(raw)
    lab.to_parquet(RAW / f"{eid}_contacts.parquet", index=False)
    (RAW / f"{eid}_contacts.events.json").write_text(json.dumps({"input": str(raw_p), "num_events": len(events),
                                                                "events": events}, indent=2))
    video = find_video(eid)
    sync = None
    hands_p = HANDS / f"{eid}_hands.parquet"
    overlay = HANDS / f"{eid}_overlay.mp4"
    aligned_p = ALIGNED / f"{eid}.parquet"
    if video is not None:
        try:
            step("hands (MediaPipe)")
            from hands import run_hands
            run_hands(str(video), str(hands_p), str(overlay), swap_handedness=True)
        except Exception as e:
            print("hands failed:", e, flush=True)
        try:
            step("clap sync")
            from sync import run_sync
            sync = run_sync(str(video), str(raw_p), str(hands_p) if hands_p.exists() else None, str(aligned_p),
                            plot=str(ALIGNED / f"{eid}_sync.png"), search_window=12.0, task=task)
        except Exception as e:
            print("sync failed:", e, flush=True)
    if not aligned_p.exists() or sync is None:
        step("no video / sync: IMU-only aligned episode")
        al = lab.copy()
        al["t_s"] = al["t_rel_s"]
        al["t_imu_s"] = al["t_rel_s"]
        al = al.iloc[::3].reset_index(drop=True)  # 100 Hz -> ~33 Hz rows like video frames
        al.to_parquet(aligned_p, index=False)
        (ALIGNED / f"{eid}.sync.json").write_text(json.dumps({"offset_s": 0.0, "task": task, "imu_only": True}))
        sync = {"offset_s": 0.0, "imu_only": True}
    offset = float(sync.get("offset_s", 0.0) or 0.0)

    step("sim twin (MuJoCo)")
    twin = None
    try:
        twin = sim_twin(eid, aligned_p, task)
    except Exception as e:
        print("twin failed:", e, flush=True)

    # ---- publish per-episode JSON for the Episode screen ----
    t_imu = lab["t_rel_s"].to_numpy(float)
    amag = lab["accel_mag"].to_numpy(float)
    if video is not None and not sync.get("imu_only"):
        vdur = video_duration(video)
    else:
        vdur = float(t_imu[-1])
    tv = np.arange(0, vdur, 0.01)
    imu = np.interp(tv + offset, t_imu, amag, left=0, right=0)
    tc = sync.get("t_clap_imu_s")  # the 3 sync claps are not task contacts
    clap = (lambda t: tc - 0.3 <= t <= tc + 2.6) if tc is not None and not sync.get("imu_only") else (lambda t: False)
    contacts = [{"kind": UI_KIND.get(e["event"], "TAP"), "t": round(e["t_s"] - offset, 3), "event": e["event"]}
                for e in events if 0 <= e["t_s"] - offset <= vdur and not clap(e["t_s"])]
    gripper, state, t_last = [], "open", 0.0
    for c in contacts:
        new = "closed" if c["event"] == "grab_onset" else "open" if c["event"] in ("set_down", "slip_or_scrape") else state
        if new != state:
            gripper.append({"t0": t_last, "t1": c["t"], "state": state})
            state, t_last = new, c["t"]
    gripper.append({"t0": t_last, "t1": vdur, "state": state})
    pub = {"id": eid, "task": task.replace("_", " "), "duration": round(vdur, 2), "fps": 30,
           "frames": int(round(vdur * 30)), "sync_offset_ms": int(round(offset * 1000)) if not sync.get("imu_only") else 0,
           "contacts": contacts, "gripper": gripper, "imu": [round(float(x), 3) for x in imu], "imu_hz": 100}
    if overlay.exists():
        shutil.copy(overlay, PUB / f"ep_{eid}_overlay.mp4")
        shutil.copy(overlay, PUB / "episode_overlay.mp4")
        pub["overlay"] = f"/files/ep_{eid}_overlay.mp4"
    elif video is not None:
        h264(video, PUB / f"ep_{eid}_overlay.mp4")
        pub["overlay"] = f"/files/ep_{eid}_overlay.mp4"
    if twin:
        shutil.copy(twin, PUB / f"ep_{eid}_twin.mp4")
        shutil.copy(twin, PUB / "sim_twin.mp4")
        pub["twin"] = f"/files/ep_{eid}_twin.mp4"
    (PUB / f"ep_{eid}.json").write_text(json.dumps(pub))
    publish_index()
    step(f"done: {len(contacts)} contacts, offset {offset * 1000:+.0f} ms")
    return pub


def video_duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def sim_twin(eid: str, aligned_p: Path, task: str) -> Path | None:
    from sim.augment import run_augment
    out = SIM / "twins" / eid
    if out.exists():
        shutil.rmtree(out)
    run_augment(str(aligned_p), n=1, render_n=1, out_dir=str(out), lerobot_out=None, task=task, seed=7)
    vids = sorted((out / "videos").glob("sim_v*.mp4"))
    return vids[0] if vids else None


def publish_index() -> None:
    """episode_<i>.json (Episode screen rail order) + dataset_stats.json."""
    eps = []
    for i, eid in enumerate([e for e in raw_episodes() if (PUB / f"ep_{e}.json").exists()]):
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
    from sim.augment import run_augment
    src = ALIGNED / f"{eid}.parquet"
    if not src.exists():
        src = RAW / f"{eid}.parquet"
    if not src.exists():
        eps = raw_episodes()
        if not eps:
            raise SystemExit("no episodes recorded yet")
        eid = eps[-1]
        src = ALIGNED / f"{eid}.parquet" if (ALIGNED / f"{eid}.parquet").exists() else RAW / f"{eid}.parquet"
    task = episode_meta(eid).get("task") or "pick_place_mug"
    for sub in ("episodes", "videos"):
        if (SIM / sub).exists():
            shutil.rmtree(SIM / sub)
    step(f"multiply {src.name} x{n}")
    print(f'PROGRESS {{"done": 0, "total": {n}, "ok": 0, "slip": 0}}', flush=True)
    run_augment(str(src), n=n, render_n=render_n, out_dir=str(SIM), lerobot_out=None, task=task, seed=seed)
    publish_variants()
    publish_index()
    step("metrics")
    try:
        metrics()
    except Exception as e:
        print("metrics failed:", e, flush=True)


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
def metrics() -> dict:
    reals = [str(RAW / f"{e}.parquet") for e in raw_episodes()]
    rep_p = SIM / "sim_to_real_report.json"
    if reals and (SIM / "episodes").exists():
        step(f"sim-to-real on {len(reals)} real episodes")
        subprocess.run([sys.executable, "-m", "sim.sim_to_real", "--sim-dir", str(SIM / "episodes"),
                        "--real", *reals, "--out-json", str(rep_p)], cwd=ROOT, check=True)
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    bal = rep.get("zero_shot_balanced_accuracy")
    auc = rep.get("zero_shot_roc_auc")
    bars = {"baseline": 50, "sim": round(100 * bal) if bal is not None else 0}
    try:
        bars.update(extra_bars(reals))
    except Exception as e:
        print("extra bars failed:", e, flush=True)
    m = {"acc": round(100 * bal) if bal is not None else 0, "auc": round(auc, 2) if auc is not None else 0,
         "acc_metric": "balanced_accuracy", "bars": bars, "n_real": len(reals),
         "n_sim": rep.get("n_sim_episodes"),
         "synthetic": all("synth" in Path(r).stem for r in reals)}
    m.update(drift_data(reals))
    (PUB / "metrics.json").write_text(json.dumps(m, indent=2))
    publish_index()
    print(json.dumps({k: v for k, v in m.items() if k != "drift"}), flush=True)
    return m


def extra_bars(reals: list[str]) -> dict:
    """Real-only (leave-one-episode-out on real) and sim+real, as balanced accuracy %."""
    import sim.sim_to_real as s2r
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    if len(reals) < 2:
        return {"real": 0, "simreal": 0}

    def windows(p, sim):
        df = s2r.to_100hz(s2r.read_any(p))
        X, st = s2r.window_features(df)
        X = s2r.episode_norm(X)
        flag = s2r.sim_sample_labels(df) if sim else s2r.real_sample_labels(df)
        return X, s2r.window_labels(flag, st)
    real = [windows(p, False) for p in reals]
    sims = [windows(str(p), True) for p in sorted((SIM / "episodes").glob("sim_v*.parquet"))
            if not p.stem.endswith("_30fps")]
    Xs = np.vstack([x for x, _ in sims]) if sims else np.zeros((0, real[0][0].shape[1]))
    ys = np.concatenate([y for _, y in sims]) if sims else np.zeros(0)
    r_only, both = [], []
    for k in range(len(real)):
        Xt, yt = real[k]
        if len(np.unique(yt)) < 2:
            continue
        Xr = np.vstack([real[i][0] for i in range(len(real)) if i != k])
        yr = np.concatenate([real[i][1] for i in range(len(real)) if i != k])
        for X, y, acc in ((Xr, yr, r_only), (np.vstack([Xs, Xr]), np.concatenate([ys, yr]), both)):
            if len(np.unique(y)) < 2:
                continue
            clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced_subsample",
                                         random_state=42, n_jobs=-1).fit(X, y)
            acc.append(balanced_accuracy_score(yt, clf.predict(Xt)))
    return {"real": round(100 * np.mean(r_only)) if r_only else 0, "simreal": round(100 * np.mean(both)) if both else 0}


def drift_data(reals: list[str]) -> dict:
    """Ship screen: camera wrist path vs naive IMU double integration, top-down, normalized to 0..1."""
    if not reals:
        return {}
    eid = Path(max(reals, key=lambda p: pd.read_parquet(p, columns=["t_rel_s"])["t_rel_s"].iloc[-1])).stem
    df = pd.read_parquet(RAW / f"{eid}.parquet")
    df = df[df["t_rel_s"] <= 10.0]
    try:
        from live_viz import drift_path
        t, xyz = drift_path(df)
    except Exception:
        t = df["t_rel_s"].to_numpy()
        a = df[["ax", "ay", "az"]].to_numpy() - df[["ax", "ay", "az"]].iloc[:50].mean().to_numpy()
        dt = np.gradient(t)
        xyz = np.cumsum(np.cumsum(a * dt[:, None], 0) * dt[:, None], 0)
    imu_m = np.asarray(xyz)[:, :2]
    cam_m = np.zeros_like(imu_m)
    hp = HANDS / f"{eid}_hands.parquet"
    al = ALIGNED / f"{eid}.parquet"
    if hp.exists() and al.exists():
        h = pd.read_parquet(hp)
        side = "right" if h.get("right_present", pd.Series([0])).sum() >= h.get("left_present", pd.Series([0])).sum() else "left"
        ok = h[f"{side}_present"] > 0
        if ok.sum() > 10:
            off = json.loads((ALIGNED / f"{eid}.sync.json").read_text()).get("offset_s", 0.0)
            tv = h.loc[ok, "t_s"].to_numpy() + off
            for i, c in enumerate(("x0", "y0")):
                v = h.loc[ok, f"{side}_{c}"].to_numpy() * 0.6  # ~60 cm field of view on the table
                cam_m[:, i] = np.interp(t, tv, v - v[0])
    end_err = float(np.linalg.norm(imu_m[-1] - cam_m[-1]))
    allp = np.vstack([imu_m, cam_m])
    lo, span = allp.min(0), max(float((allp.max(0) - allp.min(0)).max()), 1e-3)
    norm = lambda P: [[round(float(.05 + .9 * (x - lo[0]) / span), 4), round(float(.05 + .9 * (y - lo[1]) / span), 4)]
                      for x, y in P[:: max(1, len(P) // 100)]]
    return {"drift_m": round(end_err, 2), "drift": {"cam": norm(cam_m), "imu": norm(imu_m), "episode": eid,
                                                    "seconds": round(float(t[-1] - t[0]), 1)}}


# --------------------------------------------------------------------------- export
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
    ap.add_argument("job", choices=["process", "process_all", "multiply", "metrics", "export", "index"])
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
    elif a.job == "metrics":
        metrics()
    elif a.job == "export":
        export()
    elif a.job == "index":
        publish_index()
    step(f"finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
