"""Clap-sync chest-cam video (audio) with wrist IMU (jerk), then align IMU onto video frames.

    python sync.py --video data/videos/ep_001.mov --imu data/raw_imu/ep_001.parquet \
        --hands data/hands/ep_001_hands.parquet --out data/aligned/ep_001.parquet \
        --plot data/aligned/ep_001_sync.png --search-window 12.0 --task pick_place_mug

Convention: t_imu = t_video + offset_s.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks

from contact_detect import EVENT_NAMES, detect_contacts, time_axis

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
AUDIO_SR = 16000

CONT_COLS = ["ax", "ay", "az", "gx", "gy", "gz", "qx", "qy", "qz", "qw", "roll", "pitch", "yaw",
             "grav_x", "grav_y", "grav_z", "accel_mag", "gyro_mag", "jerk_mag", "hpf_accel_mag", "slip_score"]
DISC_COLS = ["contact_event_id", "contact_flag"]


# ---------------------------------------------------------------- media helpers
def extract_audio(video_path: str | Path, sr: int = AUDIO_SR) -> np.ndarray:
    """ffmpeg -vn -ac 1 -ar 16000 -> float32 mono in [-1, 1]."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1",
                        "-ar", str(sr), "-acodec", "pcm_s16le", str(wav)], check=True)
        with wave.open(str(wav), "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x.astype(np.float32) / 32768.0


def probe_video(video_path: str | Path) -> tuple[float, int]:
    """(fps, num_frames) via ffprobe, falling back to OpenCV."""
    fps, n = 0.0, 0
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=r_frame_rate,nb_frames,duration", "-of", "json", str(video_path)],
                             capture_output=True, text=True, check=True).stdout
        st = json.loads(out)["streams"][0]
        num, den = st["r_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        if st.get("nb_frames") not in (None, "N/A"):
            n = int(st["nb_frames"])
        elif st.get("duration") not in (None, "N/A") and fps:
            n = int(round(float(st["duration"]) * fps))
    except Exception:
        pass
    if not fps or not n:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = fps or float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n = n or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return fps, n


# ---------------------------------------------------------------- envelopes
def _normalize(env: np.ndarray) -> np.ndarray:
    env = env - np.median(env)
    env = np.clip(env, 0, None)
    m = env.max() if len(env) else 0.0
    if m > 0:
        env = env / m
    return env ** 2


def audio_envelope(audio: np.ndarray, sr: int, search_window: float, bin_s: float = 0.01) -> np.ndarray:
    hop = int(round(sr * bin_s))
    nb = int(np.ceil(len(audio) / hop))
    pad = np.zeros(nb * hop, np.float32)
    pad[:len(audio)] = np.abs(audio)
    env = pad.reshape(nb, hop).max(axis=1)
    env[int(search_window / bin_s):] = 0.0
    return _normalize(env)


def imu_jerk(imu: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = time_axis(imu)
    a = np.nan_to_num(imu[["ax", "ay", "az"]].to_numpy(float))
    dt = np.diff(t, prepend=t[0] - 0.01)
    dt = np.where(dt <= 1e-4, 0.01, dt)
    jerk = np.linalg.norm(np.diff(a, axis=0, prepend=a[:1]), axis=1) / dt
    return t, jerk


def imu_envelope(t: np.ndarray, jerk: np.ndarray, t_max: float, bin_s: float = 0.01) -> np.ndarray:
    grid = np.arange(0, t_max, bin_s)
    # max in bin (so a 1-sample spike survives), then interpolate gaps
    idx = np.floor(t / bin_s).astype(int)
    env = np.full(len(grid), np.nan)
    ok = (idx >= 0) & (idx < len(grid))
    tmp = pd.Series(jerk[ok]).groupby(idx[ok]).max()
    env[tmp.index.to_numpy()] = tmp.to_numpy()
    good = ~np.isnan(env)
    env = np.interp(grid, grid[good], env[good]) if good.any() else np.zeros(len(grid))
    return _normalize(env)


def _peaks(env: np.ndarray, bin_s: float, height: float = 0.25) -> np.ndarray:
    pk, _ = find_peaks(env, height=height, distance=max(1, int(0.2 / bin_s)))
    return pk


def estimate_clap_offset(audio: np.ndarray, imu: pd.DataFrame, sr: int = AUDIO_SR, search_window: float = 12.0,
                         bin_s: float = 0.01, max_lag_s: float = 8.0) -> dict:
    """Return dict with offset_s (t_imu = t_video + offset_s) and diagnostics."""
    a_env = audio_envelope(audio, sr, search_window, bin_s)
    a_env = a_env[:int(search_window / bin_s)]
    t_imu, jerk = imu_jerk(imu)
    i_win = search_window + max_lag_s
    m = t_imu < i_win
    i_env = imu_envelope(t_imu[m], jerk[m], min(i_win, t_imu[m].max() + bin_s) if m.any() else i_win, bin_s)

    corr = correlate(i_env, a_env, mode="full")
    lags = (np.arange(len(corr)) - (len(a_env) - 1)) * bin_s
    valid = np.abs(lags) <= max_lag_s
    corr_v = np.where(valid, corr, -np.inf)
    k = int(np.argmax(corr_v))
    xcorr_offset = float(lags[k])
    denom = np.sqrt((a_env ** 2).sum() * (i_env ** 2).sum()) or 1.0
    corr_peak = float(corr[k] / denom)

    a_pk = _peaks(a_env, bin_s)
    i_pk = _peaks(i_env, bin_s)
    offset = xcorr_offset
    t_clap_video = t_clap_imu = None
    if len(a_pk):
        t_clap_video = float(a_pk[0] * bin_s)
        if len(i_pk):
            target = t_clap_video + xcorr_offset
            j = int(np.argmin(np.abs(i_pk * bin_s - target)))
            if abs(i_pk[j] * bin_s - target) < 0.25:
                t_clap_imu = float(i_pk[j] * bin_s)
                offset = t_clap_imu - t_clap_video
    if t_clap_imu is None and t_clap_video is not None:
        t_clap_imu = t_clap_video + offset
    # claps = peaks that fall inside the first few seconds around the first clap
    n_a = int(np.sum((a_pk * bin_s >= (t_clap_video or 0)) & (a_pk * bin_s <= (t_clap_video or 0) + 4.0)))
    n_i = int(np.sum((i_pk * bin_s >= (t_clap_imu or 0)) & (i_pk * bin_s <= (t_clap_imu or 0) + 4.0)))
    return {"offset_s": float(round(offset, 4)), "xcorr_offset_s": float(round(xcorr_offset, 4)),
            "t_clap_video_s": t_clap_video, "t_clap_imu_s": t_clap_imu, "num_audio_claps": n_a,
            "num_imu_claps": n_i, "corr_peak": corr_peak,
            "_audio_env": a_env, "_imu_env": i_env, "_lags": lags[valid], "_corr": corr[valid], "_bin_s": bin_s}


# ---------------------------------------------------------------- alignment
def align_episode(imu_labelled: pd.DataFrame, offset_s: float, fps: float, num_frames: int,
                  hands: pd.DataFrame | None = None) -> pd.DataFrame:
    if hands is not None and len(hands):
        out = hands.copy().reset_index(drop=True)
        if "t_s" not in out:
            out["t_s"] = out["frame_idx"] / fps
    else:
        out = pd.DataFrame({"frame_idx": np.arange(num_frames), "t_s": np.arange(num_frames) / fps})
    t_v = out["t_s"].to_numpy(float)
    t_imu = t_v + offset_s
    out["t_imu_s"] = t_imu
    src_t = time_axis(imu_labelled)
    for c in CONT_COLS:
        if c in imu_labelled:
            y = imu_labelled[c].to_numpy(float)
            out[c] = np.interp(t_imu, src_t, y, left=np.nan, right=np.nan)
    half = 0.5 / fps
    lo = np.searchsorted(src_t, t_imu - half, side="left")
    hi = np.searchsorted(src_t, t_imu + half, side="right")
    for c in DISC_COLS:
        y = imu_labelled[c].to_numpy(int)
        vals = np.zeros(len(t_imu), dtype=np.int64)
        for i, (a, b) in enumerate(zip(lo, hi)):
            if b > a:
                vals[i] = y[a:b].max()
            elif 0 <= a < len(y) and src_t[0] <= t_imu[i] <= src_t[-1]:
                vals[i] = y[min(a, len(y) - 1)]
        out[c] = vals
    out["contact_event"] = [EVENT_NAMES.get(int(v), "idle") for v in out["contact_event_id"]]
    return out


def save_plot(est: dict, imu_events: list, offset: float, path: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = est["_bin_s"]
    a_env, i_env = est["_audio_env"], est["_imu_env"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    ta = np.arange(len(a_env)) * b
    ti = np.arange(len(i_env)) * b
    ax[0].plot(ta, a_env, color="#4c8bf5", lw=1, label="audio envelope (video clock)")
    ax[0].plot(ti, i_env, color="#FFB547", lw=1, alpha=0.8, label="IMU jerk envelope (IMU clock)")
    ax[0].set_title("Raw envelopes (unsynced)")
    ax[0].legend(loc="upper right")
    ax[1].plot(ta, a_env, color="#4c8bf5", lw=1, label="audio")
    ax[1].plot(ti - offset, i_env, color="#FFB547", lw=1, alpha=0.8, label=f"IMU shifted by -{offset:+.3f}s")
    for pk in _peaks(a_env, b):
        ax[1].axvline(pk * b, color="k", ls=":", lw=0.8)
    ax[1].set_xlim(0, len(a_env) * b)
    ax[1].set_title(f"Aligned: t_imu = t_video {offset:+.3f} s  (clap lines dotted)")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("video time (s)")
    ax[2].plot(est["_lags"], est["_corr"], color="#333")
    ax[2].axvline(est["xcorr_offset_s"], color="r", ls="--", label=f"xcorr {est['xcorr_offset_s']:+.3f}s")
    ax[2].axvline(offset, color="g", ls=":", label=f"snapped {offset:+.3f}s")
    ax[2].set_xlabel("lag (s)")
    ax[2].set_title(f"Cross-correlation (peak {est['corr_peak']:.2f})")
    ax[2].legend(loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _read_table(p) -> pd.DataFrame:
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def run_sync(video, imu, hands=None, out=None, plot=None, search_window: float = 12.0, task: str | None = None,
             max_lag_s: float = 8.0) -> dict:
    """Clap-sync + align. Writes aligned parquet and <stem>.sync.json. Returns the sync.json dict."""
    video = Path(video)
    imu_df = imu if isinstance(imu, pd.DataFrame) else _read_table(imu)
    hands_df = None if hands is None else (hands if isinstance(hands, pd.DataFrame) else _read_table(hands))
    audio = extract_audio(video)
    est = estimate_clap_offset(audio, imu_df, AUDIO_SR, search_window, 0.01, max_lag_s)
    offset = est["offset_s"]
    labelled, events = detect_contacts(imu_df)
    fps, nf = probe_video(video)
    if hands_df is not None and len(hands_df) and "t_s" in hands_df and len(hands_df) > 1:
        fps_h = 1.0 / np.median(np.diff(hands_df["t_s"].to_numpy(float)))
        fps = fps if fps else fps_h
    aligned = align_episode(labelled, offset, fps, nf, hands_df)
    out = Path(out) if out else Path("data/aligned") / (video.stem + ".parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(out, index=False)
    evs = [{**e, "t_imu_s": e["t_s"], "t_video_s": float(e["t_s"] - offset)} for e in events]
    for e in evs:
        e.pop("t_s", None)
    meta = {k: v for k, v in est.items() if not k.startswith("_")}
    meta.update({"fps": float(fps), "num_frames": int(len(aligned)), "events": evs,
                 "video": str(video), "imu": str(imu) if not isinstance(imu, pd.DataFrame) else None,
                 "hands": str(hands) if isinstance(hands, (str, Path)) else None, "aligned_parquet": str(out)})
    if task:
        meta["task"] = task
    if plot:
        save_plot(est, events, offset, plot)
        meta["plot"] = str(plot)
    sj = out.with_name(out.stem + ".sync.json")
    sj.write_text(json.dumps(meta, indent=2))
    meta["sync_json"] = str(sj)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--imu", required=True)
    ap.add_argument("--hands")
    ap.add_argument("--out")
    ap.add_argument("--plot")
    ap.add_argument("--search-window", type=float, default=12.0)
    ap.add_argument("--max-lag", type=float, default=8.0)
    ap.add_argument("--task")
    a = ap.parse_args()
    meta = run_sync(a.video, a.imu, a.hands, a.out, a.plot, a.search_window, a.task, a.max_lag)
    print(f"offset_s={meta['offset_s']:+.3f} (xcorr {meta['xcorr_offset_s']:+.3f}, peak {meta['corr_peak']:.2f}); "
          f"claps audio={meta['num_audio_claps']} imu={meta['num_imu_claps']}; "
          f"{meta['num_frames']} frames @ {meta['fps']:.2f} fps -> {meta['aligned_parquet']}, {meta['sync_json']}")
    for e in meta["events"]:
        print(f"  video {e['t_video_s']:6.2f}s  {e['event']}")


if __name__ == "__main__":
    main()
