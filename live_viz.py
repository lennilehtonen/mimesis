"""Live / replay IMU dashboard: |a| & |gyro|, contact events, naive double-integration drift.

    python live_viz.py --url http://127.0.0.1:8000/latest
    python live_viz.py --replay data/raw_imu/ep_001.parquet --save-png data/aligned/ep_001_dashboard.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from contact_detect import EVENT_IDS, UI_KIND, detect_contacts, time_axis

AMBER = "#FFB547"
BLUE = "#4c8bf5"
EVENT_COLORS = {"grab_onset": "#2ca02c", "impact_or_tap": "#d62728", "set_down": "#9467bd", "slip_or_scrape": "#ff7f0e"}


def drift_path(df: pd.DataFrame, init_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Naive double integration of ax, ay, az (initial-mean subtracted). Returns (t [N], xyz [N,3]) in s, m."""
    t = time_axis(df)
    if len(t) < 2:
        return t, np.zeros((len(t), 3))
    a = np.nan_to_num(df[["ax", "ay", "az"]].to_numpy(float))
    m = t <= t[0] + init_s
    if m.sum() < 5:
        m = np.arange(len(t)) < min(len(t), 50)
    a = a - a[m].mean(axis=0)
    dt = np.diff(t)[:, None]
    v = np.vstack([np.zeros((1, 3)), np.cumsum(0.5 * (a[1:] + a[:-1]) * dt, axis=0)])
    p = np.vstack([np.zeros((1, 3)), np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, axis=0)])
    return t, p


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    if "contact_event_id" in df and "accel_mag" in df and "gyro_mag" in df:
        events = []
        ids = df["contact_event_id"].to_numpy()
        t = time_axis(df)
        for name, eid in EVENT_IDS.items():
            if eid < 2:
                continue
            on = np.flatnonzero(np.diff(np.r_[0, (ids == eid).astype(int)]) == 1)
            events += [{"event": name, "t_s": float(t[k])} for k in on]
        return df, sorted(events, key=lambda e: e["t_s"])
    return detect_contacts(df)


def draw(fig, df: pd.DataFrame, title: str = ""):
    fig.clf()
    axs = fig.subplots(2, 2)
    if len(df) < 3:
        fig.suptitle(title + "  (waiting for data…)")
        return
    lab, events = _prepare(df)
    t = time_axis(lab)
    ax = axs[0, 0]
    ax.plot(t, lab["accel_mag"], color=AMBER, lw=1, label="|a| (m/s²)")
    ax2 = ax.twinx()
    ax2.plot(t, lab["gyro_mag"], color=BLUE, lw=1, alpha=0.8, label="|gyro| (rad/s)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|a| m/s²", color=AMBER)
    ax2.set_ylabel("|gyro| rad/s", color=BLUE)
    ax.set_title("Accel / gyro magnitude")

    ax = axs[0, 1]
    ax.step(t, lab["contact_event_id"], where="post", color="#555", lw=1)
    for ev in events:
        ax.axvline(ev["t_s"], color=EVENT_COLORS.get(ev["event"], "k"), lw=1.2, alpha=0.8)
        ax.text(ev["t_s"], 5.3, UI_KIND.get(ev["event"], ev["event"]), rotation=90, fontsize=7, va="top",
                color=EVENT_COLORS.get(ev["event"], "k"))
    ax.set_yticks(range(6))
    ax.set_yticklabels(["idle", "motion", "grab", "tap", "set-down", "slip"], fontsize=8)
    ax.set_ylim(-0.3, 5.5)
    ax.set_xlabel("t (s)")
    ax.set_title(f"Contact events ({len(events)})")

    td, p = drift_path(lab)
    ax = axs[1, 0]
    ax.plot(p[:, 0], p[:, 1], color=AMBER, lw=1.5)
    ax.plot(p[0, 0], p[0, 1], "o", color="g", ms=5)
    ax.plot(p[-1, 0], p[-1, 1], "x", color="r", ms=7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Naive double-integration XY path")

    ax = axs[1, 1]
    r = np.linalg.norm(p, axis=1)
    ax.plot(td, r, color="#d62728", lw=1.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|pos| (m)")
    ax.set_title(f"Drift: {r[-1]:.2f} m after {td[-1] - td[0]:.1f} s")
    fig.suptitle(title)
    fig.tight_layout()


def fetch_latest(url: str, n: int = 1000, timeout: float = 2.0) -> pd.DataFrame:
    sep = "&" if "?" in url else "?"
    full = url if "n=" in url else f"{url}{sep}n={n}"
    with urllib.request.urlopen(full, timeout=timeout) as r:
        payload = json.loads(r.read().decode())
    s = pd.DataFrame(payload.get("samples", []))
    if s.empty:
        return s
    if "time_ns" in s:
        s["timestamp_ns"] = s["time_ns"].astype(np.int64)
        s = s.sort_values("timestamp_ns").drop_duplicates("timestamp_ns").reset_index(drop=True)
    for c in ("ax", "ay", "az", "gx", "gy", "gz"):
        if c not in s:
            s[c] = 0.0
    return s


def run_live(url: str, n: int = 1000, interval: float = 0.5, save_png: str | None = None):
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure(figsize=(13, 8))
    print(f"polling {url} every {interval}s; close the window or Ctrl-C to stop", file=sys.stderr)
    df = pd.DataFrame()
    try:
        while plt.fignum_exists(fig.number):
            try:
                df = fetch_latest(url, n)
                draw(fig, df, f"LIVE {url}  ({len(df)} samples)")
            except Exception as e:  # receiver down, bad payload, …
                fig.clf()
                fig.suptitle(f"waiting for {url}: {e}")
            plt.pause(interval)
    except KeyboardInterrupt:
        pass
    if save_png and len(df):
        draw(fig, df, "LIVE snapshot")
        fig.savefig(save_png, dpi=110)


def run_replay(path: str, save_png: str | None = None, show: bool = True):
    import matplotlib
    if save_png and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    fig = plt.figure(figsize=(13, 8))
    draw(fig, df, f"REPLAY {p.name}")
    if save_png:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=110)
        print(f"-> {save_png}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="receiver /latest endpoint, e.g. http://127.0.0.1:8000/latest")
    ap.add_argument("--replay", help="parquet/csv with ax..az (raw IMU, aligned, or sim)")
    ap.add_argument("--save-png")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--show", action="store_true", help="with --replay --save-png: also open a window")
    a = ap.parse_args()
    if a.replay:
        run_replay(a.replay, a.save_png, show=a.show or not a.save_png)
    elif a.url:
        run_live(a.url, a.n, a.interval, a.save_png)
    else:
        ap.error("pass --url or --replay")


if __name__ == "__main__":
    main()
