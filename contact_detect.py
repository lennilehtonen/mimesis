"""Contact event detector for 100 Hz wrist IMU (real Sensor Logger or simulated).

Labels per sample: idle(0) motion(1) grab_onset(2) impact_or_tap(3) set_down(4) slip_or_scrape(5).

    python contact_detect.py data/raw_imu/ep_001.parquet --out data/raw_imu/ep_001_contacts.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_IDS = {"idle": 0, "motion": 1, "grab_onset": 2, "impact_or_tap": 3, "set_down": 4, "slip_or_scrape": 5}
EVENT_NAMES = {v: k for k, v in EVENT_IDS.items()}
UI_KIND = {"grab_onset": "GRAB", "impact_or_tap": "TAP", "set_down": "SET-DOWN", "slip_or_scrape": "SLIP"}


def time_axis(df: pd.DataFrame) -> np.ndarray:
    """Seconds, starting at 0, from t_rel_s / t_s / timestamp_ns (in that order)."""
    if "t_rel_s" in df:
        t = df["t_rel_s"].to_numpy(float)
    elif "t_s" in df:
        t = df["t_s"].to_numpy(float)
    elif "timestamp_ns" in df:
        ns = df["timestamp_ns"].to_numpy(np.int64)
        t = (ns - ns[0]) / 1e9
    else:
        t = np.arange(len(df)) / 100.0
    return t - t[0] if len(t) else t


def _rolling(x: np.ndarray, w: int, fn: str) -> np.ndarray:
    s = pd.Series(x).rolling(w, center=True, min_periods=1)
    return getattr(s, fn)().to_numpy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    a = out[["ax", "ay", "az"]].to_numpy(float)
    a = np.nan_to_num(a)
    g = np.nan_to_num(out[["gx", "gy", "gz"]].to_numpy(float)) if {"gx", "gy", "gz"} <= set(out) else np.zeros_like(a)
    t = time_axis(out)
    dt = np.diff(t, prepend=t[0] - 0.01) if len(t) else t
    dt = np.where(dt <= 1e-4, 0.01, dt)
    da = np.diff(a, axis=0, prepend=a[:1])
    out["accel_mag"] = np.linalg.norm(a, axis=1)
    out["gyro_mag"] = np.linalg.norm(g, axis=1)
    out["jerk_mag"] = np.linalg.norm(da, axis=1) / dt
    hp = a - np.stack([_rolling(a[:, i], 10, "mean") for i in range(3)], axis=1)
    out["hpf_accel_mag"] = np.linalg.norm(hp, axis=1)
    out["slip_score"] = np.sqrt(_rolling(out["hpf_accel_mag"].to_numpy() ** 2, 20, "mean"))
    return out


def detect_contacts(df: pd.DataFrame, jerk_thresh: float = 20.0, accel_peak_thresh: float = 1.5,
                    slip_hpf_thresh: float = 0.6, refractory_s: float = 0.25):
    """Return (labelled copy of df, list of event dicts)."""
    out = add_features(df)
    n = len(out)
    t = time_axis(out)
    jerk = out["jerk_mag"].to_numpy()
    amag = out["accel_mag"].to_numpy()
    gmag = out["gyro_mag"].to_numpy()
    slip = out["slip_score"].to_numpy()
    amag_peak = _rolling(amag, 7, "max")
    moving = (_rolling(amag, 15, "mean") > 0.35) | (_rolling(gmag, 15, "mean") > 0.35)
    ids = np.where(moving, EVENT_IDS["motion"], EVENT_IDS["idle"]).astype(int)

    events = []
    last_t = -1e9
    cand = np.where((jerk > jerk_thresh) & (amag_peak > accel_peak_thresh))[0]
    i = 0
    while i < len(cand):
        j = i
        while j + 1 < len(cand) and t[cand[j + 1]] - t[cand[j]] < 0.08:
            j += 1
        seg = cand[i:j + 1]
        k = int(seg[np.argmax(jerk[seg])])
        i = j + 1
        if t[k] - last_t < refractory_s:
            continue
        last_t = t[k]
        after = slice(k + 5, min(n, k + 60))
        before = slice(max(0, k - 60), max(0, k - 5))
        still_after = n > k + 20 and np.mean(moving[after]) < 0.35
        still_before = k > 20 and np.mean(moving[before]) < 0.35
        if not events and not still_after:
            kind = "grab_onset"
        elif still_after:
            kind = "set_down"
        elif still_before:
            kind = "grab_onset"
        else:
            kind = "impact_or_tap"
        events.append({"event": kind, "t_s": float(t[k]), "sample_idx": k, "jerk_peak_ms3": float(jerk[k]),
                       "accel_peak_ms2": float(amag_peak[k]), "gyro_mag_rads": float(gmag[k])})

    # slip / scrape: sustained high-pass vibration not explained by a single impact
    hot = slip > slip_hpf_thresh
    k = 0
    while k < n:
        if hot[k]:
            e = k
            while e + 1 < n and hot[e + 1]:
                e += 1
            if t[e] - t[k] >= 0.15 and not any(abs(ev["t_s"] - t[k]) < 0.12 for ev in events):
                m = int(k + np.argmax(slip[k:e + 1]))
                events.append({"event": "slip_or_scrape", "t_s": float(t[m]), "sample_idx": m,
                               "jerk_peak_ms3": float(jerk[m]), "accel_peak_ms2": float(amag_peak[m]),
                               "gyro_mag_rads": float(gmag[m])})
                ids[k:e + 1] = EVENT_IDS["slip_or_scrape"]
            k = e + 1
        else:
            k += 1
    events.sort(key=lambda ev: ev["t_s"])
    for ev in events:
        k = ev["sample_idx"]
        ids[max(0, k - 5):min(n, k + 6)] = EVENT_IDS[ev["event"]]
    out["contact_event_id"] = ids
    out["contact_event"] = [EVENT_NAMES[v] for v in ids]
    out["contact_flag"] = (ids >= 2).astype(int)
    return out, events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--out")
    ap.add_argument("--events-json")
    ap.add_argument("--jerk-thresh", type=float, default=20.0)
    ap.add_argument("--accel-thresh", type=float, default=1.5)
    ap.add_argument("--slip-thresh", type=float, default=0.6)
    a = ap.parse_args()
    p = Path(a.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    out, events = detect_contacts(df, a.jerk_thresh, a.accel_thresh, a.slip_thresh)
    op = Path(a.out) if a.out else p.with_name(p.stem + "_contacts.parquet")
    out.to_parquet(op, index=False)
    ej = Path(a.events_json) if a.events_json else op.with_suffix(".events.json")
    ej.write_text(json.dumps({"input": str(p), "num_events": len(events), "events": events}, indent=2))
    print(f"{len(events)} events -> {op}, {ej}")
    for ev in events:
        print(f"  {ev['t_s']:7.2f}s  {ev['event']:15s} jerk={ev['jerk_peak_ms3']:.0f}")


if __name__ == "__main__":
    main()
