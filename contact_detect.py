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


def motion_energy(out: pd.DataFrame) -> np.ndarray:
    """0.2 s smoothed |a| + 0.3 |gyro|: low while the hand rests, a burst for every reach / pick-up."""
    return _rolling((out["accel_mag"] + 0.3 * out["gyro_mag"]).to_numpy(float), 20, "mean")


def motion_bursts(e: np.ndarray, t: np.ndarray, still_thresh=None, move_thresh=None,
                  min_still_s: float = 0.25, min_move_s: float = 0.12):
    """Hysteresis segmentation into [start, end) index pairs of movement bursts.
    Thresholds adapt to this recording's noise floor (20th percentile of the energy)."""
    floor = float(np.percentile(e, 20)) if len(e) else 0.0
    lo = still_thresh if still_thresh is not None else max(0.18, 2.5 * floor)
    hi = move_thresh if move_thresh is not None else max(0.35, 1.8 * lo)
    bursts, moving, s0, last_still = [], False, None, 0
    i, n = 0, len(e)
    while i < n:
        if not moving:
            if e[i] > hi:
                # need a calm stretch before (or the start of the recording), and the move must persist
                j = i
                while j < n and e[j] > lo and t[j] - t[i] < min_move_s:
                    j += 1
                if j >= n or t[j] - t[i] >= min_move_s:
                    moving, s0 = True, i
                    i = j
                    continue
        else:
            if e[i] < lo:
                j = i
                while j < n and e[j] < lo and t[j] - t[i] < min_still_s:
                    j += 1
                if j < n and t[j] - t[i] >= min_still_s:
                    bursts.append((s0, i))
                    moving = False
                    i = j
                    continue
                if j >= n:
                    break
        i += 1
    if moving:
        bursts.append((s0, None))  # still moving at the end of the window
    return bursts, lo, hi


def detect_contacts(df: pd.DataFrame, jerk_thresh: float = 20.0, accel_peak_thresh: float = 1.5,
                    slip_hpf_thresh: float = 0.6, refractory_s: float = 0.25):
    """Return (labelled copy of df, list of event dicts).

    grab_onset    : start of a movement burst after the hand was at rest (reach / pick up)
    set_down      : the hand comes back to rest after a burst
    impact_or_tap : a sharp jerk + acceleration spike inside a burst (knock, tap, clap)
    slip_or_scrape: sustained high-pass vibration
    """
    out = add_features(df)
    n = len(out)
    t = time_axis(out)
    jerk = out["jerk_mag"].to_numpy()
    amag = out["accel_mag"].to_numpy()
    gmag = out["gyro_mag"].to_numpy()
    slip = out["slip_score"].to_numpy()
    amag_peak = _rolling(amag, 7, "max")
    e = motion_energy(out)
    bursts, lo, hi = motion_bursts(e, t)
    ids = np.where(e > lo, EVENT_IDS["motion"], EVENT_IDS["idle"]).astype(int)

    def ev(kind, k):
        return {"event": kind, "t_s": float(t[k]), "sample_idx": int(k), "jerk_peak_ms3": float(jerk[k]),
                "accel_peak_ms2": float(amag_peak[k]), "gyro_mag_rads": float(gmag[k])}

    # 1. TAP: impulsive spike, far above the local (0.5 s) acceleration level
    base = _rolling(amag, 50, "median")
    ratio = amag_peak / (base + 0.15)
    jt = max(jerk_thresh * 5.0, 8.0 * float(np.median(jerk)) if n else 0.0)   # ~100 m/s^3 at defaults
    cand = np.where((jerk > jt) & (amag_peak > 1.7 * accel_peak_thresh) & (ratio > 2.5))[0]
    taps = []
    i = 0
    while i < len(cand):
        j = i
        while j + 1 < len(cand) and t[cand[j + 1]] - t[cand[j]] < 0.15:
            j += 1
        seg = cand[i:j + 1]
        k = int(seg[np.argmax(jerk[seg])])
        i = j + 1
        if not taps or t[k] - t[taps[-1]] >= refractory_s:
            taps.append(k)
    events = [ev("impact_or_tap", k) for k in taps]
    tap_t = np.array([t[k] for k in taps])

    # 2. GRAB / SET-DOWN from movement bursts; a short burst that is just a tap's ringing is not a grab
    for s0, s1 in bursts:
        t1 = t[s1] if s1 is not None else t[-1]
        has_tap = len(tap_t) and np.any((tap_t >= t[s0] - 0.15) & (tap_t <= t1 + 0.15))
        if has_tap and t1 - t[s0] < 0.8:
            continue
        if s0 > 0 or e[0] < lo:  # a burst that starts at sample 0 has no known rest before it
            if not (len(tap_t) and np.min(np.abs(tap_t - t[s0])) < 0.15):
                events.append(ev("grab_onset", s0))
        if s1 is not None and not (len(tap_t) and np.min(np.abs(tap_t - t[s1])) < 0.15):
            events.append(ev("set_down", s1))

    # 3. SLIP / SCRAPE: sustained high-frequency vibration that dominates the gross motion
    lp = _rolling(amag, 50, "mean")
    vt = max(slip_hpf_thresh, 3.0 * float(np.median(slip)) if n else 0.0)
    hot = (slip > vt) & (slip > 0.55 * (lp + 0.1))
    k = 0
    while k < n:
        if hot[k]:
            e2 = k
            while e2 + 1 < n and hot[e2 + 1]:
                e2 += 1
            if t[e2] - t[k] >= 0.15 and not (len(tap_t) and np.any(np.abs(tap_t - t[k]) < 0.2)):
                m = int(k + np.argmax(slip[k:e2 + 1]))
                events.append(ev("slip_or_scrape", m))
                ids[k:e2 + 1] = EVENT_IDS["slip_or_scrape"]
            k = e2 + 1
        else:
            k += 1
    events.sort(key=lambda x: x["t_s"])
    for x in events:
        k = x["sample_idx"]
        ids[max(0, k - 5):min(n, k + 6)] = EVENT_IDS[x["event"]]
    out["motion_energy"] = e
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
