"""Train a contact-window classifier on simulated IMU, evaluate zero-shot on real IMU.

    python -m sim.sim_to_real --sim-dir data/sim_aug/episodes \
        --real data/raw_imu/ep_001.parquet --out-json data/sim_aug/sim_to_real_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from contact_detect import detect_contacts, time_axis  # noqa: E402

WIN, STRIDE = 20, 5
FEATURE_NAMES = [
    "amag_mean", "amag_std", "amag_max", "amag_ptp",
    "gmag_mean", "gmag_std", "gmag_max", "gmag_ptp",
    "jerk_mean", "jerk_std", "jerk_max",
    "da_mean", "da_max", "dw_mean", "dw_max",
    "jerk_p90",
]


def read_any(p) -> pd.DataFrame:
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def to_100hz(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to a uniform 100 Hz grid if the input is not already ~100 Hz (e.g. 30 fps aligned files)."""
    t = time_axis(df)
    if len(t) < 3:
        return df
    dt = float(np.median(np.diff(t)))
    if 0.008 <= dt <= 0.012:
        out = df.copy()
        out["t_rel_s"] = t
        return out
    tg = np.arange(0.0, t[-1] + 1e-9, 0.01)
    out = pd.DataFrame({"t_rel_s": tg})
    for c in ["ax", "ay", "az", "gx", "gy", "gz", "sim_contact_force_n"]:
        if c in df:
            out[c] = np.interp(tg, t, pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float))
    return out


def window_features(df: pd.DataFrame):
    a = np.nan_to_num(df[["ax", "ay", "az"]].to_numpy(float))
    w = np.nan_to_num(df[["gx", "gy", "gz"]].to_numpy(float)) if {"gx", "gy", "gz"} <= set(df) else np.zeros_like(a)
    t = time_axis(df)
    dt = np.diff(t, prepend=t[0] - 0.01)
    dt = np.where(dt <= 1e-4, 0.01, dt)
    amag, gmag = np.linalg.norm(a, axis=1), np.linalg.norm(w, axis=1)
    da = np.linalg.norm(np.diff(a, axis=0, prepend=a[:1]), axis=1)
    dw = np.linalg.norm(np.diff(w, axis=0, prepend=w[:1]), axis=1)
    jerk = da / dt
    starts = list(range(0, max(0, len(df) - WIN + 1), STRIDE))
    X = np.zeros((len(starts), 16))
    for i, s in enumerate(starts):
        sl = slice(s, s + WIN)
        A, Gm, J = amag[sl], gmag[sl], jerk[sl]
        X[i] = [A.mean(), A.std(), A.max(), np.ptp(A), Gm.mean(), Gm.std(), Gm.max(), np.ptp(Gm),
                J.mean(), J.std(), J.max(), da[sl].mean(), da[sl].max(), dw[sl].mean(), dw[sl].max(),
                np.percentile(J, 90)]
    return X, starts


def episode_norm(X: np.ndarray) -> np.ndarray:
    """Label-free domain adaptation: robust-scale each feature within one episode (median / IQR).
    Removes per-device and per-mount gain/offset differences between MuJoCo IMU and the real phone."""
    if len(X) == 0:
        return X
    med = np.median(X, 0)
    iqr = np.percentile(X, 75, 0) - np.percentile(X, 25, 0)
    return (X - med) / (iqr + 1e-6)


def window_labels(flag: np.ndarray, starts) -> np.ndarray:
    return np.array([int(flag[s:s + WIN].max() > 0) for s in starts], dtype=int)


def sim_sample_labels(df: pd.DataFrame) -> np.ndarray:
    lab, _ = detect_contacts(df, jerk_thresh=15.0, accel_peak_thresh=1.2, slip_hpf_thresh=0.5)
    flag = lab["contact_flag"].to_numpy(int)
    if "sim_contact_force_n" in df:
        F = df["sim_contact_force_n"].to_numpy(float)
        dF = np.abs(np.diff(F, prepend=F[:1]))
        flag = flag | ((dF > 0.35) | (F > 4.0)).astype(int)
    return flag


def real_sample_labels(df: pd.DataFrame) -> np.ndarray:
    lab, _ = detect_contacts(df)
    return lab["contact_flag"].to_numpy(int)


def run(sim_dir, real_paths, out_json=None) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    sim_files = sorted(p for p in Path(sim_dir).glob("*.parquet") if not p.name.endswith("_30fps.parquet"))
    if not sim_files:
        raise SystemExit(f"no simulated episodes in {sim_dir}")
    Xs, ys = [], []
    for p in sim_files:
        df = to_100hz(read_any(p))
        X, st = window_features(df)
        X = episode_norm(X)
        if len(st):
            Xs.append(X)
            ys.append(window_labels(sim_sample_labels(df), st))
    X_tr, y_tr = np.vstack(Xs), np.concatenate(ys)

    Xr, yr, real_used = [], [], []
    for p in real_paths:
        df = to_100hz(read_any(p))
        X, st = window_features(df)
        X = episode_norm(X)
        if len(st):
            Xr.append(X)
            yr.append(window_labels(real_sample_labels(df), st))
            real_used.append(str(p))
    if not Xr:
        raise SystemExit("no usable real windows")
    X_te, y_te = np.vstack(Xr), np.concatenate(yr)

    clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced_subsample",
                                 random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    classes = list(clf.classes_)
    proba = clf.predict_proba(X_te)[:, classes.index(1)] if 1 in classes else np.zeros(len(X_te))

    auc = None
    if len(np.unique(y_te)) == 2:
        try:
            auc = float(roc_auc_score(y_te, proba))
        except ValueError:
            auc = None
    try:
        bal = float(balanced_accuracy_score(y_te, pred)) if len(np.unique(np.concatenate([y_te, pred]))) > 1 \
            else float(accuracy_score(y_te, pred))
    except Exception:
        bal = None

    report = {
        "n_sim_episodes": len(sim_files),
        "real_files": real_used,
        "window_samples": WIN, "stride_samples": STRIDE,
        "features": FEATURE_NAMES,
        "train_windows": int(len(y_tr)),
        "test_windows": int(len(y_te)),
        "train_positive_rate": float(y_tr.mean()) if len(y_tr) else None,
        "test_positive_rate": float(y_te.mean()) if len(y_te) else None,
        "pred_positive_rate": float(pred.mean()) if len(pred) else None,
        "train_classes": [int(c) for c in classes],
        "zero_shot_accuracy": float(accuracy_score(y_te, pred)),
        "zero_shot_balanced_accuracy": bal,
        "zero_shot_f1": float(f1_score(y_te, pred, zero_division=0)),
        "zero_shot_roc_auc": auc,
        "feature_norm": "per-episode median/IQR (no labels)",
        "model": "RandomForestClassifier(n_estimators=150, max_depth=8, class_weight=balanced_subsample)",
        "feature_importance": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, clf.feature_importances_)},
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", default="data/sim_aug/episodes")
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--out-json", default="data/sim_aug/sim_to_real_report.json")
    a = ap.parse_args()
    r = run(a.sim_dir, a.real, a.out_json)
    print(json.dumps({k: r[k] for k in ("n_sim_episodes", "train_windows", "test_windows", "train_positive_rate",
                                         "test_positive_rate", "zero_shot_accuracy", "zero_shot_balanced_accuracy",
                                         "zero_shot_f1", "zero_shot_roc_auc")}, indent=2))
    print(f"-> {a.out_json}")


if __name__ == "__main__":
    main()
