# Simulation

> **Current pipeline (video-only).** MIMESIS now uses one iPhone as a fixed camera and no wrist IMU. `hands.py` gives 21 MediaPipe landmarks, `sim/hand3d.py` lifts them to a metric 3D hand (solvePnP with iPhone intrinsics, A4 sheet via IPPE for the table frame), `sim/cup_track.py` finds the cup and places it on the table, and `sim/retarget.py` `retarget_exact` makes the gripper copy the hand 1:1 in time. `jobs.py` (`process_video_only`, `multiply_video`) runs it. Sections below on IMU sites, IMU noise and sim-to-real describe the earlier wrist-IMU prototype.

MIMESIS retargets one real (or IMU-only) demonstration into MuJoCo and multiplies it. All simulation code is under `sim/`. Rendering uses `mujoco.Renderer` on the main thread. On headless Linux, `sim.scene.configure_gl_backend()` sets `MUJOCO_GL=egl` when `DISPLAY` is unset; on macOS the default backend is left alone.

## 1. Scene (`sim/scene.py`)

`build_scene_xml(...)` / `load_model_and_data(...)` construct a model named `mimesis_tabletop`.

| Element | Details |
|---|---|
| Integrator | `implicitfast`, `timestep=0.002` s (500 Hz), gravity `(0, 0, -9.81)` |
| Table | Box, half-size `(0.6, 0.6, 0.025)`, top at \(z=0\), friction `0.7 0.01 0.001` |
| `target_obj` | Freejoint body. **cup**: cylinder radius `0.036 * scale`, half-height `0.052 * scale`, plus a rim cap. **box**: box with the same half-extents. Mass, sliding friction, and RGBA are arguments. |
| `wrist_mocap` | Kinematic mocap body, driven every physics step from the retargeted trajectory. |
| `wrist_body` | Dynamic `freejoint` body, mass `0.35` kg, **welded** to `wrist_mocap` (`solref="0.018 1.0"`, `solimp="0.9 0.95 0.001"`). Because it is dynamic, contacts inject real impulses into the IMU site. |
| Gripper | Two slide-joint fingers (`grip_l`, `grip_r`), range `-0.005 … 0.042` m, pad friction `0.95`, position actuators `kp=350`. |
| `imu_site` | On `wrist_body` at `(0, 0, 0.02)`. Sensors: `<accelerometer name="wrist_accel">`, `<gyro name="wrist_gyro">`, `<framequat name="wrist_quat">`. |
| Cameras | `ego_cam` (chest-ish, fovy 58) and `side_cam` (three-quarter, fovy 48). `sim/augment.py` renders **`side_cam`** at **320×240**. |

## 2. Retargeting (`sim/retarget.py`)

`retarget_episode(df)` always emits 100 Hz rows with:

`t_s, mocap_x, mocap_y, mocap_z, mocap_qw, mocap_qx, mocap_qy, mocap_qz, grip_ctrl`

and, when present, copies `ax…az`, `gx…gz`, `contact_flag`, `contact_event_id` as `real_*`.

**Orientation.** If `roll`, `pitch`, `yaw` exist, they are shifted to the first sample, clipped (\(\pm 0.45\) rad roll/pitch, \(\pm 0.65\) rad yaw), converted with `Rotation.from_euler("xyz")`, and stored as MuJoCo `(w, x, y, z)`. This is a **relative** wrist attitude, not a compass heading.

**Translation and grip — MediaPipe path.** Used when `right_x0` or `left_x0` is finite on more than `max(5, 0.1 N)` samples. Wrist `(x, y)` is rolled (window 15), then mapped into a tabletop patch (see [ARCHITECTURE.md](ARCHITECTURE.md) §4). Thumb–index distance becomes `grip_ctrl` in roughly `[-0.002, 0.032]`.

**Translation and grip — IMU-only fallback.** `contact_detect.detect_contacts` supplies a `grab_onset` time and a last `set_down` time (defaults `0.30 T` and `0.78 T` if missing or unordered). A cosine-blended waypoint script then: approaches from `(0, -0.18, 0.18)` with an open jaw; closes on a cup near `(0, 0, 0.090)`; lifts (~11.5 cm) and translates to \(x=0.13\); sets down and retracts. A small amount of real `ax, ay` is added as micro-tremor (`× 0.0012`, clipped to 8 mm).

CLI:

```bash
python -m sim.retarget data/aligned/ep_001.parquet --out data/sim_aug/ep_001_retarget.parquet
```

If `--out` is omitted, the file is written next to the input as `<stem>_retarget.parquet`.

## 3. Domain randomization (`sim/augment.py`)

Each variant index `i` uses `numpy.random.default_rng(seed + i * 101)` with `--seed` defaulting to **42**.

| Parameter | Distribution |
|---|---|
| Object type | `cup` if `i % 2 == 0`, else `box` |
| Friction and mass (easy, `i % 3 != 2`) | \(\mu \sim U(0.50, 1.05)\), \(m \sim U(0.12, 0.42)\) kg |
| Friction and mass (hard / slippery, `i % 3 == 2`) | \(\mu \sim U(0.10, 0.24)\), \(m \sim U(0.45, 0.85)\) kg |
| Scale | \(U(0.82, 1.18)\) (base radius 3.6 cm, half-height 5.2 cm) |
| Initial XY | \(U(-0.015, 0.015)\) m on both axes |
| Time warp | \(U(0.88, 1.15)\) — multiplies retargeted timestamps (speeds up or slows the demo) |
| Visual | RGBA cycles through a six-color palette |

The hard subset exists so the 3×3 demo grid is not uniformly successful: low friction plus higher mass is where slips and drops appear.

CLI (defaults in parentheses):

```bash
python -m sim.augment <input.parquet> \
  --n 20 \
  --render-n 9 \
  --out-dir data/sim_aug \
  --lerobot-out data/mimesis_lerobot \
  --task pick_place_mug \
  --seed 42
```

Pass an empty `--lerobot-out ''` if you need to skip export? **No** — the argparse default is always set; to skip LeRobot export, call `run_augment(..., lerobot_out=None)` from Python. From the CLI, `if args.lerobot_out` is truthy for the default path, so a dataset **is** written unless you change the code. Use `--lerobot-out data/mimesis_lerobot` explicitly in scripts.

Outputs:

- `data/sim_aug/episodes/sim_vXXXX.parquet` — 100 Hz simulated IMU + forces + object pose
- `data/sim_aug/episodes/sim_vXXXX_30fps.parquet` — every third row, for video-aligned export
- `data/sim_aug/videos/sim_vXXXX.mp4` — first `--render-n` variants, 320×240, `side_cam`, OpenCV `mp4v` (later transcoded by `demo_grid.py` if you build a demo)
- `data/sim_aug/manifest.json` — per-variant physics parameters and outcome labels

## 4. Simulated IMU noise model

At 100 Hz (every 5 physics substeps), `simulate_variant` reads `wrist_accel` (specific force \(\mathbf{f}\) in the body frame) and `wrist_gyro`.

\[
\mathbf{g}_\text{body} = R(\mathbf{q})^{-1} \begin{bmatrix} 0 \\ 0 \\ 9.81 \end{bmatrix},
\quad
\mathbf{a}_\text{lin} = \mathbf{f} - \mathbf{g}_\text{body} + \mathbf{b}_a + \boldsymbol{\varepsilon}_a
\]

\[
\boldsymbol{\omega} = \boldsymbol{\omega}_\text{raw} + \mathbf{b}_\omega + \boldsymbol{\varepsilon}_\omega
\]

| Term | Draw (per variant, then per sample for \(\varepsilon\)) |
|---|---|
| Accel bias \(\mathbf{b}_a\) | \(\mathcal{N}(0, 0.04^2)\) m/s², i.i.d. per axis, constant over the episode |
| Gyro bias \(\mathbf{b}_\omega\) | \(\mathcal{N}(0, 0.008^2)\) rad/s, same pattern |
| Accel noise \(\boldsymbol{\varepsilon}_a\) | \(\mathcal{N}(0, 0.06^2)\) m/s² each sample |
| Gyro noise \(\boldsymbol{\varepsilon}_\omega\) | \(\mathcal{N}(0, 0.015^2)\) rad/s each sample |

`grav_*` in the parquet is \(\mathbf{g}_\text{body}\) as defined above. Quaternion is the MuJoCo `framequat` on `wrist_body`, renormalized; Euler angles are `xyz` intrinsic.

This is a simple MEMS-like model (constant bias + white noise). It does not include random walk, scale-factor error, or g-sensitivity.

## 5. Labels

**Per sample (100 Hz):**

- `sim_contact_force_n` — sum of \(\lVert F \rVert\) from `mj_contactForce` over contacts between `target_obj` and a **non-world** body (gripper pads / palm), **excluding table–object** contacts (`other_b != 0`).
- `sim_contact_flag` — `1` if that force \(> 0.25\) N.
- After the rollout, `contact_flag` is copied from `sim_contact_flag`, and `contact_event_id` is set to `3` (`impact_or_tap`) when the flag is set, else `1` (`motion`). `jerk_mag` is also computed on the simulated linear accel.

**Per episode (manifest + export task suffix):**

- `max_lift_m` — peak object \(z\) minus initial \(z\).
- `lifted` — `max_lift_m > 0.025` m (2.5 cm).
- `slipped` — `(not lifted and max_contact_force_n > 0.5)` **or** `(\(\mu < 0.25\) and lift \(< 0.05\) m and max_contact_force_n > 0.5)`.
- `success` — `lifted and not slipped`.

These are **privileged physics labels**, not human annotations. They are only as meaningful as the gripper and friction model.

## 6. Sim-to-real classifier (`sim/sim_to_real.py`)

- **Windows:** 20 samples (200 ms at 100 Hz), stride 5 (50 ms).
- **Features (16):** mean, std, max, peak-to-peak of \(\lVert a \rVert\) and \(\lVert\omega\rVert\); mean, std, max of jerk; mean and max of \(\lVert\Delta a\rVert\) and \(\lVert\Delta\omega\rVert\); 90th percentile of jerk.
- **Sim labels:** `detect_contacts` with slightly softer thresholds (`jerk_thresh=28`, `accel_peak_thresh=2.2`, `slip_hpf_thresh=0.55`), OR’ed with a physics transient: \(|\Delta F| > 0.35\) N or \(F > 4\) N.
- **Real labels:** default `detect_contacts` on the phone IMU (`jerk_thresh=45`, etc.).
- **Model:** `sklearn.ensemble.RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)`.
- **Split:** every simulated episode parquet in `--sim-dir` except `*_30fps.parquet` is **training**; every path in `--real` is **test**. There is no leak of real windows into training.

```bash
python -m sim.sim_to_real \
  --sim-dir data/sim_aug/episodes \
  --real data/raw_imu/ep_001.parquet \
  --out-json data/sim_aug/sim_to_real_report.json
```

`--real` accepts one or more parquet/csv files. Metrics written: `zero_shot_accuracy`, `zero_shot_balanced_accuracy`, `zero_shot_f1`, `zero_shot_roc_auc`, plus window counts and positive rates.

## 7. Demo composition (`sim/demo_grid.py`)

```bash
python -m sim.demo_grid \
  --real-video data/hands/ep_001_overlay.mp4 \
  --sim-dir data/sim_aug \
  --out data/mimesis_demo.mp4 \
  --out-png data/mimesis_demo.png
```

Requires `sim_aug/manifest.json` and at least one rendered `video_path`. Layout: 480×360 left (real or placeholder), 480×360 center (first non-slip twin, else variant 0), 3×3 grid of 160×120 cells (green border `OK`, red `SLIP!`), plus a 36 px footer. Output is H.264 `yuv420p`. Mid-roll frame is saved as the PNG.
