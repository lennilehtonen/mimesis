# Dataset schema

Intermediate files use pandas parquet/CSV. The packaged corpus follows a **LeRobot v3-style** directory layout (not a guarantee of bit-compatibility with every `lerobot` release). `export.py` writes `codebase_version: "v3.0-airtruth-hack"`.

## 1. Packaged layout

```
<out-dir>/
  README.md
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes.parquet
  data/chunk-000/file-000.parquet
  videos/observation.images.egocentric/chunk-000/ep_{episode_index:04d}.mp4
```

`info.json` path templates:

- `data_path`: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
- `video_path`: `videos/{video_key}/chunk-{chunk_index:03d}/ep_{episode_index:04d}.mp4`

This hackathon exporter writes a **single** shard: `chunk-000/file-000.parquet`. Videos are one MP4 per episode, not concatenated shards.

**`info.json` features**

| Key | dtype | shape | names / meaning |
|---|---|---|---|
| `observation.images.egocentric` | video | `[1080, 1920, 3]` | Declared chest-cam size. Real `.mov` files are transcoded to H.264 if not already `.mp4` (`libx264`, `-crf 23`, `-r {fps}`). Sim renders are 320×240; the declared shape is a schema hint. |
| `observation.state` | float32 | `[14]` | `ax, ay, az, gx, gy, gz, qx, qy, qz, qw, grav_x, grav_y, grav_z, jerk_mag` |
| `observation.hands` | float32 | `[22]` | See below. Missing hands are `0.0`. |
| `observation.contact_event_id` | int32 | `[1]` | `0` idle, `1` motion, `2` grab_onset, `3` impact_or_tap, `4` set_down, `5` slip_or_scrape |
| `observation.contact_flag` | int32 | `[1]` | Binary contact / transient |
| `observation.slip_score` | float32 | `[1]` | Real: rolling high-pass vibration score from `contact_detect.py`. Sim: `1.0` if the variant `slipped`, else `0.0`. |

**`observation.hands` (22)** — order used by `_extract_hand_summary`:

`left_present`, `right_present`,
`left_x0, left_y0, left_z0` (wrist),
`left_x4, left_y4, left_z4` (thumb tip),
`left_x8, left_y8, left_z8` (index tip),
`left_wrist_speed_px_s`,
then the same ten values for `right_*`.

World landmarks (`*_w*`) are **not** packed into the LeRobot feature; they remain in the hands/aligned parquets.

**`data/chunk-000/file-000.parquet` columns:** `index`, `episode_index`, `frame_index`, `task_index`, `timestamp`, plus the observation fields above (`observation.state` and `observation.hands` stored as per-row Python lists of floats).

**`meta/episodes.parquet`:** `episode_index`, `episode_id`, `task`, `task_index`, `length`, `duration_s`, `contact_frames`, `data_from_index`, `data_to_index`, `video_path`.

**`meta/tasks.parquet`:** `task_index`, `task`.

**`meta/stats.json`:** `mean`, `std` (floored at `1e-6`), `min`, `max` for `observation.state` and `observation.hands`.

**Default repo ids:** `nullhacks/airtruth-ego-imu` for `python export.py`; `nullhacks/mimesis-sim-aug` when `sim.augment` calls `export_lerobot_dataset`. `robot_type` is `human_egocentric_wrist_imu`. `fps` defaults to `30.0`.

`python export.py --aligned-dir …` discovers `*.parquet` in that folder (including accidental extras — keep only aligned episode files there). Task is read from `<stem>.sync.json` if a `task` key exists; `sync.py` does not currently write that key, so folder mode usually stores `egocentric_manipulation` unless you pass `--manifest`.

## 2. `receiver.py` raw IMU (`data/raw_imu/<episode_id>.parquet`)

Aligned to the accelerometer clock:

| Column | Unit | Source |
|---|---|---|
| `timestamp_ns` | ns since Unix epoch | Sensor Logger `time` |
| `t_rel_s` | s | relative to first sample |
| `ax, ay, az` | m/s² | calibrated linear accel |
| `gx, gy, gz` | rad/s | gyro |
| `grav_x, grav_y, grav_z` | m/s² | gravity vector |
| `qx, qy, qz, qw` | — | orientation quaternion (Sensor Logger `values`) |
| `roll, pitch, yaw` | rad | Sensor Logger Euler |
| `accel_mag`, `gyro_mag` | derived | Euclidean norms |

Sibling files: `<id>.csv`, `<id>_meta.json` (`episode_id`, `session_id`, `device_id`, `task`, `num_samples`, `duration_s`, `effective_hz`, paths).

## 3. `contact_detect.py` output

Adds to a copy of the IMU table: `accel_mag`, `gyro_mag`, `jerk_mag` (m/s³), `hpf_accel_mag`, `slip_score`, `contact_event` (string), `contact_event_id` (int), `contact_flag` (0/1).

`--events-json` default is the output parquet path with suffix `.events.json` and contains `{input, num_events, events:[{event, t_s, sample_idx, jerk_peak_ms3, accel_peak_ms2, gyro_mag_rads}]}`.

## 4. `hands.py` parquet

`frame_idx`, `t_s`, then for each of `left` and `right`:

- `{side}_present` (0/1)
- `{side}_x{i}, _y{i}, _z{i}` and `{side}_wx{i}, _wy{i}, _wz{i}` for `i = 0…20`
- `{side}_wrist_speed_px_s`, `{side}_index_speed_px_s`

## 5. `sync.py` aligned parquet

All hands columns (if `--hands` was passed) plus `t_imu_s` and interpolated:

`ax, ay, az, gx, gy, gz, qx, qy, qz, qw, roll, pitch, yaw, grav_x, grav_y, grav_z, accel_mag, gyro_mag, jerk_mag, hpf_accel_mag, slip_score, contact_event_id, contact_flag, contact_event`.

Companion: `<out_stem>.sync.json` with `offset_s`, `xcorr_offset_s`, `t_clap_video_s`, `t_clap_imu_s`, `num_audio_claps`, `num_imu_claps`, `corr_peak`, `fps`, `num_frames`, `events`.

## 6. Retarget parquet

`t_s, mocap_x, mocap_y, mocap_z, mocap_qw, mocap_qx, mocap_qy, mocap_qz, grip_ctrl` and optional `real_ax…real_contact_event_id`. Positions are meters in the MuJoCo world; `grip_ctrl` is a slide-joint command in meters.

## 7. Simulated episode parquet (`data/sim_aug/episodes/sim_vXXXX.parquet`)

`t_rel_s`, `t_s`, `ax…az`, `gx…gz`, `qw, qx, qy, qz`, `roll, pitch, yaw`, `grav_*`, `sim_contact_force_n`, `sim_contact_flag`, `obj_x, obj_y, obj_z`, `grip_ctrl`, plus post-pass `jerk_mag`, `contact_flag`, `contact_event_id`, `slip_score`.

`manifest.json` per variant: `variant_idx`, `episode_id`, `obj_type`, `obj_mass_kg`, `obj_friction`, `obj_scale`, `time_warp`, `max_lift_m`, `max_contact_force_n`, `slipped`, `success`, `video_path`, `aligned_parquet`, `task`.
