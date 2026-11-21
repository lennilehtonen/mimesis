# Receiver API

`receiver.py` is a FastAPI app that ingests **Sensor Logger** (Kelvin Choi) HTTP Push batches, keeps a live ring buffer for `live_viz.py`, and writes per-episode parquet/CSV.

## Process

```bash
python receiver.py --host 0.0.0.0 --port 8000 --out-dir data/raw_imu
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address. Must not be `127.0.0.1` if a phone on the LAN should connect. |
| `--port` | `8000` | TCP port. |
| `--out-dir` | `data/raw_imu` | Episode output directory. Overrides at startup; the module also reads `RIG_IMU_DIR` when imported (e.g. by tests) **before** main runs. Prefer the flag when launching the CLI. |
| `--import-csv-dir` | unset | If set, **do not start the server**. Import a Sensor Logger CSV folder and exit. |
| `--episode-id` | `ep_offline_001` | Used only with `--import-csv-dir`. |
| `--task` | `unlabeled_task` | Used only with `--import-csv-dir`. |

Startup prints the LAN URL to paste into Sensor Logger, e.g. `http://192.168.1.23:8000/data`.

You may also run `uvicorn receiver:app --host 0.0.0.0 --port 8000`. In that mode `--out-dir` is not applied; set `RIG_IMU_DIR` or accept `data/raw_imu`.

## HTTP endpoints

All JSON request bodies use UTF-8. Successful ingest returns HTTP **200**. Sensor Logger can show a one-shot in-app error if the server responds with **499** and a text body; this app does not use 499 itself. Malformed JSON on `/data` returns **400**.

### `POST /data`

Ingest one HTTP Push message. `Content-Type: application/json`.

**Verified payload shape** (official Sensor Logger / awesome-sensor-logger `PUSHING.md`; `userId` is optional and ignored):

```json
{
  "messageId": 0,
  "sessionId": "recording-uuid",
  "deviceId": "device-uuid",
  "payload": [
    {
      "name": "accelerometer",
      "time": 1698501144401773000,
      "values": {"x": 0.1, "y": 0.2, "z": -0.05}
    },
    {
      "name": "gyroscope",
      "time": 1698501144401773000,
      "values": {"x": 0.01, "y": -0.02, "z": 0.03}
    },
    {
      "name": "orientation",
      "time": 1698501144401773000,
      "values": {
        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0
      }
    },
    {
      "name": "gravity",
      "time": 1698501144401773000,
      "values": {"x": 0.0, "y": 0.0, "z": 9.81}
    }
  ]
}
```

- `time` is **UTC epoch nanoseconds** (int or numeric string).
- `messageId` increments per HTTP message (batches can arrive out of order; samples are later sorted by `time`).
- `sessionId` is constant for one Sensor Logger recording and keys the in-memory session.
- `payload` must be a **list**. Each item has `name` (or `sensor`) and `time`. Values usually sit under `values`; a flat item (legacy) is accepted if `x` / `y` / `z` are top-level.

**Name aliases** (`SENSOR_ALIASES`, compared case-insensitively after strip):

| Incoming `name` | Internal key |
|---|---|
| `accelerometer`, `linear_acceleration` | `accel` (primary clock) |
| `totalacceleration` | `total_accel` |
| `accelerometeruncalibrated` | `accel_raw` |
| `gyroscope` | `gyro` |
| `gyroscopeuncalibrated` | `gyro_raw` |
| `orientation`, `attitude` | `orient` |
| `gravity` | `gravity` |

Any other name is stored under its raw lowercased string but is not interpolated into the output table.

**Response:** `{"status":"ok","sessionId":"...","ingested":<int>}`.

The session is auto-flushed to disk every **5** messages (`flush_session`) and again on `POST /episode/stop`.

### `POST /episode/start`

```json
{"episode_id": "ep_001", "task": "pick_place_mug"}
```

Both fields optional. Defaults: `episode_id = ep_HHMMSS` (UTC), `task = "manipulation_task"`. Subsequent `/data` sessions created while this is active inherit that episode id and task. Response: `{"status":"started","episode_id":"...","task":"..."}`.

### `POST /episode/stop`

```json
{"session_id": "recording-uuid"}
```

`session_id` optional; defaults to the last session that posted `/data`. Flushes parquet, CSV, and `_meta.json`. Response: `{"status":"flushed","meta":{...}}` or `{"status":"no_active_session"}`.

`meta` includes `episode_id`, `session_id`, `device_id`, `task`, `num_samples`, `duration_s`, `effective_hz`, `parquet_path`, `csv_path`, `sensor_counts`, `message_count`.

### `GET /status`

```json
{
  "active_episode": {"episode_id": "...", "task": "...", "session_id": "...", "started_at": "..."},
  "sessions": {
    "<sessionId>": {"episode_id": "...", "task": "...", "messages": 0, "counts": {"accel": 0, "gyro": 0}}
  },
  "live_buffer_len": 0
}
```

### `GET /latest?n=1000`

Query parameter `n` (default **1000**) is the maximum number of recent accel-triggered live samples. Each sample is a dict with `ax, ay, az, gx, gy, gz, roll, pitch, yaw, qx, qy, qz, qw, grav_x, grav_y, grav_z, time_ns` — the in-memory `LATEST_STATE` snapshot at each accelerometer packet (gyro/orientation/gravity update the same struct and appear on the next accel tick). The ring buffer holds at most **2500** samples.

Response: `{"count": <int>, "samples": [ ... ]}`.

`live_viz.py --url http://127.0.0.1:8000/latest` polls this endpoint.

## Offline CSV import

```bash
python receiver.py --import-csv-dir /path/to/unzipped_export \
  --episode-id ep_001 --task pick_place_mug --out-dir data/raw_imu
```

Expected filenames inside the folder: `Accelerometer.csv`, `Gyroscope.csv`, `Orientation.csv`, `Gravity.csv`, optionally `AccelerometerUncalibrated.csv`. Each CSV should include a `time` column (ns) and the usual `x,y,z` or quaternion / Euler columns from Sensor Logger’s zip export.

## Local examples

```bash
# Name the next take
curl -s -X POST http://127.0.0.1:8000/episode/start \
  -H 'Content-Type: application/json' \
  -d '{"episode_id":"ep_001","task":"unscrew_bottle_cap"}'

# Health
curl -s http://127.0.0.1:8000/status | python -m json.tool

# Flush
curl -s -X POST http://127.0.0.1:8000/episode/stop \
  -H 'Content-Type: application/json' -d '{}'
```
