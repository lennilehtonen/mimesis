# Data collection

How to record demonstrations that MIMESIS can actually sync, retarget, and multiply. No native iOS app is required.

## 1. Kit

- Two iPhones (the pipeline was designed around this; any recent iPhone with a rear camera works for Phone A).
- A Mac on the same network as Phone B, with this repo and `ffmpeg` installed.
- Straps or athletic tape for Phone B (flat on the **back** of the dominant wrist or distal forearm — not flopping in the palm).
- A chest lanyard, hoodie pocket, or cap mount for Phone A, angled about 35° down so both hands and the object stay in a **0.5× Ultra-Wide** frame.
- Objects from the task list: mug, bottle, cups, laptop, backpack, cloth, USB cable, book, glass vs foam, snack bag, spoon.

## 2. Phone B — Sensor Logger

App: **Sensor Logger** by Kelvin Choi (free iOS App Store build is sufficient).

1. **Settings → Sensor Configuration**
   - **Standardise Units & Frames = ON** (required so acceleration is in m/s² and axes are right-handed).
   - **Uncalibrated Values = OFF**.
2. **Logger** tab — enable **only**:
   - Accelerometer @ **100 Hz**
   - Gyroscope @ **100 Hz**
   - Orientation @ **100 Hz**
   - Gravity @ **100 Hz**
3. Logger gear → **HTTP Push = ON** → URL `http://<MAC_IP>:8000/data`.
   Leave an optional Authorization header empty unless you added one to the server (the stock `receiver.py` does not check it).
4. Disable Low Power Mode. Keep the Sensor Logger screen awake during the take.
5. Strap the phone so the screen faces **outward/up** consistently across episodes; do not rotate the mount mid-session.

The free tier may batch HTTP posts about once per second and may limit stored recordings (export and delete as you go). Sampling frequency itself is configurable without a paid push-interval feature: timestamps inside each batch still run at ~100 Hz.

## 3. Phone A — chest camera

1. Built-in **Camera → Video**.
2. **1080p, 30 fps** (avoid 60 fps unless you have a reason; `hands.py` and `export.py` assume ~30 fps).
3. **0.5× Ultra-Wide**, not the 1× lens.
4. **Audio ON** — clap-sync is computed from the soundtrack.
5. Lock AE/AF if the scene is stable. Do not switch cameras mid-clip.
6. One file per episode (or a clearly named batch you will split later). Name AirDropped files to match the episode id, e.g. `ep_001.mov`.

Continuity Camera is optional and only needed for `python hands.py --live --camera 0` on stage. Record3D / ARKit depth is **not** on the critical path (the free Record3D trial is capped at three clips).

Automatic Ear Detection and AirPods motion APIs are **not used** in this pipeline (Phone B is a full iPhone IMU, not an AirPod). Nothing to disable on that front.

## 4. Per-episode protocol (20–30 seconds)

1. On the Mac: `python receiver.py --port 8000` is running. Optionally `POST /episode/start` with `episode_id` and `task`.
2. Confirm a test wiggle appears on `GET /status` or `python live_viz.py`.
3. Start Phone A, then Phone B (or the reverse — the claps, not the button times, define alignment).
4. **Three sharp, separated claps** (about 0.4–0.6 s apart) in the first three seconds, both hands visible, claps loud in the chest mic and shocking the wrist IMU.
5. Perform **one** task from the list below, at a natural speed, keeping hands in frame.
6. End with a clear **set-down and 0.5 s rest** on the table when the task allows it.
7. Stop both recorders. AirDrop video. `POST /episode/stop` (or rely on the receiver’s periodic flush every five HTTP messages plus process shutdown).
8. Glance at `ep_*_meta.json`: `effective_hz` should be near 100, `duration_s` should match the take.

Do not talk over the claps. Do not begin the manipulation before the third clap. If a clap is soft, redo the take — `sync.py` cannot invent a missing transient.

## 5. Task list (18 recordable tasks)

Prefer tasks with sharp contacts, rotation, friction, or bimanual coordination. Record several takes and, if possible, more than one person.

1. `pick_place_mug` — grab, lift, gentle set-down.
2. `pour_water_cup` — wrist roll and changing mass.
3. `unscrew_bottle_cap` — torque plus slip chatter.
4. `screw_bottle_cap_tight` — torque and end-stop impact.
5. `open_close_laptop` — hinge resistance and latch click.
6. `zip_unzip_backpack` — continuous scrape.
7. `wipe_table_cloth` — sustained friction.
8. `stack_three_cups` — three grab / set-down transients.
9. `type_on_keyboard` — rapid taps (100 Hz IMU resolves them better than 30 fps video).
10. `plug_usb_cable` — insertion click.
11. `unplug_usb_cable` — release spike.
12. `flip_book_pages` — light slide.
13. `tap_glass_vs_foam` — material contrast.
14. `pick_heavy_vs_empty_bottle` — jerk scaling with mass.
15. `fold_towel_bimanual` — deformable, two hands in the chest cam.
16. `open_snack_bag` — bimanual pull and pop.
17. `stir_spoon_in_mug` — periodic gyro plus rim clinks.
18. `intentional_mug_slip` — loose or wet grip, slip, and catch (hero failure case).

Use these strings as the `task` field in `POST /episode/start`, `receiver.py --task`, and `sim.augment --task`. Simulated variants are exported as `{task}_success` or `{task}_slip`.

## 6. Networking and troubleshooting

**Prefer a personal hotspot** (iPhone or Mac) if hackathon Wi-Fi uses client isolation (phones cannot open TCP to a laptop on the same SSID).

| Symptom | What to check |
|---|---|
| `GET /status` never shows messages | Phone and Mac not on the same L2 network; typo in the Push URL; Sensor Logger HTTP Push off; Mac firewall blocking port 8000; `receiver.py` not bound to `0.0.0.0`. |
| Push URL origin | Use the IPv4 printed by `receiver.py`, not `127.0.0.1` (that is the phone itself) and not a public IP. Example: `http://192.168.1.23:8000/data`. |
| Stream starts then dies | Phone slept, Low Power Mode, or app backgrounded. Keep Sensor Logger in the foreground. |
| `effective_hz` far below 100 | Too many sensors enabled, thermal throttling, or a huge HTTP interval with dropped batches. Disable unused sensors. Check gaps in `timestamp_ns`. |
| Clap-sync plot looks random | Audio muted on Phone A; claps out of frame or after `--search-window` (default 12 s); IMU not on the wrist during the claps; huge start-time mismatch (> 8 s). Re-record. |
| HTTP is hopeless | Record locally in Sensor Logger → export **Zipped CSV** → AirDrop → `python receiver.py --import-csv-dir <unzipped_folder> --episode-id ep_001 --task pick_place_mug --out-dir data/raw_imu`. The importer looks for `Accelerometer.csv`, `Gyroscope.csv`, `Orientation.csv`, `Gravity.csv` (and optional `AccelerometerUncalibrated.csv`). |
| phyphox fallback | If Sensor Logger cannot be installed, phyphox Remote Access can be polled (`http://<IPHONE_IP>/get?...` on iOS, typically port 80). This repo does not include a phyphox poller; export CSV and convert, or write a thin adapter into the same parquet schema (`t_rel_s`, `ax…az`, `gx…gz`, orientation, gravity). |

HTTP 200 is the expected success response. If you implement a custom proxy, Sensor Logger treats **HTTP 499** plus a text body as a one-shot user-visible error.

## 7. After collection, before simulation

For each episode:

```bash
python hands.py data/videos/ep_00K.mov --out data/hands/ep_00K_hands.parquet --out-video data/hands/ep_00K_overlay.mp4
python sync.py --video data/videos/ep_00K.mov --imu data/raw_imu/ep_00K.parquet \
  --hands data/hands/ep_00K_hands.parquet --out data/aligned/ep_00K.parquet --plot data/aligned/ep_00K_sync.png
```

Inspect `ep_00K_sync.png`: the three audio claps and the shifted IMU jerk should share a vertical line. Only then run `python -m sim.augment`.
