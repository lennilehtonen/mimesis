# Team plan — NullHacks Helsinki

Event clock is **Europe/Helsinki (EET, UTC+2)**. Hack start 13:30, **GitHub submission 18:30**, pitches from 18:50. **No extensions.** Feature freeze **17:45**. This plan assumes a team of four on MacBooks with Claude Code and at least two iPhones.

## Roles

| Person | Role | Owns (files and outcomes) |
|---|---|---|
| **A — Capture** | Hardware and collector | Phone mounts, Sensor Logger settings, `receiver.py`, `POST /episode/start` and `/stop`, AirDrop into `data/videos/`, calling task names, quality control of `effective_hz` in `*_meta.json` |
| **B — Vision and sync** | Alignment | `hands.py` (including model download), `sync.py`, clap-sync PNG review, `live_viz.py` drift dashboard, MediaPipe overlay MP4s |
| **C — Simulation and ML** | Multiplier | `sim/retarget.py`, `sim/augment.py` (50–200+ variants on the best takes), `sim/sim_to_real.py`, `manifest.json` / `sim_to_real_report.json` numbers for the slide |
| **D — Export and pitch** | Product | `export.py` for real episodes, `sim/demo_grid.py` stage video, README placeholders filled, repo hygiene, 5-minute script rehearsal ([PITCH.md](PITCH.md)) |

If someone is less comfortable with Python, prefer **A** or **D**. Person C should run `test_pipeline.py` once early so MuJoCo rendering is known to work on that Mac.

## Shared conventions

- One git repository from the first hour. Branch sparingly; prefer frequent commits on `main`.
- Episode ids: `ep_001`, `ep_002`, … matching video filenames (`ep_001.mov`) and IMU stems.
- Task strings: use the list in [DATA_COLLECTION.md](DATA_COLLECTION.md).
- Do not record and debug on the same machine if you can avoid it — A records while B/C process the previous batch.
- If Sensor Logger HTTP fails for more than ten minutes, switch to CSV AirDrop and do not keep fighting the network.

## Hour by hour (EET)

### 14:40–15:10 — Proof the kit, not the idea

- All four: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Confirm `ffmpeg`.
- One Mac: `python test_pipeline.py` (about one minute). If MuJoCo or EGL/rendering fails, fix that before collecting a library of clips.
- `python -c "from hands import ensure_hand_model; ensure_hand_model()"`.
- A: hotspot or shared Wi-Fi, `python receiver.py --port 8000`, Sensor Logger HTTP Push, 5-second wiggle, `GET /status`.
- Whole team: **one** 20-second protocol take (three claps + `pick_place_mug`).

**Checkpoint:** live IMU in `live_viz.py`, a video file on the Mac, and a parquet with `effective_hz` ≈ 100.

### 15:10–16:10 — First full real-to-sim loop

- B: `hands.py` + `sync.py` on `ep_001`. The sync PNG must show three shared clap lines.
- C: `python -m sim.augment data/aligned/ep_001.parquet --n 30 --render-n 9 --out-dir data/sim_aug --lerobot-out data/mimesis_lerobot --task pick_place_mug`.
- C: `python -m sim.sim_to_real --sim-dir data/sim_aug/episodes --real data/raw_imu/ep_001.parquet`.
- D: `python -m sim.demo_grid --real-video data/hands/ep_001_overlay.mp4 --sim-dir data/sim_aug --out data/mimesis_demo.mp4`.

**Checkpoint:** a 3×3 grid with both green and red cells, and a first (possibly weak) sim-to-real number you understand.

### 16:10–17:15 — Volume

- A: 15–25 clean episodes across the 18-task list, including `intentional_mug_slip`, `unscrew_bottle_cap`, and at least one bimanual task. Switch actors if possible.
- B: process in batches of ~5; drop takes with bad sync rather than “fixing” them in software.
- C: multiply the **best 8–12** takes (`--n 20` to `--n 50` each, render 9 only on the hero tasks to save time). Merge or export a unified simulated LeRobot folder.
- D: `export.py` on `data/aligned` (or a manifest with real task names), repo README results table started.

**Checkpoint:** a dataset you can state as “R real episodes, S simulated, F frames,” plus a contact-transfer report on **real** IMU files.

### 17:15–17:45 — Harden the pitch assets

- Re-render `mimesis_demo.mp4` from the best overlay + a manifest that actually contains slips.
- Save `live_viz.py --replay … --save-png` for the drift beat.
- Copy accuracy / AUC / variant counts into [PITCH.md](PITCH.md) and the root README results table. Label any leftover synthetic figures as synthetic.
- **17:45 feature freeze.** After this, only crash fixes, copy, and video trims.

### 17:45–18:20 — Rehearse and ship the repo

- D leads two timed run-throughs of the five-minute script (phone timer, actually standing up).
- README clone path: `pip install -r requirements.txt` and `python test_pipeline.py` still work.
- No new randomization features, no last-minute iOS experiments.

### 18:20–18:30 — Submit

- Final commit and push by ~18:25. Submit the **GitHub URL** before 18:30. Do not keep coding.

## Hard floor (still presentable)

If time collapses, all of the following still make a theme-valid project:

1. One real 3-clap episode, synced, with a drift plot.
2. `sim.augment --n 12 --render-n 9` producing both successes and slips.
3. `sim_to_real.py` on that episode (even if the AUC is modest — report it honestly).
4. `mimesis_demo.mp4` and a clean README.

Do not spend the last hour polishing a second simulator. Do not start Record3D or a custom iOS app after 16:00.
