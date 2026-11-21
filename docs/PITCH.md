# Pitch script and Q&A

> **Note.** This pitch describes the earlier wrist-IMU prototype, not the current video-only pipeline.

Five minutes, plus unlimited Q&A as announced at NullHacks Helsinki. Judging: (i) coolness, (ii) technical complexity (a real build, not a mock), (iii) creativity. The theme asks you to **simulate** data, not only collect it.

Fill italic placeholders with numbers from **your** `manifest.json` and `sim_to_real_report.json`. If you mention the box regression test, say **synthetic** out loud: clap-sync error **10 ms**, ROC-AUC **0.81** on a **synthetic-real** clip — not a human held-out set.

## Spoken script (~5 minutes)

**0:00–0:30 — Hook**

“Physical AI labs can scrape video. They cannot scrape contact. Real robot teleop needs hardware. A phone IMU looks like free motion capture until you integrate it — it drifts meters in a few seconds. **MIMESIS** — Greek for imitation — does something else: you record one 10-second task with two ordinary iPhones, and we simulate a thousand physically different worlds.”

**0:30–1:15 — The physics trap**

Show `live_viz.py --replay` panel 3, or the saved dashboard PNG. “This is double-integrated linear acceleration from the wrist phone, after subtracting a short bias. The trace has already walked off the table. That is why we refuse to sell XYZ from an IMU. The chest camera is our coarse kinematic ground truth. The wrist IMU is a 100 hertz contact microphone — jerk, slip chatter, set-down shocks.”

If you have the clap-sync PNG, flash it: three audio claps and three IMU jerks on the same timeline.

**1:15–2:45 — Hero demo**

Play `data/mimesis_demo.mp4`.

- Left: real chest camera, ideally with the MediaPipe overlay (`hands.py --out-video`).
- Middle: the MuJoCo twin, same timing, a two-finger gripper welded to a mocap wrist so impacts are *physical*, not painted on.
- Right: a three-by-three grid. Green borders lifted the mug. Red borders slipped or dropped — low friction, heavier mass, slightly warped timing. Those failure labels are free, because the simulator knows the contact force and the object height.

Say the scale: “*N* real clips in, *M* simulated episodes out, *K* of them labeled slips.”

**2:45–3:45 — How it is built, and the transfer number**

One sentence per stage, no architecture lecture:

“Sensor Logger on the wrist pushes 100 hertz batches into a FastAPI receiver. MediaPipe Hands tracks the chest cam. We clap-sync audio against IMU jerk. `sim/retarget.py` drives a MuJoCo mocap body. `sim/augment.py` randomizes mass, friction, size, and MEMS noise, and writes a LeRobot-style parquet-plus-mp4 dataset.

Then the check that is not a demo trick: we train a random forest **only** on 200-millisecond windows of *simulated* IMU, and we test it on the *real* phone IMU with the same contact detector. Zero-shot ROC-AUC is _TBD_ — / accuracy _TBD_. That is the slide number.”

**3:45–4:30 — Why a lab would care**

“Success demonstrations are relatively easy to film. Honest slips, drops, and contact onset times are scarce and expensive to annotate. MIMESIS turns any two-phone clip into a generator of those labels, in a format you can drop next to LeRobot. The product is not fifty seconds of one person’s hands. It is a deployable collection-plus-simulation protocol.”

**4:30–5:00 — Limits and close**

“We are honest about what is fake. The gel-less gripper is not a human hand. MediaPipe depth is not a mocap stage. The wrist phone is bulky. The synthetic test AUC is not a production claim. Next step is a watch on the wrist and a tighter calibration. The repo is MIT-licensed, the dataset folder is in LeRobot layout, and we can walk through a live retarget if you want. Happy to take questions.”

Stop talking. Let Q&A run long.

## Q&A landmines

**“So you just double-integrate the IMU for position?”**

No. We show that this fails (dashboard panel 3; even a stationary phone can drift meters in ~10 s). Translation in sim comes from retargeted camera landmarks or a contact-locked kinematic script. The IMU supplies orientation, high-rate dynamics, and contact.

**“Is the sim-to-real number on real people?”**

If you only ran `test_pipeline.py`, say: “That 0.81 AUC is on a synthetic IMU clip used as a regression test. We will / we did run the same script on the phone recordings in `data/raw_imu/` — here is that report.” Never conflate the two.

**“Isn’t this just domain randomization?”**

The randomization is real (mass, friction, scale, time-warp, MEMS noise), but the *source* trajectory is a captured human demo, clap-synced to video, and the *claim* is a multiplier plus a transfer check, not “we trained a policy with random textures.” Low-friction variants exist specifically to mint **failure labels**.

**“Sim-to-real for a whole robot policy?”**

We do not claim a VLA or a walking policy trained in four hours. We claim contact-event transfer from simulated wrist IMU to a real phone IMU, and a dataset you can train on later.

**“Why MuJoCo, not Isaac / Genesis?”**

Isaac is not viable on a Mac. Genesis install and GUI are risky in a four-hour window. Official `pip install mujoco` plus `mujoco.Renderer` runs on Apple Silicon. Headless Linux uses EGL.

**“MediaPipe world landmarks are metric, right?”**

They are in a hand-centered frame. Monocular absolute depth is weak. We treat them as coarse tabletop coordinates, or we ignore them and use the IMU-contact fallback. We do not sell them as mocap.

**“Can I buy this afternoon’s dump?”**

No, and we would not sell it as such. The thing that scales is the protocol: phones in, labeled simulated contact and slip out. One team’s takes are a proof, not inventory.

**“Why two phones instead of AirPods on the fingers?”**

Apple’s headphone motion API streams **one bud at a time**, often at 25–50 Hz, and in-ear detection can stop the stream off-ear. A phone IMU is ~100 Hz, documented, and Sensor Logger already speaks HTTP. We still cannot get drift-free XYZ from it — that is the design constraint, not a surprise.

**“Where is the code for the LeRobot features?”**

`export.py`. State is 14-D (`ax…jerk_mag`). Hands are 22-D (presence, wrist/thumb/index, wrist speed, both sides). Contact ids are 0–5 as in `contact_detect.EVENT_TO_ID`. Simulated variants are also exported by `sim.augment` with repo id `nullhacks/mimesis-sim-aug`.

**“Did you write a custom iOS app?”**

No. Sensor Logger + the built-in Camera app. That was deliberate so the collection kit is installable in minutes.
