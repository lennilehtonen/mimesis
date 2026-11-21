"""Procedural MuJoCo tabletop scene for MIMESIS (model name `mimesis_tabletop`)."""
from __future__ import annotations

import os
import sys

import numpy as np

PALETTE = [
    (0.85, 0.33, 0.25, 1.0),
    (0.22, 0.55, 0.85, 1.0),
    (0.95, 0.75, 0.20, 1.0),
    (0.35, 0.75, 0.40, 1.0),
    (0.65, 0.40, 0.80, 1.0),
    (0.92, 0.92, 0.90, 1.0),
]

# Finger inner-face offset from wrist axis at q=0 (m). Inner face sits at +-(FINGER_X0 - q).
FINGER_X0 = 0.062
WRIST_GRASP_Z = 0.090


def configure_gl_backend() -> None:
    """Headless Linux: use EGL if no DISPLAY. macOS / Windows: leave default."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"


def _xyaxes(pos, target, up=(0, 0, 1)) -> str:
    pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return " ".join(f"{v:.5f}" for v in (*x, *y))


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def build_scene_xml(obj_type: str = "cup", mass: float = 0.25, friction: float = 0.8, scale: float = 1.0,
                    xy=(0.0, 0.0), rgba=PALETTE[0]) -> str:
    r = 0.036 * scale
    hh = 0.052 * scale
    x0, y0 = float(xy[0]), float(xy[1])
    rgba_s = _f(rgba)
    fr = f"{friction:.4f} 0.005 0.0001"
    if obj_type == "cup":
        obj_geoms = (
            f'<geom name="obj_geom" type="cylinder" size="{r:.5f} {hh:.5f}" mass="{mass * 0.92:.5f}" '
            f'friction="{fr}" priority="1" condim="4" rgba="{rgba_s}"/>\n'
            f'      <geom name="obj_rim" type="cylinder" size="{r * 1.06:.5f} 0.003" pos="0 0 {hh - 0.003:.5f}" '
            f'mass="{mass * 0.08:.5f}" friction="{fr}" priority="1" condim="4" rgba="0.95 0.95 0.95 1"/>'
        )
    else:
        obj_geoms = (
            f'<geom name="obj_geom" type="box" size="{r:.5f} {r:.5f} {hh:.5f}" mass="{mass:.5f}" '
            f'friction="{fr}" priority="1" condim="4" rgba="{rgba_s}"/>'
        )
    side_pos, side_tgt = (0.40, -0.40, 0.29), (0.06, 0.0, 0.11)
    ego_pos, ego_tgt = (0.0, -0.42, 0.50), (0.05, 0.02, 0.05)
    fx = FINGER_X0
    return f"""<mujoco model="mimesis_tabletop">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="1024"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.78 0.72" rgb2="0.74 0.70 0.64" width="256" height="256"/>
    <material name="tablemat" texture="grid" texrepeat="6 6"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.25 0.28 0.33" rgb2="0.08 0.09 0.11" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="table" type="box" size="0.6 0.6 0.025" pos="0 0 -0.025" friction="0.7 0.01 0.001" material="tablemat"/>
    <camera name="ego_cam" pos="{_f(ego_pos)}" xyaxes="{_xyaxes(ego_pos, ego_tgt)}" fovy="58"/>
    <camera name="side_cam" pos="{_f(side_pos)}" xyaxes="{_xyaxes(side_pos, side_tgt)}" fovy="48"/>
    <body name="target_obj" pos="{x0:.5f} {y0:.5f} {hh + 0.0005:.5f}">
      <freejoint name="obj_free"/>
      {obj_geoms}
    </body>
    <body name="wrist_mocap" mocap="true" pos="0 -0.18 0.18">
      <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 0 0 0.3"/>
    </body>
    <body name="wrist_body" pos="0 -0.18 0.18">
      <freejoint name="wrist_free"/>
      <inertial pos="0 0 0.02" mass="0.35" diaginertia="0.0006 0.0006 0.0004"/>
      <site name="imu_site" pos="0 0 0.02" size="0.006" rgba="0 1 0 1"/>
      <geom name="palm" type="box" size="0.072 0.038 0.016" pos="0 0 0.05" mass="0.0" rgba="0.20 0.22 0.25 1"/>
      <geom name="wrist_cap" type="cylinder" size="0.026 0.02" pos="0 0 0.085" contype="0" conaffinity="0" mass="0.0" rgba="0.14 0.15 0.17 1"/>
      <geom name="accent" type="box" size="0.073 0.039 0.002" pos="0 0 0.066" contype="0" conaffinity="0" mass="0.0" rgba="0.24 0.90 1.0 1"/>
      <geom name="phone" type="box" size="0.035 0.07 0.004" pos="0 0 0.062" contype="0" conaffinity="0" mass="0.0" rgba="0 0 0 0"/>
      <body name="finger_l" pos="0 0 0">
        <joint name="grip_l" type="slide" axis="1 0 0" range="-0.005 0.042" damping="4"/>
        <geom name="pad_l" type="box" size="0.008 0.022 0.05" pos="{-(fx + 0.006):.4f} 0 -0.015" mass="0.03"
              friction="1.4 0.02 0.002" condim="4" rgba="0.30 0.32 0.36 1"/>
        <geom name="tip_l" type="box" size="0.0015 0.02 0.02" pos="{-(fx - 0.0015):.4f} 0 -0.04" contype="0" conaffinity="0" mass="0.0" rgba="0.24 0.90 1.0 1"/>
      </body>
      <body name="finger_r" pos="0 0 0">
        <joint name="grip_r" type="slide" axis="-1 0 0" range="-0.005 0.042" damping="4"/>
        <geom name="pad_r" type="box" size="0.008 0.022 0.05" pos="{fx + 0.006:.4f} 0 -0.015" mass="0.03"
              friction="1.4 0.02 0.002" condim="4" rgba="0.30 0.32 0.36 1"/>
        <geom name="tip_r" type="box" size="0.0015 0.02 0.02" pos="{fx - 0.0015:.4f} 0 -0.04" contype="0" conaffinity="0" mass="0.0" rgba="0.24 0.90 1.0 1"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld body1="wrist_mocap" body2="wrist_body" solref="0.018 1.0" solimp="0.9 0.95 0.001"/>
  </equality>
  <contact>
    <exclude body1="wrist_body" body2="finger_l"/>
    <exclude body1="wrist_body" body2="finger_r"/>
    <exclude body1="finger_l" body2="finger_r"/>
  </contact>
  <actuator>
    <position name="act_grip_l" joint="grip_l" kp="700" ctrlrange="-0.005 0.042"/>
    <position name="act_grip_r" joint="grip_r" kp="700" ctrlrange="-0.005 0.042"/>
  </actuator>
  <sensor>
    <accelerometer name="wrist_accel" site="imu_site"/>
    <gyro name="wrist_gyro" site="imu_site"/>
    <framequat name="wrist_quat" objtype="site" objname="imu_site"/>
  </sensor>
</mujoco>
"""


def load_model_and_data(obj_type: str = "cup", mass: float = 0.25, friction: float = 0.8, scale: float = 1.0,
                        xy=(0.0, 0.0), rgba=PALETTE[0]):
    import mujoco
    xml = build_scene_xml(obj_type, mass, friction, scale, xy, rgba)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


if __name__ == "__main__":
    m, d = load_model_and_data()
    print(m.nbody, "bodies", m.ngeom, "geoms", m.nu, "actuators")
