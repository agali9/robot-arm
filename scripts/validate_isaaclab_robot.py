"""Minimal Isaac Lab validation for the RobotArm project robot.

Proves the robot loads and articulates cleanly in Isaac Lab, using the RobotArm
project's own robot package (and therefore the project USD via
``utils.paths.robot_usd_path``). It does NOT train.

What it does:
  * launches Isaac Lab (headless),
  * builds the reusable ``RobotSceneCfg`` (ground + robot + light),
  * spawns + initializes the robot articulation,
  * prints joint names, DOF count, and joint limits (and cross-checks the limits
    against the documented URDF values),
  * steps a few frames at the default pose to confirm no exceptions,
  * exits cleanly.

Run it (Isaac Lab must match the running simulator's branch):

    C:\\Users\\aniru\\OneDrive\\Documents\\IsaacLab\\isaaclab.bat -p scripts\\validate_isaaclab_robot.py

If Isaac Lab is cloned but not pip-installed, set ISAACLAB_SRC to its
``source/isaaclab`` dir and run with Isaac Sim's ``python.bat`` instead.

This file lives in the RobotArm project (``scripts/``); no custom code is placed
inside the IsaacLab repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Fallback for a cloned-but-not-installed Isaac Lab (see module docstring).
_isaaclab_src = os.environ.get("ISAACLAB_SRC")
if _isaaclab_src and _isaaclab_src not in sys.path:
    sys.path.insert(0, _isaaclab_src)

from isaaclab.app import AppLauncher  # caches the real `isaaclab` package first

# --- Launch the app (headless) -------------------------------------------------------
parser = argparse.ArgumentParser(description="Validate the RobotArm robot in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="number of envs to spawn")
parser.add_argument("--steps", type=int, default=60, help="smoke-test physics steps")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Make the RobotArm robot package importable (after real isaaclab is cached) -------
_PKG_DIR = Path(__file__).resolve().parents[1] / "isaaclab"  # RobotArm/isaaclab
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from robot.terrain import RobotSceneCfg  # noqa: E402
from utils import constants as C  # noqa: E402
from utils.paths import robot_usd_path  # noqa: E402


def main() -> int:
    print(f"[validate] robot USD asset: {robot_usd_path()}")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.3])

    print("[validate] building RobotSceneCfg (ground + robot + light) ...")
    scene = InteractiveScene(RobotSceneCfg(num_envs=args.num_envs, env_spacing=2.5))

    print("[validate] sim.reset() — spawns and initializes the articulation ...")
    sim.reset()

    robot = scene["robot"]
    print(f"[validate] articulation prim : {robot.cfg.prim_path}")
    print(f"[validate] DOF count         : {robot.num_joints}")
    print(f"[validate] joint names       : {list(robot.joint_names)}")

    # Joint position limits (rad), cross-checked against the documented URDF.
    limits = robot.data.joint_pos_limits[0].detach().cpu()  # (num_joints, 2)
    print("[validate] joint position limits (rad)  [physics vs documented URDF]:")
    limits_ok = True
    for i, name in enumerate(robot.joint_names):
        lo, hi = float(limits[i, 0]), float(limits[i, 1])
        spec = C.JOINTS.get(name)
        if spec is None:
            print(f"    {name:<9} phys[{lo:+.4f}, {hi:+.4f}]  (no documented spec)")
            continue
        drift = max(abs(lo - spec.lower), abs(hi - spec.upper))
        flag = "OK" if drift < 1e-2 else "DRIFT"
        limits_ok = limits_ok and drift < 1e-2
        print(f"    {name:<9} phys[{lo:+.4f}, {hi:+.4f}]  "
              f"urdf[{spec.lower:+.4f}, {spec.upper:+.4f}]  {flag}")

    # Short smoke: hold the default pose for a few steps; confirm it stays finite.
    print(f"[validate] stepping {args.steps} steps at the default pose ...")
    default_pos = robot.data.default_joint_pos.clone()
    for _ in range(args.steps):
        robot.set_joint_position_target(default_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    joint_pos = robot.data.joint_pos[0].detach().cpu()
    finite = bool(torch.isfinite(joint_pos).all())
    print(f"[validate] joint positions finite after stepping: {finite}")

    # Assertions — any failure raises and the run reports it.
    assert robot.num_joints == 6, f"expected 6 DOF, got {robot.num_joints}"
    assert list(robot.joint_names) == list(C.JOINT_NAMES), (
        f"joint order mismatch: {list(robot.joint_names)} != {list(C.JOINT_NAMES)}")
    assert finite, "non-finite joint positions after stepping"
    assert limits_ok, "physics joint limits drifted from documented URDF values"

    print("\n[validate] VALIDATION PASSED — robot loads and articulates cleanly.")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
