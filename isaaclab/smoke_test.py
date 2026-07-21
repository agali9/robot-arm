"""Smoke test — proves the Isaac Lab robot package is wired correctly.

It builds the minimal env from :mod:`configs.robot_env_cfg`, which forces every
manager to construct (RobotCfg -> articulation, actuators, observations, actions,
events, rewards), then resets once and steps a single random action. It does NOT
train.

Run it with Isaac Lab's launcher (it starts its own headless SimulationApp — this
is separate from any interactive Isaac Sim you have open):

    # Windows
    "C:/Users/aniru/OneDrive/Documents/IsaacLab/isaaclab.bat" -p isaaclab/smoke_test.py
    # Linux/mac
    ./isaaclab.sh -p isaaclab/smoke_test.py

Import ordering matters: we import ``isaaclab.app`` and launch the app FIRST so the
real ``isaaclab`` package is cached in ``sys.modules`` before we add this package's
directory to ``sys.path`` — that way our top-level ``robot``/``utils``/``configs``
packages never shadow the real ``isaaclab``.
"""

from __future__ import annotations

import argparse
import os
import sys

# Fallback for a cloned-but-not-pip-installed Isaac Lab: set ISAACLAB_SRC to the
# repo's `source/isaaclab` directory and we add it to the path. When Isaac Lab is
# installed normally (`isaaclab -i`) this is unset and unused.
_isaaclab_src = os.environ.get("ISAACLAB_SRC")
if _isaaclab_src and _isaaclab_src not in sys.path:
    sys.path.insert(0, _isaaclab_src)

from isaaclab.app import AppLauncher  # caches the real `isaaclab` package first

# --- Launch the app (headless) -------------------------------------------------------
parser = argparse.ArgumentParser(description="RobotArm Isaac Lab package smoke test.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now make the robot package importable and pull in the rest -----------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the `isaaclab/` package dir

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from configs.robot_env_cfg import RobotArmEnvCfg  # noqa: E402


def main() -> int:
    print("[smoke] building RobotArmEnvCfg ...")
    cfg = RobotArmEnvCfg()
    cfg.scene.num_envs = 1  # keep it tiny

    print("[smoke] creating ManagerBasedRLEnv (spawns robot, builds all managers) ...")
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        print(f"[smoke] observation space: {env.observation_space}")
        print(f"[smoke] action space:      {env.action_space}")
        print(f"[smoke] action dim:        {env.action_manager.total_action_dim}")
        print(f"[smoke] joint order:       {env.scene['robot'].joint_names}")

        print("[smoke] reset() ...")
        obs, _ = env.reset()
        assert "policy" in obs, "policy observation group missing"
        print(f"[smoke] policy obs shape:  {tuple(obs['policy'].shape)}")

        print("[smoke] step() with one random action ...")
        action = 2.0 * torch.rand(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        ) - 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        assert torch.isfinite(reward).all(), "non-finite reward"
        assert torch.isfinite(obs["policy"]).all(), "non-finite observation"
        print(f"[smoke] reward: {float(reward[0]):.4f}  "
              f"terminated: {bool(terminated[0])}  truncated: {bool(truncated[0])}")
    finally:
        env.close()

    print("\n[smoke] SMOKE TEST PASSED — package wired correctly.")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
