"""Smoke-run for the registered RobotArm reach task (no learning).

Registers the Gymnasium task, builds the env through the runner scaffold, resets,
takes a few random actions over a short rollout, verifies observations/rewards are
finite, and closes cleanly. It creates NO policy, NO networks, NO training loop.

Run:
    C:\\Users\\aniru\\OneDrive\\Documents\\IsaacLab\\isaaclab.bat -p scripts\\reach_task_smoke.py
    (add --device cpu to avoid GPU contention with an open Isaac Sim GUI)

Lives in the RobotArm project; nothing is written into the IsaacLab repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_isaaclab_src = os.environ.get("ISAACLAB_SRC")
if _isaaclab_src and _isaaclab_src not in sys.path:
    sys.path.insert(0, _isaaclab_src)

from isaaclab.app import AppLauncher  # caches the real `isaaclab` package first

parser = argparse.ArgumentParser(description="Smoke-run the RobotArm reach task.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Make the RobotArm packages importable (after the real isaaclab is cached).
_PKG_DIR = Path(__file__).resolve().parents[1] / "isaaclab"  # RobotArm/isaaclab
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

import tasks  # noqa: E402  (registers RobotArm tasks with Gymnasium)
from runners import make_reach_env  # noqa: E402


def main() -> int:
    print(f"[task-smoke] task id: {tasks.REACH_TASK_ID}")
    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs, device=args.device)

    ok = True
    try:
        core = env.unwrapped  # the ManagerBasedRLEnv behind the gym wrappers
        action_dim = core.action_manager.total_action_dim
        print(f"[task-smoke] observation space: {env.observation_space}")
        print(f"[task-smoke] action space:      {env.action_space}")
        print(f"[task-smoke] num_envs: {core.num_envs}   action dim: {action_dim}")

        print("[task-smoke] reset() ...")
        obs, _ = env.reset()
        finite_obs = bool(torch.isfinite(obs["policy"]).all())

        print(f"[task-smoke] stepping {args.steps} random actions ...")
        finite_rew = True
        rew_min, rew_max = float("inf"), float("-inf")
        for _ in range(args.steps):
            action = 2.0 * torch.rand((core.num_envs, action_dim), device=core.device) - 1.0
            obs, reward, terminated, truncated, info = env.step(action)
            finite_obs = finite_obs and bool(torch.isfinite(obs["policy"]).all())
            finite_rew = finite_rew and bool(torch.isfinite(reward).all())
            rew_min, rew_max = min(rew_min, float(reward.min())), max(rew_max, float(reward.max()))

        ok = finite_obs and finite_rew
        print("\n==================== REACH TASK SMOKE REPORT ====================")
        print(f"  task id             : {tasks.REACH_TASK_ID}")
        print(f"  steps               : {args.steps}")
        print(f"  observations finite : {finite_obs}")
        print(f"  rewards finite      : {finite_rew}  (min {rew_min:.4f}, max {rew_max:.4f})")
        print(f"  RESULT              : {'PASSED' if ok else 'FAILED'}")
        print("================================================================")
    finally:
        env.close()

    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
