"""Validation for the reach environment — no learning.

Builds the reach env, resets it repeatedly, drives random actions for several
hundred steps, and checks that: the robot stays stable, the target randomizes on
reset, and rewards/observations stay finite (no NaNs, no crashes). Prints a concise
report and exits cleanly.

Run:
    C:\\Users\\aniru\\OneDrive\\Documents\\IsaacLab\\isaaclab.bat -p isaaclab\\envs\\reach\\validation.py
    (add --device cpu to avoid GPU contention with an open Isaac Sim GUI)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_isaaclab_src = os.environ.get("ISAACLAB_SRC")
if _isaaclab_src and _isaaclab_src not in sys.path:
    sys.path.insert(0, _isaaclab_src)

from isaaclab.app import AppLauncher  # caches the real `isaaclab` package first

parser = argparse.ArgumentParser(description="Validate the RobotArm reach environment.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--resets", type=int, default=8)
parser.add_argument("--steps_per_reset", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Make the RobotArm packages importable (after the real isaaclab is cached).
_PKG_DIR = Path(__file__).resolve().parents[2]  # RobotArm/isaaclab
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from envs.reach import ReachEnvCfg  # noqa: E402
from envs.reach.commands import COMMAND_NAME  # noqa: E402


def _target_of_env0(env: ManagerBasedRLEnv) -> list[float]:
    return env.command_manager.get_command(COMMAND_NAME)[0, :3].detach().cpu().tolist()


def main() -> int:
    cfg = ReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    print(f"[reach-val] building ReachEnv (num_envs={args.num_envs}) ...")
    env = ManagerBasedRLEnv(cfg=cfg)

    ok = True
    finite_obs = finite_rew = True
    max_abs_joint_pos = 0.0
    rew_min, rew_max = math.inf, -math.inf
    targets: list[list[float]] = []
    term_counts = {"time_out": 0, "target_reached": 0, "invalid_state": 0}

    try:
        action_dim = env.action_manager.total_action_dim
        print(f"[reach-val] obs dim: {env.observation_manager.group_obs_dim['policy']}  "
              f"action dim: {action_dim}")
        print(f"[reach-val] reward terms: {list(env.reward_manager.active_terms)}")
        print(f"[reach-val] termination terms: {list(env.termination_manager.active_terms)}")

        for r in range(args.resets):
            env.reset()
            targets.append(_target_of_env0(env))  # target after each reset
            for _ in range(args.steps_per_reset):
                action = 2.0 * torch.rand((env.num_envs, action_dim), device=env.device) - 1.0
                obs, reward, terminated, truncated, info = env.step(action)

                finite_obs = finite_obs and bool(torch.isfinite(obs["policy"]).all())
                finite_rew = finite_rew and bool(torch.isfinite(reward).all())
                rew_min = min(rew_min, float(reward.min()))
                rew_max = max(rew_max, float(reward.max()))
                jp = env.scene["robot"].data.joint_pos.torch
                max_abs_joint_pos = max(max_abs_joint_pos, float(jp.abs().max()))
                for name in term_counts:
                    if name in env.termination_manager.active_terms:
                        term_counts[name] += int(env.termination_manager.get_term(name).sum())
        total_steps = args.resets * args.steps_per_reset

        # --- checks ---
        unique_targets = _count_distinct(targets, eps=1e-3)
        target_randomizes = unique_targets >= 2
        joint_pos_finite = math.isfinite(max_abs_joint_pos)
        robot_stable = joint_pos_finite and max_abs_joint_pos < 10.0  # rad, generous
        ok = finite_obs and finite_rew and target_randomizes and robot_stable

        print("\n==================== REACH VALIDATION REPORT ====================")
        print(f"  total steps            : {total_steps} ({args.resets} resets)")
        print(f"  observations finite    : {finite_obs}")
        print(f"  rewards finite         : {finite_rew}   (min {rew_min:.4f}, max {rew_max:.4f})")
        print(f"  target randomizes      : {target_randomizes}  ({unique_targets} distinct across resets)")
        print(f"  sample targets (env0)  : {[[round(v,3) for v in t] for t in targets[:4]]}")
        print(f"  robot stable           : {robot_stable}  (max|joint pos| {max_abs_joint_pos:.3f} rad)")
        print(f"  termination counts     : {term_counts}")
        print(f"  RESULT                 : {'PASSED' if ok else 'FAILED'}")
        print("================================================================")
    finally:
        env.close()

    return 0 if ok else 1


def _count_distinct(points: list[list[float]], eps: float) -> int:
    distinct: list[list[float]] = []
    for p in points:
        if all(math.dist(p, q) > eps for q in distinct):
            distinct.append(p)
    return len(distinct)


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
