"""RL launch scaffolding for RobotArm tasks (isolated from the robot package).

    env_loader.make_reach_env -- build a registered task's Gymnasium env
    runner_cfg.RunnerCfg       -- typed scaffold for a future RL run (no training)

A future PPO runner imports these, adds an agent/trainer, and launches; nothing in
the environment or robot packages changes.
"""

from runners.env_loader import load_env_cfg, make_reach_env
from runners.runner_cfg import RunnerCfg

__all__ = ["load_env_cfg", "make_reach_env", "RunnerCfg"]
