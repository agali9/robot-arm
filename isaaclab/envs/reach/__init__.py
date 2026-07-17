"""Reusable reach task for the 6-DOF arm (Isaac Lab manager-based env).

Public entry point:
    ReachEnvCfg -- the ManagerBasedRLEnvCfg for the reach task.

Task-specific MDP terms live in this package (commands, observations, rewards,
terminations, events, scene, markers); the robot definition, actuators, action
group, and generic helpers are reused from the ``robot`` / ``utils`` packages.
Import inside an Isaac Lab app.
"""

from envs.reach.reach_env_cfg import ReachEnvCfg

__all__ = ["ReachEnvCfg"]
