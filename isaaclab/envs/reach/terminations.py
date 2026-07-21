"""Reach terminations.

Episodes end on: **time-out** (max length), **target reached** (EE within a
threshold of the target), or an **invalid robot state** (non-finite or runaway
joint state — a safety net during untrained/random rollouts).

Adding more conditions later is a one-liner: write a ``func(env, ...) -> BoolTensor``
of shape ``(num_envs,)`` and add a ``DoneTerm`` field below.
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from envs.reach.commands import COMMAND_NAME
from envs.reach.observations import distance_to_target
from utils import constants as C

_EE = SceneEntityCfg("robot", body_names=C.EE_BODY_NAME)
_ROBOT = SceneEntityCfg("robot")


# --- Termination functions -----------------------------------------------------------

def target_reached(env: ManagerBasedRLEnv, threshold: float = 0.05,
                   command_name: str = COMMAND_NAME,
                   asset_cfg: SceneEntityCfg = _EE) -> torch.Tensor:
    """True where the end-effector is within ``threshold`` m of the target."""
    dist = distance_to_target(env, command_name, asset_cfg).squeeze(-1)
    return dist < threshold


def robot_invalid_state(env: ManagerBasedRLEnv, max_joint_vel: float = 50.0,
                        asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """True where the articulation state is non-finite or a joint velocity blows up."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    joint_vel = asset.data.joint_vel.torch
    non_finite = ~(torch.isfinite(joint_pos).all(dim=-1) & torch.isfinite(joint_vel).all(dim=-1))
    runaway = (joint_vel.abs() > max_joint_vel).any(dim=-1)
    return non_finite | runaway


# --- Termination group ---------------------------------------------------------------

@configclass
class ReachTerminationsCfg:
    """Termination terms for the reach MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Reaching the goal is intentionally NOT a terminal event — this matches NVIDIA's
    # reference reach task (time-out only). A *non-timeout* terminal is not bootstrapped
    # by RSL-RL, so it truncates all future return; combined with the dense distance
    # reward over a full episode (worth far more than the sparse success bonus), it makes
    # the optimal policy HOVER just outside the threshold instead of reaching. Leaving
    # this disabled lets episodes run to the time-out and the success reward
    # (rewards.py) accumulate while the arm holds the target. To restore early
    # stop-on-success semantics, set this back to
    # ``DoneTerm(func=target_reached, params={"threshold": 0.05, ...})``.
    target_reached = None

    invalid_state = DoneTerm(
        func=robot_invalid_state,
        params={"max_joint_vel": 50.0, "asset_cfg": _ROBOT},
    )
