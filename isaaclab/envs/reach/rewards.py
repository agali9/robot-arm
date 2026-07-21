"""Reach rewards — modular terms with easy-to-tune weights.

Task terms use the shared distance helper from :mod:`envs.reach.observations` (same
geometry as the observations); regularization reuses built-in
``isaaclab.envs.mdp`` rewards; the collision term reuses
:func:`robot.rewards.collision_penalty` (inactive until a contact sensor exists).

Weights live directly on the :class:`RewardTermCfg` entries — the single place to
tune. They are conservative placeholders, not a trained reward.

Term         | function                    | sign | purpose
------------ | --------------------------- | ---- | -------------------------------
distance     | reduce_distance (1 - tanh)  |  +   | dense shaping toward the target
success      | success_bonus               |  +   | sparse bonus within threshold
smooth_action| mdp.action_rate_l2          |  -   | penalize jerky action changes
joint_limits | mdp.joint_pos_limits        |  -   | penalize nearing joint limits
action       | mdp.action_l2               |  -   | penalize large actions
collision    | robot.rewards.collision...  |  -   | contact penalty (needs sensor)
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from envs.reach.commands import COMMAND_NAME
from envs.reach.observations import distance_to_target
from robot.rewards import collision_penalty
from utils import constants as C

_EE = SceneEntityCfg("robot", body_names=C.EE_BODY_NAME)


# --- Task reward functions -----------------------------------------------------------

def reduce_distance(env: ManagerBasedRLEnv, std: float = 0.1,
                    command_name: str = COMMAND_NAME,
                    asset_cfg: SceneEntityCfg = _EE) -> torch.Tensor:
    """Dense shaping in [0, 1] that grows as the EE nears the target (``1 - tanh``).

    ``std`` sets the kernel width: smaller = sharper gradient near the target. Set to
    0.1 (from 0.2) after the entropy001/gamma997 experiments proved the reach plateau
    was invariant to PPO tuning; 0.1 matches the reference fine-grained tanh kernel.
    """
    dist = distance_to_target(env, command_name, asset_cfg).squeeze(-1)
    return 1.0 - torch.tanh(dist / std)


def success_bonus(env: ManagerBasedRLEnv, threshold: float = 0.05,
                  command_name: str = COMMAND_NAME,
                  asset_cfg: SceneEntityCfg = _EE) -> torch.Tensor:
    """Sparse +1 when the end-effector is within ``threshold`` m of the target."""
    dist = distance_to_target(env, command_name, asset_cfg).squeeze(-1)
    return (dist < threshold).float()


# --- Reward group --------------------------------------------------------------------

@configclass
class ReachRewardsCfg:
    """Composed reach reward. Weights are the tuning knobs (placeholders)."""

    # Task.
    distance = RewTerm(func=reduce_distance, weight=1.0,
                       params={"std": 0.1, "command_name": COMMAND_NAME, "asset_cfg": _EE})
    success = RewTerm(func=success_bonus, weight=5.0,
                      params={"threshold": 0.05, "command_name": COMMAND_NAME, "asset_cfg": _EE})
    # Regularization / safety.
    smooth_action = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)
    action = RewTerm(func=mdp.action_l2, weight=-0.001)
    # Contact penalty — inactive (returns 0) until a "contact_forces" sensor is added.
    collision = RewTerm(func=collision_penalty, weight=-1.0,
                        params={"sensor_name": "contact_forces", "threshold": 1.0})
