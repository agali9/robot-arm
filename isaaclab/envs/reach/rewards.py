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
