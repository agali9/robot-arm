"""Reach observations + shared task-space queries.

Composes the policy observation vector for the reach task and defines the
command-based task-space helpers (target position / relative / distance) that
rewards and terminations reuse — so the geometry is implemented once.

Observation vector (concatenated, in order):
    joint positions, joint velocities, end-effector position, target position,
    target-relative position, distance to target, previous action.

Reuse: proprioception uses the built-in ``isaaclab.envs.mdp`` terms; end-effector
position reuses ``utils.scene_queries``; the target comes from the reach command
(``envs.reach.commands``). Frames are consistent — EE and target are both expressed
in the env frame, and the relative/distance terms are frame-invariant.

Cameras later (no redesign): add a ``CameraCfg`` sensor to the scene and append an
image ``ObsTerm`` to ``PolicyCfg`` (or a separate image group). Existing terms are
untouched.
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

from envs.reach.commands import COMMAND_NAME
from utils import constants as C
from utils import scene_queries as sq

# End-effector entity (passed as a term param so the manager resolves its body id).
_EE = SceneEntityCfg("robot", body_names=C.EE_BODY_NAME)


# --- Shared task-space helpers (also used by rewards & terminations) -----------------

def target_position_w(env: ManagerBasedRLEnv, command_name: str = COMMAND_NAME,
                      asset_cfg: SceneEntityCfg = _EE) -> torch.Tensor:
    """Target position in the world frame, ``(N, 3)`` (command is in the base frame)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w.torch, asset.data.root_quat_w.torch, command[:, :3]
    )
    return des_pos_w


def target_position(env: ManagerBasedRLEnv, command_name: str = COMMAND_NAME,
                    asset_cfg: SceneEntityCfg = _EE) -> torch.Tensor:
    """Target position in the env frame, ``(N, 3)``."""
    return target_position_w(env, command_name, asset_cfg) - env.scene.env_origins


def target_position_rel(env: ManagerBasedRLEnv, command_name: str = COMMAND_NAME,
                        asset_cfg: SceneEntityCfg = _EE) -> torch.Tensor:
    """Target position relative to the end-effector (target - ee), ``(N, 3)``."""
    return target_position(env, command_name, asset_cfg) - sq.ee_position_w(env, asset_cfg)


def distance_to_target(env: ManagerBasedRLEnv, command_name: str = COMMAND_NAME,
                       asset_cfg: SceneEntityCfg = _EE) -> torch.Tensor:
    """Euclidean distance from the end-effector to the target, ``(N, 1)``."""
    return torch.norm(target_position_rel(env, command_name, asset_cfg), dim=-1, keepdim=True)


# --- Observation group ---------------------------------------------------------------

@configclass
class ReachObservationsCfg:
    """Observation groups for the reach task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: proprioception + task-space + last action."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=sq.ee_position_w, params={"asset_cfg": _EE})
        target_position = ObsTerm(func=target_position,
                                  params={"command_name": COMMAND_NAME, "asset_cfg": _EE})
        target_relative = ObsTerm(func=target_position_rel,
                                  params={"command_name": COMMAND_NAME, "asset_cfg": _EE})
        distance = ObsTerm(func=distance_to_target,
                           params={"command_name": COMMAND_NAME, "asset_cfg": _EE})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False   # enable per-term noise later if desired
            self.concatenate_terms = True    # single flat vector for the policy

    policy: PolicyCfg = PolicyCfg()
