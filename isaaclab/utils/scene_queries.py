"""Shared scene queries (end-effector pose, target, distance).

These small helpers are the single implementation of "where is the tool" and
"where is the target", imported by BOTH :mod:`robot.observations` and
:mod:`robot.rewards` so the two never drift apart. Each has the standard Isaac Lab
term signature ``func(env, asset_cfg) -> torch.Tensor`` of shape ``(num_envs, D)``,
so they can be used directly as observation/reward term functions.

The target is stored on the environment instance (no global state) in a buffer that
:func:`robot.reset.randomize_target_position` fills on reset. If no target has been
set yet, target queries return zeros so the package is safe to initialize before a
task wires up target commands.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from .constants import EE_BODY_NAME

# Attribute name for the per-env target buffer stored on the environment instance.
TARGET_ATTR = "_robotarm_target_w"

_ROBOT_EE = SceneEntityCfg("robot", body_names=EE_BODY_NAME)


# --- End-effector --------------------------------------------------------------------

def ee_position_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _ROBOT_EE) -> torch.Tensor:
    """End-effector position in the env frame (world minus env origin), ``(N, 3)``."""
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    # ``.torch`` gives the explicit tensor view of the ProxyArray (2.x multi-backend);
    # implicit indexing is deprecated and warns.
    return asset.data.body_pos_w.torch[:, body_id, :] - env.scene.env_origins


def ee_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _ROBOT_EE) -> torch.Tensor:
    """End-effector orientation quaternion (w, x, y, z), ``(N, 4)``."""
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    return asset.data.body_quat_w.torch[:, body_id, :]


def ee_pose_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _ROBOT_EE) -> torch.Tensor:
    """End-effector pose ``[pos(3), quat(4)]`` in the env frame, ``(N, 7)``."""
    return torch.cat([ee_position_w(env, asset_cfg), ee_quat_w(env, asset_cfg)], dim=-1)


# --- Target --------------------------------------------------------------------------

def get_target_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the current target positions in the env frame, ``(N, 3)``.

    Reads the environment's target buffer; returns zeros if not set yet.
    """
    target = getattr(env, TARGET_ATTR, None)
    if target is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return target


def set_target_w(env: ManagerBasedEnv, values: torch.Tensor,
                 env_ids: torch.Tensor | None = None) -> None:
    """Write target positions (env frame) into the environment's target buffer."""
    target = getattr(env, TARGET_ATTR, None)
    if target is None:
        target = torch.zeros((env.num_envs, 3), device=env.device)
        setattr(env, TARGET_ATTR, target)
    if env_ids is None:
        target[:] = values
    else:
        target[env_ids] = values


def target_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _ROBOT_EE) -> torch.Tensor:
    """Target position in the env frame, ``(N, 3)`` (``asset_cfg`` unused; kept for
    a uniform term signature)."""
    return get_target_w(env)


def target_position_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _ROBOT_EE) -> torch.Tensor:
    """Target position relative to the end-effector (target - ee), ``(N, 3)``."""
    return get_target_w(env) - ee_position_w(env, asset_cfg)


def distance_to_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _ROBOT_EE) -> torch.Tensor:
    """Euclidean distance from the end-effector to the target, ``(N, 1)``."""
    return torch.norm(target_position_rel(env, asset_cfg), dim=-1, keepdim=True)
