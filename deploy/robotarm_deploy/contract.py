"""Policy I/O contract — the single source of truth for deployment.

This module encodes EXACTLY how the frozen reach policy expects to be fed and how its
raw output becomes joint commands. It is intentionally **standalone** (pure Python +
numpy, no Isaac Lab import) so the deployment package is portable to a hardware machine
that has no simulator installed.

Every value here mirrors the FROZEN training configuration:

  * observation order/dims -> ``isaaclab/envs/reach/observations.py``
  * action scale / offset  -> ``isaaclab/envs/reach/reach_env_cfg.py`` (scale=1.5,
                              use_default_offset=True) and ``robot/actions.py``
  * clip_actions           -> ``isaaclab/configs/agents/rsl_rl_ppo_cfg.py`` (1.0)
  * joint names / limits   -> ``isaaclab/utils/constants.py`` (from the URDF)
  * home pose (offset)     -> ``constants.HOME_POSE`` (all zeros)

``tests/test_contract.py`` cross-checks these against the live frozen config when the
Isaac Lab package is importable, so drift is caught in CI even though runtime is
decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- Articulation identity (kinematic base->tool order; matches physics DOF order) ---
JOINT_NAMES: tuple[str, ...] = (
    "j1_joint", "j2_joint", "j3_joint", "j4_joint", "j5_joint", "j6_joint",
)
NUM_JOINTS: int = len(JOINT_NAMES)

#: End-effector body used for task-space queries (see constants.EE_BODY_NAME).
EE_BODY_NAME: str = "j6_link"

#: Home/default joint pose (rad), in JOINT_NAMES order. This is the action offset
#: (use_default_offset=True) AND the pose joint_pos_rel is measured against.
HOME_POSE: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float32)

# --- Per-joint limits (rad, N·m, rad/s), in JOINT_NAMES order (from the URDF) --------
JOINT_LOWER: np.ndarray = np.array(
    [-4.712389, -1.047198, -2.007129, -2.617994, -1.745329, -2.792527], dtype=np.float32)
JOINT_UPPER: np.ndarray = np.array(
    [4.712389, 2.181662, 2.007129, 2.617994, 1.745329, 2.792527], dtype=np.float32)
JOINT_VEL_LIMIT: np.ndarray = np.array(
    [3.14, 2.00, 1.30, 2.50, 5.20, 5.20], dtype=np.float32)
JOINT_EFFORT_LIMIT: np.ndarray = np.array(
    [30.0, 45.0, 20.0, 5.9, 4.4, 2.0], dtype=np.float32)

# --- Action processing (must replicate Isaac Lab's ActionManager exactly) -------------
#: Raw policy output is clipped to [-CLIP_ACTIONS, +CLIP_ACTIONS] before scaling.
CLIP_ACTIONS: float = 1.0
#: JointPositionAction scale: joint_target = clip(raw) * ACTION_SCALE + HOME_POSE (rad).
ACTION_SCALE: float = 1.5
#: use_default_offset=True -> the offset is HOME_POSE (above).

# --- Observation layout (concatenated, in THIS order) — total 28 dims -----------------
@dataclass(frozen=True)
class ObsField:
    name: str
    dim: int


OBS_LAYOUT: tuple[ObsField, ...] = (
    ObsField("joint_pos", NUM_JOINTS),      # joint_pos_rel = joint_pos - HOME_POSE
    ObsField("joint_vel", NUM_JOINTS),      # joint_vel_rel = joint_vel - 0
    ObsField("ee_position", 3),             # EE position, base frame
    ObsField("target_position", 3),         # target position, base frame
    ObsField("target_relative", 3),         # target - ee (base frame)
    ObsField("distance", 1),                # ||target - ee||
    ObsField("last_action", NUM_JOINTS),    # previous CLIPPED raw action
