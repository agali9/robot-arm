"""Target command (the reach goal).

Uses Isaac Lab's current ``UniformPoseCommand`` term to sample a target pose per
episode/interval and expose it to observations/rewards through the command manager.
This is the supported "CommandTerm" mechanism; it also owns the target marker.

Only **position** matters for this reach task, so the orientation ranges are fixed
(identity) and rewards track position only. The sampling box comes from
:data:`utils.constants.WORKSPACE` (configurable — edit the ranges or pass a custom
``Workspace``), keeping the target inside the robot's reachable region.

How it connects: ``reach_env_cfg.ReachEnvCfg.commands = ReachCommandsCfg()``.
Observations read it via ``mdp.generated_commands``/helpers; rewards/terminations
read it via ``env.command_manager.get_command("ee_pose")``.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from envs.reach.markers import TARGET_MARKER_CFG
from utils import constants as C

#: Name of the target command (referenced by observations/rewards/terminations).
COMMAND_NAME = "ee_pose"


@configclass
class ReachCommandsCfg:
    """Command terms for the reach MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
