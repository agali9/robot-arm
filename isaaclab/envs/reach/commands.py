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
        body_name=C.EE_BODY_NAME,             # j6_link (tool0 is a fixed frame)
        resampling_time_range=(4.0, 4.0),     # resample the target every 4 s
        debug_vis=True,                        # draw the marker (False for headless)
        position_success_threshold=0.05,
        goal_pose_visualizer_cfg=TARGET_MARKER_CFG,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Position sampling box (robot base frame) — the reachable workspace.
            # (Restored to uniform random targets after the fixed-target diagnostic
            # confirmed the action-scale root cause; now testing target-conditioned reach
            # across the full workspace with scale=1.0.)
            pos_x=C.WORKSPACE.x,
            pos_y=C.WORKSPACE.y,
            pos_z=C.WORKSPACE.z,
            # Position-only reach: keep orientation fixed (identity).
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
