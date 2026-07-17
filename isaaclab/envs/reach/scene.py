"""Reach scene — reuses the robot package's minimal scene, with a lowered ground.

The reach scene is the reusable :class:`robot.terrain.RobotSceneCfg` (robot + light)
but overrides the ground so it sits **below** the workspace. The arm base is
kinematically fixed (its USD ``root_joint``), so it needs no ground for support;
keeping the ground clear prevents contact-recoil instability when the arm sweeps
low under random/early-policy actions. The target marker is drawn by the reach
command's goal visualizer, not by a scene asset.

Extend this class to add cameras, contact sensors, or obstacles per task without
touching the robot definition.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from robot.terrain import RobotSceneCfg

#: How far below the base to place the ground plane (meters).
GROUND_Z = -0.75


@configclass
class ReachSceneCfg(RobotSceneCfg):
    """Robot + light (inherited) + a lowered ground; marker comes from the command."""

    # Override the inherited terrain with a simple ground plane set below the base.
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)),
    )
