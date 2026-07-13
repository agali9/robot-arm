"""Terrain and minimal scene support.

Provides the smallest scene the robot needs to exist cleanly in Isaac Lab: a flat
ground plane, the robot, and a light. :class:`RobotSceneCfg` is a reusable
:class:`~isaaclab.scene.InteractiveSceneCfg` that tasks extend (adding target
markers, cameras, contact sensors, obstacles, …) rather than rebuild.

How it connects: a task env cfg sets ``scene = RobotSceneCfg(num_envs=..., env_spacing=...)``.
The robot is placed under each env namespace via the ``{ENV_REGEX_NS}/Robot`` prim
path from :mod:`robot.robot_cfg`. The ground is a simple plane; swap
``TERRAIN_CFG.terrain_type`` to ``"generator"`` or ``"usd"`` for richer terrain later.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from robot.robot_cfg import ROBOT_CFG, ROBOT_PRIM_PATH

#: Flat ground plane shared by all envs.
TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    debug_vis=False,
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Minimal reusable scene: ground + robot + dome light."""

    terrain = TERRAIN_CFG

    # Fresh copy of the robot cfg pinned to the per-env prim path.
    robot = ROBOT_CFG.replace(prim_path=ROBOT_PRIM_PATH)

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )
