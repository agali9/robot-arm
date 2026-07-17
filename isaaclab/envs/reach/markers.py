"""Target marker configuration.

The reach target is drawn as a **visible green sphere** at the commanded position.
It is rendered by the command term's goal-pose visualizer (``UniformPoseCommand``),
so it needs no separate scene asset. Visibility is controlled by the command's
``debug_vis`` flag — set it ``False`` for headless training (see reach_env_cfg).

How it connects: :mod:`envs.reach.commands` assigns :data:`TARGET_MARKER_CFG` to the
command's ``goal_pose_visualizer_cfg``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg


def target_marker_cfg(
    radius: float = 0.03,
    color: tuple[float, float, float] = (0.1, 0.9, 0.1),
    prim_path: str = "/Visuals/Command/goal_position",
) -> VisualizationMarkersCfg:
    """Build a sphere marker cfg for the reach target (configurable radius/color)."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "target": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


#: Default green target sphere.
TARGET_MARKER_CFG = target_marker_cfg()
