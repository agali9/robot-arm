"""Single source of truth for robot-specific constants.

These values mirror the validated engineering data (``urdf/robot_arm.urdf`` /
``configs/robot.yaml`` in the parent project and the articulation verified live in
Isaac Sim). Everything that needs a joint name, limit, effort, or velocity imports
it from here so the numbers are never duplicated across the Isaac Lab package.

Limits are radians; effort N·m; velocity rad/s — straight from the URDF ``<limit>``
tags. The measured joint damping came from the validated USD articulation.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Articulation identity -----------------------------------------------------------

#: Driven joints in kinematic (base -> tool) order. Matches the physics DOF order
#: reported by the validated articulation.
JOINT_NAMES: tuple[str, ...] = (
    "j1_joint", "j2_joint", "j3_joint", "j4_joint", "j5_joint", "j6_joint",
)

#: End-effector rigid body used for task-space queries.
#: NOTE: the URDF's ``tool0`` flange is a *fixed frame* and is NOT a rigid body in
#: the physics articulation (fixed links are merged during import). The last dynamic
#: body is ``j6_link``; ``tool0`` sits ~9 mm from it. We therefore use ``j6_link`` as
#: the end-effector body. For exact tool-point tracking later, add a FrameTransformer
#: sensor from ``j6_link`` to the ``tool0`` frame.
