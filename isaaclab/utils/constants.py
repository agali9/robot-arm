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

