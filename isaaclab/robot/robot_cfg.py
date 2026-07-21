"""RobotCfg — the ArticulationCfg for the 6-DOF arm.

This module owns **robot geometry and physics only**: which USD to spawn, rigid-body
and articulation-root solver settings, collision/self-collision behavior, initial
state, and soft joint-limit margin. **Actuator tuning is deliberately not here** — it
comes from :mod:`robot.actuators` (``make_actuators``) so gains can be retuned
without editing the robot definition.

How it connects: every task env cfg puts ``ROBOT_CFG`` into its scene (usually as
``ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")``). Observations, actions,
events, and rewards all reference this articulation by the scene asset name
``"robot"``. Joint limits/efforts come from the validated URDF via
:mod:`utils.constants`; the USD path is resolved portably by :mod:`utils.paths`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from robot.actuators import make_actuators
from utils import constants as C
from utils.paths import robot_usd_path

#: Prim path pattern; the scene replicates the robot under each env namespace.
