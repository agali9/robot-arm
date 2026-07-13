"""Events — startup and reset randomization wired as manager event terms.

:class:`EventCfg` groups the domain-randomization and reset behavior the robot
package offers, all modular and individually removable/retunable:

* ``physics_material`` (**startup**, optional) — randomize friction/restitution.
* ``reset_joints`` (**reset**) — set joints to home + uniform noise. Widen
  ``position_range`` to randomize the initial pose; keep it small for light noise.
* ``reset_target`` (**reset**) — sample a new reach target into the env buffer.

How it connects: a task sets ``events = EventCfg()``. The reset target term feeds
the target observations/rewards; the joint-reset term defines the start state. To
disable a behavior, drop the field in a subclass or set its params to no-op ranges.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot import reset
from utils import constants as C


@configclass
class EventCfg:
    """Startup + reset events for the robot."""

    # Optional physics randomization (startup). Remove if you want a fixed robot.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.1),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 16,
        },
    )

    # Initial pose + small joint noise around the home pose.
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )

    # New reach target per reset (env-frame workspace box).
    reset_target = EventTerm(
        func=reset.randomize_target_position,
        mode="reset",
        params={"x_range": C.WORKSPACE.x, "y_range": C.WORKSPACE.y,
                "z_range": C.WORKSPACE.z},
    )
