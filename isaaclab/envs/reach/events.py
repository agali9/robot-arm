"""Reach events — reuses the robot package's event terms.

Reuse: :class:`robot.events.EventCfg` already provides optional startup physics
randomization and a reset that applies small joint perturbations around the home
pose. The reach task keeps those and **disables the buffer-based target reset**,
because the target is now owned by the reach command (``UniformPoseCommand``
resamples it), not by a reset event.

Randomization the reach env exposes (all from the reused terms):
  * target position       -> handled by the command (see envs.reach.commands),
  * small joint noise      -> robot.events.reset_joints (reset_joints_by_offset),
  * optional physics rand  -> robot.events.physics_material (startup).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from robot.events import EventCfg as RobotEventCfg


@configclass
class ReachEventCfg(RobotEventCfg):
    """Robot events, minus the buffer target reset (the command owns the target)."""

    # Disable the robot package's buffer-based target randomization for this task;
    # setting an event term to None removes it from the manager.
    reset_target = None
