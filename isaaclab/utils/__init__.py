"""Reusable helpers for the Isaac Lab robot package.

    constants     -- single source of truth for joint names/limits/efforts, EE body,
                     actuator grouping, home pose, and the task workspace box
    paths         -- portable resolution of the validated robot USD asset
    scene_queries -- end-effector pose / target / distance term functions shared by
                     observations and rewards (no duplicated kinematics)

Import order note: ``scene_queries`` imports ``isaaclab`` and is meant to run inside
an Isaac Lab app; ``constants`` and ``paths`` are dependency-light.
"""

from . import constants, paths

__all__ = ["constants", "paths", "scene_queries"]
