"""Minimal environment scaffold assembling the robot package.

This is the smallest complete ``ManagerBasedRLEnvCfg`` that wires the reusable
robot package together — scene + observations + actions + events + rewards +
terminations. It exists so the package can be **instantiated and validated** (see
``smoke_test.py``) and as the copy/paste starting point for the first real task.

It is intentionally NOT a tuned task: rewards use placeholder weights and the only
termination is time-out. A reach task would subclass this, add a target marker /
success termination, and tune rewards — without changing the robot definition.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot.actions import ActionsCfg
from robot.events import EventCfg
