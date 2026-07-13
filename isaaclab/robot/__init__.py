"""Reusable Isaac Lab robot package for the custom 6-DOF arm.

This package defines the robot **once** so future tasks import it instead of
duplicating robot-specific logic:

    robot_cfg     -- ROBOT_CFG: ArticulationCfg (geometry + physics only)
    actuators     -- make_actuators / ActuatorTuning (hoverboard DC motors + servos)
    observations  -- ObservationsCfg (proprio + task-space + last action)
    actions       -- ActionsCfg + control-mode factories (position/velocity/effort)
    events        -- EventCfg (startup + reset randomization)
    reset         -- reset functions (target randomization)
    rewards       -- RewardsCfg + modular reward functions
    terrain       -- RobotSceneCfg (ground + robot + light)

A task typically assembles these into a ManagerBasedRLEnvCfg (see
``configs.robot_env_cfg.RobotArmEnvCfg`` for a minimal, validated example).

Note: submodules import Isaac Lab, so import this package only inside an Isaac Lab
app (e.g. launched via ``isaaclab.bat``/``isaaclab.sh``).
"""

from robot.actions import ActionsCfg, ControlMode, make_arm_action
from robot.actuators import ActuatorTuning, make_actuators
from robot.events import EventCfg
from robot.observations import ObservationsCfg
from robot.rewards import RewardsCfg
from robot.robot_cfg import ROBOT_CFG, ROBOT_PRIM_PATH
from robot.terrain import RobotSceneCfg

__all__ = [
    "ROBOT_CFG", "ROBOT_PRIM_PATH",
    "ActuatorTuning", "make_actuators",
    "ObservationsCfg",
    "ActionsCfg", "ControlMode", "make_arm_action",
    "EventCfg",
    "RewardsCfg",
    "RobotSceneCfg",
]
