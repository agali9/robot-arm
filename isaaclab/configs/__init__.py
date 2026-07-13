"""Assembled environment configurations built from the robot package.

    robot_env_cfg.RobotArmEnvCfg -- minimal, validated ManagerBasedRLEnvCfg scaffold

Tasks add their own env cfgs here (subclassing RobotArmEnvCfg or composing the robot
package directly). Importing this triggers Isaac Lab imports, so use it inside an
Isaac Lab app.
"""

from configs.robot_env_cfg import RobotArmEnvCfg

__all__ = ["RobotArmEnvCfg"]
