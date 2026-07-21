"""RobotArm task registrations (top-level ``tasks`` package on the isaaclab dir).

Importing this package registers every RobotArm Gymnasium task, so RL scripts just
do ``import tasks`` and then ``gym.make(tasks.REACH_TASK_ID, cfg=...)``.
"""

from tasks.registration import REACH_TASK_ID, register

register()

__all__ = ["REACH_TASK_ID", "register"]
