"""ReachEnvCfg — the reusable reach environment configuration.

Composes the reach task from the reusable robot package + the reach-specific MDP
terms into a single ``ManagerBasedRLEnvCfg``:

    scene        : ReachSceneCfg          (ground + robot + light — reused)
    commands     : ReachCommandsCfg       (target pose command + marker)
    observations : ReachObservationsCfg   (proprio + task-space + last action)
    actions      : ActionsCfg             (joint-position targets — reused robot pkg)
    rewards      : ReachRewardsCfg         (distance/success/regularization/collision)
    terminations : ReachTerminationsCfg   (time-out / reached / invalid state)
    events       : ReachEventCfg           (joint noise + optional physics rand — reused)

Robot geometry/physics stays in ``robot.robot_cfg`` and actuator tuning in
``robot.actuators`` — this file only wires managers together. Instantiate and pass
to ``ManagerBasedRLEnv`` (see ``validation.py``); a future PPO runner consumes the
same cfg unchanged.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from envs.reach.commands import ReachCommandsCfg
from envs.reach.events import ReachEventCfg
from envs.reach.observations import ReachObservationsCfg
from envs.reach.rewards import ReachRewardsCfg
from envs.reach.scene import ReachSceneCfg
from envs.reach.terminations import ReachTerminationsCfg
from robot.actions import ActionsCfg, joint_position_action  # reuse robot action group


@configclass
