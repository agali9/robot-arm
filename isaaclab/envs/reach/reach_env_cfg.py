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
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the 6-DOF arm reach task."""

    # Scene (num_envs / env_spacing are overridden per run, e.g. by the validator).
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=64, env_spacing=2.5)

    # MDP managers.
    observations: ReachObservationsCfg = ReachObservationsCfg()
    # Joint-position control (reused). scale=1.5 -> each joint target = home +/- 1.5 rad
    # (+/-86 deg); Isaac clamps j2 at its -1.05 rad limit, all other joints within limits.
    # Raised from 0.25 (fixed-target diagnostic proved +/-0.25 rad couldn't reach) -> 1.0
    # (random targets ~45-50% success) -> 1.5, testing whether the unreached ~half of the
    # workspace was limited by the +/-1.0 rad joint envelope (edge/corner targets).
    actions: ActionsCfg = ActionsCfg(arm_action=joint_position_action(scale=1.5))
    commands: ReachCommandsCfg = ReachCommandsCfg()
    rewards: ReachRewardsCfg = ReachRewardsCfg()
    terminations: ReachTerminationsCfg = ReachTerminationsCfg()
    events: ReachEventCfg = ReachEventCfg()

    def __post_init__(self) -> None:
        # Control / simulation rates. A 200 Hz physics step keeps the explicit
        # hoverboard actuators stable; control runs at 100 Hz (decimation 2).
        self.decimation = 2
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        # Default viewer pose (used only when rendering).
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
