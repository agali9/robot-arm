"""Gymnasium registration for the RobotArm reach task.

Exposes the validated reach environment (``envs.reach.ReachEnvCfg``) through a
stable Gymnasium task id so any RL script can discover it with ``gym.make(...)``.
Registration is **lazy** — the ``env_cfg_entry_point`` is a string, so importing
this module does not import Isaac Lab or the env; the cfg is resolved only when a
runner asks for it.

No robot logic or constants are duplicated here: the id points at the existing
``ReachEnvCfg`` (which itself reuses the robot package). Future agent configs plug
in via ``agent_cfg_entry_point`` without touching the environment.
"""

from __future__ import annotations

import gymnasium as gym

#: Stable, project-namespaced task id. Import this instead of hardcoding the string.
REACH_TASK_ID = "RobotArm-Reach-v0"

# Entry points are import strings ("module:attr"); resolved by a runner at make time.
# They require the RobotArm ``isaaclab`` directory to be on sys.path (runners add it).
_ENV_CFG_ENTRY_POINT = "envs.reach.reach_env_cfg:ReachEnvCfg"
# Primary agent config used by the RSL-RL trainer (typed Python cfg).
_RSL_RL_CFG_ENTRY_POINT = "configs.agents.rsl_rl_ppo_cfg:ReachPPORunnerCfg"
# Library-agnostic reference hyperparameters (not tied to a specific trainer).
_AGENT_CFG_ENTRY_POINT = "configs.agents:reach_ppo_cfg.yaml"


def register() -> None:
    """Register the reach task with Gymnasium (idempotent)."""
    if REACH_TASK_ID in gym.registry:
        return
    gym.register(
        id=REACH_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _ENV_CFG_ENTRY_POINT,
            # Consumed by the RL runner(s); ignored by the environment itself.
            "rsl_rl_cfg_entry_point": _RSL_RL_CFG_ENTRY_POINT,
            "agent_cfg_entry_point": _AGENT_CFG_ENTRY_POINT,
        },
    )


# Register on import so ``import tasks`` / ``import tasks.registration`` is enough.
register()
