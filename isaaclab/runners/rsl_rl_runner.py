"""RSL-RL construction helpers (the chosen RL library for Isaac Lab 2.3.x).

Small, isolated glue that resolves the task's RSL-RL agent config from the gym
registry and builds the RSL-RL objects (env wrapper + ``OnPolicyRunner``). This is
what both the training entrypoint and the dry-run use, so the construction logic
lives in exactly one place and knows nothing about the robot package.

Isaac Lab / RSL-RL imports are done lazily inside the functions so importing
``runners`` (e.g. from the plain smoke run) does not require the RL libraries.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import gymnasium as gym


def load_rsl_rl_cfg(task_id: str, entry_key: str = "rsl_rl_cfg_entry_point") -> Any:
    """Instantiate the RSL-RL runner cfg registered under ``task_id``."""
    entry = gym.spec(task_id).kwargs.get(entry_key)
    if entry is None:
        raise ValueError(f"task '{task_id}' has no '{entry_key}' registered")
    if isinstance(entry, str):
        module_name, attr_name = entry.split(":")
        entry = getattr(importlib.import_module(module_name), attr_name)
    return entry() if inspect.isclass(entry) else entry


def build_runner(env, agent_cfg, log_dir: str | None = None, device: str | None = None):
    """Wrap ``env`` for RSL-RL and construct the ``OnPolicyRunner``.

    Constructing the runner builds the actor/critic networks, the PPO algorithm, and
    the optimizer — i.e. all the RL objects. This does NOT train; call
    ``runner.learn(...)`` separately to start learning.

    Args:
        env: The Gymnasium env from ``gym.make`` (or ``make_reach_env``).
        agent_cfg: An ``RslRlOnPolicyRunnerCfg`` (e.g. ``ReachPPORunnerCfg``).
        log_dir: Directory for TensorBoard/checkpoints (None => no logging dir).
        device: Torch/sim device; defaults to ``agent_cfg.device``.

    Returns:
        ``(runner, wrapped_env)``.
    """
    from importlib import metadata

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    # Migrate the agent cfg to the installed rsl-rl version BEFORE ``to_dict()``.
    # Isaac Lab's ``RslRlMLPModelCfg`` still carries legacy fields (``stochastic``,
    # ``init_noise_std``, ...) that rsl-rl >= 5.0.0 replaced with ``distribution_cfg``.
    # The supported entrypoints (scripts/reinforcement_learning/rsl_rl/train.py) call
    # this shim; it strips those deprecated keys so they don't reach ``MLPModel`` as
    # unexpected kwargs (which otherwise fails runner construction). Skipping it is what
    # caused the "MLPModel.__init__() got an unexpected keyword argument 'stochastic'".
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    device = device or agent_cfg.device
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    return runner, wrapped_env
