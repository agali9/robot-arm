"""Load a registered RobotArm task and build its Gymnasium environment.

Small, self-contained helpers (no dependency on ``isaaclab_tasks``) that mirror the
supported Isaac Lab flow: resolve the ``env_cfg_entry_point`` from the gym registry,
instantiate the env cfg, then ``gym.make(task_id, cfg=cfg)``. This is what a future
RL runner uses to construct the environment; it is isolated from the robot package.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import gymnasium as gym


def load_env_cfg(task_id: str) -> Any:
    """Instantiate the env cfg registered under ``task_id``'s ``env_cfg_entry_point``.

    Supports a string ``"module:Class"`` entry point (the default) or a class/instance.
    """
    entry = gym.spec(task_id).kwargs.get("env_cfg_entry_point")
    if entry is None:
        raise ValueError(f"task '{task_id}' has no 'env_cfg_entry_point' registered")
    if isinstance(entry, str):
        module_name, attr_name = entry.split(":")
        entry = getattr(importlib.import_module(module_name), attr_name)
    return entry() if inspect.isclass(entry) else entry


def make_reach_env(
    task_id: str,
    num_envs: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
):
    """Register (if needed), build the env cfg, and return the Gymnasium env.

    Args:
        task_id: The registered task id (e.g. ``tasks.REACH_TASK_ID``).
        num_envs: Override the number of environments (else the cfg default).
        device: Override the sim device (e.g. ``"cpu"``, ``"cuda:0"``).
        seed: Override the env seed (RL runners set this from the agent cfg).
        render_mode: Passed through to ``gym.make`` (``None`` for headless).
    """
    import tasks  # noqa: F401  (ensures the task is registered)

    cfg = load_env_cfg(task_id)
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    if device is not None:
        cfg.sim.device = device
    if seed is not None:
        cfg.seed = int(seed)
    return gym.make(task_id, cfg=cfg, render_mode=render_mode)
