"""RL runner configuration scaffold.

A typed, minimal configuration describing *how* a future RL run would be launched:
which registered task, how many envs, which device, and where the agent
(PPO/other) config comes from. **It does not train** — the training fields are inert
placeholders so a real runner (RSL-RL / skrl / rl_games) can be dropped in later
without changing the environment or robot packages.

This module is isolated from the robot package; it only references a task id and an
optional agent-config entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

from tasks.registration import REACH_TASK_ID


@dataclass
class RunnerCfg:
    """How to launch (or, for now, just construct) a task environment."""

    # --- Task / environment -----------------------------------------------------------
    task_id: str = REACH_TASK_ID
    num_envs: int = 64
    device: str = "cuda:0"
    seed: int = 42

    # --- Agent config (future) --------------------------------------------------------
    # Path or "module:file.yaml" entry point for the RL algorithm's hyperparameters.
    # None until an RL runner is added; the registered task also carries a default
    # under its "agent_cfg_entry_point" kwarg.
    agent_cfg_entry_point: str | None = None

    # --- Training controls (INERT — no training is implemented) -----------------------
    max_iterations: int = 0            # 0 => construct/validate only, never train
    log_dir: str | None = None
    experiment_name: str = "robotarm_reach"

    def make_env(self, render_mode: str | None = None):
        """Build the Gymnasium env described by this cfg (no training)."""
        from runners.env_loader import make_reach_env

        return make_reach_env(
            self.task_id, num_envs=self.num_envs, device=self.device,
            render_mode=render_mode,
        )
