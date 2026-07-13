"""Reset logic (custom reset functions used as event terms).

Isaac Lab performs resets through *event terms* with ``mode="reset"``. This module
holds the robot-specific reset function that isn't already provided by
``isaaclab.envs.mdp``: sampling a new **target position** into the environment's
target buffer (read back by observations and rewards).

The other requested reset behaviors reuse built-in mdp terms, wired in
:mod:`robot.events`:

* **initial pose / small joint noise** — ``mdp.reset_joints_by_offset`` with a
  configurable ``position_range`` (small = joint noise, larger = pose randomization),
* **optional physics randomization** — ``mdp.randomize_rigid_body_material`` (startup).

Keeping the reset *functions* here and their *wiring* in ``events.py`` makes both
easy to extend without touching the rest of the package.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv

from utils import constants as C
from utils import scene_queries as sq


def _uniform(n: int, rng: tuple[float, float], device: torch.device) -> torch.Tensor:
    return torch.rand(n, device=device) * (rng[1] - rng[0]) + rng[0]


def randomize_target_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = C.WORKSPACE.x,
    y_range: tuple[float, float] = C.WORKSPACE.y,
    z_range: tuple[float, float] = C.WORKSPACE.z,
) -> None:
    """Sample a uniform target position (env frame) for the given envs on reset.

    The target is stored on the environment via :func:`utils.scene_queries.set_target_w`
    (per-env buffer, created on first use). Ranges define the reachable workspace box
    and are overridable per task through the event term ``params``.
    """
    n = len(env_ids)
    device = env.device
    positions = torch.stack(
        [_uniform(n, x_range, device),
         _uniform(n, y_range, device),
         _uniform(n, z_range, device)],
        dim=-1,
    )
    sq.set_target_w(env, positions, env_ids)
