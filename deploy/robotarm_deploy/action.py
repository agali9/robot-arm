"""Action processing — raw policy output -> commanded joint position targets.

Replicates Isaac Lab's ActionManager pipeline EXACTLY (the exported policy does NOT
include it — it emits only the raw actor mean):

    raw  --clip[-1,1]-->  clipped  --*scale + home-->  target_rad  --clamp limits-->  cmd

Keeping this here (not in the policy, not in the hardware driver) means the policy stays
task-only and every backend receives identical, already-safe joint targets.
"""

from __future__ import annotations

import numpy as np

from . import contract as C


class ActionProcessor:
    """Stateless converter from raw policy actions to clamped joint position targets.

    Also tracks the last *clipped* raw action, which is fed back into the observation
    (``last_action`` term) — so this object owns the single definition of "last action".
    """

    def __init__(self) -> None:
        self._last_clipped = np.zeros(C.ACTION_DIM, dtype=np.float32)

    @property
    def last_clipped_action(self) -> np.ndarray:
        """The most recent clipped raw action (what the obs ``last_action`` term uses)."""
        return self._last_clipped.copy()

    def reset(self) -> None:
        self._last_clipped[:] = 0.0

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        """Return joint position targets (rad, JOINT_NAMES order) for ``raw_action``.

        Steps mirror the frozen config: clip -> scale+offset -> clamp to joint limits.
        Updates ``last_clipped_action`` as a side effect.
        """
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw.shape[0] != C.ACTION_DIM:
            raise ValueError(f"expected {C.ACTION_DIM} actions, got {raw.shape[0]}")

        clipped = np.clip(raw, -C.CLIP_ACTIONS, C.CLIP_ACTIONS)
        self._last_clipped = clipped.astype(np.float32)

        target = clipped * C.ACTION_SCALE + C.HOME_POSE            # rad, absolute target
        target = np.clip(target, C.JOINT_LOWER, C.JOINT_UPPER)     # respect URDF limits
        return target.astype(np.float32)
