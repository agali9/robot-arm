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
