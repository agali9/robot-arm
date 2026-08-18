"""Contract tests — dims/order, and cross-check against the FROZEN training constants.

The deployment contract is a standalone copy (for portability); these tests catch drift
by importing the pure-Python ``isaaclab/utils/constants.py`` (no Isaac Sim needed) and
asserting the joint names, limits, and home pose still match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # deploy/
from robotarm_deploy import contract as C  # noqa: E402


def test_dims():
    assert C.OBS_DIM == 28
    assert C.ACTION_DIM == 6
    assert sum(f.dim for f in C.OBS_LAYOUT) == C.OBS_DIM


def test_obs_slices_cover_vector():
    sl = C.obs_slices()
    covered = sorted((s.start, s.stop) for s in sl.values())
    # contiguous, non-overlapping, covers [0, 28)
    assert covered[0][0] == 0 and covered[-1][1] == C.OBS_DIM
    for (a0, a1), (b0, b1) in zip(covered, covered[1:]):
        assert a1 == b0


def test_contract_matches_frozen_constants():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "isaaclab" / "utils"))
    sys.path.insert(0, str(root / "isaaclab"))
    try:
        from utils import constants as K  # frozen training constants (pure python)
    except Exception:
        import pytest
        pytest.skip("frozen constants not importable in this environment")
