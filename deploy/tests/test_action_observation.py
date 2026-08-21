"""Tests for the action processor and observation builder (pure contract logic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.action import ActionProcessor  # noqa: E402
from robotarm_deploy.observation import ObservationBuilder, RobotState  # noqa: E402


def test_action_clip_scale_clamp():
    ap = ActionProcessor()
    # An out-of-range raw action clips to +/-1 then scales by 1.5, then clamps to limits.
    raw = np.array([5.0, -5.0, 0.0, 0.5, -0.5, 2.0], dtype=np.float32)
    target = ap.process(raw)
    # clipped = [1,-1,0,0.5,-0.5,1]; scaled = clipped*1.5
    expected = np.clip(np.array([1, -1, 0, 0.5, -0.5, 1]) * C.ACTION_SCALE,
                       C.JOINT_LOWER, C.JOINT_UPPER)
    np.testing.assert_allclose(target, expected, atol=1e-6)
    # last clipped action is tracked for the obs term
    np.testing.assert_allclose(ap.last_clipped_action,
                               np.array([1, -1, 0, 0.5, -0.5, 1], dtype=np.float32))


def test_action_within_joint_limits_always():
    ap = ActionProcessor()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        target = ap.process(rng.standard_normal(6).astype(np.float32) * 10)
        assert np.all(target >= C.JOINT_LOWER - 1e-6)
        assert np.all(target <= C.JOINT_UPPER + 1e-6)


def test_observation_layout_and_values():
    ob = ObservationBuilder()
    jp = np.arange(6, dtype=np.float32) * 0.1
    jv = np.arange(6, dtype=np.float32) * -0.2
    ee = np.array([0.3, 0.0, 0.35], dtype=np.float32)
    tgt = np.array([0.4, 0.2, 0.30], dtype=np.float32)
    last = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    obs = ob.build(RobotState(jp, jv, ee, stamp=1.0), tgt, last)
    assert obs.shape == (C.OBS_DIM,)
    sl = C.obs_slices()
    np.testing.assert_allclose(obs[sl["joint_pos"]], jp - C.HOME_POSE)
    np.testing.assert_allclose(obs[sl["joint_vel"]], jv)
    np.testing.assert_allclose(obs[sl["ee_position"]], ee)
    np.testing.assert_allclose(obs[sl["target_position"]], tgt)
    np.testing.assert_allclose(obs[sl["target_relative"]], tgt - ee)
    np.testing.assert_allclose(obs[sl["distance"]], [np.linalg.norm(tgt - ee)], atol=1e-6)
    np.testing.assert_allclose(obs[sl["last_action"]], last)


if __name__ == "__main__":
    test_action_clip_scale_clamp()
    test_action_within_joint_limits_always()
    test_observation_layout_and_values()
    print("action+observation: all tests PASSED")
