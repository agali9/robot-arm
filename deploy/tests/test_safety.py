"""Tests for the safety layer (NaN, clip, slew-rate, watchdog, e-stop, staleness)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.safety import SafetyConfig, SafetyLayer, SafetyState  # noqa: E402


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def _layer(**cfg):
    clk = FakeClock()
    return SafetyLayer(SafetyConfig(**cfg), clock=clk), clk


def test_nan_action_trips_hold():
    sl, _ = _layer()
    rep = sl.approve(np.array([np.nan, 0, 0, 0, 0, 0]), C.HOME_POSE, 0.01, state_stamp=0.0)
    assert rep.holding and sl.state is SafetyState.HOLD
    assert "nan_action" in rep.faults


def test_nan_observation_trips():
    sl, _ = _layer()
    assert sl.check_observation(np.zeros(C.OBS_DIM)) is True
    assert sl.check_observation(np.array([np.inf] + [0] * (C.OBS_DIM - 1))) is False
    assert sl.state is SafetyState.HOLD


def test_clip_and_scale_normal():
    sl, _ = _layer(vel_limit_scale=1.0)
    # large-but-finite action -> clipped to +/-1, scaled, slew-limited from home
    rep = sl.approve(np.full(6, 10.0, dtype=np.float32), C.HOME_POSE, 0.01, state_stamp=0.0)
    assert not rep.holding and rep.state is SafetyState.OK
    np.testing.assert_allclose(rep.clipped_action, np.ones(6), atol=1e-6)
    # target must respect joint limits and the per-cycle slew cap
    assert np.all(rep.target <= C.JOINT_UPPER + 1e-6)
    max_step = C.JOINT_VEL_LIMIT * 0.01
    assert np.all(np.abs(rep.target - C.HOME_POSE) <= max_step + 1e-6)


def test_slew_rate_limits_velocity():
    sl, _ = _layer()
    # command a big jump; the first cycle can move at most vel_limit*dt per joint
    rep = sl.approve(np.ones(6, dtype=np.float32), C.HOME_POSE, dt=0.01, state_stamp=0.0)
    step = np.abs(rep.target - C.HOME_POSE)
    assert np.all(step <= C.JOINT_VEL_LIMIT * 0.01 + 1e-6)


def test_estop_latches_until_reset():
    sl, clk = _layer()
    sl.estop()
    rep = sl.approve(np.zeros(6), C.HOME_POSE, 0.01, state_stamp=clk())
    assert rep.state is SafetyState.ESTOP and rep.holding
    sl.reset()
    assert sl.state is SafetyState.OK


def test_watchdog_timeout():
    sl, clk = _layer(watchdog_timeout_s=0.1)
    sl.note_inference()
    clk.advance(0.2)                       # no inference for 200 ms
    assert sl.check_watchdog() is False
    assert sl.state is SafetyState.HOLD


def test_stale_state_trips():
    sl, clk = _layer(state_max_age_s=0.05)
    clk.advance(1.0)
    assert sl.check_state_fresh(state_stamp=0.0) is False   # 1 s old >> 50 ms
    assert sl.state is SafetyState.HOLD


def test_hold_commands_last_good_target():
    sl, _ = _layer()
    good = sl.approve(np.full(6, 0.2, dtype=np.float32), C.HOME_POSE, 0.01, state_stamp=0.0)
    last_good = good.target.copy()
    bad = sl.approve(np.array([np.nan] * 6), good.target, 0.01, state_stamp=0.0)
    assert bad.holding
    np.testing.assert_allclose(bad.target, last_good)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASSED")
    print("safety: all tests PASSED")
