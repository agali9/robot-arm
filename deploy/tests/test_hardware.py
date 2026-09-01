"""Tests for the hardware bring-up layer: calibration, encoders, safety, dry-run, sequence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.hardware import calibration as calib  # noqa: E402
from robotarm_deploy.hardware.bringup import BringUpSequence, BringUpState  # noqa: E402
from robotarm_deploy.hardware.encoders import EncoderBank, SimulatedEncoder  # noqa: E402
from robotarm_deploy.hardware.hardware_safety import HardwareLimits, HardwareSafetyMonitor  # noqa: E402
from robotarm_deploy.hardware.simulated import make_simulated_hardware  # noqa: E402


class _Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def adv(self, dt): self.t += dt


# --- calibration ---------------------------------------------------------------------
def test_calibration_roundtrip_and_frames():
    j = calib.JointCalibration("j1_joint", zero_offset=0.5, direction=-1,
                               soft_lower=-1.0, soft_upper=1.0, homed=True)
    # mech -> joint -> mech is consistent
    for mech in (-0.3, 0.0, 0.5, 1.2):
        joint = j.to_joint(mech)
        assert abs(j.to_mech(joint) - mech) < 1e-6
    # direction/offset applied: mech==zero_offset -> joint 0
    assert abs(j.to_joint(0.5)) < 1e-6


def test_calibration_persistence(tmp_path=None):
    import tempfile
    cal = calib.RobotCalibration.identity()
    cal.joints["j2_joint"].zero_offset = 0.123
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cal.json"
        cal.save(p)
        back = calib.RobotCalibration.load(p)
    assert abs(back.joints["j2_joint"].zero_offset - 0.123) < 1e-9
    assert back.joints["j1_joint"].name == "j1_joint"


def test_calibration_tools():
    mech = np.linspace(0.1, 0.4, 20)
    assert calib.verify_direction(mech, np.ones(20)) == 1
    assert calib.verify_direction(mech, -np.ones(20)) == -1
    assert calib.check_encoder_sanity(mech)[0] is True
    assert calib.check_encoder_sanity(np.full(20, 0.2))[0] is False       # stuck
    bad = mech.copy(); bad[10] += 5.0
    assert calib.check_encoder_sanity(bad)[0] is False                    # jump


# --- encoders ------------------------------------------------------------------------
def test_encoder_bank_positions_and_velocity():
    cal = calib.RobotCalibration.identity()
    encs = {n: SimulatedEncoder(n) for n in C.JOINT_NAMES}
    clk = _Clock()
    bank = EncoderBank(encs, cal, clock=clk)
    encs["j1_joint"].set_mech(0.2)
    pos, vel, _ = bank.read()
    assert abs(pos[0] - 0.2) < 1e-6 and np.allclose(vel, 0.0)
    clk.adv(0.1); encs["j1_joint"].set_mech(0.3)
    pos, vel, _ = bank.read()
    assert vel[0] > 0.0                                                    # moved +


# --- hardware safety -----------------------------------------------------------------
def test_output_gate_requires_both_switches():
    m = HardwareSafetyMonitor(HardwareLimits(), clock=_Clock())
    m.reset()
    assert m.output_enabled is False
    m.set_policy_enable(True); assert m.output_enabled is False
    m.set_manual_enable(True); assert m.output_enabled is True
    m.estop(); assert m.output_enabled is False and m.tripped


def test_soft_limit_and_disagreement_trip():
    clk = _Clock()
    m = HardwareSafetyMonitor(HardwareLimits(max_disagreement=0.2), clock=clk)
    m.reset(); m.set_manual_enable(True); m.set_policy_enable(True)
    good = np.zeros(C.NUM_JOINTS, dtype=np.float32)
    ok, _ = m.check(good, good); assert ok
    # commanded far from measured -> disagreement
    m.reset(); m.set_manual_enable(True); m.set_policy_enable(True)
    cmd = good.copy(); cmd[0] = 1.0
    ok, faults = m.check(good, cmd)
    assert not ok and "joint_disagreement" in faults


def test_encoder_timeout_trips():
    clk = _Clock()
    m = HardwareSafetyMonitor(HardwareLimits(encoder_timeout_s=0.05), clock=clk)
    m.reset()
    clk.adv(0.2)
    ok, faults = m.check(np.zeros(C.NUM_JOINTS), None)
    assert not ok and "encoder_timeout" in faults


def test_over_current_hook_trips():
    from robotarm_deploy.hardware.motors import MotorFeedback
    m = HardwareSafetyMonitor(HardwareLimits(over_current_a=10.0), clock=_Clock())
    m.reset()
    fb = {"j1_joint": MotorFeedback(current_a=25.0)}
    ok, faults = m.check(np.zeros(C.NUM_JOINTS), None, feedback=fb)
    assert not ok and any("over_current" in f for f in faults)


# --- dry-run + bring-up sequence -----------------------------------------------------
def test_dry_run_intercepts_commands(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "cmds.jsonl"
        backend, motors, encoders, cal, monitor = make_simulated_hardware(
            dry_run=True, dry_run_log=log)
        backend.connect()
        monitor.reset()
        backend.send_joint_targets(np.full(C.NUM_JOINTS, 0.1, dtype=np.float32))
        motors.close()
        # command logged, but the follower encoder did NOT move (not transmitted)
        assert log.exists() and log.read_text().strip() != ""
        pos, _, _ = encoders.read()
        assert np.allclose(pos, 0.0)


def test_bringup_sequence_gates_and_reaches_dry_run():
    backend, motors, encoders, cal, monitor = make_simulated_hardware(dry_run=True)
    seq = BringUpSequence(backend, motors, encoders, cal, monitor)
    # cannot skip: init_motors before power_on fails
    assert seq.init_motors().ok is False
    assert seq.power_on().ok
    assert seq.init_motors().ok
    assert seq.verify_encoders().ok
    assert seq.home().ok
    assert seq.calibrate().ok
    assert seq.verify_safety().ok
    assert seq.enable_jog().ok
    assert seq.enable_dry_run().ok
    assert seq.state is BringUpState.DRY_RUN
    # LIVE refuses without confirm, then arms with confirm
    assert seq.enable_live(confirm=False).ok is False
    assert seq.enable_live(confirm=True).ok
    assert seq.state is BringUpState.LIVE and monitor.output_enabled
    assert seq.estop().ok and monitor.tripped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  {fn.__name__} PASSED")
    print(f"hardware: all {len(fns)} tests PASSED")
