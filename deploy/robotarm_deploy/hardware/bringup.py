"""Bring-up state machine — the safe, gated path from powered-off to live policy.

Enforces the ordering the hardware requires, so the first real motion is predictable:

    OFF -> POWERED_ON -> MOTORS_INIT -> ENCODERS_VERIFIED -> HOMED -> CALIBRATED
        -> SAFETY_VERIFIED -> JOG_READY -> DRY_RUN -> LIVE

Each transition runs its checks and REFUSES to advance if a prerequisite is missing.
``estop()`` and ``shutdown()`` are reachable from any state. Motor output is permitted
only in ``LIVE`` (and only while the safety monitor's two enable switches are on) — in
``DRY_RUN`` the policy runs but commands are intercepted/logged, never transmitted.

This orchestrates the hardware objects; it does not import the policy. The inference loop
(``InferenceApp``) is wired to the ``HardwareBackend`` and driven only after ``enable_live``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .. import contract as C
from . import calibration as calib
from .encoders import EncoderBank
from .hardware_backend import HardwareBackend
from .hardware_safety import HardwareSafetyMonitor
from .motors import MotorBank


class BringUpState(str, Enum):
    OFF = "off"
    POWERED_ON = "powered_on"
    MOTORS_INIT = "motors_init"
    ENCODERS_VERIFIED = "encoders_verified"
    HOMED = "homed"
    CALIBRATED = "calibrated"
    SAFETY_VERIFIED = "safety_verified"
    JOG_READY = "jog_ready"
    DRY_RUN = "dry_run"
    LIVE = "live"
    ESTOP = "estop"
    SHUTDOWN = "shutdown"


# Allowed forward progression (ESTOP/SHUTDOWN are always allowed, handled separately).
_ORDER = [BringUpState.OFF, BringUpState.POWERED_ON, BringUpState.MOTORS_INIT,
          BringUpState.ENCODERS_VERIFIED, BringUpState.HOMED, BringUpState.CALIBRATED,
          BringUpState.SAFETY_VERIFIED, BringUpState.JOG_READY]


@dataclass
class StepResult:
    ok: bool
    state: BringUpState
    message: str = ""
    detail: dict = field(default_factory=dict)


class BringUpSequence:
    """Drives one robot through bring-up. Call the steps in order; each is idempotent-ish."""

    def __init__(self, backend: HardwareBackend, motors: MotorBank, encoders: EncoderBank,
                 calibration: calib.RobotCalibration, monitor: HardwareSafetyMonitor) -> None:
        self.backend = backend
        self.motors = motors
        self.encoders = encoders
        self.cal = calibration
        self.monitor = monitor
        self.state = BringUpState.OFF
        self.log: list[str] = []

    def _record(self, ok: bool, state: BringUpState, msg: str, **detail) -> StepResult:
        tag = "OK " if ok else "FAIL"
        self.log.append(f"[{tag}] {state.value}: {msg}")
        if ok:
            self.state = state
        return StepResult(ok, self.state, msg, detail)

    def _require(self, state: BringUpState) -> str | None:
        """Return an error message if we're not at least at ``state`` (in ESTOP if tripped)."""
        if self.state in (BringUpState.ESTOP, BringUpState.SHUTDOWN):
            return f"blocked in {self.state.value}; reset()/restart required"
        if state in _ORDER and self.state in _ORDER and _ORDER.index(self.state) < _ORDER.index(state):
            return f"prerequisite not met (at {self.state.value}, need {state.value})"
        return None

    # --- steps ------------------------------------------------------------------------
    def power_on(self) -> StepResult:
        self.backend.connect()
        return self._record(True, BringUpState.POWERED_ON, "logic powered; backend connected")

    def init_motors(self) -> StepResult:
        if (err := self._require(BringUpState.POWERED_ON)):
            return self._record(False, self.state, err)
        alive = self.motors.alive()
        dead = [n for n, a in alive.items() if not a]
        if dead:
            return self._record(False, BringUpState.POWERED_ON,
                                f"motors not responding: {dead}", alive=alive)
        self.motors.enable_all()
        return self._record(True, BringUpState.MOTORS_INIT,
                            f"{C.NUM_JOINTS} motors enabled", alive=alive)

    def verify_encoders(self) -> StepResult:
        if (err := self._require(BringUpState.MOTORS_INIT)):
            return self._record(False, self.state, err)
        alive = self.encoders.alive()
        dead = [n for n, a in alive.items() if not a]
        if dead:
            return self._record(False, BringUpState.MOTORS_INIT,
                                f"encoders not responding: {dead}", alive=alive)
        pos = self.encoders.read_joint_positions()
        if not np.all(np.isfinite(pos)):
            return self._record(False, BringUpState.MOTORS_INIT, "non-finite encoder read")
        # plausibility: measured positions within hard limits
        if np.any(pos < C.JOINT_LOWER - 0.2) or np.any(pos > C.JOINT_UPPER + 0.2):
            return self._record(False, BringUpState.MOTORS_INIT,
                                "encoder read implausible vs joint limits", pos=pos.tolist())
        return self._record(True, BringUpState.ENCODERS_VERIFIED,
                            "encoders alive + plausible", pos=pos.tolist())

    def home(self) -> StepResult:
        if (err := self._require(BringUpState.ENCODERS_VERIFIED)):
            return self._record(False, self.state, err)
        for n in C.JOINT_NAMES:
            self.cal.joints[n].homed = True
        return self._record(True, BringUpState.HOMED,
                            "all joints homed (encoder zero established)")

    def calibrate(self, calibration: calib.RobotCalibration | None = None) -> StepResult:
        if (err := self._require(BringUpState.HOMED)):
            return self._record(False, self.state, err)
        if calibration is not None:
            self.cal.joints = calibration.joints
        if not self.cal.all_homed():
            return self._record(False, BringUpState.HOMED, "calibration reports unhomed joints")
        return self._record(True, BringUpState.CALIBRATED, "calibration applied")

    def verify_safety(self) -> StepResult:
        if (err := self._require(BringUpState.CALIBRATED)):
            return self._record(False, self.state, err)
        ok, problems = self.monitor.preflight(self.cal)
        if not ok:
            return self._record(False, BringUpState.CALIBRATED,
                                "preflight failed", problems=problems)
        self.monitor.reset()
        return self._record(True, BringUpState.SAFETY_VERIFIED, "safety preflight passed")

    def enable_jog(self) -> StepResult:
        if (err := self._require(BringUpState.SAFETY_VERIFIED)):
            return self._record(False, self.state, err)
        return self._record(True, BringUpState.JOG_READY,
                            "manual jog permitted (operator-supervised)")

    def enable_dry_run(self) -> StepResult:
        if self.state not in (BringUpState.JOG_READY, BringUpState.DRY_RUN):
            return self._record(False, self.state, "reach JOG_READY before DRY_RUN")
        self.motors.dry_run = True
        self.monitor.set_policy_enable(True)   # policy runs; MotorBank intercepts commands
        return self._record(True, BringUpState.DRY_RUN,
                            "policy dry-run: commands logged, NOT transmitted")

    def enable_live(self, confirm: bool = False) -> StepResult:
        if self.state != BringUpState.DRY_RUN:
            return self._record(False, self.state, "must pass DRY_RUN before LIVE")
        if not confirm:
            return self._record(False, self.state,
                                "LIVE requires confirm=True (explicit operator arming)")
        ok, faults = self.monitor.check(self.encoders.read_joint_positions(), None)
        if not ok:
            return self._record(False, BringUpState.DRY_RUN,
                                f"safety not green: {faults}")
        self.motors.dry_run = False
        self.monitor.set_manual_enable(True)
        self.monitor.set_policy_enable(True)
        return self._record(True, BringUpState.LIVE,
                            "LIVE: motor output ENABLED (monitor-gated)")

    # --- always reachable -------------------------------------------------------------
    def estop(self) -> StepResult:
        self.monitor.estop()
        self.motors.disable_all()
        return self._record(True, BringUpState.ESTOP, "E-STOP: motors disabled, output latched off")

    def disable_policy(self) -> StepResult:
        """Software policy-disable without a full e-stop (returns toward DRY_RUN)."""
        self.monitor.set_policy_enable(False)
        self.motors.dry_run = True
        return self._record(True, BringUpState.DRY_RUN, "policy output disabled")

    def shutdown(self) -> StepResult:
        self.monitor.set_policy_enable(False)
        self.monitor.set_manual_enable(False)
        self.backend.disconnect()
        return self._record(True, BringUpState.SHUTDOWN, "graceful shutdown complete")
