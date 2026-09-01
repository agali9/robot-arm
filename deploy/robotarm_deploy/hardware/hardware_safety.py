"""Hardware safety monitor — the hardware-side checks, layered on the policy SafetyLayer.

The frozen :class:`safety.SafetyLayer` guards the policy side (NaN, clip, slew, watchdog,
e-stop). Hardware adds failure modes a policy can't see. This monitor runs alongside and
must be GREEN for motor output to be permitted:

    soft limits (calibrated) · hard limits (URDF) · encoder timeout · motor timeout ·
    joint-disagreement (commanded vs measured) · over-current hook · temperature hook ·
    power/voltage hook · manual enable switch · policy enable switch.

``output_enabled`` is the single authority the bring-up uses to decide whether a command
may reach a motor. It is only True when BOTH physical switches are on AND no fault is
latched. Any tripped fault latches until :meth:`reset` (operator-acknowledged).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import contract as C


@dataclass
class HardwareLimits:
    """Hardware safety thresholds. Defaults are conservative."""

    soft_lower: np.ndarray = field(default_factory=lambda: C.JOINT_LOWER.copy())
    soft_upper: np.ndarray = field(default_factory=lambda: C.JOINT_UPPER.copy())
    hard_lower: np.ndarray = field(default_factory=lambda: C.JOINT_LOWER.copy())
    hard_upper: np.ndarray = field(default_factory=lambda: C.JOINT_UPPER.copy())
    #: Max allowed |commanded - measured| joint position (rad) before "disagreement".
    max_disagreement: float = 0.35
    encoder_timeout_s: float = 0.05
    motor_timeout_s: float = 0.05
    over_current_a: float = 30.0
    over_temperature_c: float = 80.0
    min_voltage_v: float = 30.0


class HardwareSafetyMonitor:
    """Runtime hardware safety. One instance per robot; checked every control cycle."""

    def __init__(self, limits: HardwareLimits | None = None,
                 clock=time.monotonic) -> None:
        self.limits = limits or HardwareLimits()
        self._clock = clock
        self._faults: list[str] = []
        self._tripped = False
        self._manual_enable = False     # physical enable / dead-man switch
        self._policy_enable = False     # software: operator armed the policy
        self._last_encoder_t = 0.0
        self._last_motor_t = 0.0

    # --- switches ---------------------------------------------------------------------
    def set_manual_enable(self, on: bool) -> None:
        self._manual_enable = bool(on)

    def set_policy_enable(self, on: bool) -> None:
        self._policy_enable = bool(on)

    @property
    def output_enabled(self) -> bool:
        """True only if both switches are on and nothing is tripped."""
        return self._manual_enable and self._policy_enable and not self._tripped

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def faults(self) -> tuple[str, ...]:
        return tuple(self._faults)

    def trip(self, fault: str) -> None:
        self._tripped = True
        if fault not in self._faults:
            self._faults.append(fault)

    def estop(self) -> None:
        self.trip("estop")
        self._policy_enable = False

    def reset(self) -> None:
        self._faults.clear()
        self._tripped = False
        now = self._clock()
        self._last_encoder_t = now
        self._last_motor_t = now

    # --- heartbeats -------------------------------------------------------------------
    def note_encoder(self) -> None:
        self._last_encoder_t = self._clock()

    def note_motor(self) -> None:
        self._last_motor_t = self._clock()

    # --- preflight (static, before enabling output) -----------------------------------
    def preflight(self, calibration) -> tuple[bool, list[str]]:
        """Static checks that must pass before motors may be commanded."""
        problems: list[str] = []
        if not calibration.all_homed():
            problems.append("not all joints homed")
        lo, hi = calibration.soft_limits()
        if np.any(lo < self.limits.hard_lower - 1e-6):
            problems.append("a soft_lower is below the URDF hard limit")
        if np.any(hi > self.limits.hard_upper + 1e-6):
            problems.append("a soft_upper is above the URDF hard limit")
        if np.any(lo >= hi):
            problems.append("soft_lower >= soft_upper for some joint")
        return (len(problems) == 0), problems

    # --- per-cycle runtime check ------------------------------------------------------
    def check(self, measured_pos: np.ndarray, commanded_pos: np.ndarray | None,
              feedback: dict | None = None, voltage_v: float | None = None,
              now: float | None = None) -> tuple[bool, tuple[str, ...]]:
        """Run all runtime checks. Trips (latches) on any violation. Returns (ok, faults)."""
        now = self._clock() if now is None else now
        m = np.asarray(measured_pos, dtype=np.float32).reshape(-1)

        if not np.all(np.isfinite(m)):
            self.trip("encoder_nonfinite")
        if np.any(m < self.limits.hard_lower - 1e-3) or np.any(m > self.limits.hard_upper + 1e-3):
            self.trip("hard_limit_exceeded")
        if np.any(m < self.limits.soft_lower - 1e-3) or np.any(m > self.limits.soft_upper + 1e-3):
            self.trip("soft_limit_exceeded")
        if (now - self._last_encoder_t) > self.limits.encoder_timeout_s:
            self.trip("encoder_timeout")
        if (now - self._last_motor_t) > self.limits.motor_timeout_s:
            self.trip("motor_timeout")
        if commanded_pos is not None:
            c = np.asarray(commanded_pos, dtype=np.float32).reshape(-1)
            if float(np.abs(c - m).max()) > self.limits.max_disagreement:
                self.trip("joint_disagreement")
        if feedback:
            for name, fb in feedback.items():
                cur = getattr(fb, "current_a", float("nan"))
                temp = getattr(fb, "temperature_c", float("nan"))
                if np.isfinite(cur) and cur > self.limits.over_current_a:
                    self.trip(f"over_current:{name}")
                if np.isfinite(temp) and temp > self.limits.over_temperature_c:
                    self.trip(f"over_temperature:{name}")
        if voltage_v is not None and np.isfinite(voltage_v) and voltage_v < self.limits.min_voltage_v:
            self.trip("under_voltage")

        return (not self._tripped), self.faults
