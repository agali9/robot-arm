"""Safety layer — defense-in-depth between the policy and any actuator backend.

The policy is a black box trained in sim; nothing downstream should trust it blindly.
Every command passes through these independent checks before reaching hardware:

    NaN/inf detection (obs + action) · policy-output clamping · joint-limit enforcement ·
    velocity (slew-rate) limiting · watchdog (inference stalled) · command timeout ·
    state-staleness / loss-of-communication · emergency-stop latch.

On ANY violation the layer engages a latched **safe-hold**: it stops issuing new motion
and repeatedly commands the last known-good target until an explicit ``reset()``. This
module is pure/numpy and has no I/O, so it is fully unit-testable (``tests/test_safety.py``)
and identical across sim and hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import contract as C


class SafetyState(str, Enum):
    OK = "ok"
    HOLD = "hold"          # latched safe-hold (a fault tripped); needs reset()
    ESTOP = "estop"        # external emergency stop asserted


@dataclass
class SafetyConfig:
    """Tunable safety envelope. Defaults are conservative (from the URDF limits)."""

    joint_lower: np.ndarray = field(default_factory=lambda: C.JOINT_LOWER.copy())
    joint_upper: np.ndarray = field(default_factory=lambda: C.JOINT_UPPER.copy())
    joint_vel_limit: np.ndarray = field(default_factory=lambda: C.JOINT_VEL_LIMIT.copy())
    #: Extra factor applied to the velocity limit when slew-limiting targets (<=1 = strict).
    vel_limit_scale: float = 1.0
    #: Max age (s) of a robot-state snapshot before it is "stale" -> loss of comms.
    state_max_age_s: float = 0.1
    #: Watchdog: max interval (s) between successful inference cycles before HOLD.
    watchdog_timeout_s: float = 0.2
    #: Command timeout (s): if no command is dispatched within this, hold.
    command_timeout_s: float = 0.2


@dataclass
class SafetyReport:
    """Per-cycle outcome from the safety layer."""

    state: SafetyState
    target: np.ndarray                    # the command actually approved (rad)
    faults: tuple[str, ...] = ()
    holding: bool = False
    #: The clipped raw action used this cycle (zeros on hold). This is the single source
    #: of truth for the observation ``last_action`` term, so it never drifts.
    clipped_action: np.ndarray = field(default_factory=lambda: np.zeros(C.ACTION_DIM,
                                                                        dtype=np.float32))


class SafetyLayer:
    """Stateful safety supervisor. One instance per robot; call once per control cycle."""

    def __init__(self, cfg: SafetyConfig | None = None,
                 clock: "callable[[], float]" = time.monotonic) -> None:
        self.cfg = cfg or SafetyConfig()
        self._clock = clock
        self._state = SafetyState.OK
        self._last_good_target = C.HOME_POSE.copy()
        self._last_inference_t = self._clock()
        self._last_command_t = self._clock()
        self._faults: list[str] = []

    # --- state ------------------------------------------------------------------------
    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def engaged(self) -> bool:
        """True when NOT free to issue new motion (HOLD or ESTOP)."""
        return self._state is not SafetyState.OK

    def estop(self) -> None:
        """Assert emergency stop (latched)."""
        self._state = SafetyState.ESTOP

    def trip(self, fault: str) -> None:
        """Latch a safe-HOLD for a named fault (unless already ESTOPped)."""
        if self._state is not SafetyState.ESTOP:
            self._state = SafetyState.HOLD
        if fault not in self._faults:
            self._faults.append(fault)

    def reset(self) -> None:
        """Clear HOLD/ESTOP and faults (operator-acknowledged). Resets watchdogs."""
        self._state = SafetyState.OK
        self._faults.clear()
        now = self._clock()
        self._last_inference_t = now
        self._last_command_t = now

    # --- checks -----------------------------------------------------------------------
    @staticmethod
    def _finite(x: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(x)))

    def check_observation(self, obs: np.ndarray) -> bool:
        """Reject non-finite observations (sensor glitch / bad FK)."""
        if not self._finite(obs):
            self.trip("nan_observation")
            return False
        return True

    def check_state_fresh(self, state_stamp: float, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        if (now - state_stamp) > self.cfg.state_max_age_s:
            self.trip("stale_state")
            return False
        return True

    def check_watchdog(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        if (now - self._last_inference_t) > self.cfg.watchdog_timeout_s:
