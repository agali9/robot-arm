"""HardwareBackend — the real-robot ActuatorInterface (motors + encoders + calibration).

Composes the hardware pieces into the same interface the inference loop already uses:

    read_state()          <- EncoderBank (calibrated joint pos/vel) + FK (EE) + timestamp
    send_joint_targets()  -> calibration -> MotorBank (gated by the safety monitor)

The policy/obs/action/policy-side-safety code is unchanged; swapping ``SimBackend`` for
``HardwareBackend`` is the whole port. Output is HARD-GATED by
``HardwareSafetyMonitor.output_enabled`` — if the enable switches are off or any fault is
latched, targets are NOT transmitted (motors hold), so the backend is safe by default and
supports dry-run transparently (``MotorBank(dry_run=True)``).
"""

from __future__ import annotations

import numpy as np

from .. import contract as C
from ..observation import RobotState
from .base import ActuatorInterface, CommunicationError
from .encoders import EncoderBank
from .hardware_safety import HardwareSafetyMonitor
from .motors import MotorBank


class HardwareBackend(ActuatorInterface):
    """ActuatorInterface backed by real motors + encoders (or simulated ones for dry-run)."""

    name = "hardware"

    def __init__(self, motor_bank: MotorBank, encoder_bank: EncoderBank,
                 calibration, kinematics, monitor: HardwareSafetyMonitor) -> None:
        self._motors = motor_bank
        self._encoders = encoder_bank
        self._cal = calibration
        self._kin = kinematics
        self._monitor = monitor
        self._last_commanded = C.HOME_POSE.copy()
        self._connected = False

    @property
    def monitor(self) -> HardwareSafetyMonitor:
        return self._monitor

    @property
    def last_commanded(self) -> np.ndarray:
        return self._last_commanded.copy()

    def connect(self) -> None:
        # Motors are enabled explicitly by the bring-up sequence, not on connect(), so a
        # HardwareBackend can be constructed and read (encoders) before any motor is live.
        self._connected = True
        self._monitor.reset()

    def disconnect(self) -> None:
        try:
            self._motors.disable_all()
        finally:
            self._motors.close()
            self._connected = False

    def read_state(self) -> RobotState:
        if not self._connected:
            raise CommunicationError("HardwareBackend not connected")
        pos, vel, stamp = self._encoders.read()
        self._monitor.note_encoder()
        ee = np.asarray(self._kin.ee_position(pos), dtype=np.float32).reshape(3)
        return RobotState(joint_pos=pos, joint_vel=vel, ee_position=ee, stamp=stamp)

    def send_joint_targets(self, targets: np.ndarray) -> None:
        """Transmit joint targets to motors — ONLY if the safety monitor permits it."""
        if not self._connected:
            raise CommunicationError("HardwareBackend not connected")
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        self._last_commanded = targets.copy()
        # Commands flow to the MotorBank when LIVE (output_enabled) OR in dry-run (where the
        # MotorBank logs instead of transmitting). Otherwise hold — do not touch motors.
        if not (self._monitor.output_enabled or self._motors.dry_run):
            return
        mech = self._cal.to_mech_vec(targets)
        self._motors.send_mech_targets(mech)   # transmits (live) or logs (dry-run)
        self._monitor.note_motor()

    def motor_feedback(self):
        return self._motors.feedback()
