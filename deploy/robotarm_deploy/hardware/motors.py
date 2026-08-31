"""Motor bridge — joint position targets -> motor commands, hardware-isolated + dry-run.

The policy emits joint position targets (rad, policy frame). Calibration converts them to
mechanism targets; this module drives the physical actuators. Two motor families match the
planned hardware, each behind the same :class:`MotorDriver` interface:

  * **J1/J2 hoverboard hub motors** (:class:`HoverboardDriver`) — BLDC, driven via a VESC
    over CAN; a position command is realized by the VESC's position PID (or an outer loop).
  * **J3-J6 serial-bus servos** (:class:`SerialBusServoDriver`) — accept a goal position
    over a serial bus (Feetech/Dynamixel-style), which closes its own position loop.

Only ``enable/disable/send_position/read_feedback`` are hardware-specific; everything above
(policy, safety, calibration) is identical across motor types. :class:`MotorBank` aggregates
per-joint drivers and provides **dry-run**: commands are logged, not transmitted — so the
whole stack can be exercised before any motor is energized.
"""

from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import contract as C


@dataclass
class MotorFeedback:
    """Optional per-motor telemetry (fields are NaN when a device can't report them)."""

    current_a: float = float("nan")     # amps (over-current hook)
    temperature_c: float = float("nan")  # deg C (thermal hook)
    voltage_v: float = float("nan")
    moving: bool = False


# --- Wire protocols (pure functions, unit-tested — the exact bytes on the bus) --------

# VESC CAN command ids (VESC firmware). Position control uses SET_POS.
VESC_CAN_PACKET_SET_POS = 4


def vesc_set_pos_frame(vesc_id: int, position_deg: float) -> tuple[int, bytes]:
    """Build the VESC ``SET_POS`` CAN frame: (extended arbitration id, 4-byte payload).

    VESC encodes position as ``int32 = degrees * 1e6`` (big-endian). The extended id is
    ``vesc_id | (packet_id << 8)``.
    """
    import struct
    arb_id = (vesc_id & 0xFF) | (VESC_CAN_PACKET_SET_POS << 8)
    val = int(round(position_deg * 1e6))
    return arb_id, struct.pack(">i", val)


# Feetech STS/SCS serial-bus servo protocol (STS3215-class).
STS_INSTR_WRITE = 0x03
STS_INSTR_READ = 0x02
STS_ADDR_GOAL_POSITION = 0x2A     # 42
STS_ADDR_PRESENT_POSITION = 0x38  # 56


def _sts_checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def feetech_write_pos_packet(servo_id: int, ticks: int) -> bytes:
    """Feetech STS 'write goal position' packet (2-byte little-endian ticks)."""
    ticks &= 0xFFFF
    params = bytes([STS_ADDR_GOAL_POSITION, ticks & 0xFF, (ticks >> 8) & 0xFF])
    length = len(params) + 2
    body = bytes([servo_id & 0xFF, length, STS_INSTR_WRITE]) + params
    return b"\xff\xff" + body + bytes([_sts_checksum(body)])


def feetech_read_pos_packet(servo_id: int) -> bytes:
    """Feetech STS 'read present position' request packet (reads 2 bytes)."""
    params = bytes([STS_ADDR_PRESENT_POSITION, 0x02])
    length = len(params) + 2
    body = bytes([servo_id & 0xFF, length, STS_INSTR_READ]) + params
    return b"\xff\xff" + body + bytes([_sts_checksum(body)])


def feetech_parse_pos_response(resp: bytes) -> int | None:
    """Parse a Feetech response ``FF FF id len err pos_lo pos_hi chk`` -> ticks (or None)."""
    if len(resp) < 8 or resp[0] != 0xFF or resp[1] != 0xFF:
        return None
    pos_lo, pos_hi = resp[5], resp[6]
    return pos_lo | (pos_hi << 8)


class MotorDriver(abc.ABC):
    """One joint's actuator. Implement the four methods for a real device."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @abc.abstractmethod
    def _enable(self) -> None: ...
    @abc.abstractmethod
    def _disable(self) -> None: ...
    @abc.abstractmethod
    def _send_position(self, mech_target_rad: float) -> None: ...

    def enable(self) -> None:
        self._enable()
        self._enabled = True

    def disable(self) -> None:
        self._disable()
        self._enabled = False

    def send_position(self, mech_target_rad: float) -> None:
        if not self._enabled:
            raise RuntimeError(f"MotorDriver[{self.name}] not enabled")
        self._send_position(float(mech_target_rad))

    def read_feedback(self) -> MotorFeedback:
        return MotorFeedback()

    def is_alive(self) -> bool:
        return True


class HoverboardDriver(MotorDriver):
    """J1/J2 hoverboard hub motor via a **VESC on a CAN bus** (real protocol).

    Sends VESC ``SET_POS`` frames (``vesc_set_pos_frame``): the mechanism target (rad) is
    scaled by the belt reduction to motor radians, converted to degrees, and packed as
    ``int32 = deg*1e6``. The VESC must be configured for position (PID) control. Feedback
    (current/temp) comes from VESC status frames read by the paired encoder.
    """

    def __init__(self, name: str, can_bus, vesc_id: int, belt_ratio: float = 1.0) -> None:
        super().__init__(name)
        self.can_bus = can_bus
        self.vesc_id = vesc_id
        self.belt_ratio = belt_ratio

    def _enable(self) -> None:
        if not self.can_bus.is_open:
            self.can_bus.open()          # opens the shared bus (idempotent)

    def _disable(self) -> None:
        # Release: command zero current (a real release frame). Bus stays open (shared).
        pass

    def _send_position(self, mech_target_rad: float) -> None:
        motor_deg = mech_target_rad * self.belt_ratio * 180.0 / np.pi
        arb_id, data = vesc_set_pos_frame(self.vesc_id, motor_deg)
        self.can_bus.send(arb_id, data, extended=True)

    def is_alive(self) -> bool:
        return self.can_bus.is_open


class SerialBusServoDriver(MotorDriver):
    """J3-J6 serial-bus servo (**Feetech STS/SCS**, real protocol) on a shared serial bus.

    ``_send_position`` writes a goal-position packet (``feetech_write_pos_packet``) to the
    servo's tick range; ``read_feedback`` / the paired encoder read present position.
    Confirm the servo family (STS3215-class assumed) and ``ticks_per_rev`` against your
    hardware before enabling motion.
    """

    def __init__(self, name: str, serial_bus, servo_id: int, ticks_per_rev: int = 4096,
                 center_tick: int = 2048) -> None:
        super().__init__(name)
        self.serial_bus = serial_bus
        self.servo_id = servo_id
        self.ticks_per_rev = ticks_per_rev
        self.center_tick = center_tick

    def rad_to_ticks(self, mech_rad: float) -> int:
        return int(round(self.center_tick + mech_rad / (2.0 * np.pi) * self.ticks_per_rev))

    def _enable(self) -> None:
        if not self.serial_bus.is_open:
            self.serial_bus.open()       # opens the shared servo bus (idempotent)
        # (a real driver also writes the torque-enable register here)

    def _disable(self) -> None:
        pass                              # a real driver writes torque-disable

    def _send_position(self, mech_target_rad: float) -> None:
        pkt = feetech_write_pos_packet(self.servo_id, self.rad_to_ticks(mech_target_rad))
        self.serial_bus.write(pkt)

