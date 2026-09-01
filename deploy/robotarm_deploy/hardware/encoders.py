"""Encoder abstraction — raw sensor counts -> mechanism angle (rad), hardware-isolated.

Encoders differ (absolute magnetic, incremental quadrature, serial-bus servo feedback);
the policy must never see any of that. Each :class:`EncoderDriver` converts its raw
reading to a **mechanism angle in radians at the joint output**; the calibration layer
then maps mechanism -> policy joint angle. Only normalized joint positions leave the
:class:`EncoderBank`.

Velocity is derived by finite-difference here (with a light low-pass) unless a driver
reports it directly, so the observation's ``joint_vel`` term is populated on hardware.

Real transports (SPI/I2C/quadrature/serial) live behind ``read_raw()`` — the ONE method a
new encoder implements. ``SimulatedEncoder`` mirrors a commanded position for dry-run and
tests.
"""

from __future__ import annotations

import abc
import time

import numpy as np

from .. import contract as C


class EncoderDriver(abc.ABC):
    """One joint's position sensor. Implement ``read_raw`` for a real device."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def read_raw(self) -> float:
        """Return the raw mechanism angle (rad) at the joint output (no calibration)."""

    def is_alive(self) -> bool:
        """Best-effort liveness (override for a real bus heartbeat). Default True."""
        return True


class AbsoluteEncoder(EncoderDriver):
    """Absolute encoder: a single reading gives the angle (no homing needed).

    ``counts_per_rev`` and ``gear_ratio`` map raw counts to joint-output radians:
    ``mech = 2*pi * (counts / counts_per_rev) / gear_ratio``. ``read_counts`` is the raw
    device hook (stubbed until wired).
    """

    def __init__(self, name: str, counts_per_rev: int = 4096, gear_ratio: float = 1.0,
                 read_counts=None) -> None:
        super().__init__(name)
        self.counts_per_rev = counts_per_rev
        self.gear_ratio = gear_ratio
        self._read_counts = read_counts

    def read_raw(self) -> float:
        if self._read_counts is None:
            raise NotImplementedError(
                f"AbsoluteEncoder[{self.name}]: wire read_counts to the device (SPI/I2C).")
        counts = float(self._read_counts())
        return 2.0 * np.pi * (counts / self.counts_per_rev) / self.gear_ratio


class IncrementalEncoder(EncoderDriver):
    """Incremental encoder: reports counts relative to power-on; needs homing.

    After :meth:`set_home` (called at a known joint angle), ``read_raw`` returns the
    absolute mechanism angle. Before homing it raises — enforcing "home before use".
    """

    def __init__(self, name: str, counts_per_rev: int = 4096, gear_ratio: float = 1.0,
                 read_counts=None) -> None:
        super().__init__(name)
        self.counts_per_rev = counts_per_rev
        self.gear_ratio = gear_ratio
        self._read_counts = read_counts
        self._home_counts: float | None = None
        self._home_angle: float = 0.0

    def set_home(self, home_mech_angle: float = 0.0) -> None:
        if self._read_counts is None:
            raise NotImplementedError(f"IncrementalEncoder[{self.name}]: wire read_counts.")
        self._home_counts = float(self._read_counts())
        self._home_angle = float(home_mech_angle)

    @property
    def homed(self) -> bool:
        return self._home_counts is not None

    def read_raw(self) -> float:
        if not self.homed:
            raise RuntimeError(f"IncrementalEncoder[{self.name}] not homed; call set_home().")
        counts = float(self._read_counts())
        d_rev = (counts - self._home_counts) / self.counts_per_rev / self.gear_ratio
        return self._home_angle + 2.0 * np.pi * d_rev


class SerialServoEncoder(EncoderDriver):
    """J3-J6 position feedback from the serial-bus servo (Feetech STS) — shares the bus.

    Requests present position (``feetech_read_pos_packet``), parses the response, and maps
    ticks -> mechanism angle: ``(ticks - center_tick)/ticks_per_rev * 2*pi``. Same bus as the
    servo driver (half-duplex request/response).
    """

    def __init__(self, name: str, serial_bus, servo_id: int, ticks_per_rev: int = 4096,
                 center_tick: int = 2048) -> None:
        super().__init__(name)
        self.serial_bus = serial_bus
        self.servo_id = servo_id
        self.ticks_per_rev = ticks_per_rev
        self.center_tick = center_tick

    def is_alive(self) -> bool:
        return self.serial_bus.is_open

    def read_raw(self) -> float:
        from .motors import feetech_parse_pos_response, feetech_read_pos_packet
        self.serial_bus.flush_input()
        self.serial_bus.write(feetech_read_pos_packet(self.servo_id))
        resp = self.serial_bus.read(8)
        ticks = feetech_parse_pos_response(resp)
        if ticks is None:
            raise RuntimeError(f"SerialServoEncoder[{self.name}]: no/invalid response")
        return (ticks - self.center_tick) / self.ticks_per_rev * 2.0 * np.pi


class VescCanEncoder(EncoderDriver):
    """J1/J2 position feedback from the VESC over CAN — shares the bus with the driver.

    VESCs broadcast status frames; this caches the latest reported motor position for
    ``vesc_id`` and maps it back to the joint (divide by belt reduction). Position over CAN
    is firmware-dependent (enable a status frame carrying position, e.g. STATUS_5 tacho or a
    custom app); ``_extract_position_deg`` is the single hook to match your VESC config.
    """

    def __init__(self, name: str, can_bus, vesc_id: int, belt_ratio: float = 1.0) -> None:
        super().__init__(name)
        self.can_bus = can_bus
        self.vesc_id = vesc_id
        self.belt_ratio = belt_ratio
        self._last_motor_deg: float | None = None

    def is_alive(self) -> bool:
        return self.can_bus.is_open

    def _extract_position_deg(self, msg) -> float | None:
        """Return motor position (deg) if ``msg`` is a position status frame for our VESC.

        Placeholder matching the common convention (int32 big-endian deg*1e6 in the frame).
        Adjust the packet id + byte layout to your VESC firmware's status configuration.
        """
        import struct
        if (msg.arbitration_id & 0xFF) != (self.vesc_id & 0xFF):
            return None
        if len(msg.data) < 4:
            return None
        return struct.unpack(">i", bytes(msg.data[:4]))[0] / 1e6

    def read_raw(self) -> float:
        # Drain pending frames; keep the latest position for our VESC.
        for _ in range(16):
            msg = self.can_bus.recv(timeout=0.001)
            if msg is None:
                break
            deg = self._extract_position_deg(msg)
            if deg is not None:
                self._last_motor_deg = deg
        if self._last_motor_deg is None:
            raise RuntimeError(f"VescCanEncoder[{self.name}]: no position status yet")
        return (self._last_motor_deg * np.pi / 180.0) / self.belt_ratio


class SimulatedEncoder(EncoderDriver):
    """Follows a commanded mechanism angle (for dry-run / tests). Always homed."""

    def __init__(self, name: str, initial: float = 0.0) -> None:
        super().__init__(name)
        self._mech = float(initial)

    def set_mech(self, mech: float) -> None:
        self._mech = float(mech)

    def read_raw(self) -> float:
        return self._mech


class EncoderBank:
    """All joint encoders + calibration -> normalized joint positions/velocities.

    This is the ONLY object that produces the policy's ``joint_pos``/``joint_vel``. It
    holds the calibration so upstream code deals purely in policy-frame radians.
    """

    def __init__(self, encoders: dict[str, EncoderDriver], calibration,
                 vel_filter_alpha: float = 0.4, clock=time.monotonic) -> None:
        self._enc = encoders
        self._cal = calibration
        self._alpha = float(vel_filter_alpha)
        self._clock = clock
        self._last_pos: np.ndarray | None = None
        self._last_t: float | None = None
        self._vel = np.zeros(C.NUM_JOINTS, dtype=np.float32)

    def alive(self) -> dict[str, bool]:
        return {n: self._enc[n].is_alive() for n in C.JOINT_NAMES}

    def read_joint_positions(self) -> np.ndarray:
        """Return calibrated joint positions (rad, JOINT_NAMES order)."""
        mech = np.array([self._enc[n].read_raw() for n in C.JOINT_NAMES], dtype=np.float32)
        return self._cal.to_joint_vec(mech)

    def read(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(joint_pos, joint_vel, stamp)``; velocity is filtered finite-diff."""
        now = self._clock()
        pos = self.read_joint_positions()
        if self._last_pos is not None and self._last_t is not None:
            dt = max(now - self._last_t, 1e-4)
            raw_vel = (pos - self._last_pos) / dt
            self._vel = self._alpha * raw_vel + (1.0 - self._alpha) * self._vel
        self._last_pos, self._last_t = pos, now
        return pos.astype(np.float32), self._vel.astype(np.float32), now
