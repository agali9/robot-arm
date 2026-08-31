"""Physical transports — CAN (python-can) and serial (pyserial) buses, hardware-isolated.

Both the motor drivers and the encoders on a given bus SHARE one transport instance (one
CAN bus for J1/J2 VESCs, one serial bus for J3-J6 servos), so framing and access are
centralized. The third-party libs are imported lazily inside :meth:`open` — the package
imports fine on a machine without them or without hardware, and a clear error is raised only
when a real connection is attempted.

Neither transport is opened by construction: a ``HardwareBackend`` can exist (and dry-run)
before any bus is live.
"""

from __future__ import annotations

import time


class CanBus:
    """Thin wrapper over ``python-can`` for the VESC (hoverboard) bus."""

    def __init__(self, channel: str = "can0", interface: str = "socketcan",
                 bitrate: int = 500000) -> None:
        self.channel = channel
        self.interface = interface
        self.bitrate = bitrate
        self._bus = None

    @property
    def is_open(self) -> bool:
        return self._bus is not None

    def open(self) -> None:
        if self._bus is not None:
            return
        try:
            import can  # python-can
        except Exception as exc:
            raise RuntimeError(
                "python-can not installed; `pip install python-can` on the robot host "
                f"(needed for the {self.channel} VESC bus).") from exc
        self._bus = can.interface.Bus(channel=self.channel, interface=self.interface,
                                      bitrate=self.bitrate)

    def send(self, arbitration_id: int, data: bytes, extended: bool = True) -> None:
        if self._bus is None:
            raise RuntimeError("CanBus not open")
        import can
        self._bus.send(can.Message(arbitration_id=arbitration_id, data=data,
                                   is_extended_id=extended))

    def recv(self, timeout: float = 0.005):
        if self._bus is None:
            raise RuntimeError("CanBus not open")
        return self._bus.recv(timeout=timeout)   # a can.Message or None

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.shutdown()
