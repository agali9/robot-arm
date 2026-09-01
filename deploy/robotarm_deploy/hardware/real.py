"""Real hardware factory — assemble the physical HardwareBackend from config.

Wires the real transports, drivers, encoders, loaded calibration, and the URDF FK (with
the validated base->env transform) into the same ``HardwareBackend`` the inference loop
already uses. Nothing is opened/energized here — buses open lazily on motor/encoder use,
and output stays gated by the safety monitor. Build this, run the bring-up sequence, and
only go LIVE after a clean dry-run.

Joint→device map matches the hardware plan: **J1/J2 hoverboard motors on a VESC/CAN bus**,
**J3–J6 serial-bus servos**. IDs/ratios/ports live in :class:`HardwareConfig` — the only
thing to fill in for a specific robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import contract as C
from ..kinematics import UrdfKinematicsProvider
from .calibration import RobotCalibration
from .encoders import EncoderBank, SerialServoEncoder, VescCanEncoder
from .hardware_backend import HardwareBackend
from .hardware_safety import HardwareLimits, HardwareSafetyMonitor
from .motors import HoverboardDriver, MotorBank, SerialBusServoDriver
from .transports import CanBus, SerialBus

_PROJECT = Path(__file__).resolve().parents[3]

#: Hardware plan: which joints are hoverboard (VESC/CAN) vs serial-bus servos.
HOVERBOARD_JOINTS: tuple[str, ...] = ("j1_joint", "j2_joint")
SERVO_JOINTS: tuple[str, ...] = ("j3_joint", "j4_joint", "j5_joint", "j6_joint")


@dataclass
class HardwareConfig:
    """Per-robot wiring. Fill in the real ids/ports/ratios before bring-up."""

    can_channel: str = "can0"
    can_interface: str = "socketcan"
    can_bitrate: int = 500000
    serial_port: str = "/dev/ttyUSB0"
    serial_baud: int = 1000000
    #: VESC CAN ids for J1/J2 (motor:joint belt ratios).
    vesc_ids: dict[str, int] = field(default_factory=lambda: {"j1_joint": 1, "j2_joint": 2})
    belt_ratios: dict[str, float] = field(default_factory=lambda: {"j1_joint": 1.0, "j2_joint": 1.0})
    #: Serial servo ids + tick scaling for J3-J6.
    servo_ids: dict[str, int] = field(default_factory=lambda: {
        "j3_joint": 3, "j4_joint": 4, "j5_joint": 5, "j6_joint": 6})
    servo_ticks_per_rev: int = 4096
    servo_center_tick: int = 2048
    urdf_path: str = str(_PROJECT / "urdf" / "robot_arm.urdf")
    base_transform_path: str = str(_PROJECT / "configs" / "base_transform.json")
    calibration_path: str = str(_PROJECT / "configs" / "hardware_calibration.json")
    dry_run_log: str | None = None


def make_real_hardware(cfg: HardwareConfig | None = None, dry_run: bool = True):
    """Build the real ``HardwareBackend`` stack. Returns ``(backend, motors, encoders, cal, monitor)``.

    ``dry_run=True`` (default) keeps motor writes intercepted+logged until the operator
    explicitly goes LIVE — safe by construction.
    """
    cfg = cfg or HardwareConfig()

    can_bus = CanBus(cfg.can_channel, cfg.can_interface, cfg.can_bitrate)
    serial_bus = SerialBus(cfg.serial_port, cfg.serial_baud)

    motors, encoders = {}, {}
    for j in HOVERBOARD_JOINTS:
        motors[j] = HoverboardDriver(j, can_bus, cfg.vesc_ids[j], cfg.belt_ratios.get(j, 1.0))
        encoders[j] = VescCanEncoder(j, can_bus, cfg.vesc_ids[j], cfg.belt_ratios.get(j, 1.0))
    for j in SERVO_JOINTS:
        motors[j] = SerialBusServoDriver(j, serial_bus, cfg.servo_ids[j],
                                        cfg.servo_ticks_per_rev, cfg.servo_center_tick)
        encoders[j] = SerialServoEncoder(j, serial_bus, cfg.servo_ids[j],
                                        cfg.servo_ticks_per_rev, cfg.servo_center_tick)

    # Calibration (persistent) — fall back to identity if the file is missing (unhomed;
    # the safety preflight will then refuse to go live until a real calibration is loaded).
    if Path(cfg.calibration_path).is_file():
        cal = RobotCalibration.load(cfg.calibration_path)
    else:
        cal = RobotCalibration.identity()

    # URDF FK with the validated base->env transform (falls back to base frame if missing).
    if Path(cfg.base_transform_path).is_file():
        kin = UrdfKinematicsProvider.from_config(cfg.urdf_path, cfg.base_transform_path)
    else:
        kin = UrdfKinematicsProvider(cfg.urdf_path)

    # Safety limits: hard = URDF, soft = calibrated soft limits.
    lo, hi = cal.soft_limits()
    limits = HardwareLimits(soft_lower=lo, soft_upper=hi,
                            hard_lower=C.JOINT_LOWER.copy(), hard_upper=C.JOINT_UPPER.copy())
    monitor = HardwareSafetyMonitor(limits)

    mbank = MotorBank(motors, dry_run=dry_run, dry_run_log=cfg.dry_run_log)
    ebank = EncoderBank(encoders, cal)
    backend = HardwareBackend(mbank, ebank, cal, kin, monitor)
    return backend, mbank, ebank, cal, monitor
