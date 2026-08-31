"""Simulated hardware stack — a fully in-memory robot for bring-up + dry-run + tests.

Wires ``SimulatedMotor`` <-> ``SimulatedEncoder`` (perfect follower: the encoder mirrors the
last commanded mechanism angle) into the same objects a real robot uses — MotorBank,
EncoderBank, calibration, safety monitor, HardwareBackend. This lets the entire bring-up
sequence and the policy dry-run run end-to-end with NO hardware and NO ROS, so the plumbing
is verified before a single motor is energized.

Not a physics model — an idealized actuator. Its purpose is to validate the *stack*
(encoders -> obs -> policy -> action -> safety -> motor bridge), not dynamics.
"""

from __future__ import annotations

from pathlib import Path

from .. import contract as C
from ..kinematics import ConstantKinematics
from .calibration import RobotCalibration
from .encoders import EncoderBank, SimulatedEncoder
from .hardware_backend import HardwareBackend
from .hardware_safety import HardwareLimits, HardwareSafetyMonitor
from .motors import MotorBank, SimulatedMotor


def make_simulated_hardware(dry_run: bool = True, dry_run_log: str | Path | None = None,
                            calibration: RobotCalibration | None = None):
    """Build a complete simulated hardware stack.

    Returns ``(backend, motors, encoders, calibration, monitor)`` — the exact tuple the
    :class:`bringup.BringUpSequence` expects. ``dry_run=True`` makes the MotorBank log
    instead of "transmit" (here transmit = update the follower encoder), matching how a
    real dry-run intercepts commands.
    """
    cal = calibration or RobotCalibration.identity()

    encoders_map = {n: SimulatedEncoder(n, initial=0.0) for n in C.JOINT_NAMES}
    # couple each motor to its encoder so a (transmitted) command moves the measured angle
    motors_map = {n: SimulatedMotor(n, encoder=encoders_map[n]) for n in C.JOINT_NAMES}

    motors = MotorBank(motors_map, dry_run=dry_run, dry_run_log=dry_run_log)
    encoders = EncoderBank(encoders_map, cal)
    monitor = HardwareSafetyMonitor(HardwareLimits())
    kin = ConstantKinematics()
    backend = HardwareBackend(motors, encoders, cal, kin, monitor)
    return backend, motors, encoders, cal, monitor
