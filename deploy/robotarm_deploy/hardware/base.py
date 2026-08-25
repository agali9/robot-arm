"""Hardware abstraction — the single interface every actuator backend implements.

The policy and inference loop talk ONLY to this interface, so swapping simulation for
ROS 2 (or a future CAN / VESC driver) changes one line at startup and nothing else. The
contract is intentionally minimal and command-mode-agnostic to joint *position targets*
(the reach policy's action space):

    connect()                      # bring the backend up
    read_state()  -> RobotState    # synchronized joint pos/vel + EE + timestamp
    send_joint_targets(targets)    # command absolute joint position targets (rad)
    disconnect()

Backends must NOT scale/clip/offset — they receive final, already-safe joint targets
(rad, JOINT_NAMES order) from the inference loop's ActionProcessor/SafetyLayer. A backend
that cannot measure EE position (real hardware) computes it via a
``kinematics.KinematicsProvider`` and fills ``RobotState.ee_position``.
"""

from __future__ import annotations

import abc

import numpy as np

from ..observation import RobotState


class ActuatorInterface(abc.ABC):
    """Abstract base for all actuator backends (sim, ROS 2, CAN, VESC, ...)."""

    #: Human-readable backend name (for logging).
    name: str = "abstract"

    @abc.abstractmethod
    def connect(self) -> None:
        """Bring the backend up (open sim handle / ROS node / bus). Idempotent."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Tear the backend down cleanly. Safe to call if never connected."""

    @abc.abstractmethod
    def read_state(self) -> RobotState:
        """Return the latest synchronized robot state (joint pos/vel, EE, timestamp).

        Raises ``CommunicationError`` if no fresh state is available.
        """

    @abc.abstractmethod
    def send_joint_targets(self, targets: np.ndarray) -> None:
        """Command absolute joint position targets (rad, JOINT_NAMES order).

        Targets are already clipped, scaled, and joint-limited by the inference loop.
        """

    # --- convenience ------------------------------------------------------------------
    def hold(self, state: RobotState | None = None) -> None:
        """Command a safe hold at the current measured position (best-effort)."""
        s = state or self.read_state()
        self.send_joint_targets(np.asarray(s.joint_pos, dtype=np.float32))

    def __enter__(self) -> "ActuatorInterface":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()


class CommunicationError(RuntimeError):
    """Raised by a backend when robot state/commands cannot be exchanged."""
