"""Forward kinematics providers — supply the end-effector position for the observation.

The 28-dim observation needs the EE position (base frame), which is *derived* from joint
angles, not a raw sensor. Where that comes from depends on the backend:

  * **Simulation**: the sim already tracks the EE body pose -> ``SimKinematicsProvider``
    just forwards whatever the sim backend read.
  * **Hardware**: compute FK from joint angles -> ``UrdfKinematicsProvider`` (to be
    implemented alongside the hardware driver, using ``urdf/robot_arm.urdf``).

This is the ONLY task-space quantity a hardware integrator must supply beyond raw joint
feedback, so it is isolated here behind a tiny protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from . import contract as C


class KinematicsProvider(Protocol):
    """Maps joint angles (rad, JOINT_NAMES order) -> EE position (m, base frame)."""

    def ee_position(self, joint_pos: np.ndarray) -> np.ndarray: ...


class SimKinematicsProvider:
    """Passthrough provider: the sim backend already measured the EE position.

    The inference loop reads EE straight from ``RobotState.ee_position`` in sim, so this
    provider is only used if some path needs the protocol shape; it returns the cached
    value set by the backend.
    """

    def __init__(self) -> None:
        self._cached = np.zeros(3, dtype=np.float32)

    def update(self, ee_position: np.ndarray) -> None:
        self._cached = np.asarray(ee_position, dtype=np.float32).reshape(3)

    def ee_position(self, joint_pos: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._cached.copy()


class ConstantKinematics:
    """Returns a fixed EE position regardless of joint angles.

    A placeholder for **plumbing** verification only (hardware dry-run / tests): it lets the
    full stack run (encoders -> obs -> policy -> action -> safety -> motor) without a real FK
    backend. Reaching behaviour is NOT meaningful with this provider — use
    ``UrdfKinematicsProvider`` (or the sim) for correct end-effector tracking.
    """

    def __init__(self, ee: tuple[float, float, float] = (0.3, 0.0, 0.3)) -> None:
        self._ee = np.asarray(ee, dtype=np.float32)

    def ee_position(self, joint_pos: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._ee.copy()


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis XYZ rpy -> 3x3 rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle_to_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation about a (unit) axis by ``theta`` radians."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s, C = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> quaternion (w, x, y, z) — Isaac's ordering."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


class UrdfKinematicsProvider:
    """Forward kinematics from ``urdf/robot_arm.urdf`` — pure numpy, no external deps.

    Parses the serial chain ``base_link -> ... -> end_link`` (default ``j6_link`` =
    ``contract.EE_BODY_NAME``) with the stdlib XML parser and composes, per joint,
    ``Trans(origin.xyz) @ Rot(origin.rpy) @ Rot(joint_axis, theta)``. Returns the EE
    position (and pose) in the **base-link frame**, which equals the env frame the policy
    was trained on (the robot base sits at the env origin). Joint order follows
    ``contract.JOINT_NAMES``.

    Reusable by both deployment (EE for the observation) and calibration (expected EE at a
    pose). Validate once against Isaac Sim (``deploy/scripts/verify_kinematics.py``): the two
    must agree to < 1 mm before enabling hardware.
    """

    def __init__(self, urdf_path: str, end_link: str = C.EE_BODY_NAME,
                 base_link: str = "base_link",
                 base_position=(0.0, 0.0, 0.0),
                 base_quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> None:
        self.urdf_path = str(urdf_path)
        self.end_link = end_link
        self.base_link = base_link
        # base_link -> policy(env) frame, fitted once vs Isaac (deploy/scripts/verify_kinematics.py).
        self._base_t = np.asarray(base_position, dtype=np.float64).reshape(3)
        self._base_R = _quat_to_matrix(np.asarray(base_quat_wxyz, dtype=np.float64))
        self._chain = self._build_chain()   # ordered [(joint_name, xyz, R_origin, axis)]
        chain_joints = [c[0] for c in self._chain]
        # The actuated joints in the chain must match the policy joint set/order.
        if [j for j in chain_joints if j in C.JOINT_NAMES] != list(C.JOINT_NAMES):
            raise ValueError(f"URDF chain joints {chain_joints} != contract {C.JOINT_NAMES}")

    @classmethod
