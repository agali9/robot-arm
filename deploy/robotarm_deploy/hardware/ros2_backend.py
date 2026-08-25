"""ROS 2 backend — an ActuatorInterface that talks to the robot over ROS 2 topics.

This is the seam between the (framework-neutral) inference loop and a real robot running
ROS 2 (target: **Jazzy**). It owns a ``RobotInterfaceNode`` (see ``ros2/interface_node``)
and adapts it to the ``ActuatorInterface`` API:

    read_state()          <- latest /joint_states (+ EE via a KinematicsProvider)
    send_joint_targets()  -> /joint_position_targets

``rclpy`` is imported lazily inside ``connect()`` so the deployment package imports fine in
Isaac's Python (no rclpy). Run this backend only from a sourced ROS 2 environment.

The EE position is not on the wire (it is not a raw sensor); this backend computes it from
the measured joint angles via an injected ``KinematicsProvider`` (``UrdfKinematicsProvider``
for hardware). That FK provider + this backend's topic names are the ONLY robot-specific
pieces to finalize for a given robot.
"""

from __future__ import annotations

import time

import numpy as np

from .. import contract as C
from ..observation import RobotState
from .base import ActuatorInterface, CommunicationError


class Ros2Backend(ActuatorInterface):
    """ActuatorInterface over ROS 2. Requires a sourced ROS 2 (Jazzy) environment."""

    name = "ros2"

    def __init__(self, kinematics, node_name: str = "robotarm_inference",
                 state_topic: str = "/joint_states",
                 command_topic: str = "/joint_position_targets",
                 spin_timeout_s: float = 0.02) -> None:
        self._kin = kinematics                    # KinematicsProvider (FK for EE)
        self._node_name = node_name
        self._state_topic = state_topic
        self._command_topic = command_topic
        self._spin_timeout_s = spin_timeout_s
        self._rclpy = None
        self._node = None

    def connect(self) -> None:
        import rclpy  # lazy: only available in a sourced ROS 2 env
        from ..ros2.interface_node import RobotInterfaceNode

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = RobotInterfaceNode(
            node_name=self._node_name, state_topic=self._state_topic,
            command_topic=self._command_topic, joint_names=list(C.JOINT_NAMES))

    def disconnect(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
            self._rclpy = None

    def _spin_once(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=self._spin_timeout_s)

    def read_state(self) -> RobotState:
        if self._node is None:
            raise CommunicationError("Ros2Backend not connected")
        self._spin_once()
        snap = self._node.latest_joint_state()   # (pos, vel, stamp) or None
        if snap is None:
            raise CommunicationError("no /joint_states received yet")
        jp, jv, stamp = snap
        ee = np.asarray(self._kin.ee_position(jp), dtype=np.float32).reshape(3)
        return RobotState(joint_pos=jp.astype(np.float32), joint_vel=jv.astype(np.float32),
                          ee_position=ee, stamp=stamp)

    def send_joint_targets(self, targets: np.ndarray) -> None:
        if self._node is None:
            raise CommunicationError("Ros2Backend not connected")
        self._node.publish_joint_targets(np.asarray(targets, dtype=np.float32))
        self._spin_once()
