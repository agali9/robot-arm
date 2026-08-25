"""ROS 2 interface node — the RL-agnostic bridge to robot topics (ROS 2 Jazzy).

Responsibilities (and nothing else — it never imports the policy):
  * subscribe to ``/joint_states`` (``sensor_msgs/JointState``), reorder joints to the
    policy's ``contract.JOINT_NAMES`` by *name* (robust to arbitrary wire order),
  * cache the latest state with its header timestamp (for staleness / sync),
  * publish desired joint position targets on the command topic.

Command message type is configurable. Default is ``sensor_msgs/JointState`` (self-
describing: carries joint names + target positions), which bridges cleanly to a custom
motor-controller node (CAN / VESC). For ``ros2_control`` users, switch ``command_msg`` to
``float64multiarray`` (JointGroupPositionController) — one flag, no other change.

This module imports ``rclpy`` at load time, so import it only from a sourced ROS 2
environment (the ``Ros2Backend`` does this lazily).
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .. import contract as C


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class RobotInterfaceNode(Node):
    """Bidirectional joint-state / joint-target bridge. Independent of the RL algorithm."""

    def __init__(self, node_name: str = "robotarm_inference",
                 state_topic: str = "/joint_states",
                 command_topic: str = "/joint_position_targets",
                 joint_names: list[str] | None = None,
                 command_msg: str = "jointstate") -> None:
        super().__init__(node_name)
        self._joint_names = list(joint_names or C.JOINT_NAMES)
        self._name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
        self._command_msg = command_msg.lower()

        self._latest_pos: np.ndarray | None = None
        self._latest_vel: np.ndarray | None = None
        self._latest_stamp: float = 0.0

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(JointState, state_topic, self._on_state, qos)
        if self._command_msg == "float64multiarray":
            self._pub = self.create_publisher(Float64MultiArray, command_topic, qos)
        else:
            self._pub = self.create_publisher(JointState, command_topic, qos)

    # --- inbound: /joint_states -------------------------------------------------------
    def _on_state(self, msg: JointState) -> None:
        """Reorder incoming joints to contract order by name; cache with timestamp."""
        pos = np.zeros(len(self._joint_names), dtype=np.float32)
        vel = np.zeros(len(self._joint_names), dtype=np.float32)
        have_vel = len(msg.velocity) == len(msg.name)
        for i, name in enumerate(msg.name):
            j = self._name_to_idx.get(name)
            if j is None:
                continue                      # ignore joints the policy doesn't control
            pos[j] = msg.position[i]
            if have_vel:
                vel[j] = msg.velocity[i]
        self._latest_pos = pos
        self._latest_vel = vel
        self._latest_stamp = _stamp_to_sec(msg.header.stamp)

    def latest_joint_state(self):
        """Return ``(pos, vel, stamp_sec)`` in contract order, or ``None`` if none yet."""
        if self._latest_pos is None:
            return None
        return self._latest_pos.copy(), self._latest_vel.copy(), self._latest_stamp

    # --- outbound: joint position targets ---------------------------------------------
    def publish_joint_targets(self, targets: np.ndarray) -> None:
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if self._command_msg == "float64multiarray":
            msg = Float64MultiArray()
            msg.data = targets.tolist()
        else:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(self._joint_names)
            msg.position = targets.tolist()
        self._pub.publish(msg)
