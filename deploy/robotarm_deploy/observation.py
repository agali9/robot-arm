"""Observation assembly — build the 28-dim policy input from robot state + target.

The policy is frame- and hardware-agnostic: it consumes a fixed 28-vector. This module
is the ONE place that assembles that vector, in the exact order the frozen policy was
trained on (see ``contract.OBS_LAYOUT``). It performs no I/O and no kinematics — it takes
already-measured joint state, an already-computed end-effector position, the target, and
the previous action, all in the **robot base frame**.

  obs = [ joint_pos_rel(6), joint_vel_rel(6), ee_pos(3), target(3),
          target-ee(3), distance(1), last_action(6) ]

``joint_pos_rel = joint_pos - HOME_POSE`` and ``joint_vel_rel = joint_vel`` (default vel
is zero), matching ``mdp.joint_pos_rel`` / ``mdp.joint_vel_rel``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import contract as C


@dataclass
class RobotState:
    """A single synchronized snapshot of the robot, in the base frame.

    ``ee_position`` is DERIVED (forward kinematics), not a raw sensor: the simulation
    backend reads it from the sim; a hardware backend computes it via FK from
    ``joint_pos`` (see ``kinematics.KinematicsProvider``).
    """

    joint_pos: np.ndarray                 # (6,) rad, JOINT_NAMES order
    joint_vel: np.ndarray                 # (6,) rad/s
    ee_position: np.ndarray               # (3,) m, base frame
    stamp: float = 0.0                    # seconds (monotonic or ROS time)

    def validate(self) -> None:
        for name, arr, n in (("joint_pos", self.joint_pos, C.NUM_JOINTS),
                             ("joint_vel", self.joint_vel, C.NUM_JOINTS),
                             ("ee_position", self.ee_position, 3)):
            a = np.asarray(arr).reshape(-1)
            if a.shape[0] != n:
                raise ValueError(f"{name}: expected {n} values, got {a.shape[0]}")


class ObservationBuilder:
    """Assemble the flat 28-dim observation vector (numpy float32)."""

    def build(self, state: RobotState, target_position: np.ndarray,
              last_action: np.ndarray) -> np.ndarray:
        state.validate()
        jp = np.asarray(state.joint_pos, dtype=np.float32).reshape(-1)
        jv = np.asarray(state.joint_vel, dtype=np.float32).reshape(-1)
        ee = np.asarray(state.ee_position, dtype=np.float32).reshape(-1)
        tgt = np.asarray(target_position, dtype=np.float32).reshape(-1)
        last = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if tgt.shape[0] != 3:
            raise ValueError(f"target_position must be length 3, got {tgt.shape[0]}")
        if last.shape[0] != C.ACTION_DIM:
            raise ValueError(f"last_action must be length {C.ACTION_DIM}")

        rel = tgt - ee
        dist = np.array([np.linalg.norm(rel)], dtype=np.float32)
        obs = np.concatenate([
            jp - C.HOME_POSE,     # joint_pos_rel
            jv,                   # joint_vel_rel (default vel = 0)
            ee,                   # ee_position (base frame)
            tgt,                  # target_position (base frame)
            rel,                  # target - ee
            dist,                 # distance
            last,                 # last (clipped) action
        ]).astype(np.float32)
        assert obs.shape[0] == C.OBS_DIM, f"assembled obs dim {obs.shape[0]} != {C.OBS_DIM}"
        return obs
