"""Joint calibration — map raw mechanism angles <-> policy joint angles, persistently.

The policy is trained in a specific joint frame (``contract.JOINT_NAMES`` order, radians,
home = 0, URDF sign convention). Real encoders read arbitrary raw angles with their own
zero and sign. Calibration is the *only* place that reconciles the two, so the policy and
every other layer stay hardware-agnostic:

    joint_angle = direction * (mech_angle - zero_offset)          # measure  (encoder -> policy)
    mech_target = zero_offset + direction * joint_target          # command  (policy -> motor)

Calibration is persisted to JSON so it survives power cycles and is auditable. The
calibration *tools* here are pure functions over sampled data (no I/O) so they are unit
tested; the bring-up scripts feed them live samples.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import contract as C


@dataclass
class JointCalibration:
    """Per-joint calibration. Angles in radians (mechanism side)."""

    name: str
    zero_offset: float = 0.0        # mech angle (rad) that corresponds to policy 0
    direction: int = 1              # +1 or -1: mech sign -> policy (URDF) sign
    soft_lower: float = 0.0         # policy-frame soft limit (rad), inside URDF hard limit
    soft_upper: float = 0.0
    homed: bool = False

    def to_joint(self, mech_angle: float) -> float:
        return float(self.direction) * (mech_angle - self.zero_offset)

    def to_mech(self, joint_angle: float) -> float:
        return self.zero_offset + float(self.direction) * joint_angle


@dataclass
class RobotCalibration:
    """Full-robot calibration + provenance, persisted to JSON."""

    joints: dict[str, JointCalibration] = field(default_factory=dict)
    created: str = ""
    notes: str = ""

    # --- vectorized apply (JOINT_NAMES order) -----------------------------------------
    def to_joint_vec(self, mech: np.ndarray) -> np.ndarray:
        return np.array([self.joints[n].to_joint(float(m))
                         for n, m in zip(C.JOINT_NAMES, mech)], dtype=np.float32)

    def to_mech_vec(self, joint: np.ndarray) -> np.ndarray:
        return np.array([self.joints[n].to_mech(float(j))
                         for n, j in zip(C.JOINT_NAMES, joint)], dtype=np.float32)

    def soft_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.joints[n].soft_lower for n in C.JOINT_NAMES], dtype=np.float32)
        hi = np.array([self.joints[n].soft_upper for n in C.JOINT_NAMES], dtype=np.float32)
        return lo, hi

    def all_homed(self) -> bool:
        return all(self.joints[n].homed for n in C.JOINT_NAMES)

    # --- persistence ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"created": self.created or datetime.now().isoformat(timespec="seconds"),
                "notes": self.notes,
                "joints": {n: asdict(j) for n, j in self.joints.items()}}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RobotCalibration":
        data = json.loads(Path(path).read_text())
        joints = {n: JointCalibration(**j) for n, j in data["joints"].items()}
        return cls(joints=joints, created=data.get("created", ""), notes=data.get("notes", ""))

    @classmethod
    def identity(cls) -> "RobotCalibration":
        """A safe default: zero offset, +1 direction, soft = URDF hard limits, unhomed."""
        joints = {}
        for i, n in enumerate(C.JOINT_NAMES):
            joints[n] = JointCalibration(name=n, zero_offset=0.0, direction=1,
                                         soft_lower=float(C.JOINT_LOWER[i]),
                                         soft_upper=float(C.JOINT_UPPER[i]), homed=False)
