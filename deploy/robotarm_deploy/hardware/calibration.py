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
        return cls(joints=joints, created=datetime.now().isoformat(timespec="seconds"),
                   notes="identity (uncalibrated)")


# --- Calibration tools (pure functions over sampled data) ----------------------------

def zero_offset_from_home(mech_at_home: float) -> float:
    """Zero-offset such that the current mechanism angle maps to policy 0 (home pose)."""
    return float(mech_at_home)


def verify_direction(mech_samples: np.ndarray, commanded_dir: np.ndarray) -> int:
    """Infer joint direction (+1/-1) from a monotonic jog.

    ``mech_samples`` are mechanism angles recorded while commanding motion in a known
    (+) policy direction (``commanded_dir`` > 0). Returns +1 if the mechanism angle rose
    with the commanded (+) direction, else -1.
    """
    dm = np.polyfit(np.arange(len(mech_samples)), mech_samples, 1)[0]  # slope
    cd = float(np.mean(commanded_dir))
    return 1 if (dm * cd) >= 0 else -1


def check_encoder_sanity(mech_samples: np.ndarray, *, max_jump: float = 0.5,
                         min_range: float = 1e-3) -> tuple[bool, str]:
    """Sanity-check a stream of encoder angles: finite, no wild jumps, some motion range.

    Returns ``(ok, message)``. ``max_jump`` = max plausible per-sample step (rad);
    ``min_range`` guards against a dead/stuck encoder during a jog.
    """
    s = np.asarray(mech_samples, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        return False, "non-finite encoder samples"
    if s.size >= 2:
        if float(np.abs(np.diff(s)).max()) > max_jump:
            return False, f"encoder jump > {max_jump} rad (wiring/parity glitch?)"
        if float(s.max() - s.min()) < min_range:
            return False, "encoder did not move during jog (stuck/disconnected?)"
    return True, "ok"


def verify_soft_within_hard(cal: JointCalibration, hard_lower: float,
                            hard_upper: float, margin: float = 0.0) -> tuple[bool, str]:
    """Ensure a joint's soft limits sit inside the URDF hard limits (with optional margin)."""
    if cal.soft_lower < hard_lower + margin:
        return False, f"{cal.name}: soft_lower {cal.soft_lower} < hard {hard_lower}"
    if cal.soft_upper > hard_upper - margin:
        return False, f"{cal.name}: soft_upper {cal.soft_upper} > hard {hard_upper}"
    if cal.soft_lower >= cal.soft_upper:
        return False, f"{cal.name}: soft_lower >= soft_upper"
    return True, "ok"
