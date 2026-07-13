"""Asset path resolution for the Isaac Lab robot package.

The ArticulationCfg needs a USD file to spawn. We resolve it in a portable,
overridable way instead of hardcoding one machine's path:

1. ``ROBOTARM_USD`` environment variable, if set (highest priority — lets you point
   at any asset without editing code).
2. The **project** source-of-truth USD: ``RobotArm/isaac/robotArm.usd``.
3. The legacy in-repo ``isaac/robot_arm.usda`` as a last-resort fallback.

The project USD (``robotArm.usd``) is the single default so the robot definition
stays inside the RobotArm project and does not depend on any path outside it.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels up from this file (isaaclab/utils/paths.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Project source-of-truth asset (validated, already imported/tested in Isaac Sim).
_PROJECT_USD = PROJECT_ROOT / "isaac" / "robotArm.usd"
# Legacy in-repo asset, kept only as a fallback if the project USD is absent.
_LEGACY_USD = PROJECT_ROOT / "isaac" / "robot_arm.usda"


def robot_usd_path() -> str:
    """Return the USD path to spawn the robot from (see module docstring)."""
    env = os.environ.get("ROBOTARM_USD")
    if env:
        return str(Path(env).expanduser())
    if _PROJECT_USD.exists():
        return str(_PROJECT_USD)
    return str(_LEGACY_USD)
