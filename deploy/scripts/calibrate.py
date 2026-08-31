"""Calibration tool — create / verify persistent joint calibration.

Calibration reconciles raw encoder angles with the policy joint frame (zero-offset,
direction, soft limits). On real hardware, the bring-up script feeds live jog samples to
the ``calibration`` tools; this CLI manages the persistent file and validates it.

    python deploy\\scripts\\calibrate.py --init  --out configs\\hardware_calibration.json
    python deploy\\scripts\\calibrate.py --verify configs\\hardware_calibration.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.hardware import calibration as calib  # noqa: E402


def _verify(cal: calib.RobotCalibration) -> bool:
    ok = True
    print(f"{'joint':<10}{'dir':>4}{'zero':>9}{'soft_lo':>10}{'soft_hi':>10}{'homed':>7}  check")
    for i, n in enumerate(C.JOINT_NAMES):
        j = cal.joints[n]
        good, msg = calib.verify_soft_within_hard(j, float(C.JOINT_LOWER[i]),
                                                  float(C.JOINT_UPPER[i]))
        ok = ok and good
        print(f"{n:<10}{j.direction:>4}{j.zero_offset:>9.3f}{j.soft_lower:>10.3f}"
              f"{j.soft_upper:>10.3f}{str(j.homed):>7}  {'ok' if good else msg}")
    print(f"\n  all soft-within-hard: {ok}   all homed: {cal.all_homed()}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage joint calibration.")
    ap.add_argument("--init", action="store_true", help="create an identity calibration")
    ap.add_argument("--verify", type=str, default=None, help="verify a calibration JSON")
    ap.add_argument("--out", type=str, default="configs/hardware_calibration.json")
    ap.add_argument("--demo-tools", action="store_true", help="show tools on synthetic data")
    args = ap.parse_args()

    if args.demo_tools:
        # Illustrate the pure calibration tools on a synthetic monotonic jog.
        mech = np.linspace(0.10, 0.35, 25) + np.random.default_rng(0).normal(0, 1e-4, 25)
        print("direction (expect +1):", calib.verify_direction(mech, np.ones(25)))
        print("encoder sanity:", calib.check_encoder_sanity(mech))
        print("stuck-encoder sanity:", calib.check_encoder_sanity(np.full(25, 0.2)))
        return 0

    if args.init:
        cal = calib.RobotCalibration.identity()
        cal.notes = "identity template — replace zero_offset/direction/soft limits per joint"
        cal.save(args.out)
        print(f"[calibrate] wrote identity calibration to {args.out}")
        _verify(cal)
        return 0

    if args.verify:
        cal = calib.RobotCalibration.load(args.verify)
        print(f"[calibrate] loaded {args.verify}")
        return 0 if _verify(cal) else 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
