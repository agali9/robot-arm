"""Hardware bring-up runner — drives the gated power-on -> live sequence (dry-run safe).

Runs the :class:`BringUpSequence` step by step, printing the checklist. By default it uses
the fully SIMULATED hardware stack (no hardware, no ROS, no Isaac) so the entire pipeline —
motors, encoders, calibration, safety monitor, and (optionally) the exported policy in
dry-run — can be validated before touching metal.

    # full simulated bring-up + policy dry-run (commands logged, never transmitted):
    isaaclab.bat -p deploy\\scripts\\bringup.py --policy deploy\\exported\\policy.pt \
        --dry-run-steps 200 --log deploy\\logs\\dryrun_cmds.jsonl

    # go LIVE in the SIM stack (still no real motors; demonstrates the LIVE gate):
    python deploy\\scripts\\bringup.py --go-live

This script NEVER commands real hardware: with the simulated stack "transmit" only updates
an in-memory follower encoder. For a real robot, swap ``make_simulated_hardware`` for the
real MotorBank/EncoderBank/KinematicsProvider and re-run the identical sequence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.hardware.bringup import BringUpSequence, BringUpState  # noqa: E402
from robotarm_deploy.hardware.calibration import RobotCalibration  # noqa: E402
from robotarm_deploy.hardware.real import HardwareConfig, make_real_hardware  # noqa: E402
from robotarm_deploy.hardware.simulated import make_simulated_hardware  # noqa: E402


def _print(res) -> bool:
    tag = "OK  " if res.ok else "FAIL"
    print(f"  [{tag}] -> {res.state.value:<18} {res.message}")
    return res.ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Hardware bring-up (simulated stack).")
    ap.add_argument("--policy", type=str, default=None, help="exported policy for dry-run")
    ap.add_argument("--calibration", type=str, default=None, help="calibration JSON to load")
    ap.add_argument("--dry-run-steps", type=int, default=100)
    ap.add_argument("--log", type=str, default=None, help="dry-run command log (JSONL)")
    ap.add_argument("--go-live", action="store_true", help="also demonstrate the LIVE gate")
    ap.add_argument("--real", action="store_true",
                    help="use the REAL hardware stack (requires the robot host + buses)")
    args = ap.parse_args()

    if args.real:
        # Real stack: reads real encoders / drives real motors (dry-run gated). On a machine
        # without the CAN/serial buses the encoder-verify gate will correctly refuse.
        cfg = HardwareConfig(calibration_path=args.calibration or HardwareConfig().calibration_path,
                             dry_run_log=args.log)
        backend, motors, encoders, cal, monitor = make_real_hardware(cfg, dry_run=True)
        print("[bringup] REAL hardware stack (dry-run gated).")
    else:
        cal = RobotCalibration.load(args.calibration) if args.calibration else RobotCalibration.identity()
        backend, motors, encoders, cal, monitor = make_simulated_hardware(
            dry_run=True, dry_run_log=args.log, calibration=cal)
    seq = BringUpSequence(backend, motors, encoders, cal, monitor)

    print("=================== HARDWARE BRING-UP ===================")
    steps = [seq.power_on, seq.init_motors, seq.verify_encoders, seq.home,
             seq.calibrate, seq.verify_safety, seq.enable_jog, seq.enable_dry_run]
    for step in steps:
        if not _print(step()):
            print("  bring-up halted (a gate failed).")
            return 1

    assert seq.state is BringUpState.DRY_RUN
    print("\n--- policy dry-run (commands intercepted + logged, NOT transmitted) ---")
    if args.policy:
        code = _dry_run_policy(backend, monitor, args.policy, args.dry_run_steps)
        if code != 0:
            return code
    else:
        print("  (no --policy given; skipping policy dry-run)")

    if args.go_live:
        print("\n--- LIVE gate (simulated stack) ---")
        _print(seq.enable_live(confirm=False))   # refused: needs explicit confirm
        _print(seq.enable_live(confirm=True))     # armed
        print(f"  output_enabled={monitor.output_enabled}  motors.dry_run={motors.dry_run}")

    _print(seq.shutdown())
    print("========================================================")
    print("\n".join(seq.log[-4:]))
    return 0


def _dry_run_policy(backend, monitor, policy_path: str, steps: int) -> int:
    """Run the InferenceApp against the (dry-run) HardwareBackend and log commands."""
    from robotarm_deploy.inference_app import InferenceApp
    from robotarm_deploy.safety import SafetyConfig, SafetyLayer

    # relaxed timeouts so the stepped, non-realtime dry-run doesn't trip the watchdog
    safety = SafetyLayer(SafetyConfig(watchdog_timeout_s=1e9, command_timeout_s=1e9,
                                      state_max_age_s=1e9))
    app = InferenceApp(policy_path, backend, safety=safety, control_dt=0.01)
    app.set_target(np.array([0.40, 0.20, 0.30], dtype=np.float32))
    n_cmds = 0
    for i in range(steps):
        info = app.step()
        monitor.note_encoder(); monitor.note_motor()          # feed hardware watchdogs
        ok, faults = monitor.check(backend.read_state().joint_pos, backend.last_commanded)
        if not ok:
            print(f"  [FAIL] hardware safety tripped at step {i}: {faults}")
            return 1
        n_cmds += 1
    lat = app.policy.latency.snapshot()
    print(f"  dry-run OK: {n_cmds} policy cycles, commands logged (not transmitted).")
    print(f"  inference latency mean={lat.get('mean_ms', float('nan')):.3f} ms  "
          f"hardware safety=GREEN  motor output=SUPPRESSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
