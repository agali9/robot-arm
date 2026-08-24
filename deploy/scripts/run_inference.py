"""Run the deployment InferenceApp against a chosen backend (inference only).

Exercises the FULL deployment stack — exported policy + ObservationBuilder +
ActionProcessor + SafetyLayer + ActuatorInterface — through one backend-agnostic loop.

Backends:
  * ``--backend sim``  : drives the Isaac Sim reach articulation (needs Isaac Sim; this
    script launches it). Proves the whole stack runs end-to-end in sim.
  * ``--backend ros2`` : drives a real/simulated robot over ROS 2 (run from a sourced
    ROS 2 Jazzy env with rclpy; a UrdfKinematicsProvider must be supplied for EE FK).

    # sim:
    isaaclab.bat -p deploy\\scripts\\run_inference.py --backend sim ^
        --policy deploy\\exported\\policy.pt --steps 600 --target 0.40 0.20 0.30

No training. Reports inference latency and loop timing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Run deployment inference (sim or ros2).")
parser.add_argument("--backend", choices=["sim", "ros2"], default="sim")
parser.add_argument("--policy", type=str, default="deploy/exported/policy.pt")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--target", type=float, nargs=3, default=[0.40, 0.20, 0.30])
parser.add_argument("--device", type=str, default="cpu")

_PROJECT = Path(__file__).resolve().parents[2]


def _run_sim(args) -> int:
    # Launch Isaac Sim FIRST (SimBackend drives the articulation).
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    app_args, _ = ap.parse_known_args(["--headless"])
    app_args.device = args.device
    launcher = AppLauncher(app_args)
    app = launcher.app

    sys.path.insert(0, str(_PROJECT / "isaaclab"))
    sys.path.insert(0, str(_PROJECT / "deploy"))
    import numpy as np
    from robotarm_deploy.hardware.sim_backend import SimBackend
    from robotarm_deploy.inference_app import InferenceApp
    from robotarm_deploy.safety import SafetyConfig, SafetyLayer

    code = 1
    try:
        backend = SimBackend(device=args.device)
        backend.connect()
        safety = SafetyLayer(SafetyConfig(watchdog_timeout_s=1.0, command_timeout_s=1.0,
                                          state_max_age_s=1e9,   # relaxed for stepped sim
                                          vel_limit_scale=8.0))  # let the demo complete the move
        app_inf = InferenceApp(args.policy, backend, safety=safety,
                               control_dt=backend.control_dt)
        s0 = backend.read_state()
        print(f"[infer] home EE={np.round(s0.ee_position,3)}  target={np.round(args.target,3)}  "
              f"init dist={np.linalg.norm(np.array(args.target)-s0.ee_position)*100:.1f}cm", flush=True)
        app_inf.set_target(np.array(args.target, dtype=np.float32))
        summary = app_inf.run(args.steps, verbose_every=100)
        backend.disconnect()
        print("\n==================== INFERENCE SUMMARY ====================")
        print(f"  backend            : sim")
        print(f"  inference latency  : {summary['inference_latency']}")
        print(f"  loop timing        : {summary['loop_timing']}")
        print(f"  final safety state : {summary['final_safety']}")
        print("==========================================================")
        code = 0
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close()
    return code


def _run_ros2(args) -> int:
    # No Isaac Sim; requires a sourced ROS 2 env with rclpy.
    sys.path.insert(0, str(_PROJECT / "deploy"))
    import numpy as np
    from robotarm_deploy.hardware.ros2_backend import Ros2Backend
    from robotarm_deploy.inference_app import InferenceApp
    from robotarm_deploy.kinematics import UrdfKinematicsProvider

    kin = UrdfKinematicsProvider(str(_PROJECT / "urdf" / "robot_arm.urdf"))
    backend = Ros2Backend(kinematics=kin)
    backend.connect()
    try:
        app_inf = InferenceApp(args.policy, backend)
        app_inf.set_target(np.array(args.target, dtype=np.float32))
        summary = app_inf.run(args.steps, verbose_every=100)
        print("\nINFERENCE SUMMARY (ros2):", summary)
    finally:
        backend.disconnect()
    return 0


if __name__ == "__main__":
    a, _ = parser.parse_known_args()
    sys.exit(_run_sim(a) if a.backend == "sim" else _run_ros2(a))
