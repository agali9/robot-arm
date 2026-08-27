# RobotArm Deployment (`deploy/`)

Portable inference layer around the **frozen** reach policy. Same policy runs in Isaac Sim
and (later) on hardware over ROS 2 — moving to hardware needs only a driver + FK, nothing
in the policy/obs/action/safety path. Full design: [`docs/Deployment.md`](../docs/Deployment.md).

## Layout
```
deploy/
  robotarm_deploy/           # the portable package (numpy-only core; no sim/ROS/RL imports)
    contract.py              # obs/action spec — single source of truth (28 obs, 6 act, scale 1.5)
    observation.py           # assemble the 28-dim observation
    action.py                # raw action -> clip -> scale -> joint-limit clamp
    policy.py                # load TorchScript/ONNX, run inference, latency
    safety.py                # limits · slew · watchdog · e-stop · NaN · staleness
    kinematics.py            # EE forward-kinematics providers (sim / URDF stub)
    inference_app.py         # backend-agnostic control loop
    ros2/                    # RobotInterfaceNode (pub targets / sub joint states)
    hardware/                # ActuatorInterface ABC + SimBackend + Ros2Backend, PLUS the
                             # real-robot bring-up layer:
                             #   calibration.py, encoders.py, motors.py (hoverboard/servo),
                             #   hardware_safety.py, hardware_backend.py, bringup.py,
                             #   simulated.py (in-memory stack for dry-run/tests)
  scripts/
    export_policy.py         # checkpoint -> policy.pt + policy.onnx (+ validate)
    verify_in_sim.py         # run exported policy in Isaac; check obs/action/success parity
    run_inference.py         # run the InferenceApp (sim or ros2 backend)
    bringup.py               # gated hardware bring-up (power-on -> ... -> live) + dry-run
    calibrate.py             # create / verify persistent joint calibration
  tests/                     # pure-python unit tests (contract, action/obs, safety, hardware)
  exported/                  # policy.pt, policy.onnx, metadata.json (generated)
```

## Quick start
```bat
:: 1) export + validate (exact match to checkpoint)
isaaclab.bat -p deploy\scripts\export_policy.py ^
    --checkpoint logs\reach\2026-07-19_16-33-14_scale15_random\model_400.pt ^
    --out deploy\exported --device cpu

:: 2) unit tests (no Isaac needed, but run via any python with numpy)
isaaclab.bat -p deploy\tests\test_safety.py

:: 3) verify the exported policy in Isaac Sim (obs/action/success parity)
isaaclab.bat -p deploy\scripts\verify_in_sim.py --policy deploy\exported\policy.pt --device cpu

:: 4) run the full stack in sim
isaaclab.bat -p deploy\scripts\run_inference.py --backend sim --policy deploy\exported\policy.pt
```

## Dependencies
- Core package: `numpy`, plus `torch` (TorchScript) or `onnxruntime` (ONNX) at inference.
- ROS 2 layer: a sourced **ROS 2 Jazzy** environment (`rclpy`, `sensor_msgs`, `std_msgs`) —
  not Isaac's bundled Python.
- Hardware FK: an FK backend (`pytorch_kinematics` / `kinpy` / `pinocchio`) for
  `UrdfKinematicsProvider` (the one piece still to implement for hardware).

**Not yet controlling the real robot** — this makes the policy portable so hardware bring-up
is only a driver away.
