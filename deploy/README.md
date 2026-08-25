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
