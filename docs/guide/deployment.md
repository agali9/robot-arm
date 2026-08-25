# Deployment

Run the trained reach policy outside Isaac Lab. Same code path for sim and (eventually) hardware.

Status: validated in simulation. Not yet commanding the physical arm.

## Idea

The policy is a pure function: 28-dim observation → 6-dim raw action. Everything robot-specific (FK, motor drivers, ROS topics) sits behind an `ActuatorInterface` so the policy never changes when you swap backends.

```
checkpoint  →  export (TorchScript/ONNX)  →  InferenceApp  →  backend (sim / ROS / hardware)
```

## Policy contract

Defined in `deploy/robotarm_deploy/contract.py` (guarded by tests).

**Observation (28 values):**

- joint position − home (6)
- joint velocity (6)
- end-effector position, base frame (3)
- target position, base frame (3)
- target − ee (3)
- distance scalar (1)
- last clipped action (6)

**Action pipeline (outside the network):**

```
raw → clip ±1 → × 1.5 rad → add home (0) → clamp to joint limits → send
```

Joint order: `j1` … `j6`. End link: `j6_link`.

## Export

```bat
isaaclab.bat -p deploy\scripts\export_policy.py ^
    --checkpoint logs\reach\2026-07-19_16-33-14_scale15_random\model_400.pt ^
    --out deploy\exported --device cpu
```

Writes `policy.pt` (TorchScript), `policy.onnx`, and `metadata.json`. TorchScript matches the checkpoint exactly (max |Δaction| = 0 over random obs).

## Run in sim

```bat
:: Verify obs/action parity with training env
isaaclab.bat -p deploy\scripts\verify_in_sim.py --policy deploy\exported\policy.pt

:: Full control loop
isaaclab.bat -p deploy\scripts\run_inference.py --backend sim ^
    --policy deploy\exported\policy.pt --steps 600 --target 0.40 0.20 0.30
```



## Control loop

Each cycle at 100 Hz (`control_dt=0.01`):

1. Read joint state from backend
2. Build observation
3. Policy inference
4. Safety layer (clip, scale, slew limit, watchdog, e-stop)
5. Send joint targets

Safety violations latch a hold on the last good command until `reset()`.

## Safety layer


| Check                           | On failure      |
| ------------------------------- | --------------- |
| NaN/inf in obs or action        | Hold            |
| Joint limit violation           | Clamp or hold   |
| Slew rate exceeded              | Limit step size |
| Inference stall / comms timeout | Hold            |
| E-stop                          | Latch off       |


Unit tests in `deploy/tests/` — no Isaac required.

## ROS 2 (Jazzy)

`RobotInterfaceNode` bridges topics:

- In: `/joint_states`
- Out: `/joint_position_targets`

Run from a sourced ROS environment (not Isaac's Python):

```bash
source /opt/ros/jazzy/setup.bash
python deploy/scripts/run_inference.py --backend ros2 --policy deploy/exported/policy.pt
```

EE position is computed via FK on the backend, not read from a sensor topic.

## Package layout

```
deploy/
├── robotarm_deploy/
│   ├── contract.py, observation.py, action.py, policy.py, safety.py
│   ├── kinematics.py          FK providers
│   ├── inference_app.py       main loop
│   ├── ros2/                  topic bridge
│   └── hardware/              motors, encoders, calibration, bringup
├── scripts/                   export, verify, run, bringup, calibrate
├── tests/
└── exported/                  generated policy files
```



## Moving to hardware

Only these pieces are robot-specific:

1. FK from URDF (`UrdfKinematicsProvider` — implemented, validated < 0.1 mm vs sim)
2. Motor driver node (VESC/CAN for J1/J2, serial bus for J3–J6)
3. Safety timeouts tuned to real control rate
4. Calibration saved to `configs/hardware_calibration.json`

See [hardware.md](hardware.md) for the bring-up sequence.