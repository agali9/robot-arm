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

