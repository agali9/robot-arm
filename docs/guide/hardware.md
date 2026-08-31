# Hardware Bring-Up

Getting the reach policy onto the real arm. Same inference code as sim; you swap in real motor and encoder drivers.

**Not live yet.** Stack is built and dry-run tested; physical motion still needs config, calibration, and the gated checklist below.

## Architecture

```
Policy  →  SafetyLayer  →  HardwareBackend
                              ├── EncoderBank (raw → joint rad)
                              ├── Calibration (mech ↔ policy frame)
                              ├── MotorBank (joint targets → motor commands)
                              └── HardwareSafetyMonitor
```

The policy only sees normalized joint positions in `j1..j6` order with home = 0. Motors only get mechanism-frame position targets. Nothing hardware-specific leaks into the policy path.

## Motors


| Joints | Driver                 | Bus                        | Notes                                |
| ------ | ---------------------- | -------------------------- | ------------------------------------ |
| J1, J2 | `HoverboardDriver`     | VESC over CAN              | Position mode, belt ratio in scaling |
| J3–J6  | `SerialBusServoDriver` | Serial (Feetech STS-style) | Goal position, servo closes loop     |


`MotorBank` can run in **dry-run** mode: full pipeline executes, commands log to JSONL instead of transmitting.

## Encoders

`EncoderBank` converts raw readings to policy-frame joint angles and filtered velocities.

- Absolute encoders: one read gives angle
- Incremental: need `set_home()` at a known pose first



## Calibration

Per joint, stored in `configs/hardware_calibration.json`:


| Field              | Purpose                              |
| ------------------ | ------------------------------------ |
| `zero_offset`      | Mech angle at home pose              |
| `direction`        | ±1 if encoder counts opposite policy |
| `soft_lower/upper` | Limits just inside mechanical stops  |
| `homed`            | Whether zero is established          |


```bat
python deploy\scripts\calibrate.py --init --out configs\hardware_calibration.json
python deploy\scripts\calibrate.py --verify configs\hardware_calibration.json
```

Procedure per joint: move to home → set zero → jog + direction check → find soft limits → save.

## Hardware safety

Runs alongside the policy safety layer. Motors only enable when:

- Calibration loaded, all joints homed
- Encoders alive and plausible
- Soft limits inside URDF hard limits
- No latched fault
- **Both** enable switches on (physical dead-man + software arm)

Faults latch until operator `reset()`. `estop()` drops everything immediately.

## Bring-up states

Run `deploy/scripts/bringup.py`. Each gate blocks the next.


| Step | State             | What happens                              |
| ---- | ----------------- | ----------------------------------------- |
