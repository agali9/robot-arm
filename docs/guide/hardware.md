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
| 0    | OFF               | Powered down                              |
| 1    | POWERED_ON        | Logic on, backend connects                |
| 2    | MOTORS_INIT       | All 6 motors respond                      |
| 3    | ENCODERS_VERIFIED | Reads finite and plausible                |
| 4    | HOMED             | Zero established                          |
| 5    | CALIBRATED        | JSON loaded and applied                   |
| 6    | SAFETY_VERIFIED   | Monitor preflight green                   |
| 7    | JOG_READY         | Manual jog each joint                     |
| 8    | DRY_RUN           | Policy runs, commands logged only         |
| 9    | LIVE              | `enable_live(confirm=True)` — motors move |


```bat
:: Safe rehearsal (no motor output)
isaaclab.bat -p deploy\scripts\bringup.py --policy deploy\exported\policy.pt ^
    --dry-run-steps 200 --log deploy\logs\dryrun_cmds.jsonl
```



## First real move

Keep a hand on the E-stop. All six conditions must be true before live output:

1. Calibration loaded, all homed
2. Encoders sane
3. Soft limits valid
4. Hardware safety green
5. Dead-man + policy enable both on
6. `enable_live(confirm=True)` called explicitly

**First move tips:**

- Pick a near, central target — small motion
- Start with low `vel_limit_scale` (slew limiter bounds speed)
- Watch for `joint_disagreement`, over-current, over-temp trips


| Step | Action                                          |
| ---- | ----------------------------------------------- |
| 0    | Power off, area clear                           |
| 1    | Logic power, then motor bus                     |
| 2    | Dry-run 200 steps, check logged commands        |
| 3–6  | Encoders, homing, calibration, safety preflight |
| 7    | Motor enable                                    |
| 8    | Manual jog each joint ±small                    |
| 9    | Policy dry-run (no transmit)                    |
| 10   | Arm both enable switches                        |
| 11   | First live move at low slew                     |
| 12   | Ramp slew gradually                             |


Abort: E-stop → diagnose → fix → re-home → restart from step 0.

## Before first power-on

Fill in `HardwareConfig`:

- CAN channel, serial port
- VESC IDs and belt ratios (J1, J2)
- Servo IDs and tick scaling (J3–J6)

Confirm servo register map matches your hardware (driver assumes Feetech STS3215-class). Verify VESC position feedback frame layout.

FK is done: `UrdfKinematicsProvider` validated against sim with base transform in `configs/base_transform.json`.

## Failure recovery


| Symptom                   | Likely cause                         |
| ------------------------- | ------------------------------------ |
| `encoder_timeout`         | Wiring, sensor power                 |
| `joint_disagreement`      | Stall, slipped belt, wrong direction |
| `soft_limit_exceeded`     | Bad calibration                      |
| Wrong commands in dry-run | Fix calibration/FK — do not go live  |


Pattern: E-stop → power safe → fix root cause → re-home → re-run bring-up from the top.