# Reach Environment

The first reusable Isaac Lab environment for the 6-DOF arm: a **manager-based reach
task**. Move the end-effector to a randomized target. **No training code** — this is
the environment only, ready for a future PPO runner.

Targets Isaac Lab **2.3.x** (validated against 2.3.2). Lives entirely in the RobotArm
project; reuses the `robot` / `utils` packages; nothing is written into IsaacLab.

## Layout

```
isaaclab/envs/reach/
├─ reach_env_cfg.py   # ReachEnvCfg — composes everything into a ManagerBasedRLEnvCfg
├─ scene.py           # ReachSceneCfg — ground (lowered) + robot + light (reused)
├─ commands.py        # ReachCommandsCfg — UniformPoseCommand target (+ marker)
├─ markers.py         # target sphere VisualizationMarkersCfg
├─ observations.py    # ReachObservationsCfg + shared task-space queries
├─ rewards.py         # ReachRewardsCfg + reward functions
├─ terminations.py    # ReachTerminationsCfg + termination functions
├─ events.py          # ReachEventCfg — reuses robot events (drops buffer target)
├─ validation.py      # random-action stability validation (no learning)
└─ __init__.py        # exports ReachEnvCfg
```

## Architecture & data flow

```
                 ┌───────────────────────── ReachEnvCfg ─────────────────────────┐
 reset ─────────▶│ events (reuse robot): joint noise + optional physics random    │
                 │ commands: UniformPoseCommand resamples target ──▶ marker sphere │
                 │                                                                 │
 policy action ─▶│ actions (reuse robot): JointPositionAction (scale 0.25)  ──────┼─▶ robot drives
                 │                                                                 │      │
                 │ observations: joint pos/vel, EE pos, target, target-rel,        │◀─────┘ sim step
                 │               distance, last action  ──────────────────────────┼─▶ obs vector (28)
                 │ rewards: distance + success − smooth − limits − action − coll.  │
                 │ terminations: time-out | target reached | invalid state         │
                 └─────────────────────────────────────────────────────────────────┘
```

**Reuse (no duplicated robot logic):** the robot articulation (`robot.robot_cfg.ROBOT_CFG`),
actuators (`robot.actuators`), action group (`robot.actions.ActionsCfg`), scene
(`robot.terrain.RobotSceneCfg`), events (`robot.events.EventCfg`), EE/kinematic
helpers (`utils.scene_queries`), and constants (`utils.constants`) all come from the
reusable packages. Only *task-specific* MDP terms live here.

### Observation flow (28-dim policy vector, concatenated)
`joint_pos(6)` + `joint_vel(6)` + `ee_position(3)` + `target_position(3)` +
`target_relative(3)` + `distance(1)` + `last_action(6)`. Proprioception uses built-in
`mdp` terms; EE position reuses `utils.scene_queries`; target terms read the command.
EE and target are both in the env frame; relative/distance are frame-invariant.
**Cameras later:** add a `CameraCfg` sensor to `ReachSceneCfg` and one image `ObsTerm`
to `PolicyCfg` — no other change.

### Action flow
`ActionsCfg.arm_action = JointPositionAction` over all 6 joints, delta-from-home,
`scale=0.25`. Velocity/effort/hybrid are one-line swaps via `robot.actions` factories
(`make_arm_action(ControlMode.EFFORT)`), no architecture change.

### Reward flow (weights are the tuning knobs — placeholders, not trained)
| term | func | weight | purpose |
| --- | --- | --- | --- |
| distance | `reduce_distance` (1−tanh) | +1.0 | dense shaping toward target |
| success | `success_bonus` (<5cm) | +5.0 | sparse reach bonus |
| smooth_action | `mdp.action_rate_l2` | −0.01 | penalize jerky actions |
| joint_limits | `mdp.joint_pos_limits` | −0.1 | keep off the limits |
| action | `mdp.action_l2` | −0.001 | penalize large actions |
| collision | `robot.rewards.collision_penalty` | −1.0 | **0 until a contact sensor exists** |

### Reset flow
On reset: `events.reset_joints` sets joints to home + small noise; the command
resamples a new target (into its workspace box); optional startup physics
randomization. The buffer-based target reset from the robot package is disabled here
(the command owns the target).

### Termination flow
`time_out` (episode length), `target_reached` (EE within 5 cm), `invalid_state`
(non-finite or runaway joint velocity — safety net). Add more by writing a
`func(env,...) -> BoolTensor(num_envs)` and a `DoneTerm` field.

## Target & marker
The target is a `UniformPoseCommand` (the current Isaac Lab CommandTerm), sampled per
reset / every 4 s inside `utils.constants.WORKSPACE` (configurable). It is drawn as a
green **sphere** (`markers.TARGET_MARKER_CFG`). Disable the marker for headless
training with `cfg.commands.ee_pose.debug_vis = False`.

## Validation (no learning)

```bat
C:\Users\aniru\OneDrive\Documents\IsaacLab\isaaclab.bat -p isaaclab\envs\reach\validation.py --device cpu
```
Resets repeatedly, drives random actions for several hundred steps, and verifies:
robot stability, target randomization, and finite rewards/observations (no NaNs/
crashes). Latest run: rewards ∈ [−0.001, 0.005], 0 invalid states, targets randomize,
**PASSED**.

## How future PPO training plugs in

`ReachEnvCfg` is a standard `ManagerBasedRLEnvCfg`, so an RL runner consumes it
unchanged:

1. Register the task (Gymnasium id) pointing `env_cfg_entry_point` at `ReachEnvCfg`,
   and add an agent cfg (e.g. RSL-RL / skrl PPO) — *that* is the only new code.
2. Wrap the env: `ManagerBasedRLEnv(cfg)` → `RslRlVecEnvWrapper` (or skrl wrapper).
3. Run the trainer. The policy consumes the 28-dim observation and outputs the 6-dim
   joint-position action; rewards/terminations already flow from this env.
4. For headless throughput: raise `scene.num_envs`, set `commands.ee_pose.debug_vis =
   False`, and (optionally) run on GPU.

No PPO/networks/inference are included here by design.
