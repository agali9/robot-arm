# Isaac Lab

Reusable robot package for training in Isaac Lab 2.3.x. Defines the arm once (geometry, actuators, observations, rewards) so tasks import it instead of copying config.

## Layout

```
isaaclab/
├── robot/           articulation, actuators, obs/actions/events/rewards
├── utils/           joint constants, USD path resolution
├── configs/         base env scaffold
├── tasks/           reach task registration
├── envs/reach/      reach MDP (commands, terminations)
└── scripts/         train, eval, plot
```

Import model: add `isaaclab/` to `sys.path` and import `robot`, `utils`, `configs` — not as a nested `isaaclab` package (that would shadow NVIDIA's).

## Robot model

- 6 revolute joints `j1_joint` … `j6_joint`, tool frame `tool0`
- J1/J2: hoverboard motors → `DCMotorCfg`
- J3–J6: servos → `ImplicitActuatorCfg`
- Limits and efforts from URDF (`utils/constants.py`)
- USD: `isaac/robot_arm.usda` (override with `ROBOTARM_USD` env var)



## Reach task

Registered as `RobotArm-Reach-v0`. Random target pose in a workspace box, joint-position actions, distance/success/smoothness rewards. Frozen baseline checkpoint: `logs/reach/.../model_400.pt` (~85% success in sim).

## Training workflow

```bat
:: Quick sanity check (~1-2 min)
isaaclab.bat -p scripts\train_reach.py --num_envs 64 --max_iterations 15 ^
    --save_interval 5 --run_name validation --headless

:: Full baseline
isaaclab.bat -p scripts\train_reach.py --num_envs 4096 --max_iterations 1000 ^
    --run_name baseline --headless

:: Plot curves (no sim needed)
isaaclab.bat -p scripts\plot_reach_curves.py --run_dir logs\reach\<run>

:: Eval a checkpoint
isaaclab.bat -p scripts\eval_reach.py --checkpoint logs\reach\<run> --episodes 100 --headless
```

Monitor with `tensorboard --logdir logs/reach`.

## Run folders

Each run is self-contained under `logs/reach/<timestamp>_<name>/`:

```
model_*.pt              checkpoints
events.out.tfevents.*   TensorBoard
params/env.yaml         env config snapshot
params/agent.yaml       PPO config snapshot
params/metadata.json    seed, versions, hyperparams
curves.csv / curves.png exported metrics
eval.json               rollout eval results
```

Compare runs using `metadata.json`, `curves.csv`, and `eval.json`.

## Smoke test

```bat
isaaclab.bat -p isaaclab/smoke_test.py
```

Builds the env, resets, steps once, checks observations are finite. No training.

## Hyperparameters

PPO config: `isaaclab/configs/agents/rsl_rl_ppo_cfg.py`. Based on NVIDIA's Franka reach task with annotated deviations. Change defaults there; override run size on the CLI.

## Key metrics (TensorBoard)


| What           | Tag                              |
| -------------- | -------------------------------- |
| Mean reward    | `Train/mean_reward`              |
| Success rate   | `Metrics/success_rate`           |
| Distance error | `Metrics/ee_pose/position_error` |
| Policy loss    | `Loss/surrogate`                 |
| Learning rate  | `Loss/learning_rate`             |


`plot_reach_curves.py` exports these to CSV automatically.