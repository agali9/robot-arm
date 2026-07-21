# RobotArm RL Integration (Gymnasium task + runner scaffold)

The glue that exposes the validated reach environment as a standard **Gymnasium
task** and provides a **runner scaffold** for future RL training. This is *setup and
registration only* — no PPO, no networks, no training loop.

## Task ID

```
RobotArm-Reach-v0
```

Import it instead of hardcoding the string:

```python
import tasks
tasks.REACH_TASK_ID        # "RobotArm-Reach-v0"
```

## Where things live (all inside the RobotArm project)

```
isaaclab/
├─ tasks/
│  ├─ __init__.py               # `import tasks` registers all RobotArm tasks
│  └─ registration/__init__.py  # gym.register(RobotArm-Reach-v0 -> ReachEnvCfg)
├─ runners/
│  ├─ env_loader.py             # load_env_cfg(task_id) + make_reach_env(...)
│  └─ runner_cfg.py             # RunnerCfg scaffold (task, num_envs, device, agent cfg)
├─ configs/
│  └─ agents/reach_ppo_cfg.yaml # placeholder PPO hyperparameters (not trained)
└─ envs/reach/                  # the validated reach environment (unchanged)
scripts/
└─ reach_task_smoke.py          # registers + makes + steps the task (no learning)
```

The registration is **lazy**: `env_cfg_entry_point` is the string
`"envs.reach.reach_env_cfg:ReachEnvCfg"`, so importing `tasks` does not import Isaac
Lab — the cfg is resolved only when a runner builds the env. **No robot constants,
USD paths, workspace, reward, reset, or scene logic are duplicated** — the id points
straight at the existing `ReachEnvCfg`, which reuses the `robot`/`utils` packages.

## How future RL scripts instantiate the env

Preferred (through the runner scaffold):

```python
import tasks                       # registers RobotArm-Reach-v0
from runners import make_reach_env

env = make_reach_env(tasks.REACH_TASK_ID, num_envs=4096, device="cuda:0")
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)   # action: (num_envs, 6)
env.close()
```

Equivalent explicit Isaac Lab flow (what `make_reach_env` does):

```python
import gymnasium as gym
import tasks
from runners import load_env_cfg

cfg = load_env_cfg(tasks.REACH_TASK_ID)   # instantiates ReachEnvCfg from the registry
cfg.scene.num_envs = 4096
env = gym.make(tasks.REACH_TASK_ID, cfg=cfg)
```

`env.unwrapped` is the `ManagerBasedRLEnv`; use it for manager access (e.g.
`env.unwrapped.action_manager.total_action_dim`).

> **Device:** use **GPU** (`cuda:0`) — the reach env is validated stable on GPU
> PhysX. CPU PhysX diverges for this articulation under random/untrained actions, so
> `--device cpu` is only safe once a trained (smooth) policy drives it. RL training
> runs on GPU regardless.

## Smoke run (no learning)

```bat
C:\Users\aniru\OneDrive\Documents\IsaacLab\isaaclab.bat -p scripts\reach_task_smoke.py --device cuda:0
```
Registers the task, builds it via `gym.make`, resets, takes 40 random actions, checks
observations/rewards are finite, and closes. Latest run: obs `(16, 28)`, actions
`(16, 6)`, rewards ∈ [−0.0007, 0.0046], **PASSED**.

## How this connects to RobotCfg and the reach environment

```
RobotCfg (robot.robot_cfg)  ─┐
actuators / actions / utils ─┤  reused, unchanged
                             ▼
envs.reach.ReachEnvCfg  ──▶  tasks/registration (gym id)  ──▶  runners.make_reach_env  ──▶  gym env
                                                                      ▲
                                          configs/agents/*.yaml  ─────┘ (agent hyperparams, future)
```

The task/runner layer is **isolated from the robot package** — it only references a
task id and (optionally) an agent-config entry point. Nothing here changes the robot
definition or the environment.

## How future PPO training plugs in

`RobotArm-Reach-v0` is a standard registered task, so an RL runner drops in without
touching the env:

1. Choose an RL library (RSL-RL / skrl / rl-games). Adapt
   `configs/agents/reach_ppo_cfg.yaml` to its schema (or register a library-specific
   `*_cfg_entry_point` on the task, like the stock Isaac Lab tasks do).
2. In a new `runners/train.py`: launch the app, `import tasks`, build the env with
   `make_reach_env(..., device="cuda:0")`, wrap it in the library's vec-env wrapper
   (e.g. `RslRlVecEnvWrapper`), construct the agent from the agent cfg, and run its
   trainer. Set `commands.ee_pose.debug_vis = False` and raise `num_envs` for throughput.
3. `RunnerCfg` (in `runners/runner_cfg.py`) is the typed place to hold the task id,
   env count, device, seed, agent-config path, and (currently inert) training controls.

No PPO / networks / inference are included here by design.
