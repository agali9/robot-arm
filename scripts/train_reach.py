"""PPO training entrypoint for RobotArm-Reach-v0 (RSL-RL).

Production-quality launcher: builds the registered reach env, constructs the RSL-RL
``OnPolicyRunner`` from the task's agent config, and runs training. It reuses the
existing Gymnasium registration, env loader, and RSL-RL runner helpers — no robot
constants, reward, reset, or scene logic is duplicated here.

    # first short validation run (few envs / few iterations) — do this before a full run:
    isaaclab.bat -p scripts\\train_reach.py --num_envs 64 --max_iterations 15 ^
        --save_interval 5 --run_name validation --headless

    # a fuller run later:
    isaaclab.bat -p scripts\\train_reach.py --num_envs 4096 --max_iterations 1000 --headless

Each run gets its own self-contained directory under ``logs/reach/<timestamp>_<name>/``
holding: RSL-RL checkpoints (``model_*.pt``), TensorBoard events, and a ``params/``
snapshot (``env.yaml``, ``agent.yaml``, ``metadata.json``) for reproducibility and
cross-experiment comparison. Tune hyperparameters in
``isaaclab/configs/agents/rsl_rl_ppo_cfg.py``; override run size on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO on RobotArm-Reach-v0 (RSL-RL).")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=None, help="override agent cfg")
parser.add_argument("--num_steps_per_env", type=int, default=None, help="override agent cfg")
parser.add_argument("--save_interval", type=int, default=None, help="override agent cfg")
parser.add_argument("--seed", type=int, default=None, help="override agent cfg")
parser.add_argument("--run_name", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_PKG_DIR = _PROJECT_DIR / "isaaclab"
sys.path.insert(0, str(_PKG_DIR))

import tasks  # noqa: E402  (registers RobotArm tasks)
from runners.env_loader import load_env_cfg, make_reach_env  # noqa: E402
from runners.rsl_rl_runner import build_runner, load_rsl_rl_cfg  # noqa: E402


def _dump_run_metadata(log_dir: str, agent_cfg, env_cfg, num_envs: int, device: str) -> None:
    """Snapshot the exact configs + run metadata into ``<log_dir>/params/``.

    Mirrors the supported reference trainer (``dump_yaml(params/env.yaml|agent.yaml)``)
    so every run is reproducible and comparable. ``metadata.json`` is a compact,
    machine-readable header for cross-experiment comparison and checkpoint provenance.
    """
    from importlib import metadata as _md

    from isaaclab.utils.io import dump_yaml

    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)

    def _ver(pkg: str) -> str:
        try:
            return _md.version(pkg)
        except Exception:
            return "unknown"

    metadata = {
        "task": tasks.REACH_TASK_ID,
        "created": datetime.now().isoformat(timespec="seconds"),
        "num_envs": num_envs,
        "device": device,
        "seed": agent_cfg.seed,
        "max_iterations": agent_cfg.max_iterations,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval": agent_cfg.save_interval,
        "experiment_name": agent_cfg.experiment_name,
        "run_name": agent_cfg.run_name,
        "algorithm": {
            "clip_param": agent_cfg.algorithm.clip_param,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "schedule": agent_cfg.algorithm.schedule,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
            "desired_kl": agent_cfg.algorithm.desired_kl,
        },
        "actor_hidden_dims": list(agent_cfg.actor.hidden_dims),
        "critic_hidden_dims": list(agent_cfg.critic.hidden_dims),
        "versions": {
            "python": platform.python_version(),
            "rsl_rl_lib": _ver("rsl-rl-lib"),
            "isaaclab": _ver("isaaclab"),
            "torch": _ver("torch"),
        },
    }
    with open(os.path.join(params_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def main() -> int:
    agent_cfg = load_rsl_rl_cfg(tasks.REACH_TASK_ID)
    # Command-line overrides (keep hyperparameters easy to tune / size to control).
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.run_name:
        agent_cfg.run_name = args.run_name
    agent_cfg.device = args.device

    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs,
                         device=args.device, seed=agent_cfg.seed)

    # One self-contained directory per run: logs/reach/<timestamp>_<run_name>/.
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = f"{stamp}_{agent_cfg.run_name}" if agent_cfg.run_name else stamp
    log_dir = str(_PROJECT_DIR / "logs" / agent_cfg.experiment_name / run)
    os.makedirs(log_dir, exist_ok=True)

    runner, wrapped_env = build_runner(env, agent_cfg, log_dir=log_dir, device=args.device)

    # Snapshot configs + metadata for reproducibility (after runner build so the
    # migrated agent cfg is what gets recorded).
    _dump_run_metadata(log_dir, agent_cfg, load_env_cfg(tasks.REACH_TASK_ID),
                       wrapped_env.num_envs, args.device)

    print(f"[train] task={tasks.REACH_TASK_ID}  num_envs={wrapped_env.num_envs}  "
          f"device={args.device}")
    print(f"[train] max_iterations={agent_cfg.max_iterations}  "
          f"num_steps_per_env={agent_cfg.num_steps_per_env}  "
          f"save_interval={agent_cfg.save_interval}")
    print(f"[train] log_dir={log_dir}")

    # Start training. (For the first proof, use a small --max_iterations.)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"[train] done. logs + checkpoints in: {log_dir}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
