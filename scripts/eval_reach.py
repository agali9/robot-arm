"""Evaluate a trained RobotArm-Reach-v0 checkpoint (RSL-RL PPO).

Loads a checkpoint, runs a batch of evaluation episodes with the *deterministic*
inference policy, and prints aggregate metrics. No training, no video.

    isaaclab.bat -p scripts\\eval_reach.py --checkpoint logs\\reach\\<run>\\model_50.pt ^
        --num_envs 64 --episodes 100 --headless

If ``--checkpoint`` is a directory (a run folder), the latest ``model_*.pt`` inside is
used. Metrics reported (position-only reach; threshold matches the env, 0.05 m):

    success rate       fraction of episodes whose EE came within threshold of the target
    avg reward         mean total (undiscounted) return per episode
    avg distance       mean EE->target distance over all steps (tracking quality, m)
    avg final distance mean EE->target distance on the last step before an episode ends
    avg episode length mean steps per episode

Also writes ``eval.json`` next to the checkpoint for cross-experiment comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a reach PPO checkpoint (RSL-RL).")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a model_*.pt file OR a run directory containing them.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=100, help="min completed episodes to score")
parser.add_argument("--threshold", type=float, default=0.05, help="success distance (m)")
parser.add_argument("--max_steps", type=int, default=4000, help="hard cap on eval steps")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_PKG_DIR = _PROJECT_DIR / "isaaclab"
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

import tasks  # noqa: E402  (registers RobotArm tasks)
from envs.reach.commands import COMMAND_NAME  # noqa: E402
from runners.env_loader import make_reach_env  # noqa: E402
from runners.rsl_rl_runner import build_runner, load_rsl_rl_cfg  # noqa: E402


def _resolve_checkpoint(path_str: str) -> Path:
    """Return a concrete ``model_*.pt`` path (latest by iteration if a dir is given)."""
    p = Path(path_str)
    if p.is_dir():
        models = list(p.glob("model_*.pt"))
        if not models:
            raise FileNotFoundError(f"no model_*.pt found in directory: {p}")

        def _it(m: Path) -> int:
            match = re.search(r"model_(\d+)\.pt", m.name)
            return int(match.group(1)) if match else -1

        return max(models, key=_it)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def main() -> int:
    ckpt = _resolve_checkpoint(args.checkpoint)
    print(f"[eval] checkpoint: {ckpt}")

    agent_cfg = load_rsl_rl_cfg(tasks.REACH_TASK_ID)
    agent_cfg.device = args.device

    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs,
                         device=args.device, seed=agent_cfg.seed)
    runner, wrapped_env = build_runner(env, agent_cfg, log_dir=None, device=args.device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=args.device)
    print(f"[eval] policy loaded  num_envs={wrapped_env.num_envs}  episodes>={args.episodes}")

    n = wrapped_env.num_envs
    dev = args.device
    ep_return = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    ep_min_dist = torch.full((n,), float("inf"), device=dev)

    # Completed-episode accumulators.
    done_returns: list[float] = []
    done_lengths: list[float] = []
    done_success: list[float] = []
    done_final_dist: list[float] = []
    dist_sum = 0.0
    dist_count = 0

