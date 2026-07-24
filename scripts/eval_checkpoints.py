"""Analysis-only: evaluate a whole checkpoint progression in ONE Isaac session.

Post-run research tooling (not part of the frozen training pipeline). Boots Isaac
once, then loads each ``model_*.pt`` in a run directory in turn and scores the
*deterministic* (mean) inference policy over several episodes, building an
evaluation learning curve. Writes ``eval_curve.csv`` into the run directory.

    isaaclab.bat -p scripts\\eval_checkpoints.py --run_dir logs\\reach\\<run> ^
        --num_envs 64 --episodes 60 --stride 100 --headless

``--stride N`` samples every Nth checkpoint (plus the first and last). Metrics per
checkpoint: success rate, avg reward, avg distance, avg final distance, avg episode
length — identical definitions to scripts/eval_reach.py, computed from the command
term's ``position_error`` metric.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a checkpoint progression (RSL-RL).")
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=60)
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--stride", type=int, default=100, help="sample every Nth checkpoint iter")
parser.add_argument("--max_steps", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_PKG_DIR = _PROJECT_DIR / "isaaclab"
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

import tasks  # noqa: E402
from envs.reach.commands import COMMAND_NAME  # noqa: E402
from runners.env_loader import make_reach_env  # noqa: E402
from runners.rsl_rl_runner import build_runner, load_rsl_rl_cfg  # noqa: E402


def _ckpt_iter(p: Path) -> int:
    m = re.search(r"model_(\d+)\.pt", p.name)
    return int(m.group(1)) if m else -1


def _select(run_dir: Path, stride: int) -> list[Path]:
    models = sorted(run_dir.glob("model_*.pt"), key=_ckpt_iter)
    if not models:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    iters = [_ckpt_iter(m) for m in models]
    keep = [m for m, it in zip(models, iters) if it % stride == 0]
    for edge in (models[0], models[-1]):  # always include first + last
        if edge not in keep:
            keep.append(edge)
    return sorted(set(keep), key=_ckpt_iter)


def _score(env, wrapped_env, runner, cmd_term, device, n, episodes, threshold, max_steps):
    ep_return = torch.zeros(n, device=device)
    ep_len = torch.zeros(n, device=device)
    ep_min = torch.full((n,), float("inf"), device=device)
    returns, lengths, success, final_d = [], [], [], []
    dist_sum, dist_cnt = 0.0, 0

    obs, _ = wrapped_env.reset()
    step = 0
    # no_grad (not inference_mode): inference_mode would mark the command-term metric
    # tensors as "inference tensors", which the NEXT checkpoint's reset() cannot
