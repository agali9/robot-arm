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
    # inplace-zero outside inference_mode.
    with torch.no_grad():
        while len(returns) < episodes and step < max_steps:
            actions = runner.get_inference_policy(device=device)(obs)
            obs, rew, dones, _ = wrapped_env.step(actions)
            rew = rew.view(-1); dones = dones.view(-1).bool()
            dist = cmd_term.metrics["position_error"].view(-1)
            dist_sum += float(dist.sum()); dist_cnt += n
            ep_return += rew; ep_len += 1.0
            ep_min = torch.minimum(ep_min, dist)
            if dones.any():
                for i in dones.nonzero(as_tuple=False).view(-1).tolist():
                    returns.append(float(ep_return[i])); lengths.append(float(ep_len[i]))
                    success.append(1.0 if float(ep_min[i]) < threshold else 0.0)
                    final_d.append(float(dist[i]))
                idx = dones.nonzero(as_tuple=False).view(-1)
                ep_return[idx] = 0.0; ep_len[idx] = 0.0; ep_min[idx] = float("inf")
            step += 1
    m = max(len(returns), 1)
    return {
        "episodes": len(returns),
        "success_rate": sum(success) / m,
        "avg_reward": sum(returns) / m,
        "avg_distance_m": dist_sum / max(dist_cnt, 1),
        "avg_final_distance_m": sum(final_d) / m,
        "avg_episode_length": sum(lengths) / m,
    }


def main() -> int:
    run_dir = Path(args.run_dir)
    ckpts = _select(run_dir, args.stride)
    print(f"[evalck] {len(ckpts)} checkpoints: {[_ckpt_iter(c) for c in ckpts]}", flush=True)

    agent_cfg = load_rsl_rl_cfg(tasks.REACH_TASK_ID)
    agent_cfg.device = args.device
    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs,
                         device=args.device, seed=agent_cfg.seed)
    runner, wrapped_env = build_runner(env, agent_cfg, log_dir=None, device=args.device)
    cmd_term = env.unwrapped.command_manager.get_term(COMMAND_NAME)

    rows = []
    for c in ckpts:
        runner.load(str(c))
        r = _score(env, wrapped_env, runner, cmd_term, args.device, wrapped_env.num_envs,
                   args.episodes, args.threshold, args.max_steps)
        r["iter"] = _ckpt_iter(c)
        rows.append(r)
        print(f"[evalck] iter {r['iter']:>4}  succ {r['success_rate']*100:5.1f}%  "
              f"reward {r['avg_reward']:6.3f}  dist {r['avg_distance_m']*100:5.2f}cm  "
              f"final {r['avg_final_distance_m']*100:5.2f}cm  eplen {r['avg_episode_length']:.0f}",
              flush=True)

    out = run_dir / "eval_curve.csv"
    cols = ["iter", "episodes", "success_rate", "avg_reward", "avg_distance_m",
            "avg_final_distance_m", "avg_episode_length"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"[evalck] wrote {out}", flush=True)

    best = min(rows, key=lambda r: r["avg_final_distance_m"])
    print(f"[evalck] BEST by final-distance: iter {best['iter']}  "
          f"final {best['avg_final_distance_m']*100:.2f}cm  succ {best['success_rate']*100:.1f}%", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    code = 1
    try:
        code = main()
    except Exception:
        print("\n[evalck] EXCEPTION:", flush=True); traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(code)
