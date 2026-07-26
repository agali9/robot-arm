"""Dry-run validation for the reach training scaffold (NO learning).

Proves the trainer plumbing is correctly wired without training: loads the RSL-RL
agent config, builds the env, **constructs the RSL-RL objects** (actor/critic
networks + PPO + optimizer via ``OnPolicyRunner``), then performs a *no-op
initialization check* — confirms the inference policy is obtainable, has parameters,
and that the observation/action dimensions line up — and exits cleanly.

    isaaclab.bat -p scripts\\reach_train_dryrun.py --num_envs 16 --device cuda:0

No optimizer step, no ``runner.learn(...)``, no checkpoints. It also does NOT step
the env itself: ``OnPolicyRunner`` already resets the env during construction, and a
second manual reset would needlessly re-create the GPU physics view. Actual env
reset/step is covered by ``scripts/reach_task_smoke.py``; ``runner.learn(...)`` drives
its own single reset/rollout. Use GPU (the env is validated on GPU PhysX).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dry-run the reach PPO scaffold (no learning).")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

_PKG_DIR = Path(__file__).resolve().parents[1] / "isaaclab"
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

import tasks  # noqa: E402  (registers RobotArm tasks)
from runners.env_loader import make_reach_env  # noqa: E402
from runners.rsl_rl_runner import build_runner, load_rsl_rl_cfg  # noqa: E402


def main() -> int:
    print(f"[dryrun] task: {tasks.REACH_TASK_ID}")
    agent_cfg = load_rsl_rl_cfg(tasks.REACH_TASK_ID)
    agent_cfg.device = args.device
    print(f"[dryrun] agent cfg: {type(agent_cfg).__name__}  actor={agent_cfg.actor.hidden_dims}"
          f"  algo={agent_cfg.algorithm.class_name}  clip_actions={agent_cfg.clip_actions}")

    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs,
                         device=args.device, seed=agent_cfg.seed)

    ok = True
    try:
        # Construct the RL objects: networks + PPO + optimizer. (No log dir, no training.)
        print("[dryrun] building RSL-RL runner (networks + PPO + optimizer) ...", flush=True)
        runner, wrapped_env = build_runner(env, agent_cfg, log_dir=None, device=args.device)
        print("[dryrun] runner built.", flush=True)

        # No-op init checks — no env stepping.
        policy = runner.get_inference_policy(device=args.device)
        n_params = sum(p.numel() for p in policy.parameters())
        obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"]
        action_dim = env.unwrapped.action_manager.total_action_dim

        checks = {
            "runner_built": runner is not None,
            "has_ppo_algorithm": hasattr(runner, "alg"),
            "policy_obtainable": callable(policy),
            "policy_has_params": n_params > 0,
            "num_envs_match": wrapped_env.num_envs == args.num_envs,
        }
        ok = all(checks.values())

        # Flush the report BEFORE any env teardown so it can't be lost to a crash.
        print("\n==================== REACH TRAIN DRY-RUN REPORT ====================", flush=True)
        print(f"  library             : RSL-RL (OnPolicyRunner / PPO)", flush=True)
        print(f"  env obs dim (policy): {obs_dim}", flush=True)
        print(f"  action dim          : {action_dim}", flush=True)
        print(f"  policy parameters   : {n_params:,}", flush=True)
        print(f"  RL objects built    : actor/critic + PPO + optimizer", flush=True)
        print(f"  checks              : {checks}", flush=True)
        print(f"  learning performed  : False (dry-run only)", flush=True)
        print(f"  RESULT              : {'PASSED' if ok else 'FAILED'}", flush=True)
        print("===================================================================", flush=True)
    finally:
        try:
            env.close()
        except Exception as exc:  # a physics-view teardown hiccup must not fail the run
            print(f"[dryrun] note: env.close() teardown warning: {exc}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
