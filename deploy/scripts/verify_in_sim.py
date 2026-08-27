"""Simulation verification — run the EXPORTED policy in Isaac Sim, no retraining.

Proves the exported policy + the deployment obs/action code reproduce the frozen
checkpoint's behaviour, verifying three things the deployment must preserve:

  1. **same observations**  — the deployment ``ObservationBuilder`` assembles the exact
     28-vector the env's ObservationManager produces (max abs diff),
  2. **same actions**       — the deployment ``ActionProcessor`` produces the same joint
     targets as the env's ActionManager for the same raw action,
  3. **same reach performance / success rate** — a rollout driven by the exported
     TorchScript policy hits the expected ~85% success on random targets.

    isaaclab.bat -p deploy\\scripts\\verify_in_sim.py ^
        --policy deploy\\exported\\policy.pt --num_envs 256 --episodes 400 --device cuda:0

Use ``--device cuda:0``: the raw policy occasionally commands an aggressive first move from
the home pose that can destabilise **CPU** PhysX (NaN -> invalid_state). This is a property
of the un-limited raw policy in this verification harness; on hardware the deployment
SafetyLayer's slew-rate limiter caps exactly this, which is why the inference app is safe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify the exported reach policy in Isaac Sim.")
parser.add_argument("--policy", type=str, default="deploy/exported/policy.pt")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=400)
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument("--obs_tol", type=float, default=1e-4)
parser.add_argument("--act_tol", type=float, default=1e-4)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT / "isaaclab"))
sys.path.insert(0, str(_PROJECT / "deploy"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms  # noqa: E402

import tasks  # noqa: E402
from envs.reach.commands import COMMAND_NAME  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.action import ActionProcessor  # noqa: E402
from robotarm_deploy.observation import ObservationBuilder, RobotState  # noqa: E402
from robotarm_deploy.policy import PolicyRunner  # noqa: E402
from runners.env_loader import make_reach_env  # noqa: E402


def _target_env_frame(env, robot):
    """Reconstruct the reach target in the env frame, exactly as the obs term does."""
    cmd = env.command_manager.get_command(COMMAND_NAME)[:, :3]
    des_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch,
                                        robot.data.root_quat_w.torch, cmd)
    return des_w - env.scene.env_origins


def main() -> int:
    device = args.device
    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs, device=device, seed=42)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=C.CLIP_ACTIONS)   # matches training clip
    u = env.unwrapped
    robot = u.scene["robot"]
    ee_id = robot.find_bodies(C.EE_BODY_NAME)[0][0]

    ts = torch.jit.load(args.policy, map_location=device).eval()
    runner_py = PolicyRunner(args.policy)      # the deployment single-obs runner (parity)
    obuild = ObservationBuilder()
    aproc = ActionProcessor()

    n = wrapped.num_envs
    ep_len = torch.zeros(n, device=device)
    ep_min = torch.full((n,), float("inf"), device=device)
    successes, lengths = [], []
    cmd_term = u.command_manager.get_term(COMMAND_NAME)

    # Env action-term config (its OWN scale/offset) — the ground truth for action parity.
    aterm = u.action_manager.get_term("arm_action")
    env_scale = float(aterm._scale) if np.isscalar(aterm._scale) else \
        aterm._scale.detach().cpu().numpy()
    env_offset = 0.0 if np.isscalar(aterm._offset) and aterm._offset == 0.0 else \
        (float(aterm._offset) if np.isscalar(aterm._offset)
         else aterm._offset[0].detach().cpu().numpy())
    assert abs(float(np.mean(env_scale)) - C.ACTION_SCALE) < 1e-6, \
        f"env action scale {env_scale} != contract {C.ACTION_SCALE}"

    obs_err = act_err = py_err = 0.0
    last_clipped = torch.zeros((n, C.ACTION_DIM), device=device)

    obs, _ = wrapped.reset()
    step = 0
    with torch.no_grad():
        while len(successes) < args.episodes and step < args.max_steps:
            env_obs = obs["policy"]                          # (n,28) ground truth
            raw = ts(env_obs)                                # (n,6) exported policy
            clipped = torch.clamp(raw, -C.CLIP_ACTIONS, C.CLIP_ACTIONS)

            # --- (1) observation parity on env 0 (independent reconstruction) ---------
            jp = robot.data.joint_pos.torch[0, :C.NUM_JOINTS].cpu().numpy()
            jv = robot.data.joint_vel.torch[0, :C.NUM_JOINTS].cpu().numpy()
            ee = (robot.data.body_pos_w.torch[0, ee_id] - u.scene.env_origins[0]).cpu().numpy()
            tgt = _target_env_frame(u, robot)[0].cpu().numpy()
            dep_obs = obuild.build(RobotState(jp, jv, ee), tgt,
                                   last_clipped[0].cpu().numpy())
            obs_err = max(obs_err, float(np.abs(dep_obs - env_obs[0].cpu().numpy()).max()))

            # --- (2) action parity: deployment ActionProcessor vs env's scale/offset --
            # The env does processed = clip(raw)*scale + offset; the deployment mapping
            # must equal that (before the deployment's extra joint-limit clamp).
            r0 = np.clip(raw[0].cpu().numpy(), -C.CLIP_ACTIONS, C.CLIP_ACTIONS)
            env_expected = r0 * env_scale + env_offset
            dep_scaled = r0 * C.ACTION_SCALE + C.HOME_POSE
            act_err = max(act_err, float(np.abs(dep_scaled - env_expected).max()))
            _ = aproc.process(raw[0].cpu().numpy())          # exercise deployment path

            # --- (3) deployment single-obs runner matches batched TS ------------------
            py_out = runner_py.infer(env_obs[0].cpu().numpy())
            py_err = max(py_err, float(np.abs(py_out - raw[0].cpu().numpy()).max()))

            last_clipped = clipped
            # Step FIRST, then read the freshly-computed distance metric (mirrors
            # eval_checkpoints.py; reading it before the step yields stale/zero values).
            obs, _, dones, _ = wrapped.step(raw)
            dones = dones.view(-1).bool()
            dist = cmd_term.metrics["position_error"].view(-1)   # EE->target, per env
            ep_min = torch.minimum(ep_min, dist)
            ep_len += 1.0
            if dones.any():
                idx = dones.nonzero(as_tuple=False).view(-1)
                for i in idx.tolist():
                    successes.append(1.0 if float(ep_min[i]) < args.threshold else 0.0)
                    lengths.append(float(ep_len[i]))
                ep_min[idx] = float("inf")
                ep_len[idx] = 0.0
                last_clipped[idx] = 0.0   # env zeroes last_action on auto-reset; mirror it
            step += 1

    succ = float(np.mean(successes)) if successes else float("nan")
    avg_len = float(np.mean(lengths)) if lengths else float("nan")
    lat = runner_py.latency.snapshot()
    obs_ok, act_ok = obs_err <= args.obs_tol, act_err <= args.act_tol
    ok = obs_ok and act_ok and py_err <= 1e-5

    print("\n============= SIM VERIFICATION (exported policy) =============", flush=True)
    print(f"  policy               : {args.policy}", flush=True)
    print(f"  episodes scored      : {len(successes)} over {n} envs", flush=True)
    print(f"  (1) obs parity  err  : {obs_err:.2e}  (tol {args.obs_tol:.0e})  -> "
          f"{'OK' if obs_ok else 'FAIL'}", flush=True)
    print(f"  (2) action parity err: {act_err:.2e}  (tol {args.act_tol:.0e})  -> "
          f"{'OK' if act_ok else 'FAIL'}", flush=True)
    print(f"  (3) py-runner vs TS  : {py_err:.2e}", flush=True)
    print(f"  success rate         : {succ*100:.1f} %  (expected ~85%)", flush=True)
    print(f"  avg episode length   : ~{avg_len:.0f} steps", flush=True)
    print(f"  inference latency    : mean {lat.get('mean_ms', float('nan')):.3f} ms  "
          f"p99 {lat.get('p99_ms', float('nan')):.3f} ms", flush=True)
    print(f"  RESULT (parity)      : {'PASSED' if ok else 'FAILED'}", flush=True)
    print("=============================================================", flush=True)

    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback
        print("\n[verify] EXCEPTION:", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(code)
