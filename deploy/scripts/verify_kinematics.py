"""Validate the URDF FK provider against Isaac Sim (the frozen USD), and fit base->env.

The policy's ``ee_position`` observation is in the ENV frame (Isaac ``body_pos_w -
env_origins``). ``UrdfKinematicsProvider`` computes FK in the BASE_LINK frame. The two
differ by the robot's fixed base placement in the env. This script:

  1. sets many random joint configs in Isaac, reads the true j6_link EE (env frame),
  2. computes the numpy URDF FK (base frame) for the same configs,
  3. fits the rigid transform base->env (Kabsch) and reports the residual,
  4. writes the fitted base pose to ``configs/base_transform.json`` so the FK provider can
     return EE in the exact frame the policy expects.

PASS if the residual after the fitted transform is < 1 mm (URDF matches the USD).

    isaaclab.bat -p deploy\\scripts\\verify_kinematics.py --num_envs 64 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate URDF FK vs Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--tol_mm", type=float, default=1.0)
parser.add_argument("--out", type=str, default="configs/base_transform.json")
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

import tasks  # noqa: E402
from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.kinematics import UrdfKinematicsProvider, _matrix_to_quat  # noqa: E402
from runners.env_loader import make_reach_env  # noqa: E402


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Best-fit rotation+translation mapping P -> Q (both (N,3)). Returns (R, t)."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp


def main() -> int:
    device = args.device
    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs, device=device, seed=1)
    u = env.unwrapped
    u.reset()
    robot = u.scene["robot"]
    ee_id = robot.find_bodies(C.EE_BODY_NAME)[0][0]
    fk = UrdfKinematicsProvider(str(_PROJECT / "urdf" / "robot_arm.urdf"))

    # Collect (joint_pos, ee) pairs from the STABLE running arm — no teleport (the explicit
    # DC-motor actuators are only conditionally stable and blow up when teleported). Small
    # random actions diversify the configs; the arm holds them normally.
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    wrapped = RslRlVecEnvWrapper(env, clip_actions=C.CLIP_ACTIONS)
    wrapped.reset()
    rng = np.random.default_rng(0)
    qs, ees = [], []
    with torch.no_grad():
        for _ in range(30):
            act = torch.tensor(rng.uniform(-1, 1, size=(args.num_envs, C.NUM_JOINTS)),
                               dtype=torch.float32, device=device)
            wrapped.step(act)
            q = robot.data.joint_pos.torch[:, :C.NUM_JOINTS].cpu().numpy()
            ee = (robot.data.body_pos_w.torch[:, ee_id] - u.scene.env_origins).cpu().numpy()
            good = np.isfinite(q).all(1) & np.isfinite(ee).all(1)
            qs.append(q[good]); ees.append(ee[good])
    q_all = np.concatenate(qs)
    ee_isaac = np.concatenate(ees)
    ee_fk = np.stack([fk.ee_position(q_all[i]) for i in range(len(q_all))])  # base frame
    n = len(q_all)

    R, t = kabsch(ee_fk, ee_isaac)                               # base -> env
    ee_fk_env = (R @ ee_fk.T).T + t
    err_mm = np.linalg.norm(ee_fk_env - ee_isaac, axis=1) * 1e3
    max_mm, mean_mm = float(err_mm.max()), float(err_mm.mean())
    rot_dev = float(np.abs(R - np.eye(3)).max())
    ok = max_mm < args.tol_mm

    base_quat = _matrix_to_quat(R)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "base_position": t.tolist(), "base_quat_wxyz": base_quat.tolist(),
        "rotation_max_dev_from_identity": rot_dev,
        "residual_mm": {"max": max_mm, "mean": mean_mm},
        "note": "base_link pose in the policy (env) frame; add to URDF FK to match training."},
        indent=2))

    print("\n============= URDF FK vs ISAAC (frozen USD) =============", flush=True)
    print(f"  samples              : {n} random joint configs", flush=True)
    print(f"  base->env translation: {np.round(t, 4).tolist()} m", flush=True)
    print(f"  base->env rotation   : max|R-I| = {rot_dev:.2e} (0 => pure translation)", flush=True)
    print(f"  residual (fitted)    : max {max_mm:.3f} mm  mean {mean_mm:.3f} mm", flush=True)
    print(f"  wrote base transform : {out}", flush=True)
    print(f"  RESULT               : {'PASSED (< %.1f mm)' % args.tol_mm if ok else 'FAILED'}",
          flush=True)
    print("========================================================", flush=True)
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback
        print("\n[verify_kin] EXCEPTION:", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(code)
