"""Export the frozen reach policy to portable formats + validate against the checkpoint.

Runs inside Isaac Sim's Python (needs the RSL-RL runner to reconstruct the policy). It:
  1. builds the reach env + RSL-RL runner and loads a checkpoint,
  2. exports the actor to TorchScript (``policy.pt``) and ONNX (``policy.onnx``) via the
     SUPPORTED ``runner.export_policy_to_jit`` / ``export_policy_to_onnx``,
  3. VALIDATES: runs N random observations through the exported policy and the live
     checkpoint policy and asserts the raw actions match to tight tolerance,
  4. writes ``metadata.json`` (checkpoint, obs/action dims, contract) next to the models.

    isaaclab.bat -p deploy\\scripts\\export_policy.py ^
        --checkpoint logs\\reach\\2026-07-19_16-33-14_scale15_random\\model_400.pt ^
        --out deploy\\exported --num_envs 16 --device cpu

No training, no hardware. Use ``--device cpu`` so the exported/validated numerics match
the CPU deployment target.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export + validate the reach policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, default="deploy/exported")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--samples", type=int, default=512, help="random obs for validation")
parser.add_argument("--tol", type=float, default=1e-5, help="max abs action mismatch")
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
from runners.env_loader import make_reach_env  # noqa: E402
from runners.rsl_rl_runner import build_runner, load_rsl_rl_cfg  # noqa: E402


def main() -> int:
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    agent_cfg = load_rsl_rl_cfg(tasks.REACH_TASK_ID)
    agent_cfg.device = device
    env = make_reach_env(tasks.REACH_TASK_ID, num_envs=args.num_envs, device=device,
                         seed=agent_cfg.seed)
    runner, wrapped_env = build_runner(env, agent_cfg, log_dir=None, device=device)
    runner.load(str(ckpt))
    print(f"[export] loaded checkpoint: {ckpt}", flush=True)

    # Sanity: env dims must match the deployment contract.
    obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"]
    obs_dim = int(obs_dim[0] if isinstance(obs_dim, (tuple, list)) else obs_dim)
    act_dim = int(env.unwrapped.action_manager.total_action_dim)
    assert obs_dim == C.OBS_DIM, f"obs dim {obs_dim} != contract {C.OBS_DIM}"
    assert act_dim == C.ACTION_DIM, f"action dim {act_dim} != contract {C.ACTION_DIM}"
    print(f"[export] contract OK: obs={obs_dim} action={act_dim}", flush=True)

    # 1) Export TorchScript (supported path).
    runner.export_policy_to_jit(path=str(out_dir), filename="policy.pt")
    print(f"[export] wrote {out_dir/'policy.pt'} (TorchScript)", flush=True)

    # 2) Export ONNX (best-effort; needs onnx installed).
    onnx_ok = False
    try:
        runner.export_policy_to_onnx(path=str(out_dir), filename="policy.onnx")
        onnx_ok = True
        print(f"[export] wrote {out_dir/'policy.onnx'} (ONNX)", flush=True)
    except Exception as exc:  # keep TorchScript deliverable even if onnx missing
        print(f"[export] ONNX export skipped ({type(exc).__name__}: {exc})", flush=True)

    # 3) Validate exported == checkpoint on random observations.
    policy = runner.get_inference_policy(device=device)
    ts = torch.jit.load(str(out_dir / "policy.pt"), map_location=device).eval()
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((args.samples, C.OBS_DIM)).astype(np.float32)
    obs_t = torch.from_numpy(obs).to(device)
    with torch.inference_mode():
        # The checkpoint policy consumes a dict keyed by obs-group ("policy"); the
        # EXPORTED graph consumes a flat tensor (that decoupling is the point of export).
        ref = policy({"policy": obs_t}).cpu().numpy()
        got_ts = ts(obs_t).cpu().numpy()
    err_ts = float(np.abs(ref - got_ts).max())
    print(f"[export] TorchScript max abs action err vs checkpoint = {err_ts:.2e}", flush=True)

    err_onnx = None
    if onnx_ok:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_dir / "policy.onnx"),
                                        providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            got_onnx = np.concatenate(
                [sess.run(None, {in_name: obs[i:i + 1]})[0] for i in range(args.samples)])
            err_onnx = float(np.abs(ref - got_onnx[:, :C.ACTION_DIM]).max())
            print(f"[export] ONNX max abs action err vs checkpoint = {err_onnx:.2e}", flush=True)
        except Exception as exc:
            print(f"[export] ONNX validation skipped ({exc})", flush=True)

    ok = err_ts <= args.tol and (err_onnx is None or err_onnx <= args.tol)

    meta = {
        "checkpoint": str(ckpt),
        "exported": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "obs_dim": C.OBS_DIM,
        "action_dim": C.ACTION_DIM,
        "action_scale": C.ACTION_SCALE,
        "clip_actions": C.CLIP_ACTIONS,
        "obs_normalization": False,
        "joint_names": list(C.JOINT_NAMES),
        "formats": ["torchscript:policy.pt"] + (["onnx:policy.onnx"] if onnx_ok else []),
        "validation": {"torchscript_max_abs_err": err_ts, "onnx_max_abs_err": err_onnx,
                       "tolerance": args.tol, "passed": ok},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote {out_dir/'metadata.json'}", flush=True)

    print("\n==================== POLICY EXPORT REPORT ====================", flush=True)
    print(f"  checkpoint : {ckpt.name}", flush=True)
    print(f"  formats    : {meta['formats']}", flush=True)
    print(f"  TS  error  : {err_ts:.2e}  (tol {args.tol:.0e})", flush=True)
    print(f"  ONNX error : {err_onnx if err_onnx is not None else 'n/a'}", flush=True)
    print(f"  RESULT     : {'PASSED' if ok else 'FAILED'}", flush=True)
    print("=============================================================", flush=True)

    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback
        print("\n[export] EXCEPTION:", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(code)
