# robot-arm

A 6-DOF robot arm I'm building for a school/personal project. The first two joints use hoverboard hub motors, the rest are servos.

Right now there's sim + RL training in Isaac Lab, and a deploy folder for running the trained policy. Hardware side is still a work in progress — J1 and J2 are built, J3-J6 are just CAD.

## What's in the repo

```
urdf/        robot model
isaaclab/    Isaac Lab env and reach task
deploy/      export policy and run inference
scripts/     training scripts
docs/guide/  build notes
```

## Training

Needs Isaac Lab and a GPU.

```bat
isaaclab.bat -p scripts\train_reach.py --num_envs 64 --max_iterations 15 --headless
```

Checkpoints go to `logs/reach/`. There's already an exported policy in `deploy/exported/`.

## Run the policy in sim

```bat
isaaclab.bat -p deploy\scripts\run_inference.py --backend sim --policy deploy\exported\policy.pt
```

## Tests (no Isaac needed)

```bash
python deploy/tests/test_safety.py
```

More info in `deploy/README.md` and `docs/guide/build.md`.
