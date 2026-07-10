# Isaac Sim

Load and check the robot in Isaac Sim before training or deployment.

## Assets

| File | Purpose |
| --- | --- |
| `isaac/robot_arm.usda` | Scene with the arm articulation |
| `urdf/robot_arm.urdf` | Joint tree, limits, inertial/collision geometry |
| `configs/robot.yaml` | Joint names, limits, home pose (source of truth for software) |

Override the USD path with the `ROBOTARM_USD` env var if you use a different export.

## Open the scene

1. Launch Isaac Sim.
2. File → Open → `isaac/robot_arm.usda`.
3. Press Play and confirm the arm appears with six revolute joints (`j1_joint` … `j6_joint`).

Check the articulation in the Stage tree: base link through `j6_link`, tool frame at `tool0`.

## What to verify

- Scale and orientation look right (Y-up, arm standing on base).
- All six joints move independently in the property panel or with a short script.
- Collision meshes are present (not just visuals).
- Joint limits match `configs/robot.yaml`.

If something is off, fix the URDF/USD export — don't patch limits only in sim.

## Joint model (sim)

