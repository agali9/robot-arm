# RobotArm — Isaac Lab Robot Package

A **reusable Isaac Lab robot package** for the custom 6-DOF arm. It defines the robot
once (geometry, physics, actuators) plus modular observation / action / event / reward
building blocks, so future tasks (reach, pick, etc.) import it instead of duplicating
robot-specific logic.

Targets **Isaac Lab 2.3.x** (manager-based workflow), validated against the install at
`.../IsaacLab` (2.3.2).

> This package is independent of the project's MCP server / recording / live-control
> framework. It does not modify the robot, URDF, USD, or project layout.

## Layout

```
isaaclab/
├─ robot/
│  ├─ robot_cfg.py     # ROBOT_CFG: ArticulationCfg (geometry + physics ONLY)
│  ├─ actuators.py     # ActuatorTuning + make_actuators (J1/J2 DC motors, J3-J6 servos)
│  ├─ observations.py  # ObservationsCfg (joint pos/vel, EE pose, target, distance, last action)
│  ├─ actions.py       # ActionsCfg + position/velocity/effort factories
│  ├─ events.py        # EventCfg (startup + reset randomization)
│  ├─ reset.py         # reset functions (target randomization)
