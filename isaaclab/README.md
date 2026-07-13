# RobotArm — Isaac Lab Robot Package

A **reusable Isaac Lab robot package** for the custom 6-DOF arm. It defines the robot
once (geometry, physics, actuators) plus modular observation / action / event / reward
building blocks, so future tasks (reach, pick, etc.) import it instead of duplicating
robot-specific logic.

Targets **Isaac Lab 2.3.x** (manager-based workflow), validated against the install at
`.../IsaacLab` (2.3.2).

> This package is independent of the project's MCP server / recording / live-control
> framework. It does not modify the robot, URDF, USD, or project layout.

