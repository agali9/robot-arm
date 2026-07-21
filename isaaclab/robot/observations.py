"""Observation definitions for future tasks.

Defines :class:`ObservationsCfg` — a manager-based observation group with the terms
reach/manipulation tasks need: joint positions & velocities (relative to defaults),
end-effector pose, target position, target-relative position, distance to target,
and the previous action.

How it connects: a task env cfg sets ``observations = ObservationsCfg()``. Proprio
