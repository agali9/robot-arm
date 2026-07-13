"""Actuator configuration — kept separate from robot geometry (RobotCfg).

Two physically distinct groups, matching the hardware:

* **J1 / J2 — hoverboard hub motors with belt reductions.** Modeled with
  :class:`~isaaclab.actuators.DCMotorCfg`, an *explicit* DC-motor model with a
  torque–speed saturation curve — appropriate for BLDC hub motors. The joint-space
  effort/velocity limits already reflect the belt reduction (they come from the
  URDF), so the reduction ratios here are documentation for motor-side modeling.
* **J3–J6 — serial-bus servos.** Position-controlled, so modeled with
  :class:`~isaaclab.actuators.ImplicitActuatorCfg` (PD handled in the solver).

Tuning lives in :class:`ActuatorTuning`. **Uncertain values are dataclass fields
with labeled defaults** — `damping` defaults to values measured from the validated
USD; position `stiffness` and hoverboard `saturation_effort` / belt reductions are
starting points to tune. Pass a customized :class:`ActuatorTuning` to
:func:`make_actuators` to retune without touching RobotCfg.
"""
