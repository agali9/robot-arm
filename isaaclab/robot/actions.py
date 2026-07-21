"""Action definitions.

The initial action space is **joint position targets** (delta from the home pose,
which pairs well with the position-controlled actuators in :mod:`robot.actuators`).

Designed to grow into velocity / effort / hybrid control without an architecture
change: each control mode has a small factory that returns the matching Isaac Lab
action term, and :class:`ActionsCfg` simply selects one. Hybrid control (e.g.
position for the arm, effort for the wrist) is expressed by adding more term fields
to :class:`ActionsCfg` — the manager runs them together.
