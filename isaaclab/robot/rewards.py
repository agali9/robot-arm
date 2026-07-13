"""Reward terms — modular functions + a :class:`RewardsCfg` that composes them.

One small function per reward component, so tasks can reuse/reweight/disable each
independently. Robot-specific rewards (reach, distance, collision) use the shared
:mod:`utils.scene_queries` helpers so they measure the same EE/target quantities as
