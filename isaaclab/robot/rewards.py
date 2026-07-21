"""Reward terms — modular functions + a :class:`RewardsCfg` that composes them.

One small function per reward component, so tasks can reuse/reweight/disable each
independently. Robot-specific rewards (reach, distance, collision) use the shared
:mod:`utils.scene_queries` helpers so they measure the same EE/target quantities as
the observations; generic shaping (smoothness, limits, action penalty) reuses the
built-in ``isaaclab.envs.mdp`` rewards.

**Weights here are conservative placeholders, not a tuned reward.** The goal is
correct structure and reuse; tune weights when you build the actual task.
