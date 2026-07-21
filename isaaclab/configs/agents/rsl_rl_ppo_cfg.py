"""RSL-RL PPO agent configuration for RobotArm-Reach-v0.

Typed config (no YAML schema drift) matching the Isaac Lab 2.3.x RSL-RL API — the
same structure the stock reach tasks use. This is the single place to tune training
hyperparameters for the reach task; it references **no robot constants** and knows
nothing about the robot package.

Baseline philosophy (see docs/experiments): this is a *reference-aligned baseline*,
not a tuned config. Values track NVIDIA's Franka reach task
(``manager_based/manipulation/reach/config/franka/agents/rsl_rl_ppo_cfg.py``) so the
custom robot is the only variable when comparing against a known-good result. Every
deliberate deviation is called out inline with a ``# DEVIATION`` note and justified.

Deviations from the Franka reference (all others are identical):
  * ``entropy_coef`` 0.001 — now EQUAL to the Franka reference. The first baseline used
    0.005 for exploration, but that run showed the entropy bonus dominating once reward
    saturated (action std diverged 0.75 -> 19.4, policy never reached targets). This
    experiment (run_name ``entropy001``) reverts it to 0.001 as a single-variable test.
  * ``experiment_name`` — cosmetic (log folder name).
Network width, epochs, minibatches, clip, gamma, lambda, lr schedule, desired_kl and
grad-norm are kept IDENTICAL to the reference so results are directly comparable. The
policy/value MLP is [64, 64] like the reference: the observation (28-dim) and action
(6-dim) sizes are essentially the same scale as Franka reach (28-dim obs, 7-dim act),
so no larger network is warranted for a baseline.

For the first short validation run, override size/length on the command line
(``--num_envs``, ``--max_iterations``, ``--save_interval``) rather than editing here.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO on-policy runner config for the reach task (reference-aligned baseline)."""

    # --- Rollout / schedule -----------------------------------------------------------
    num_steps_per_env = 24        # rollout length per env before each PPO update (== ref)
    max_iterations = 1000         # full-run default; the first run should override small
    save_interval = 50            # checkpoint every N iterations (== ref)
    experiment_name = "reach"     # DEVIATION (cosmetic): log folder name -> logs/reach/
    run_name = ""

    # Bound raw policy actions to [-1, 1] before they reach the env. Essential for this
    # arm: an untrained/high-variance policy would otherwise emit large joint targets
    # and destabilize physics. This matches the action range the env is validated on.
    clip_actions = 1.0

    # --- Policy / value networks (MLP) ------------------------------------------------
    # Kept at the reference [64, 64] (see module docstring): obs/action scale ~ Franka.
    actor = RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=False,
    )

    # --- PPO algorithm ----------------------------------------------------------------
    # All values match the Franka reference EXCEPT entropy_coef (see docstring).
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,       # == Franka reference (was 0.005; see exp entropy001)
        num_learning_epochs=8,    # == ref (Franka)
        num_mini_batches=4,       # == ref
        learning_rate=1.0e-3,     # == ref (adaptive schedule adjusts it online)
        schedule="adaptive",      # == ref: KL-adaptive LR toward desired_kl
        gamma=0.99,               # == ref (gamma997 exp ruled out horizon as the bottleneck)
        lam=0.95,                 # == ref
        desired_kl=0.01,          # == ref
        max_grad_norm=1.0,        # == ref
    )
