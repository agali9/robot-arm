"""Extract and plot learning curves from a reach training run (TensorBoard events).

Reads the ``events.out.tfevents.*`` file(s) in a run directory, exports every scalar
to a tidy CSV (for cross-experiment comparison in any tool), and renders a PNG grid
of the key training curves. Does NOT launch Isaac Sim, so it runs fast with any
Python that has ``tensorboard`` installed (the Isaac Sim python does).

    isaaclab.bat -p scripts\\plot_reach_curves.py --run_dir logs\\reach\\<run>
    # or, plain python with tensorboard+matplotlib available:
    python scripts\\plot_reach_curves.py --run_dir logs\\reach\\<run>

Outputs (written into the run dir): ``curves.csv``, ``curves.png`` (if matplotlib is
present) and a printed first-vs-last summary. If ``--run_dir`` is omitted, the most
recent run under ``logs/reach/`` is used.

Metric -> TensorBoard tag mapping (RSL-RL + Isaac Lab emit these natively):
    episode reward     Train/mean_reward
    episode length     Train/mean_episode_length
    policy loss        Loss/surrogate
    value loss         Loss/value_function
    entropy            Loss/entropy
    learning rate      Loss/learning_rate
    KL / exploration   Loss/... (KL) , Policy/mean_noise_std
    success rate       Episode_Termination/target_reached (fraction of episodes reached)
    distance to target Metrics/ee_pose/position_error
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parents[1]

# Curated panels: (title, [candidate tags/substrings, first hit wins]). Ordered so the
# exact tag name this RSL-RL version emits is tried before a looser substring.
_PANELS = [
    ("Episode reward", ["Train/mean_reward", "mean_reward"]),
    ("Episode length", ["Train/mean_episode_length", "mean_episode_length"]),
    ("Distance to target (m)", ["Metrics/ee_pose/position_error", "position_error"]),
    ("Success rate", ["Metrics/success_rate", "Episode_Termination/target_reached"]),
    ("Policy (surrogate) loss", ["Loss/surrogate", "surrogate"]),
    ("Value loss", ["Loss/value", "value_function", "value"]),
    ("Entropy", ["Loss/entropy", "entropy"]),
    ("Learning rate", ["Loss/learning_rate", "learning_rate"]),
    ("Policy noise std", ["Policy/mean_std", "mean_std", "noise_std"]),
    ("Termination: time-out", ["Episode_Termination/time_out", "time_out"]),
]


def _latest_run() -> Path:
    root = _PROJECT_DIR / "logs" / "reach"
    runs = [p for p in root.glob("*") if p.is_dir()] if root.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no runs found under {root}; pass --run_dir")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} for every scalar in the run's event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    return out


def _pick(tags: dict, substrings: list[str]) -> str | None:
    for s in substrings:
        if s in tags:  # exact tag match first
            return s
    for s in substrings:  # then substring match
        for tag in tags:
            if s.lower() in tag.lower():
                return tag
    return None


def _write_csv(scalars: dict, out: Path) -> None:
