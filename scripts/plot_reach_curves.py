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
    rows: list[tuple[str, int, float]] = []
    for tag, series in scalars.items():
        for step, val in series:
            rows.append((tag, step, val))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "step", "value"])
        w.writerows(rows)


def _plot(scalars: dict, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional
        print(f"[plot] matplotlib unavailable ({exc}); skipping PNG (CSV still written).")
        return False

    panels = [(title, _pick(scalars, subs)) for title, subs in _PANELS]
    panels = [(t, tag) for t, tag in panels if tag]
    if not panels:
        print("[plot] no known scalar tags found; skipping PNG.")
        return False

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for (title, tag), ax in zip(panels, axes.flat):
        ax.axis("on")
        series = scalars[tag]
        xs = [s for s, _ in series]
        ys = [v for _, v in series]
        ax.plot(xs, ys, color="#2b6cb0", linewidth=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Reach learning curves — {out.parent.name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot reach training learning curves.")
    ap.add_argument("--run_dir", type=str, default=None, help="run folder (default: latest)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir) if a.run_dir else _latest_run()
    print(f"[plot] run_dir: {run_dir}")

    scalars = _load_scalars(run_dir)
    if not scalars:
        print("[plot] no scalar data found in event files.")
        return 1
    print(f"[plot] {len(scalars)} scalar tags found.")

    csv_out = run_dir / "curves.csv"
    _write_csv(scalars, csv_out)
    print(f"[plot] wrote {csv_out}")

    png_out = run_dir / "curves.png"
    if _plot(scalars, png_out):
        print(f"[plot] wrote {png_out}")

    # First-vs-last summary for the curated metrics.
    print("\n---------------- learning-curve summary (first -> last) ----------------")
    for title, subs in _PANELS:
        tag = _pick(scalars, subs)
        if not tag or not scalars[tag]:
            continue
        first = scalars[tag][0][1]
        last = scalars[tag][-1][1]
        print(f"  {title:<28} {first:>12.4f} -> {last:>12.4f}   [{tag}]")
    print("------------------------------------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
