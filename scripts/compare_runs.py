"""
Post-hoc comparison of completed runs.

Loads eval_history.csv from each run directory and plots:
  - Lap completion over training
  - Lap time over training
  - Completion pct over training

Usage
-----
    python scripts/compare_runs.py                   # compare A, B, C
    python scripts/compare_runs.py --modes A B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from racing_rl.config.loader import load_config_for_mode
from racing_rl.utils.path_utils import get_run_dir, _REPO_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare A/B/C training curves")
    p.add_argument("--modes", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    colours = {"A": "#ff6b6b", "B": "#4ecdc4", "C": "#ffe66d"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
    fig.patch.set_facecolor("#1a1a1a")

    for mode in args.modes:
        cfg = load_config_for_mode(mode)
        run_dir = get_run_dir(cfg)
        history_path = run_dir / "eval_history.csv"

        if not history_path.exists():
            print(f"No eval_history.csv for mode {mode} ({run_dir}), skipping.")
            continue

        df = pd.read_csv(history_path)
        col = colours.get(mode, "white")
        label = f"Obs {mode}"

        axes[0].plot(df["eval_index"], df["completion_pct"], color=col, label=label, lw=2)
        # For lap time, only show rows where lap was completed
        completed = df[df["lap_complete"] == True]
        if not completed.empty:
            axes[1].plot(completed["eval_index"], completed["lap_time"], color=col, label=label, lw=2)
        axes[2].plot(df["eval_index"], df["mean_speed"], color=col, label=label, lw=2)

    titles = ["Completion % over evaluations", "Lap time [s] (completed laps only)", "Mean speed [m/s]"]
    ylabels = ["Completion [%]", "Lap time [s]", "Mean speed [m/s]"]

    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(t, color="white", fontsize=10)
        ax.set_xlabel("Eval index", color="white")
        ax.set_ylabel(yl, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.legend(labelcolor="white", facecolor="#333333", framealpha=0.5)

    plt.tight_layout()

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "outputs" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_curves_comparison.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Comparison curves saved: {out_path}")


if __name__ == "__main__":
    main()
