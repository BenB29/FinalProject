"""
Comprehensive post-experiment analysis.

Generates all plots for the report from completed A/B/C experiment runs:
  1. Training curves (completion %, lap time, mean speed, max slip)
  2. Racing line comparison overlay on track
  3. Speed heatmaps per mode on the track
  4. Telemetry comparison (speed, steering, throttle/brake vs arc-length)
  5. Summary statistics table (printed + CSV)

Usage
-----
    python scripts/analyse_experiment.py --config configs/suzuka_full.yaml
    python scripts/analyse_experiment.py   # defaults to circuit_lite
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from racing_rl.config.loader import load_config, load_config_for_mode
from racing_rl.tracks.parametric import build_track
from racing_rl.utils.path_utils import get_run_dir, _REPO_ROOT

COLOURS = {"A": "#ff6b6b", "B": "#4ecdc4", "C": "#ffe66d"}
DT = 0.05  # sim timestep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full experiment analysis")
    p.add_argument("--modes", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    p.add_argument("--config", "-c", default=None)
    return p.parse_args()


def _load_cfg(args, mode):
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config_for_mode(mode)
    cfg.obs_mode = mode
    cfg.experiment_name = f"experiment_{cfg.track.name}/obs_{mode}"
    return cfg


def _get_trajectory(cfg, track):
    """Run best model deterministically and return trajectory."""
    from stable_baselines3 import SAC
    from racing_rl.env.racing_env import RacingEnv

    run_dir = get_run_dir(cfg)
    model_path = run_dir / "models" / "best_model.zip"
    if not model_path.exists():
        model_path = run_dir / "models" / "final_model.zip"
    if not model_path.exists():
        return None

    model = SAC.load(str(model_path))
    env = RacingEnv(cfg, eval_mode=True)
    obs, _ = env.reset(seed=cfg.seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    traj = env.get_trajectory()
    env.close()
    return traj


def plot_training_curves(args, track_name, out_dir):
    """4-panel training curves: completion, lap time, mean speed, max slip."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(f"Training Curves — {track_name}", color="white", fontsize=14, y=0.98)

    for mode in args.modes:
        cfg = _load_cfg(args, mode)
        history_path = get_run_dir(cfg) / "eval_history.csv"
        if not history_path.exists():
            print(f"  No eval_history.csv for mode {mode}, skipping.")
            continue

        df = pd.read_csv(history_path)
        col = COLOURS[mode]
        label = f"Obs {mode}"

        axes[0, 0].plot(df["eval_index"], df["completion_pct"], color=col, label=label, lw=2, alpha=0.85)
        completed = df[df["lap_complete"] == True]
        if not completed.empty:
            axes[0, 1].plot(completed["eval_index"], completed["lap_time"], color=col, label=label, lw=2, alpha=0.85)
        axes[1, 0].plot(df["eval_index"], df["mean_speed"], color=col, label=label, lw=2, alpha=0.85)
        axes[1, 1].plot(df["eval_index"], df["max_slip"], color=col, label=label, lw=2, alpha=0.85)

    titles = ["Completion %", "Lap Time [s] (completed only)", "Mean Speed [m/s]", "Max Slip Ratio"]
    ylabels = ["Completion [%]", "Lap time [s]", "Speed [m/s]", "Slip ratio"]

    for ax, t, yl in zip(axes.flat, titles, ylabels):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(t, color="white", fontsize=11)
        ax.set_xlabel("Eval index", color="white")
        ax.set_ylabel(yl, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.legend(labelcolor="white", facecolor="#333333", framealpha=0.5)
        ax.grid(True, color="#333333", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_speed_heatmaps(args, track, track_name, trajectories, out_dir):
    """One subplot per mode showing speed as colour along the trajectory."""
    modes_with_traj = [(m, t) for m, t in trajectories.items() if t is not None]
    if not modes_with_traj:
        return

    n = len(modes_with_traj)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), dpi=120)
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(f"Speed Heatmap — {track_name}", color="white", fontsize=14, y=0.98)

    if n == 1:
        axes = [axes]

    centre = track.centreline_xy
    left = track.left_boundary_xy
    right = track.right_boundary_xy

    for ax, (mode, traj) in zip(axes, modes_with_traj):
        ax.set_facecolor("#1a1a1a")
        # Track boundaries
        ax.plot(left[:, 0], left[:, 1], color="#555555", lw=0.8)
        ax.plot(right[:, 0], right[:, 1], color="#555555", lw=0.8)
        ax.plot(centre[:, 0], centre[:, 1], color="#333333", lw=0.5, ls="--")

        arr = np.array(traj)
        x, y, speed = arr[:, 0], arr[:, 1], arr[:, 2]

        sc = ax.scatter(x, y, c=speed, cmap="RdYlGn", s=3, vmin=0, vmax=80)
        ax.set_title(f"Obs {mode}", color="white", fontsize=11)
        ax.set_aspect("equal")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Speed [m/s]", color="white")
        cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    path = out_dir / "speed_heatmaps.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_telemetry(args, track, track_name, trajectories, out_dir):
    """Speed, steering, and throttle/brake traces vs distance for each mode."""
    modes_with_traj = [(m, t) for m, t in trajectories.items() if t is not None]
    if not modes_with_traj:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=120, sharex=True)
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(f"Telemetry Comparison — {track_name}", color="white", fontsize=14, y=0.98)

    for mode, traj in modes_with_traj:
        arr = np.array(traj)
        x, y = arr[:, 0], arr[:, 1]
        speed = arr[:, 2]
        steering = arr[:, 4]
        accel_cmd = arr[:, 5]

        # Compute cumulative distance
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dist = np.cumsum(np.hypot(dx, dy))

        col = COLOURS[mode]
        label = f"Obs {mode}"

        axes[0].plot(dist, speed, color=col, label=label, lw=1.2, alpha=0.85)
        axes[1].plot(dist, np.degrees(steering), color=col, label=label, lw=1.2, alpha=0.85)
        axes[2].plot(dist, accel_cmd, color=col, label=label, lw=1.2, alpha=0.85)

    ylabels = ["Speed [m/s]", "Steering [deg]", "Accel cmd [-1, 1]"]
    titles = ["Speed vs Distance", "Steering Angle vs Distance", "Throttle / Brake vs Distance"]

    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(t, color="white", fontsize=11)
        ax.set_ylabel(yl, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.legend(labelcolor="white", facecolor="#333333", framealpha=0.5)
        ax.grid(True, color="#333333", alpha=0.3)

    axes[2].set_xlabel("Distance [m]", color="white")
    axes[2].axhline(0, color="#666666", lw=0.5, ls="--")

    plt.tight_layout()
    path = out_dir / "telemetry_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_racing_lines(args, track, track_name, trajectories, out_dir):
    """All racing lines overlaid on the track map."""
    modes_with_traj = [(m, t) for m, t in trajectories.items() if t is not None]
    if not modes_with_traj:
        return

    fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    centre = track.centreline_xy
    left = track.left_boundary_xy
    right = track.right_boundary_xy

    ax.fill(
        np.concatenate([left[:, 0], right[::-1, 0]]),
        np.concatenate([left[:, 1], right[::-1, 1]]),
        color="#2a2a2a", zorder=0
    )
    ax.plot(left[:, 0], left[:, 1], color="#666666", lw=1.5)
    ax.plot(right[:, 0], right[:, 1], color="#666666", lw=1.5)
    ax.plot(centre[:, 0], centre[:, 1], color="#444444", lw=0.5, ls="--")

    for mode, traj in modes_with_traj:
        arr = np.array(traj)
        ax.plot(arr[:, 0], arr[:, 1], color=COLOURS[mode], label=f"Obs {mode}",
                lw=0.9, alpha=0.6)

    ax.set_title(f"Racing Line Comparison — {track_name}", color="white", fontsize=13)
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(labelcolor="white", facecolor="#333333", framealpha=0.7, fontsize=11)

    plt.tight_layout()
    path = out_dir / "racing_lines.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(args, out_dir):
    """Print and save final metrics comparison."""
    rows = []
    for mode in args.modes:
        cfg = _load_cfg(args, mode)
        history_path = get_run_dir(cfg) / "eval_history.csv"
        if not history_path.exists():
            continue
        df = pd.read_csv(history_path)
        completed = df[df["lap_complete"] == True]
        rows.append({
            "mode": mode,
            "total_evals": len(df),
            "laps_completed": int(completed.shape[0]),
            "best_lap_time": f"{completed['lap_time'].min():.2f}" if not completed.empty else "N/A",
            "final_completion_%": f"{df['completion_pct'].iloc[-1]:.1f}",
            "peak_completion_%": f"{df['completion_pct'].max():.1f}",
            "best_mean_speed": f"{df['mean_speed'].max():.1f}",
            "lowest_max_slip": f"{df['max_slip'].min():.3f}",
            "first_lap_eval": str(completed["eval_index"].min()) if not completed.empty else "N/A",
        })

    if not rows:
        print("  No data to summarise.")
        return

    # Print
    print(f"\n{'='*80}")
    print(f"  Summary Table")
    print(f"{'='*80}")
    header = list(rows[0].keys())
    print("  " + "  ".join(f"{h:>18}" for h in header))
    print("  " + "-" * (20 * len(header)))
    for row in rows:
        print("  " + "  ".join(f"{v:>18}" for v in row.values()))

    # Save CSV
    csv_path = out_dir / "summary_table.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")


def main() -> None:
    args = parse_args()

    base_cfg = _load_cfg(args, "B")
    track_name = base_cfg.track.name
    track = build_track(track_name, base_cfg.track.half_width)

    out_dir = _REPO_ROOT / "outputs" / f"experiment_{track_name}" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalysing experiment: {track_name}")
    print(f"Output: {out_dir}\n")

    # 1. Training curves
    print("1. Training curves...")
    plot_training_curves(args, track_name, out_dir)

    # 2. Get trajectories for remaining plots
    print("2. Loading best trajectories...")
    trajectories = {}
    for mode in args.modes:
        cfg = _load_cfg(args, mode)
        traj = _get_trajectory(cfg, track)
        trajectories[mode] = traj
        status = f"{len(traj)} steps" if traj else "no model"
        print(f"  Mode {mode}: {status}")

    # 3. Racing line overlay
    print("3. Racing line overlay...")
    plot_racing_lines(args, track, track_name, trajectories, out_dir)

    # 4. Speed heatmaps
    print("4. Speed heatmaps...")
    plot_speed_heatmaps(args, track, track_name, trajectories, out_dir)

    # 5. Telemetry comparison
    print("5. Telemetry comparison...")
    plot_telemetry(args, track, track_name, trajectories, out_dir)

    # 6. Summary table
    print("6. Summary statistics...")
    print_summary_table(args, out_dir)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
