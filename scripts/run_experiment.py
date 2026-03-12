"""
Batch experiment runner: trains obs modes A, B, and C sequentially under
identical physics/reward/budget settings, then saves a side-by-side comparison.

Usage
-----
    python scripts/run_experiment.py                 # train A, B, C at 1M steps each
    python scripts/run_experiment.py --timesteps 200000 --modes A B
    python scripts/run_experiment.py --skip-train --modes A B C   # compare only

Output directory structure:
    outputs/
        obs_A/  ...
        obs_B/  ...
        obs_C/  ...
        comparison/
            racing_line_comparison.png
            metrics_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from racing_rl.config.loader import load_config_for_mode
from racing_rl.evaluation.best_tracker import BestModelTracker
from racing_rl.plotting.track_plot import plot_comparison
from racing_rl.tracks.parametric import build_track
from racing_rl.training.trainer import train
from racing_rl.utils.path_utils import get_run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch A/B/C experiment runner")
    p.add_argument(
        "--modes", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"]
    )
    p.add_argument("--timesteps", "-t", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; only generate comparison plot from existing runs.",
    )
    return p.parse_args()


def load_best_trajectory(cfg, track) -> list:
    """Load the trajectory from the best model's PNG metadata — or re-evaluate."""
    from stable_baselines3 import SAC
    from racing_rl.evaluation.evaluator import run_evaluation
    from racing_rl.utils.path_utils import get_run_dir

    run_dir = get_run_dir(cfg)
    model_path = run_dir / "models" / "best_model.zip"
    if not model_path.exists():
        model_path = run_dir / "models" / "final_model.zip"
    if not model_path.exists():
        print(f"  No model found for mode {cfg.obs_mode}, skipping.")
        return []

    model = SAC.load(str(model_path))
    metrics, _ = run_evaluation(
        model=model,
        cfg=cfg,
        track=track,
        out_dir=run_dir / "eval_pngs",
        eval_index=9999,
        n_episodes=3,
        is_best=False,
    )
    # Re-run a single deterministic episode to get trajectory
    from racing_rl.env.racing_env import RacingEnv
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


def main() -> None:
    args = parse_args()

    # ---- Train ---------------------------------------------------------- #
    if not args.skip_train:
        for mode in args.modes:
            print(f"\n{'='*60}")
            print(f"  Training obs mode {mode}")
            print(f"{'='*60}")
            cfg = load_config_for_mode(mode)
            if args.timesteps:
                cfg.training.total_timesteps = args.timesteps
            if args.seed is not None:
                cfg.seed = args.seed
            train(cfg)

    # ---- Compare -------------------------------------------------------- #
    print(f"\n{'='*60}")
    print("  Generating comparison plot")
    print(f"{'='*60}")

    # Use mode B config for track (same track for all)
    ref_cfg = load_config_for_mode("B")
    track = build_track(ref_cfg.track.name, ref_cfg.track.half_width)

    trajectories = {}
    summary_rows = []

    for mode in args.modes:
        cfg = load_config_for_mode(mode)
        if args.seed is not None:
            cfg.seed = args.seed

        traj = load_best_trajectory(cfg, track)
        if traj:
            trajectories[f"Obs {mode}"] = traj

        # Load best metrics
        run_dir = get_run_dir(cfg)
        tracker = BestModelTracker(run_dir)
        best = tracker.best
        if best:
            summary_rows.append({
                "mode": mode,
                "lap_complete": best.lap_complete,
                "lap_time": best.lap_time,
                "completion_pct": best.completion_pct,
                "mean_speed": best.mean_speed,
                "max_slip": best.max_slip,
            })

    # Save comparison PNG
    comp_dir = Path(ref_cfg.output.base_dir)
    if not comp_dir.is_absolute():
        from racing_rl.utils.path_utils import _REPO_ROOT
        comp_dir = _REPO_ROOT / comp_dir
    comp_dir = comp_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    comp_png = comp_dir / "racing_line_comparison.png"
    plot_comparison(
        track=track,
        trajectories=trajectories,
        title=f"Racing Line Comparison — Obs A / B / C",
        out_path=comp_png,
    )
    print(f"Comparison PNG saved: {comp_png}")

    # Save metrics CSV
    if summary_rows:
        csv_path = comp_dir / "metrics_summary.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Metrics CSV saved:   {csv_path}")

        print(f"\n{'Obs':<6} {'Complete':<10} {'Lap time':>10} {'Completion':>12} {'Mean spd':>10} {'Max slip':>10}")
        print("-" * 62)
        for row in summary_rows:
            print(
                f"  {row['mode']:<4} {str(row['lap_complete']):<10} "
                f"{row['lap_time']:>10.2f} {row['completion_pct']:>11.1f}% "
                f"{row['mean_speed']:>9.1f} {row['max_slip']:>10.3f}"
            )


if __name__ == "__main__":
    main()
