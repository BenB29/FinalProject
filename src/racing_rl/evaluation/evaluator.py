"""
Deterministic evaluation runner.

Runs N episodes with the policy in deterministic mode, averages key metrics,
saves a PNG, and returns an EvalMetrics object.

Why deterministic evaluation?
------------------------------
SAC is stochastic during training (entropy term).  For fair comparison and
stable best-model selection, we use the *mean* of the actor output (no
sampling noise) during evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from racing_rl.config.schema import RacingConfig
from racing_rl.env.racing_env import RacingEnv
from racing_rl.evaluation.best_tracker import EvalMetrics
from racing_rl.plotting.track_plot import save_eval_png
from racing_rl.tracks.base import BaseTrack


def run_evaluation(
    model,                          # SB3 model with .predict()
    cfg: RacingConfig,
    track: BaseTrack,
    out_dir: Path,
    eval_index: int,
    n_episodes: int = 3,
    is_best: bool = False,
) -> Tuple[EvalMetrics, Path]:
    """
    Run deterministic evaluation and save PNG.

    Parameters
    ----------
    model:
        Trained SB3 SAC model.
    cfg:
        Full racing config (physics / obs mode / etc.).
    track:
        Pre-built track object (shared with training env).
    out_dir:
        Directory for PNG output.
    eval_index:
        Integer counter for this evaluation (used in filenames).
    n_episodes:
        Number of full episodes to average.  The best-trajectory episode
        (by lap progress, then lap time) is used for the PNG.
    is_best:
        If True, '_BEST' is appended to the PNG filename.

    Returns
    -------
    (EvalMetrics, png_path, best_trajectory)
    """
    env = RacingEnv(cfg, eval_mode=True)

    episode_results = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)   # deterministic seed per episode
        done = False
        traj: List[Tuple[float, float, float]] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if env.eval_mode:
                # trajectory is appended inside env already
                pass
            done = terminated or truncated

        traj = env.get_trajectory()
        lap_complete = info.get("lap_complete", False)
        lap_time = info.get("episode_lap_time", info.get("lap_time", 0.0))
        completion_pct = info.get("episode_lap_progress", 0.0) * 100.0
        speeds = [p[2] for p in traj] if traj else [0.0]
        mean_speed = float(np.mean(speeds))
        max_slip = info.get("episode_max_slip", 0.0)

        episode_results.append({
            "lap_complete": lap_complete,
            "lap_time": lap_time,
            "completion_pct": completion_pct,
            "mean_speed": mean_speed,
            "max_slip": max_slip,
            "trajectory": traj,
        })

    # Select the representative episode:
    # prefer completed lap (lowest lap time), else highest completion pct
    def _sort_key(r):
        completed = 1 if r["lap_complete"] else 0
        # For completed: minimise time; for partial: maximise completion
        return (completed, -r["lap_time"] if r["lap_complete"] else r["completion_pct"])

    best_ep = max(episode_results, key=_sort_key)

    # Average numeric metrics across all episodes
    avg_lap_time = float(np.mean([r["lap_time"] for r in episode_results]))
    avg_completion = float(np.mean([r["completion_pct"] for r in episode_results]))
    avg_mean_speed = float(np.mean([r["mean_speed"] for r in episode_results]))
    avg_max_slip = float(np.mean([r["max_slip"] for r in episode_results]))
    any_complete = any(r["lap_complete"] for r in episode_results)

    # Save PNG using the best episode's trajectory
    png_path = save_eval_png(
        track=track,
        trajectory=best_ep["trajectory"],
        out_dir=out_dir,
        eval_index=eval_index,
        lap_time=avg_lap_time,
        completion_pct=avg_completion,
        is_best=is_best,
        mean_speed=avg_mean_speed,
        max_slip=avg_max_slip,
        obs_mode=cfg.obs_mode,
        speed_max=cfg.obs.speed_max,
    )

    metrics = EvalMetrics(
        eval_index=eval_index,
        lap_complete=any_complete,
        lap_time=avg_lap_time,
        completion_pct=avg_completion,
        mean_speed=avg_mean_speed,
        max_slip=avg_max_slip,
        png_path=str(png_path),
    )

    env.close()
    return metrics, png_path, best_ep["trajectory"]
