"""
Custom SB3 callback for periodic evaluation with:
  - deterministic episode rollout
  - best-model promotion
  - automatic PNG export
  - CSV logging
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from racing_rl.config.schema import RacingConfig
from racing_rl.evaluation.best_tracker import BestModelTracker, EvalMetrics
from racing_rl.evaluation.evaluator import run_evaluation
from racing_rl.tracks.base import BaseTrack
from racing_rl.utils.logging_utils import get_logger

log = get_logger(__name__)


class RacingEvalCallback(BaseCallback):
    """
    Periodic evaluation callback integrated with the best-model tracker.

    Parameters
    ----------
    cfg:
        Full racing config.
    track:
        Pre-built track object (same instance used in training env).
    run_dir:
        Root directory for this run's outputs (models + PNGs + logs).
    eval_freq:
        Evaluate every *eval_freq* environment steps.
    n_eval_episodes:
        Episodes per evaluation.
    verbose:
        SB3 verbosity level.
    """

    def __init__(
        self,
        cfg: RacingConfig,
        track: BaseTrack,
        run_dir: Path,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 3,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.track = track
        self.run_dir = Path(run_dir)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._eval_index = 0
        self._png_dir = self.run_dir / "eval_pngs"
        self._models_dir = self.run_dir / "models"
        self._png_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = BestModelTracker(self.run_dir)

    # ------------------------------------------------------------------ #
    #  BaseCallback interface                                              #
    # ------------------------------------------------------------------ #

    def _init_callback(self) -> None:
        super()._init_callback()
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq:
            self._last_eval_timestep = self.num_timesteps
            self._run_eval()
        return True   # returning False would stop training early

    def _on_training_end(self) -> None:
        """Run a final evaluation at the end of training."""
        log.info("Training ended — running final evaluation.")
        self._run_eval()
        self.tracker.save_eval_history()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _run_eval(self) -> None:
        self._eval_index += 1
        idx = self._eval_index

        # Save latest model checkpoint
        latest_path = self._models_dir / "latest_model"
        self.model.save(str(latest_path))

        # ---- Preliminary eval run without best label -------------------- #
        metrics, png_path, best_traj = run_evaluation(
            model=self.model,
            cfg=self.cfg,
            track=self.track,
            out_dir=self._png_dir,
            eval_index=idx,
            n_episodes=self.n_eval_episodes,
            is_best=False,   # tentative — will re-save if promoted
        )

        # ---- Promotion check -------------------------------------------- #
        promoted = self.tracker.evaluate_and_maybe_promote(
            metrics=metrics,
            model_save_path=Path(str(latest_path) + ".zip"),
        )

        if promoted:
            # Re-save PNG with BEST tag, then sync the path in metrics + JSON
            _rename_to_best(png_path)
            metrics.png_path = str(png_path).replace(".png", "_BEST.png")
            self.tracker.refresh_best_metrics()   # persist updated png_path
            log.info(
                "★ NEW BEST  eval=%04d  lap_complete=%s  "
                "lap_time=%.2fs  completion=%.1f%%",
                idx,
                metrics.lap_complete,
                metrics.lap_time,
                metrics.completion_pct,
            )

        else:
            log.info(
                "  eval=%04d  lap_complete=%s  "
                "lap_time=%.2fs  completion=%.1f%%",
                idx,
                metrics.lap_complete,
                metrics.lap_time,
                metrics.completion_pct,
            )

        # Always log to tensorboard-style scalars if available
        if self.logger is not None:
            self.logger.record("eval/lap_complete", int(metrics.lap_complete))
            self.logger.record("eval/lap_time", metrics.lap_time)
            self.logger.record("eval/completion_pct", metrics.completion_pct)
            self.logger.record("eval/mean_speed", metrics.mean_speed)
            self.logger.record("eval/max_slip", metrics.max_slip)
            self.logger.dump(self.num_timesteps)


def _rename_to_best(original: Path) -> None:
    """Rename eval PNG file to include _BEST suffix."""
    original = Path(original)
    if not original.exists():
        return
    new_name = original.stem + "_BEST" + original.suffix
    best_path = original.parent / new_name
    # Remove old BEST file for previous best if needed
    for old in original.parent.glob("*_BEST.png"):
        if old != best_path:
            old.rename(str(old).replace("_BEST.png", "_demoted.png"))
    original.rename(best_path)
