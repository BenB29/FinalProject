"""
Best-model tracker: tracks the best evaluation result and controls promotion.

Promotion rules (in priority order)
-------------------------------------
1. A run that completes the lap beats one that does not — always.
2. Among completed-lap runs, LOWER lap time wins.
3. Among partial-lap (non-completing) runs, HIGHER completion percentage wins.
4. Tie-break: lower max_slip wins; then higher mean_speed wins.

Guarantee: once a model is recorded as BEST, it can only be replaced by a
strictly better model according to the rules above.

Persistence
-----------
The best metrics are written to ``best_metrics.json`` in the run output dir
so they survive restarts.  On init the tracker reads this file if it exists.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EvalMetrics:
    """Metrics collected during a single evaluation run."""

    eval_index: int
    lap_complete: bool
    lap_time: float          # total episode time [s]; meaningful only if lap_complete
    completion_pct: float    # [0, 100]
    mean_speed: float        # [m/s]
    max_slip: float          # peak slip ratio
    png_path: str = ""       # path to the saved PNG for this eval


class BestModelTracker:
    """
    Tracks the best model across evaluations and handles safe promotion.

    Parameters
    ----------
    run_dir:
        Root directory for this training run.  Best model and metrics are
        stored inside this directory.
    """

    BEST_MODEL_FILENAME = "best_model.zip"
    LATEST_MODEL_FILENAME = "latest_model.zip"
    BEST_METRICS_FILENAME = "best_metrics.json"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._best: Optional[EvalMetrics] = None
        self._eval_history: list[EvalMetrics] = []

        # Load persisted best metrics from a previous run, if any
        self._load_best_metrics()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def best(self) -> Optional[EvalMetrics]:
        return self._best

    @property
    def best_model_path(self) -> Path:
        return self.run_dir / self.BEST_MODEL_FILENAME

    @property
    def latest_model_path(self) -> Path:
        return self.run_dir / self.LATEST_MODEL_FILENAME

    def evaluate_and_maybe_promote(
        self,
        metrics: EvalMetrics,
        model_save_path: Path,
    ) -> bool:
        """
        Compare *metrics* against current best.  If better, promote to best.

        Parameters
        ----------
        metrics:
            Metrics from the latest evaluation.
        model_save_path:
            Path to the model checkpoint file to promote (if better).
            This is copied to ``best_model.zip`` in the run directory.

        Returns
        -------
        promoted:
            True if this eval is the new best; False otherwise.
        """
        self._eval_history.append(metrics)

        is_better = self._is_better(metrics, self._best)
        if is_better:
            self._promote(metrics, model_save_path)

        return is_better

    def record_eval(self, metrics: EvalMetrics) -> None:
        """Record an eval without triggering model promotion (no model file)."""
        self._eval_history.append(metrics)

    def refresh_best_metrics(self) -> None:
        """Re-save best_metrics.json with the current state of self._best.

        Call this after externally updating metrics fields (e.g. png_path)
        so the persisted JSON stays in sync.
        """
        if self._best is not None:
            self._save_best_metrics()

    # ------------------------------------------------------------------ #
    #  Comparison logic                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_better(
        candidate: EvalMetrics,
        current_best: Optional[EvalMetrics],
    ) -> bool:
        """Return True iff *candidate* is strictly better than *current_best*."""
        if current_best is None:
            return True

        # Rule 1: lap completion trumps partial lap
        if candidate.lap_complete and not current_best.lap_complete:
            return True
        if not candidate.lap_complete and current_best.lap_complete:
            return False

        # Rule 2: both complete → lower lap time
        if candidate.lap_complete and current_best.lap_complete:
            if candidate.lap_time < current_best.lap_time:
                return True
            if candidate.lap_time > current_best.lap_time:
                return False
            # Tie-break: lower max slip
            if candidate.max_slip < current_best.max_slip:
                return True
            return False

        # Rule 3: both partial → higher completion pct
        if candidate.completion_pct > current_best.completion_pct:
            return True
        if candidate.completion_pct < current_best.completion_pct:
            return False

        # Tie-break: lower max slip
        if candidate.max_slip < current_best.max_slip:
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _promote(self, metrics: EvalMetrics, model_src: Path) -> None:
        """Copy model file to best_model.zip and persist metrics."""
        import time
        model_src = Path(model_src)
        if model_src.exists():
            dest = self.best_model_path
            # Retry — OneDrive can briefly lock files during sync
            for attempt in range(5):
                try:
                    shutil.copy2(model_src, dest)
                    break
                except (PermissionError, OSError):
                    if attempt < 4:
                        time.sleep(0.5)
                    else:
                        raise

        self._best = metrics
        self._save_best_metrics()

    def _save_best_metrics(self) -> None:
        path = self.run_dir / self.BEST_METRICS_FILENAME
        with path.open("w") as fh:
            json.dump(asdict(self._best), fh, indent=2)

    def _load_best_metrics(self) -> None:
        path = self.run_dir / self.BEST_METRICS_FILENAME
        if not path.exists():
            return
        try:
            with path.open("r") as fh:
                d = json.load(fh)
            self._best = EvalMetrics(**d)
        except Exception:
            # Corrupted file — ignore and start fresh
            self._best = None

    def save_eval_history(self) -> None:
        """Write full evaluation history to CSV."""
        import csv
        path = self.run_dir / "eval_history.csv"
        if not self._eval_history:
            return
        fieldnames = list(asdict(self._eval_history[0]).keys())
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for m in self._eval_history:
                writer.writerow(asdict(m))
