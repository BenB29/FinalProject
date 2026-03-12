"""
Output-path helpers.

Run directory layout:
    outputs/
        {experiment_name}/
            models/
                latest_model.zip
                best_model.zip
                final_model.zip
            eval_pngs/
                eval_0001_laptime_*.png
                ...
            best_metrics.json
            eval_history.csv
"""
from __future__ import annotations

from pathlib import Path

from racing_rl.config.schema import RacingConfig


_REPO_ROOT = Path(__file__).resolve().parents[3]   # src/racing_rl/utils/path_utils.py -> root


def make_run_dir(cfg: RacingConfig) -> Path:
    """
    Create and return the output directory for a training run.

    Uses ``cfg.output.base_dir / cfg.experiment_name``.
    The path is created if it does not exist.
    """
    base = Path(cfg.output.base_dir)
    if not base.is_absolute():
        base = _REPO_ROOT / base
    run_dir = base / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_run_dir(cfg: RacingConfig) -> Path:
    """Return the run directory without creating it."""
    base = Path(cfg.output.base_dir)
    if not base.is_absolute():
        base = _REPO_ROOT / base
    return base / cfg.experiment_name
