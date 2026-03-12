"""Tests for the best-model tracker and promotion logic."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from racing_rl.evaluation.best_tracker import BestModelTracker, EvalMetrics


def _metrics(
    eval_index=1,
    lap_complete=False,
    lap_time=100.0,
    completion_pct=50.0,
    mean_speed=20.0,
    max_slip=0.5,
) -> EvalMetrics:
    return EvalMetrics(
        eval_index=eval_index,
        lap_complete=lap_complete,
        lap_time=lap_time,
        completion_pct=completion_pct,
        mean_speed=mean_speed,
        max_slip=max_slip,
    )


@pytest.fixture
def tmp_run_dir(tmp_path):
    return tmp_path / "run"


# ---- Initial state --------------------------------------------------------

def test_initial_best_is_none(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    assert tracker.best is None


def test_first_eval_always_promoted(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    m = _metrics(eval_index=1, lap_complete=False, completion_pct=30.0)
    promoted = tracker.evaluate_and_maybe_promote(m, model_save_path=Path("/nonexistent"))
    assert promoted
    assert tracker.best is not None
    assert tracker.best.eval_index == 1


# ---- Promotion rules ------------------------------------------------------

def test_completed_lap_beats_partial(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    # Establish a partial-lap best
    partial = _metrics(eval_index=1, lap_complete=False, completion_pct=80.0)
    tracker.evaluate_and_maybe_promote(partial, Path("/nonexistent"))
    # A completed lap should beat it regardless
    complete = _metrics(eval_index=2, lap_complete=True, lap_time=120.0, completion_pct=100.0)
    promoted = tracker.evaluate_and_maybe_promote(complete, Path("/nonexistent"))
    assert promoted
    assert tracker.best.lap_complete is True


def test_partial_does_not_beat_completed(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    complete = _metrics(eval_index=1, lap_complete=True, lap_time=100.0, completion_pct=100.0)
    tracker.evaluate_and_maybe_promote(complete, Path("/nonexistent"))
    # Even a very high completion partial should NOT beat completed
    partial = _metrics(eval_index=2, lap_complete=False, completion_pct=99.0)
    promoted = tracker.evaluate_and_maybe_promote(partial, Path("/nonexistent"))
    assert not promoted


def test_lower_lap_time_beats_higher(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    slow = _metrics(eval_index=1, lap_complete=True, lap_time=120.0)
    tracker.evaluate_and_maybe_promote(slow, Path("/nonexistent"))
    fast = _metrics(eval_index=2, lap_complete=True, lap_time=90.0)
    promoted = tracker.evaluate_and_maybe_promote(fast, Path("/nonexistent"))
    assert promoted
    assert tracker.best.lap_time == pytest.approx(90.0)


def test_higher_lap_time_not_promoted(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    fast = _metrics(eval_index=1, lap_complete=True, lap_time=90.0)
    tracker.evaluate_and_maybe_promote(fast, Path("/nonexistent"))
    slow = _metrics(eval_index=2, lap_complete=True, lap_time=100.0)
    promoted = tracker.evaluate_and_maybe_promote(slow, Path("/nonexistent"))
    assert not promoted
    assert tracker.best.lap_time == pytest.approx(90.0)


def test_higher_completion_pct_promoted_when_both_partial(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    low = _metrics(eval_index=1, lap_complete=False, completion_pct=40.0)
    tracker.evaluate_and_maybe_promote(low, Path("/nonexistent"))
    high = _metrics(eval_index=2, lap_complete=False, completion_pct=70.0)
    promoted = tracker.evaluate_and_maybe_promote(high, Path("/nonexistent"))
    assert promoted


def test_tie_break_by_lower_max_slip(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    m1 = _metrics(eval_index=1, lap_complete=True, lap_time=100.0, max_slip=1.5)
    tracker.evaluate_and_maybe_promote(m1, Path("/nonexistent"))
    m2 = _metrics(eval_index=2, lap_complete=True, lap_time=100.0, max_slip=0.8)
    promoted = tracker.evaluate_and_maybe_promote(m2, Path("/nonexistent"))
    assert promoted


# ---- Persistence ----------------------------------------------------------

def test_best_metrics_persisted_to_json(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    m = _metrics(eval_index=5, lap_complete=True, lap_time=88.5)
    tracker.evaluate_and_maybe_promote(m, Path("/nonexistent"))
    json_path = tmp_run_dir / "best_metrics.json"
    assert json_path.exists()
    with json_path.open() as f:
        data = json.load(f)
    assert data["lap_time"] == pytest.approx(88.5)
    assert data["eval_index"] == 5


def test_best_metrics_loaded_on_restart(tmp_run_dir):
    # First tracker session
    t1 = BestModelTracker(tmp_run_dir)
    m = _metrics(eval_index=3, lap_complete=True, lap_time=95.0)
    t1.evaluate_and_maybe_promote(m, Path("/nonexistent"))

    # Second tracker (simulates restart)
    t2 = BestModelTracker(tmp_run_dir)
    assert t2.best is not None
    assert t2.best.eval_index == 3
    assert t2.best.lap_time == pytest.approx(95.0)


def test_worse_model_does_not_overwrite_best(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    good = _metrics(eval_index=1, lap_complete=True, lap_time=90.0)
    tracker.evaluate_and_maybe_promote(good, Path("/nonexistent"))
    bad = _metrics(eval_index=2, lap_complete=True, lap_time=120.0)
    tracker.evaluate_and_maybe_promote(bad, Path("/nonexistent"))
    assert tracker.best.lap_time == pytest.approx(90.0)


# ---- Eval history ---------------------------------------------------------

def test_eval_history_saved_as_csv(tmp_run_dir):
    tracker = BestModelTracker(tmp_run_dir)
    for i in range(3):
        m = _metrics(eval_index=i + 1)
        tracker.record_eval(m)
    tracker.save_eval_history()
    csv_path = tmp_run_dir / "eval_history.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "eval_index" in content
