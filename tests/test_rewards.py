"""Tests for the reward calculator."""
from __future__ import annotations

import numpy as np
import pytest

from racing_rl.config.schema import RewardConfig
from racing_rl.physics.vehicle import VehicleState
from racing_rl.rewards.reward import RewardCalculator
from racing_rl.tracks.base import TrackQuery


def _make_rc() -> RewardCalculator:
    return RewardCalculator(RewardConfig())


def _make_state(speed=20.0, slip_ratio=0.0) -> VehicleState:
    s = VehicleState(speed=speed)
    s.slip_ratio = slip_ratio
    return s


def _make_tq() -> TrackQuery:
    return TrackQuery(
        s=10.0,
        lateral_offset=0.0,
        track_heading=0.0,
        curvature=0.0,
        lookahead_curvatures=np.zeros(3),
        dist_to_left=5.0,
        dist_to_right=5.0,
        cx=0.0,
        cy=0.0,
    )


def _zero_action() -> np.ndarray:
    return np.array([0.0, 0.0])


# ---- Progress reward ------------------------------------------------------

def test_positive_progress_gives_positive_reward():
    rc = _make_rc()
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=5.0, action=_zero_action(), action_prev=_zero_action())
    assert bd.progress > 0.0


def test_zero_progress_gives_zero_progress_reward():
    rc = _make_rc()
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=0.0, action=_zero_action(), action_prev=_zero_action())
    assert bd.progress == pytest.approx(0.0)


def test_negative_progress_gives_zero_progress_reward():
    """Reversing should not give a negative progress reward (just zero)."""
    rc = _make_rc()
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=-3.0, action=_zero_action(), action_prev=_zero_action())
    assert bd.progress == pytest.approx(0.0)


# ---- Slip penalty ---------------------------------------------------------

def test_no_slip_penalty_below_threshold():
    rc = _make_rc()
    # slip_ratio = 0.5, well below SLIP_THRESHOLD = 0.85
    bd = rc.compute(_make_state(slip_ratio=0.5), _make_state(slip_ratio=0.5),
                    _make_tq(), delta_s=1.0, action=_zero_action(),
                    action_prev=_zero_action())
    assert bd.slip_penalty == pytest.approx(0.0)


def test_high_slip_gives_negative_penalty():
    rc = _make_rc()
    bd = rc.compute(_make_state(slip_ratio=2.0), _make_state(slip_ratio=2.0),
                    _make_tq(), delta_s=1.0, action=_zero_action(),
                    action_prev=_zero_action())
    assert bd.slip_penalty < 0.0


# ---- Steering smoothness --------------------------------------------------

def test_steering_change_gives_penalty():
    rc = _make_rc()
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=1.0,
                    action=np.array([1.0, 0.0]),
                    action_prev=np.array([0.0, 0.0]))
    assert bd.steer_smooth_penalty < 0.0


def test_zero_steering_change_no_smooth_penalty():
    rc = _make_rc()
    a = np.array([0.5, 0.0])
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=1.0, action=a, action_prev=a)
    assert bd.steer_smooth_penalty == pytest.approx(0.0)


# ---- Event rewards --------------------------------------------------------

def test_off_track_penalty_applied():
    rc = _make_rc()
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=0.0, action=_zero_action(), action_prev=_zero_action(),
                    off_track=True)
    assert bd.event == pytest.approx(rc.cfg.off_track_penalty)


def test_lap_completion_bonus():
    rc = _make_rc()
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=1.0, action=_zero_action(), action_prev=_zero_action(),
                    lap_complete=True)
    assert bd.event == pytest.approx(rc.cfg.lap_completion_bonus)


def test_off_track_takes_priority_over_lap_complete():
    """Both flags shouldn't happen, but if they did, off_track fires first."""
    rc = _make_rc()
    bd = rc.compute(_make_state(), _make_state(), _make_tq(),
                    delta_s=0.0, action=_zero_action(), action_prev=_zero_action(),
                    off_track=True, lap_complete=True)
    # off_track is checked first in the code
    assert bd.event == pytest.approx(rc.cfg.off_track_penalty)


# ---- Total is sum of parts ------------------------------------------------

def test_total_equals_sum_of_components():
    rc = _make_rc()
    bd = rc.compute(_make_state(slip_ratio=1.5), _make_state(slip_ratio=1.5),
                    _make_tq(), delta_s=3.0,
                    action=np.array([0.3, 0.5]),
                    action_prev=np.array([-0.2, 0.2]))
    expected = (
        bd.progress + bd.speed_bonus + bd.slip_penalty
        + bd.steer_smooth_penalty + bd.accel_smooth_penalty + bd.event
    )
    assert abs(bd.total - expected) < 1e-9
