"""Tests for the observation builder."""
from __future__ import annotations

import math
import numpy as np
import pytest

from racing_rl.config.schema import ObsNormConfig, TrackConfig
from racing_rl.observations.builder import ObservationBuilder, _wrap_angle
from racing_rl.physics.vehicle import VehicleState
from racing_rl.tracks.base import TrackQuery


def _make_tq(
    lateral_offset=0.0,
    track_heading=0.0,
    n_lookahead=5,
) -> TrackQuery:
    return TrackQuery(
        s=0.0,
        lateral_offset=lateral_offset,
        track_heading=track_heading,
        curvature=0.0,
        lookahead_curvatures=np.zeros(n_lookahead),
        dist_to_left=5.0,
        dist_to_right=5.0,
        cx=0.0,
        cy=0.0,
    )


def _make_state(speed=20.0, heading=0.0) -> VehicleState:
    return VehicleState(speed=speed, heading=heading)


def _make_builder(mode: str, n_lookahead: int = 5) -> ObservationBuilder:
    norm = ObsNormConfig(speed_max=80.0, offset_max=5.0, curvature_scale=50.0)
    track_cfg = TrackConfig(
        lookahead_distances=[10.0, 25.0, 50.0, 80.0, 120.0][:n_lookahead]
    )
    return ObservationBuilder(mode, norm, track_cfg)


# ---- Dimension tests ------------------------------------------------------

def test_obs_dim_A():
    b = _make_builder("A")
    assert b.obs_dim == 2


def test_obs_dim_B():
    b = _make_builder("B")
    assert b.obs_dim == 4


def test_obs_dim_C():
    b = _make_builder("C", n_lookahead=5)
    assert b.obs_dim == 9   # 4 + 5


def test_obs_array_shape_A():
    b = _make_builder("A")
    obs = b.build(_make_state(), _make_tq())
    assert obs.shape == (2,)
    assert obs.dtype == np.float32


def test_obs_array_shape_B():
    b = _make_builder("B")
    obs = b.build(_make_state(), _make_tq())
    assert obs.shape == (4,)


def test_obs_array_shape_C():
    b = _make_builder("C")
    obs = b.build(_make_state(), _make_tq(n_lookahead=5))
    assert obs.shape == (9,)


# ---- Value tests ----------------------------------------------------------

def test_speed_normalised_correctly():
    b = _make_builder("A")
    obs = b.build(_make_state(speed=40.0), _make_tq())
    # speed_norm = 40 / 80 = 0.5
    assert abs(obs[0] - 0.5) < 1e-5


def test_zero_speed_gives_zero_norm():
    b = _make_builder("A")
    obs = b.build(_make_state(speed=0.0), _make_tq())
    assert abs(obs[0]) < 1e-6


def test_offset_normalised_correctly():
    b = _make_builder("A")
    obs = b.build(_make_state(), _make_tq(lateral_offset=2.5))
    # offset_norm = 2.5 / 5.0 = 0.5
    assert abs(obs[1] - 0.5) < 1e-5


def test_zero_heading_error_gives_sin0_cos1(track):
    """When vehicle heading == track heading, sin=0, cos=1."""
    b = _make_builder("B")
    # Both heading and track heading = 0
    state = _make_state(heading=0.0)
    tq = _make_tq(track_heading=0.0)
    obs = b.build(state, tq)
    assert abs(obs[2]) < 1e-5    # sin(0) == 0
    assert abs(obs[3] - 1.0) < 1e-5  # cos(0) == 1


def test_pi_heading_error_gives_sin0_cosNeg1():
    b = _make_builder("B")
    state = _make_state(heading=math.pi)
    tq = _make_tq(track_heading=0.0)
    obs = b.build(state, tq)
    assert abs(obs[2]) < 1e-5    # sin(pi) ~= 0
    assert abs(obs[3] + 1.0) < 1e-5  # cos(pi) == -1


def test_lookahead_curvatures_present_in_C():
    b = _make_builder("C", n_lookahead=3)
    norm = ObsNormConfig(curvature_scale=50.0)
    kappas = np.array([0.01, -0.02, 0.005])
    tq = TrackQuery(
        s=0.0,
        lateral_offset=0.0,
        track_heading=0.0,
        curvature=0.0,
        lookahead_curvatures=kappas,
        dist_to_left=5.0,
        dist_to_right=5.0,
        cx=0.0,
        cy=0.0,
    )
    obs = b.build(_make_state(), tq)
    # obs[4:7] should be kappas * 50
    expected = kappas * 50.0
    np.testing.assert_allclose(obs[4:7], expected, atol=1e-5)


# ---- Wrap-angle utility ---------------------------------------------------

def test_wrap_angle_zero():
    assert _wrap_angle(0.0) == pytest.approx(0.0)


def test_wrap_angle_pi():
    # pi wraps to -pi (convention: (-pi, pi])
    w = _wrap_angle(math.pi)
    assert abs(abs(w) - math.pi) < 1e-9


def test_wrap_angle_large_positive():
    w = _wrap_angle(3 * math.pi)
    # 3π wraps to ±π — just ensure it lands in [-pi, pi]
    assert -math.pi <= w <= math.pi


def test_wrap_angle_negative():
    w = _wrap_angle(-math.pi / 2)
    assert abs(w - (-math.pi / 2)) < 1e-9
