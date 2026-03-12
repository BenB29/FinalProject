"""Tests for the track system."""
from __future__ import annotations

import math
import numpy as np
import pytest

from racing_rl.tracks.parametric import build_circuit_lite


def test_track_builds_without_error():
    track = build_circuit_lite()
    assert track is not None


def test_track_length_positive(track):
    assert track.length > 0


def test_track_length_reasonable(track):
    # Our circuit should be between 300 and 2000 m
    assert 300 < track.length < 2000, f"Unexpected length: {track.length:.1f} m"


def test_centreline_array_shape(track):
    cl = track.centreline_xy
    assert cl.ndim == 2
    assert cl.shape[1] == 2
    assert cl.shape[0] > 100


def test_boundary_shapes_match(track):
    lb = track.left_boundary_xy
    rb = track.right_boundary_xy
    cl = track.centreline_xy
    assert lb.shape == cl.shape
    assert rb.shape == cl.shape


def test_boundaries_offset_from_centreline(track):
    """Left and right boundary points should be ~half_width from centreline."""
    cl = track.centreline_xy
    lb = track.left_boundary_xy
    rb = track.right_boundary_xy
    # Sample a few points
    for i in range(0, len(cl), len(cl) // 10):
        dl = np.linalg.norm(lb[i] - cl[i])
        dr = np.linalg.norm(rb[i] - cl[i])
        assert abs(dl - track.half_width) < 0.5, f"Left boundary error at i={i}: {dl:.2f}"
        assert abs(dr - track.half_width) < 0.5, f"Right boundary error at i={i}: {dr:.2f}"


def test_start_position_returns_triple(track):
    result = track.start_position()
    assert len(result) == 3
    x, y, h = result
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert isinstance(h, float)


def test_query_on_centreline(track):
    """Querying a point ON the centreline should give near-zero lateral offset."""
    cl = track.centreline_xy
    mid_idx = len(cl) // 2
    x, y = cl[mid_idx]
    tq = track.query(x, y, [10.0, 20.0])
    assert abs(tq.lateral_offset) < 0.5, f"Expected ~0 offset, got {tq.lateral_offset:.3f}"


def test_query_offset_sign(track):
    """
    Querying a point displaced laterally from centreline should give
    the correct sign for lateral_offset.
    """
    cl = track.centreline_xy
    lb = track.left_boundary_xy
    rb = track.right_boundary_xy
    # Pick a sample away from the ends
    i = len(cl) // 3
    cx, cy = cl[i]
    lx, ly = lb[i]  # left of travel: expect positive offset

    tq_l = track.query(float(lx), float(ly), [10.0])
    tq_r = track.query(float(rb[i, 0]), float(rb[i, 1]), [10.0])

    assert tq_l.lateral_offset > 0, f"Left of centreline should be positive, got {tq_l.lateral_offset}"
    assert tq_r.lateral_offset < 0, f"Right of centreline should be negative, got {tq_r.lateral_offset}"


def test_is_off_track_centreline_point(track):
    """A point on the centreline should NOT be off-track."""
    cl = track.centreline_xy
    x, y = float(cl[0, 0]), float(cl[0, 1])
    assert not track.is_off_track(x, y)


def test_is_off_track_far_outside(track):
    """A point far from the track should be off-track."""
    x, y = 99999.0, 99999.0
    assert track.is_off_track(x, y)


def test_lookahead_curvatures_shape(track):
    cl = track.centreline_xy
    x, y = float(cl[0, 0]), float(cl[0, 1])
    distances = [10.0, 25.0, 50.0]
    tq = track.query(x, y, distances)
    assert len(tq.lookahead_curvatures) == 3


def test_heading_at_start_is_finite(track):
    _, _, h = track.start_position()
    assert math.isfinite(h)


def test_query_arc_length_in_range(track):
    cl = track.centreline_xy
    for i in [0, len(cl) // 4, len(cl) // 2]:
        x, y = float(cl[i, 0]), float(cl[i, 1])
        tq = track.query(x, y, [10.0])
        assert 0.0 <= tq.s <= track.length + 1.0, f"s={tq.s} out of range"
