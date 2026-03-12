"""Tests for vehicle physics."""
from __future__ import annotations

import math
import numpy as np
import pytest

from racing_rl.config.schema import VehicleConfig
from racing_rl.physics.vehicle import VehiclePhysics, VehicleState


@pytest.fixture
def physics():
    return VehiclePhysics(VehicleConfig())


@pytest.fixture
def straight_state():
    """Vehicle at origin, heading east, zero speed."""
    return VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.0)


@pytest.fixture
def moving_state():
    """Vehicle at origin, heading east, speed 20 m/s."""
    return VehicleState(x=0.0, y=0.0, heading=0.0, speed=20.0)


# ---- Action application ---------------------------------------------------

def test_throttle_increases_speed(physics, straight_state):
    action = np.array([0.0, 1.0])   # full throttle, no steering
    ns, _ = physics.step(straight_state, action)
    assert ns.speed > straight_state.speed


def test_brake_decreases_speed(physics, moving_state):
    action = np.array([0.0, -1.0])  # full brake
    ns, _ = physics.step(moving_state, action)
    assert ns.speed < moving_state.speed


def test_speed_does_not_exceed_max(physics):
    cfg = VehicleConfig()
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=cfg.max_speed - 0.1)
    action = np.array([0.0, 1.0])
    for _ in range(20):
        state, _ = physics.step(state, action)
    assert state.speed <= cfg.max_speed + 0.01


def test_speed_never_goes_negative(physics):
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=0.0)
    action = np.array([0.0, -1.0])  # brake when already stopped
    ns, _ = physics.step(state, action)
    assert ns.speed >= 0.0


# ---- Steering & heading ---------------------------------------------------

def test_steering_rate_limit(physics):
    """Steering angle should not jump by more than max_rate * dt in one step."""
    cfg = physics.cfg
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=20.0)
    action = np.array([1.0, 0.5])   # full steering cmd
    ns, _ = physics.step(state, action)
    max_delta = cfg.max_steering_rate * cfg.dt
    assert abs(ns.steering_angle - state.steering_angle) <= max_delta + 1e-9


def test_steering_clamped_to_max(physics):
    cfg = physics.cfg
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0,
                         steering_angle=cfg.max_steering_angle - 0.001)
    # Step many times with full left command
    for _ in range(50):
        state, _ = physics.step(state, np.array([1.0, 0.3]))
    assert state.steering_angle <= cfg.max_steering_angle + 1e-6


def test_right_steer_turns_right(physics):
    """Full right steering cmd should result in rightward heading change."""
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=20.0)
    action = np.array([-1.0, 0.3])   # right = negative in our sign convention
    for _ in range(10):
        state, _ = physics.step(state, action)
    # Heading should decrease (right turn when going east)
    assert state.heading < 0.0, f"Expected negative heading, got {state.heading}"


def test_left_steer_turns_left(physics):
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=20.0)
    action = np.array([1.0, 0.3])    # left = positive
    for _ in range(10):
        state, _ = physics.step(state, action)
    assert state.heading > 0.0


# ---- Traction circle / grip -----------------------------------------------

def test_slip_ratio_positive(physics, moving_state):
    """Slip ratio should always be >= 0."""
    action = np.array([0.5, 0.8])
    _, slip = physics.step(moving_state, action)
    assert slip >= 0.0


def test_high_combined_demand_does_not_crash(physics):
    """Extreme simultaneous braking and steering should not raise."""
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=60.0)
    action = np.array([1.0, -1.0])   # full steer + full brake
    ns, slip = physics.step(state, action)
    assert math.isfinite(ns.speed)
    assert math.isfinite(slip)


# ---- Position update ------------------------------------------------------

def test_straight_ahead_moves_in_x(physics):
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=20.0)
    action = np.array([0.0, 0.3])
    ns, _ = physics.step(state, action)
    assert ns.x > 0.0
    assert abs(ns.y) < 0.1


def test_distance_increases(physics):
    state = VehicleState(x=0.0, y=0.0, heading=0.0, speed=20.0)
    action = np.array([0.0, 0.5])
    ns, _ = physics.step(state, action)
    assert ns.distance_travelled > state.distance_travelled


def test_elapsed_time_increments(physics):
    state = VehicleState()
    action = np.array([0.0, 0.0])
    ns, _ = physics.step(state, action)
    assert abs(ns.elapsed_time - physics.cfg.dt) < 1e-9
