"""Tests for the Gymnasium environment."""
from __future__ import annotations

import numpy as np
import pytest

from racing_rl.config.loader import load_config_for_mode
from racing_rl.env.racing_env import RacingEnv


@pytest.fixture(params=["A", "B", "C"])
def env(request):
    cfg = load_config_for_mode(request.param)
    # Short episodes for speed
    cfg.env.max_steps = 200
    cfg.env.no_progress_steps = 50
    e = RacingEnv(cfg, eval_mode=False)
    yield e
    e.close()


@pytest.fixture
def env_b():
    cfg = load_config_for_mode("B")
    cfg.env.max_steps = 200
    e = RacingEnv(cfg, eval_mode=False)
    yield e
    e.close()


# ---- Space / shape checks -------------------------------------------------

def test_observation_space_shape(env):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32


def test_action_space_bounds(env):
    low, high = env.action_space.low, env.action_space.high
    np.testing.assert_array_equal(low, [-1.0, -1.0])
    np.testing.assert_array_equal(high, [1.0, 1.0])


def test_obs_mode_A_dim():
    cfg = load_config_for_mode("A")
    e = RacingEnv(cfg)
    obs, _ = e.reset()
    assert obs.shape == (2,)
    e.close()


def test_obs_mode_B_dim():
    cfg = load_config_for_mode("B")
    e = RacingEnv(cfg)
    obs, _ = e.reset()
    assert obs.shape == (4,)
    e.close()


def test_obs_mode_C_dim():
    cfg = load_config_for_mode("C")
    e = RacingEnv(cfg)
    obs, _ = e.reset()
    assert obs.shape == (4 + len(cfg.track.lookahead_distances),)
    e.close()


# ---- Step return types ----------------------------------------------------

def test_step_returns_correct_types(env_b):
    obs, _ = env_b.reset()
    action = env_b.action_space.sample()
    obs, reward, terminated, truncated, info = env_b.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_info_has_required_keys(env_b):
    env_b.reset()
    action = env_b.action_space.sample()
    _, _, _, _, info = env_b.step(action)
    required = ["s", "lateral_offset", "speed", "slip_ratio", "lap_progress", "lap_time"]
    for k in required:
        assert k in info, f"Missing key: {k}"


# ---- Episode flow ---------------------------------------------------------

def test_episode_terminates_within_max_steps(env_b):
    env_b.reset()
    done = False
    steps = 0
    while not done:
        action = env_b.action_space.sample()
        _, _, term, trunc, _ = env_b.step(action)
        done = term or trunc
        steps += 1
    assert steps <= env_b.cfg.env.max_steps + 1


def test_reset_gives_finite_obs(env_b):
    obs, _ = env_b.reset()
    assert np.all(np.isfinite(obs)), f"Non-finite obs after reset: {obs}"


def test_deterministic_reset_same_obs():
    """Eval mode with same seed should give identical reset observations."""
    cfg = load_config_for_mode("B")
    e = RacingEnv(cfg, eval_mode=True)
    obs1, _ = e.reset(seed=42)
    obs2, _ = e.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    e.close()


# ---- Termination conditions -----------------------------------------------

def test_off_track_terminates():
    """Forcing the car far off-track should trigger termination."""
    cfg = load_config_for_mode("B")
    cfg.env.max_steps = 1000
    e = RacingEnv(cfg, eval_mode=False)
    e.reset()
    # Teleport vehicle state off-track
    e._state.x = 99999.0
    e._state.y = 99999.0
    action = np.array([0.0, 0.0])
    _, _, terminated, _, info = e.step(action)
    assert terminated
    assert info.get("off_track")
    e.close()


# ---- Eval mode trajectory logging -----------------------------------------

def test_eval_mode_logs_trajectory():
    cfg = load_config_for_mode("B")
    cfg.env.max_steps = 50
    e = RacingEnv(cfg, eval_mode=True)
    e.reset()
    done = False
    while not done:
        action = e.action_space.sample()
        _, _, term, trunc, _ = e.step(action)
        done = term or trunc
    traj = e.get_trajectory()
    assert len(traj) > 0
    assert len(traj[0]) == 6  # (x, y, speed, heading, steering, accel_cmd)
    e.close()


def test_non_eval_mode_empty_trajectory():
    cfg = load_config_for_mode("B")
    cfg.env.max_steps = 20
    e = RacingEnv(cfg, eval_mode=False)
    e.reset()
    done = False
    while not done:
        action = e.action_space.sample()
        _, _, term, trunc, _ = e.step(action)
        done = term or trunc
    # Non-eval mode should NOT log trajectory
    assert len(e.get_trajectory()) == 0
    e.close()
