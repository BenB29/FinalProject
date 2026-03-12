"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src importable during tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from racing_rl.config.loader import load_config_for_mode
from racing_rl.config.schema import RacingConfig
from racing_rl.tracks.parametric import build_circuit_lite, ParametricTrack


@pytest.fixture(scope="session")
def track() -> ParametricTrack:
    return build_circuit_lite(half_width=5.0)


@pytest.fixture(params=["A", "B", "C"])
def cfg_all_modes(request) -> RacingConfig:
    return load_config_for_mode(request.param)


@pytest.fixture(scope="session")
def cfg_b() -> RacingConfig:
    return load_config_for_mode("B")
