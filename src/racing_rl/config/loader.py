"""
Config loader: merges base.yaml + per-mode override yaml into a RacingConfig.

Usage:
    cfg = load_config("configs/obs_b.yaml")
    cfg = load_config_for_mode("C")
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .schema import (
    EnvConfig,
    ObsNormConfig,
    OutputConfig,
    RacingConfig,
    RewardConfig,
    TrackConfig,
    TrainingConfig,
    VehicleConfig,
)

# --------------------------------------------------------------------------- #
#  Internal helpers                                                             #
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[3]   # src/racing_rl/config/loader.py -> root
_CONFIGS_DIR = _REPO_ROOT / "configs"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _dict_to_config(d: Dict[str, Any]) -> RacingConfig:
    """Convert a flat/nested dict (from YAML) to a typed RacingConfig."""
    cfg = RacingConfig()

    # Top-level scalars
    for key in ("experiment_name", "obs_mode", "seed", "device"):
        if key in d:
            setattr(cfg, key, d[key])

    # Sub-configs  (only update fields that exist in the dataclass)
    def _apply(sub_cfg, sub_dict: dict):
        for k, v in sub_dict.items():
            if hasattr(sub_cfg, k):
                setattr(sub_cfg, k, v)

    if "track" in d:
        _apply(cfg.track, d["track"])
    if "vehicle" in d:
        _apply(cfg.vehicle, d["vehicle"])
    if "obs" in d:
        _apply(cfg.obs, d["obs"])
    if "reward" in d:
        _apply(cfg.reward, d["reward"])
    if "env" in d:
        _apply(cfg.env, d["env"])
    if "training" in d:
        _apply(cfg.training, d["training"])
    if "output" in d:
        _apply(cfg.output, d["output"])

    return cfg


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def load_config(override_yaml: Optional[str | Path] = None) -> RacingConfig:
    """
    Load config by merging ``configs/base.yaml`` with an optional override file.

    Args:
        override_yaml: Path to a YAML file that overrides base values.
                       Can be absolute or relative to the repo root.

    Returns:
        Fully populated :class:`RacingConfig`.
    """
    base_dict = _load_yaml(_CONFIGS_DIR / "base.yaml")

    if override_yaml is not None:
        override_path = Path(override_yaml)
        if not override_path.is_absolute():
            override_path = _REPO_ROOT / override_path
        override_dict = _load_yaml(override_path)
        merged = _deep_merge(base_dict, override_dict)
    else:
        merged = base_dict

    return _dict_to_config(merged)


def load_config_for_mode(obs_mode: str) -> RacingConfig:
    """
    Convenience loader: returns config for observation mode A, B, or C.

    Uses ``configs/obs_{mode.lower()}.yaml`` as the override file.
    """
    mode = obs_mode.upper()
    if mode not in ("A", "B", "C"):
        raise ValueError(f"obs_mode must be A, B, or C — got {obs_mode!r}")
    override = _CONFIGS_DIR / f"obs_{mode.lower()}.yaml"
    return load_config(override)
