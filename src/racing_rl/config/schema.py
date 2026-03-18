"""
Config schema using Python dataclasses.

All tuneable hyper-parameters live here so every subsystem can import a
single typed object instead of pulling raw dicts around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


# --------------------------------------------------------------------------- #
#  Sub-configs                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class TrackConfig:
    name: str = "circuit_lite"
    half_width: float = 5.0              # metres
    lookahead_distances: List[float] = field(
        default_factory=lambda: [10.0, 25.0, 50.0, 80.0, 120.0]
    )


@dataclass
class VehicleConfig:
    mass: float = 780.0                  # kg
    wheelbase: float = 3.0              # m
    max_speed: float = 80.0             # m/s
    max_steering_angle: float = 0.30    # rad
    max_steering_rate: float = 0.60     # rad/s
    max_throttle_accel: float = 15.0    # m/s^2
    max_brake_decel: float = 22.0       # m/s^2
    drag_coeff: float = 0.55            # lumped c_d * A  (air density absorbed)
    roll_resist_coeff: float = 0.012
    mu_grip: float = 3.8                # combined peak tyre friction coeff
    gravity: float = 9.81              # m/s^2
    dt: float = 0.05                   # simulation timestep  [s]


@dataclass
class ObsNormConfig:
    speed_max: float = 80.0            # normalise speed to [0, 1]
    offset_max: float = 5.0           # track half-width  (normalise offset)
    curvature_scale: float = 50.0     # multiply curvature by this for obs scaling


@dataclass
class RewardConfig:
    w_progress: float = 2.0           # reward per metre of arc-length progress
    w_speed: float = 0.005            # small bonus for speed  (anti-crawl)
    w_slip: float = -3.0              # penalty per unit over-grip slip
    w_steer_smooth: float = -0.15     # penalty for steering-rate magnitude
    w_accel_smooth: float = -0.20     # penalty for accel-command jerk
    lap_completion_bonus: float = 500.0
    off_track_penalty: float = -200.0
    min_progress_per_step: float = 0.10   # m forward progress to avoid stall-term


@dataclass
class EnvConfig:
    max_steps: int = 4000
    no_progress_steps: int = 200
    eval_deterministic: bool = True


@dataclass
class TrainingConfig:
    total_timesteps: int = 1_000_000
    n_eval_episodes: int = 3
    eval_freq: int = 20_000
    learning_rate: float = 3e-4
    buffer_size: int = 500_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    ent_coef: str = "auto"
    learning_starts: int = 5_000
    gradient_steps: int = 1
    train_freq: int = 1
    n_envs: int = 1


@dataclass
class OutputConfig:
    base_dir: str = "outputs"


# --------------------------------------------------------------------------- #
#  Top-level config                                                             #
# --------------------------------------------------------------------------- #

ObsMode = Literal["A", "B", "C"]


@dataclass
class RacingConfig:
    """Single source of truth for one training run."""

    experiment_name: str = "default"
    obs_mode: ObsMode = "B"
    seed: int = 42
    device: str = "auto"

    track: TrackConfig = field(default_factory=TrackConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    obs: ObsNormConfig = field(default_factory=ObsNormConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Derived: number of lookahead curvature samples (set after construction)
    @property
    def n_lookahead(self) -> int:
        return len(self.track.lookahead_distances)

    @property
    def obs_dim(self) -> int:
        """Observation vector dimension for the configured mode."""
        base = 2  # speed + lateral offset
        if self.obs_mode == "A":
            return base
        if self.obs_mode == "B":
            return base + 2   # + sin/cos heading error
        if self.obs_mode == "C":
            return base + 2 + self.n_lookahead
        raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
