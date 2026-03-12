"""
Observation vector builder for modes A, B, and C.

Mode A  (2 dims)
    [ speed_norm, lateral_offset_norm ]

Mode B  (4 dims)
    [ speed_norm, lateral_offset_norm, sin_heading_err, cos_heading_err ]

Mode C  (4 + N dims)
    [ speed_norm, lateral_offset_norm, sin_heading_err, cos_heading_err,
      kappa_1_norm, kappa_2_norm, ..., kappa_N_norm ]

Normalisation
-------------
* speed_norm      = speed / speed_max           → [0, 1]
* lateral_offset  = offset / half_width         → [-1, 1] roughly
* sin/cos of heading error  are already in [-1, 1]
* curvature_norm  = curvature * curvature_scale → values around [-1, 1]
  (a curvature of 0.02 1/m at curvature_scale=50 gives 1.0, corresponding
   to a radius of 50 m — a tight hairpin for this car)

Important: there is NO track-position term in any mode (e.g. no arc-length s,
no absolute x/y).  The agent must generalise around the whole circuit from
local observations only.
"""
from __future__ import annotations

import math

import numpy as np

from racing_rl.config.schema import ObsMode, ObsNormConfig, TrackConfig
from racing_rl.physics.vehicle import VehicleState
from racing_rl.tracks.base import TrackQuery


class ObservationBuilder:
    """
    Builds normalised observation vectors from raw vehicle state and track query.

    Parameters
    ----------
    obs_mode:
        One of 'A', 'B', 'C'.
    obs_norm:
        Normalisation parameters from config.
    track_cfg:
        Track config (for lookahead distances and half-width).
    """

    def __init__(
        self,
        obs_mode: ObsMode,
        obs_norm: ObsNormConfig,
        track_cfg: TrackConfig,
    ) -> None:
        self.obs_mode = obs_mode.upper()
        self.obs_norm = obs_norm
        self.track_cfg = track_cfg
        if self.obs_mode not in ("A", "B", "C"):
            raise ValueError(f"Unknown obs_mode: {obs_mode!r}")

    @property
    def obs_dim(self) -> int:
        if self.obs_mode == "A":
            return 2
        if self.obs_mode == "B":
            return 4
        # Mode C: 4 base + N lookahead curvature samples
        return 4 + len(self.track_cfg.lookahead_distances)

    def build(
        self,
        vehicle: VehicleState,
        track_query: TrackQuery,
    ) -> np.ndarray:
        """
        Build the observation vector.

        Parameters
        ----------
        vehicle:
            Current vehicle state.
        track_query:
            Result of TrackQuery at the vehicle's position.

        Returns
        -------
        obs:
            Float32 array of shape (obs_dim,).
        """
        norm = self.obs_norm

        # ---- Shared base features ---------------------------------------- #
        speed_norm = vehicle.speed / norm.speed_max

        # Normalise lateral offset by track half-width
        # Clamp to [-2, 2] to avoid exploding observations for off-track states
        offset_norm = np.clip(
            track_query.lateral_offset / norm.offset_max, -2.0, 2.0
        )

        if self.obs_mode == "A":
            obs = np.array([speed_norm, offset_norm], dtype=np.float32)
            return obs

        # ---- Heading error features (B, C) ------------------------------- #
        heading_error = _wrap_angle(vehicle.heading - track_query.track_heading)
        sin_he = math.sin(heading_error)
        cos_he = math.cos(heading_error)

        if self.obs_mode == "B":
            obs = np.array([speed_norm, offset_norm, sin_he, cos_he], dtype=np.float32)
            return obs

        # ---- Lookahead curvature features (C) ---------------------------- #
        kappa_norms = (
            track_query.lookahead_curvatures * norm.curvature_scale
        ).astype(np.float32)
        # Clamp to prevent extreme values from noisy track sections
        kappa_norms = np.clip(kappa_norms, -3.0, 3.0)

        obs = np.array(
            [speed_norm, offset_norm, sin_he, cos_he, *kappa_norms],
            dtype=np.float32,
        )
        return obs


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #

def _wrap_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
