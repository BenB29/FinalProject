"""
Abstract base class for track representations.

Every track must expose the public interface defined here so that the
environment, observation builder, and plotting utilities all work against
a single contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TrackQuery:
    """Result of querying the track for a vehicle position."""

    # Nearest-point arc-length along the centreline  [m]
    s: float
    # Signed lateral offset: positive = left of travel direction  [m]
    lateral_offset: float
    # Track heading at nearest point (global frame)  [rad]
    track_heading: float
    # Signed curvature at nearest point  [1/m]  (positive = left-curving)
    curvature: float
    # Lookahead curvatures at configured distances  [1/m]
    lookahead_curvatures: np.ndarray
    # Distance from left boundary  [m]
    dist_to_left: float
    # Distance from right boundary  [m]
    dist_to_right: float
    # Nearest centreline point coords  [m]
    cx: float
    cy: float


class BaseTrack(ABC):
    """Minimal interface all track implementations must satisfy."""

    @property
    @abstractmethod
    def length(self) -> float:
        """Total centreline arc-length  [m]."""

    @property
    @abstractmethod
    def half_width(self) -> float:
        """Nominal track half-width  [m]."""

    @abstractmethod
    def query(
        self,
        x: float,
        y: float,
        lookahead_distances: List[float],
    ) -> TrackQuery:
        """
        Compute full track-relative geometry for a vehicle at (x, y).

        Args:
            x, y: Vehicle position in global frame  [m].
            lookahead_distances: Distances ahead on the centreline at which
                to sample curvature  [m].

        Returns:
            :class:`TrackQuery` with all track-relative information.
        """

    @abstractmethod
    def is_off_track(self, x: float, y: float) -> bool:
        """True if the vehicle centre is outside the track boundaries."""

    @abstractmethod
    def start_position(self) -> Tuple[float, float, float]:
        """Returns (x, y, heading) of the canonical start/finish position."""

    # ------------------------------------------------------------------ #
    #  Plotting helpers (concrete, uses the abstract query internally)     #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def centreline_xy(self) -> np.ndarray:
        """(N, 2) array of centreline sample points."""

    @property
    @abstractmethod
    def left_boundary_xy(self) -> np.ndarray:
        """(N, 2) array of left boundary sample points."""

    @property
    @abstractmethod
    def right_boundary_xy(self) -> np.ndarray:
        """(N, 2) array of right boundary sample points."""
