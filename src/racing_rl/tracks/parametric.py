"""
Parametric track built from a closed cubic B-spline through control points.

Design notes
------------
* A set of waypoints defines the circuit shape.  scipy interpolates a smooth
  closed curve through them.
* The curve is then densely sampled (every SAMPLE_DS metres) to build a
  lookup table: (x, y, heading, curvature, arc_length).
* A KD-tree over the sample points gives O(log N) nearest-point queries.
* Signed lateral offset is computed from the cross-product of the tangent
  with the vector from centreline to vehicle.
* Curvature uses the Frenet–Serret formula on the parametric derivatives.

Assumptions
-----------
* Track half-width is uniform (configurable, default 5 m).  A variable-width
  extension would slot in by replacing the scalar with a per-sample array.
* The car starts at s=0 heading in the direction of the centreline tangent.
* Arc-length is measured counter-clockwise (positive direction of travel).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree

from .base import BaseTrack, TrackQuery

# Resolution of the pre-sampled centreline table
SAMPLE_DS = 0.5   # metres between samples


# --------------------------------------------------------------------------- #
#  Default "circuit_lite" waypoints                                            #
# --------------------------------------------------------------------------- #

# A compact but varied circuit (roughly 900 m lap):
#   T1  – fast right-hander off the straight
#   T2  – tight left hairpin (heavy braking)
#   T3  – medium right
#   T4/5 – chicane
#   T6  – sweeping fast left back onto the straight
#
# Coordinates in metres; car travels counter-clockwise when viewed from above.
# Start/finish is at (0, 0) with the car heading in the +X direction.

_CIRCUIT_LITE_WAYPOINTS: np.ndarray = np.array([
    [  0,    0],   # S/F line  (start of main straight)
    [ 80,   -5],   # mid-straight
    [170,  -10],   # end of main straight / T1 entry
    [220,  -25],   # T1 mid (fast right)
    [240,  -65],   # T1 apex
    [225, -105],   # T1 exit
    [190, -130],   # short link
    [165, -150],   # T2 entry (hairpin)
    [135, -165],   # T2 apex  (tight left)
    [100, -150],   # T2 exit
    [ 75, -130],   # T3 entry
    [ 55, -105],   # T3 apex (medium right)
    [ 60,  -75],   # T3 exit
    [ 75,  -55],   # chicane entry straight
    [ 90,  -45],   # T4 right
    [ 95,  -30],   # T4/T5 mid
    [ 80,  -20],   # T5 left
    [ 55,  -15],   # T5 exit
    [ 20,  -10],   # T6 entry (sweeping left)
    [-10,   -5],   # T6 mid
    [-15,    0],   # T6 exit / back to S/F
], dtype=float)


# --------------------------------------------------------------------------- #
#  ParametricTrack                                                             #
# --------------------------------------------------------------------------- #

class ParametricTrack(BaseTrack):
    """
    A smooth closed track built from cubic-spline interpolation of waypoints.

    Parameters
    ----------
    waypoints:
        (N, 2) array of (x, y) control points.  The spline closes the loop
        automatically.
    half_width:
        Uniform track half-width  [m].
    sample_ds:
        Arc-length spacing for the pre-sampled lookup table  [m].
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        half_width: float = 5.0,
        sample_ds: float = SAMPLE_DS,
    ) -> None:
        self._half_width = half_width
        self._sample_ds = sample_ds
        self._build_from_waypoints(waypoints)

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def _build_from_waypoints(self, waypoints: np.ndarray) -> None:
        """Fit a closed B-spline and build the lookup table + KD-tree."""
        pts = np.asarray(waypoints, dtype=float)

        # splprep wants the loop closed: repeat the first point at the end
        pts_closed = np.vstack([pts, pts[0]])
        tck, _ = splprep([pts_closed[:, 0], pts_closed[:, 1]], s=0, per=True, k=3)

        # Initial rough sampling to estimate total arc length
        u_fine = np.linspace(0, 1, 10_000, endpoint=False)
        x_fine, y_fine = splev(u_fine, tck)
        dx = np.diff(x_fine, append=x_fine[0])
        dy = np.diff(y_fine, append=y_fine[0])
        ds_fine = np.hypot(dx, dy)
        arc_fine = np.concatenate([[0.0], np.cumsum(ds_fine[:-1])])
        total_length = float(np.sum(ds_fine))

        # Resample at uniform arc-length spacing
        n_samples = max(int(total_length / self._sample_ds) + 1, 500)
        arc_targets = np.linspace(0, total_length, n_samples, endpoint=False)
        u_uniform = np.interp(arc_targets, arc_fine, u_fine)

        x, y = splev(u_uniform, tck)
        # 1st and 2nd derivatives w.r.t. parameter u
        dx_du, dy_du = splev(u_uniform, tck, der=1)
        d2x_du2, d2y_du2 = splev(u_uniform, tck, der=2)

        speed_sq = dx_du**2 + dy_du**2
        speed = np.sqrt(speed_sq)

        # Heading (tangent direction)
        heading = np.arctan2(dy_du, dx_du)

        # Signed curvature: κ = (x' y'' - y' x'') / |r'|^3
        curvature = (dx_du * d2y_du2 - dy_du * d2x_du2) / (speed_sq * speed + 1e-12)

        self._n = n_samples
        self._total_length = total_length
        self._arc = arc_targets            # [n] arc-length of each sample
        self._cx = x                       # [n] centreline x
        self._cy = y                       # [n] centreline y
        self._heading = heading            # [n] tangent heading
        self._curvature = curvature        # [n] signed curvature

        # KD-tree for fast nearest-neighbour lookup
        self._kd = KDTree(np.column_stack([x, y]))

    # ------------------------------------------------------------------ #
    #  BaseTrack interface                                                 #
    # ------------------------------------------------------------------ #

    @property
    def length(self) -> float:
        return self._total_length

    @property
    def half_width(self) -> float:
        return self._half_width

    def _nearest_idx(self, x: float, y: float) -> int:
        """Return index of the nearest centreline sample (global KD-tree)."""
        _, idx = self._kd.query([x, y])
        return int(idx)

    def _nearest_idx_local(
        self, x: float, y: float, hint_idx: int, radius: int = 200,
    ) -> int:
        """
        Return nearest centreline index, searching only within *radius*
        samples of *hint_idx*.

        This prevents the KD-tree from jumping to the wrong section of
        track on figure-8 layouts (e.g. Suzuka) where two parts of the
        centreline are physically close but far apart in arc-length.
        """
        n = self._n
        # Build index range wrapping around the circular buffer
        lo = hint_idx - radius
        hi = hint_idx + radius
        indices = np.arange(lo, hi + 1) % n

        dx = self._cx[indices] - x
        dy = self._cy[indices] - y
        dists_sq = dx * dx + dy * dy
        best_local = int(indices[np.argmin(dists_sq)])
        return best_local

    def query(
        self,
        x: float,
        y: float,
        lookahead_distances: List[float],
        hint_idx: int = -1,
    ) -> TrackQuery:
        """
        Full track-relative geometry query for vehicle at (x, y).

        Parameters
        ----------
        hint_idx:
            If >= 0, constrain the nearest-point search to a local window
            around this index.  Essential for figure-8 tracks where a
            global KD-tree lookup would snap to the wrong track section
            at the crossover.  Pass -1 for a global (unconstrained) search.
        """
        if hint_idx >= 0:
            idx = self._nearest_idx_local(x, y, hint_idx)
        else:
            idx = self._nearest_idx(x, y)

        # Nearest centreline point
        cx = float(self._cx[idx])
        cy = float(self._cy[idx])
        s = float(self._arc[idx])
        heading = float(self._heading[idx])
        kappa = float(self._curvature[idx])

        # Signed lateral offset
        # Normal vector (points left of travel direction):  n = (-sin h, cos h)
        vec_x = x - cx
        vec_y = y - cy
        # dot with normal: offset > 0 means car is to the left
        lateral_offset = -vec_x * np.sin(heading) + vec_y * np.cos(heading)

        # Distances to boundaries
        dist_to_left = self._half_width - lateral_offset
        dist_to_right = self._half_width + lateral_offset

        # Lookahead curvatures
        la_curvatures = self._sample_lookahead(idx, lookahead_distances)

        return TrackQuery(
            s=s,
            lateral_offset=lateral_offset,
            track_heading=heading,
            curvature=kappa,
            lookahead_curvatures=la_curvatures,
            dist_to_left=dist_to_left,
            dist_to_right=dist_to_right,
            cx=cx,
            cy=cy,
        )

    def _sample_lookahead(
        self, start_idx: int, distances: List[float]
    ) -> np.ndarray:
        """
        Sample curvature at specified arc-length distances ahead of start_idx.

        Distances wrap around the circuit (modulo total_length).
        """
        s0 = self._arc[start_idx]
        result = np.zeros(len(distances), dtype=float)
        for i, d in enumerate(distances):
            s_target = (s0 + d) % self._total_length
            # Binary search on the arc array (which is sorted)
            ahead_idx = int(np.searchsorted(self._arc, s_target) % self._n)
            result[i] = self._curvature[ahead_idx]
        return result

    def is_off_track(self, x: float, y: float, hint_idx: int = -1) -> bool:
        """True if the vehicle is outside the track boundaries."""
        idx = self._nearest_idx_local(x, y, hint_idx) if hint_idx >= 0 else self._nearest_idx(x, y)
        cx = self._cx[idx]
        cy = self._cy[idx]
        heading = self._heading[idx]
        vec_x = x - cx
        vec_y = y - cy
        lateral_offset = -vec_x * np.sin(heading) + vec_y * np.cos(heading)
        # Use Euclidean distance as secondary check for safety
        dist_from_centre = np.hypot(vec_x, vec_y)
        return abs(lateral_offset) > self._half_width or dist_from_centre > self._half_width * 1.5

    def start_position(self) -> Tuple[float, float, float]:
        """Start at s=0 with the centreline heading."""
        return float(self._cx[0]), float(self._cy[0]), float(self._heading[0])

    # ------------------------------------------------------------------ #
    #  Plotting helpers                                                    #
    # ------------------------------------------------------------------ #

    @property
    def centreline_xy(self) -> np.ndarray:
        return np.column_stack([self._cx, self._cy])

    @property
    def left_boundary_xy(self) -> np.ndarray:
        # Left normal offset
        nx = -np.sin(self._heading)
        ny = np.cos(self._heading)
        return np.column_stack([
            self._cx + nx * self._half_width,
            self._cy + ny * self._half_width,
        ])

    @property
    def right_boundary_xy(self) -> np.ndarray:
        nx = -np.sin(self._heading)
        ny = np.cos(self._heading)
        return np.column_stack([
            self._cx - nx * self._half_width,
            self._cy - ny * self._half_width,
        ])

    @property
    def arc_lengths(self) -> np.ndarray:
        return self._arc.copy()

    @property
    def curvatures(self) -> np.ndarray:
        return self._curvature.copy()


# --------------------------------------------------------------------------- #
#  Factory function                                                            #
# --------------------------------------------------------------------------- #

def build_circuit_lite(half_width: float = 5.0) -> ParametricTrack:
    """Return the default 'circuit_lite' track."""
    return ParametricTrack(_CIRCUIT_LITE_WAYPOINTS, half_width=half_width)


def build_track(name: str, half_width: float = 5.0) -> ParametricTrack:
    """
    Build a track by name.

    Supports:
      - 'circuit_lite'  -- built-in compact test circuit
      - Any name matching a .geojson file in data/tracks/  (e.g. 'Suzuka')

    For GeoJSON tracks the sample spacing is 1 m (vs 0.5 m for the test
    circuit) because real circuits are 4-6 km long.
    """
    if name == "circuit_lite":
        return build_circuit_lite(half_width)

    # Try loading a GeoJSON file from data/tracks/
    from pathlib import Path
    from .geojson_loader import load_geojson_track
    _data_dir = Path(__file__).resolve().parents[3] / "data" / "tracks"
    geojson_path = _data_dir / f"{name}.geojson"
    if geojson_path.exists():
        return load_geojson_track(geojson_path, half_width=half_width, sample_ds=1.0)

    available = ["circuit_lite"]
    if _data_dir.exists():
        available += [p.stem for p in _data_dir.glob("*.geojson")]
    raise ValueError(f"Unknown track: {name!r}.  Available: {available}")
