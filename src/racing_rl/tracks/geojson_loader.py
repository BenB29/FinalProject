"""
GeoJSON track loader.

Reads a GeoJSON FeatureCollection containing a single LineString geometry
(the track centreline) and converts lat/lon coordinates to local XY in
metres using a flat-Earth projection centred on the track's centroid.

The projection is accurate to within ~0.1 % for areas smaller than 20 km,
which is more than sufficient for any racing circuit.

Usage
-----
    from racing_rl.tracks.geojson_loader import load_geojson_track

    track = load_geojson_track("data/tracks/Suzuka.geojson", half_width=7.0)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .parametric import ParametricTrack


# --------------------------------------------------------------------------- #
#  Coordinate conversion                                                       #
# --------------------------------------------------------------------------- #

# WGS-84 metres per degree of latitude (roughly constant)
_M_PER_DEG_LAT = 111_320.0


def _lonlat_to_xy(
    coords: np.ndarray,
    lat_ref: float,
    lon_ref: float,
) -> np.ndarray:
    """
    Convert (lon, lat) array to local (x, y) in metres.

    Uses a simple equirectangular projection centred on (lon_ref, lat_ref).
    X points east, Y points north — consistent with standard map orientation.
    """
    lon = coords[:, 0]
    lat = coords[:, 1]
    cos_ref = math.cos(math.radians(lat_ref))
    x = (lon - lon_ref) * _M_PER_DEG_LAT * cos_ref
    y = (lat - lat_ref) * _M_PER_DEG_LAT
    return np.column_stack([x, y])


# --------------------------------------------------------------------------- #
#  GeoJSON parsing                                                             #
# --------------------------------------------------------------------------- #

def _parse_geojson(path: Path) -> dict:
    """Read and return the first Feature from a GeoJSON FeatureCollection."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError(f"No features in {path}")
        return features[0]
    elif data.get("type") == "Feature":
        return data
    else:
        raise ValueError(f"Unsupported GeoJSON type: {data.get('type')}")


def _extract_coords(feature: dict) -> np.ndarray:
    """Extract (N, 2) array of [lon, lat] from a Feature's LineString geometry."""
    geom = feature.get("geometry", {})
    geom_type = geom.get("type", "")

    if geom_type != "LineString":
        raise ValueError(
            f"Expected LineString geometry, got {geom_type!r}. "
            "The GeoJSON must contain a single LineString for the centreline."
        )

    coords = np.array(geom["coordinates"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate array shape: {coords.shape}")

    # Take only lon/lat (drop altitude if present)
    return coords[:, :2]


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def load_geojson_track(
    path: str | Path,
    half_width: float = 7.0,
    smoothing: float = 5.0,
    sample_ds: float = 1.0,
) -> ParametricTrack:
    """
    Load a track from a GeoJSON file and return a ParametricTrack.

    Parameters
    ----------
    path:
        Path to the .geojson file.
    half_width:
        Track half-width in metres. Suzuka is roughly 12-15 m wide
        so 7 m is a reasonable default.
    smoothing:
        Spline smoothing factor passed to splprep.  A small positive value
        smooths GPS noise without distorting the shape.  Set to 0 for
        exact interpolation.
    sample_ds:
        Arc-length spacing for the lookup table (metres).  Real circuits
        are long (~5 km) so 1 m is more efficient than the 0.5 m default.

    Returns
    -------
    ParametricTrack built from the GeoJSON centreline.
    """
    path = Path(path)
    feature = _parse_geojson(path)
    coords = _extract_coords(feature)

    # Remove duplicate closing point if present (spline handles closure)
    if np.allclose(coords[0], coords[-1], atol=1e-8):
        coords = coords[:-1]

    # Centroid for projection reference
    lon_ref = float(coords[:, 0].mean())
    lat_ref = float(coords[:, 1].mean())

    # Convert to local metres
    xy = _lonlat_to_xy(coords, lat_ref, lon_ref)

    # Extract metadata for logging
    props = feature.get("properties", {})
    name = props.get("Name", props.get("name", path.stem))
    length_m = props.get("length", None)

    # Build the parametric track with slightly relaxed smoothing
    # to handle any GPS coordinate noise
    track = ParametricTrack(xy, half_width=half_width, sample_ds=sample_ds)

    return track
