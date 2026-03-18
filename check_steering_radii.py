"""
Check all tracks for corners tighter than the vehicle's minimum turning radius.

For each track:
  - Loads the track via build_track()
  - Finds the minimum radius of curvature (1 / max |curvature|) along the centreline
  - Reads the matching config yaml for max_steering_angle (falls back to 0.40 rad)
  - Computes R_min = wheelbase / tan(max_steering_angle)  with wheelbase = 3.0 m
  - Flags IMPOSSIBLE if min_radius < R_min
"""

import sys
import os
import math
import glob
import pathlib

# Add src to path so racing_rl can be imported
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import yaml

from racing_rl.tracks.parametric import build_track

WHEELBASE = 3.0          # metres
DEFAULT_STEERING = 0.40  # rad – fallback if no config found
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRACKS_DIR  = PROJECT_ROOT / "data" / "tracks"


def find_config_for_track(track_name: str):
    """
    Return the path to the yaml config whose track.name matches track_name.
    Falls back to a case-insensitive filename match, then returns None.
    """
    # Primary: search every yaml for a track.name field that equals track_name
    for yaml_path in CONFIGS_DIR.glob("*.yaml"):
        try:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                tname = cfg.get("track", {}).get("name", "")
                if tname == track_name:
                    return yaml_path
        except Exception:
            pass

    # Secondary: lowercase filename match (e.g., "Bahrain" -> "bahrain.yaml")
    candidate = CONFIGS_DIR / f"{track_name.lower()}.yaml"
    if candidate.exists():
        return candidate

    return None


def get_max_steering_angle(track_name: str) -> tuple[float, str]:
    """Return (max_steering_angle_rad, source_description)."""
    cfg_path = find_config_for_track(track_name)
    if cfg_path is None:
        return DEFAULT_STEERING, f"DEFAULT ({DEFAULT_STEERING} rad)"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        angle = cfg["vehicle"]["max_steering_angle"]
        return float(angle), cfg_path.name
    except Exception as e:
        return DEFAULT_STEERING, f"DEFAULT (parse error: {e})"


def analyse_tracks():
    geojson_files = sorted(TRACKS_DIR.glob("*.geojson"))
    if not geojson_files:
        print(f"No .geojson files found in {TRACKS_DIR}")
        return

    rows = []
    for gj in geojson_files:
        track_name = gj.stem  # e.g. "Bahrain", "Monaco"

        # Load the track
        try:
            track = build_track(track_name)
        except Exception as e:
            rows.append((track_name, None, None, None, f"LOAD ERROR: {e}"))
            continue

        # Minimum radius of curvature anywhere on the centreline
        abs_curvature = np.abs(track.curvatures)
        max_abs_kappa = float(abs_curvature.max())
        if max_abs_kappa < 1e-9:
            min_radius = float("inf")
        else:
            min_radius = 1.0 / max_abs_kappa

        # Vehicle's minimum turning radius from config
        max_steer, cfg_src = get_max_steering_angle(track_name)
        r_min_vehicle = WHEELBASE / math.tan(max_steer)

        status = "IMPOSSIBLE" if min_radius < r_min_vehicle else "OK"
        rows.append((track_name, min_radius, r_min_vehicle, cfg_src, status))

    # ---- Print table -------------------------------------------------------
    col_w = [14, 12, 17, 24, 10]
    header = (
        f"{'Track':<{col_w[0]}} "
        f"{'Min Radius':>{col_w[1]}} "
        f"{'R_min (vehicle)':>{col_w[2]}} "
        f"{'Config':>{col_w[3]}} "
        f"{'Status':<{col_w[4]}}"
    )
    sep = "-" * (sum(col_w) + len(col_w))
    print()
    print(header)
    print(sep)

    impossible_count = 0
    error_count = 0
    for row in rows:
        track_name, min_radius, r_min_vehicle, cfg_src, status = row
        if status.startswith("LOAD ERROR"):
            print(f"{'  ' + track_name:<{col_w[0]}} {'N/A':>{col_w[1]}} {'N/A':>{col_w[2]}} {'N/A':>{col_w[3]}} {status}")
            error_count += 1
        else:
            marker = " <-- !!!" if status == "IMPOSSIBLE" else ""
            print(
                f"{track_name:<{col_w[0]}} "
                f"{min_radius:>{col_w[1]}.2f} "
                f"{r_min_vehicle:>{col_w[2]}.2f} "
                f"{cfg_src:>{col_w[3]}} "
                f"{status:<{col_w[4]}}"
                f"{marker}"
            )
            if status == "IMPOSSIBLE":
                impossible_count += 1

    print(sep)
    print(f"\nSummary: {len(rows)} tracks checked, {impossible_count} IMPOSSIBLE, {error_count} load errors.")
    if impossible_count:
        print("\nIMPOSSIBLE tracks have at least one corner whose radius is smaller than")
        print(f"the vehicle's minimum turning radius (wheelbase={WHEELBASE} m / tan(max_steering_angle)).")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    analyse_tracks()
