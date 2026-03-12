"""
Plotting utilities for track, racing lines, and evaluation PNGs.

All functions accept a BaseTrack and optional trajectory data, and either
save to disk or return a Figure.

PNG naming convention (mandatory)
----------------------------------
    eval_{index:04d}_laptime_{lap_time:.2f}_completion_{pct:.0f}[_BEST].png

e.g.
    eval_0012_laptime_88.24_completion_100_BEST.png
    eval_0013_laptime_89.10_completion_100.png
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for server / subprocess)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from racing_rl.tracks.base import BaseTrack


# --------------------------------------------------------------------------- #
#  Colour map for speed                                                        #
# --------------------------------------------------------------------------- #
_SPEED_CMAP = matplotlib.colormaps.get_cmap("plasma")


# --------------------------------------------------------------------------- #
#  Core plotting function                                                      #
# --------------------------------------------------------------------------- #

def plot_racing_line(
    track: BaseTrack,
    trajectory: Optional[List[Tuple[float, float, float]]] = None,
    title: str = "Racing Line",
    ax: Optional[plt.Axes] = None,
    colour_by_speed: bool = True,
    speed_max: float = 80.0,
    mark_start: bool = True,
) -> plt.Figure:
    """
    Draw track boundaries, centreline, and an optional driven trajectory.

    Parameters
    ----------
    track:
        Any :class:`BaseTrack` implementation.
    trajectory:
        List of (x, y, speed) tuples from the episode.  If None, only the
        track is drawn.
    title:
        Figure title string.
    ax:
        Pre-existing Axes to draw into.  If None, a new figure is created.
    colour_by_speed:
        If True, colour the trajectory by speed using the plasma colourmap.
    speed_max:
        Reference speed for colour normalisation  [m/s].
    mark_start:
        If True, draw a start/finish marker.

    Returns
    -------
    fig: matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 16), dpi=180)
    else:
        fig = ax.figure

    # ---- Track boundaries ------------------------------------------------ #
    cl = track.centreline_xy
    lb = track.left_boundary_xy
    rb = track.right_boundary_xy

    # Close the loops
    cl_closed = np.vstack([cl, cl[0]])
    lb_closed = np.vstack([lb, lb[0]])
    rb_closed = np.vstack([rb, rb[0]])

    ax.fill(
        np.concatenate([lb_closed[:, 0], rb_closed[::-1, 0]]),
        np.concatenate([lb_closed[:, 1], rb_closed[::-1, 1]]),
        color="#2a2a2a",
        alpha=0.85,
        zorder=1,
        label="Track surface",
    )
    ax.plot(lb_closed[:, 0], lb_closed[:, 1], "w-", lw=1.2, zorder=2, label="Boundaries")
    ax.plot(rb_closed[:, 0], rb_closed[:, 1], "w-", lw=1.2, zorder=2)
    ax.plot(cl_closed[:, 0], cl_closed[:, 1], "--", color="#888888", lw=0.8,
            zorder=3, label="Centreline", alpha=0.7)

    # ---- Start / finish marker ------------------------------------------ #
    if mark_start:
        sx, sy = cl[0]
        # Short perpendicular line across the track
        h0 = np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0])
        nx, ny = -np.sin(h0), np.cos(h0)
        hw = track.half_width
        ax.plot(
            [sx - nx * hw, sx + nx * hw],
            [sy - ny * hw, sy + ny * hw],
            "r-", lw=3, zorder=6, label="Start/Finish",
        )

    # ---- Racing-line trajectory ----------------------------------------- #
    if trajectory:
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        speeds = [p[2] for p in trajectory]

        if colour_by_speed and len(xs) > 1:
            norm = mcolors.Normalize(vmin=0, vmax=speed_max)
            for i in range(len(xs) - 1):
                c = _SPEED_CMAP(norm(speeds[i]))
                ax.plot(xs[i:i+2], ys[i:i+2], color=c, lw=1.2, zorder=5, solid_capstyle="round")
            # Colourbar
            sm = plt.cm.ScalarMappable(cmap=_SPEED_CMAP, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Speed [m/s]", fontsize=9)
        else:
            ax.plot(xs, ys, color="#00ff88", lw=1.2, zorder=5, label="Racing line")

    # ---- Cosmetics ------------------------------------------------------- #
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_xlabel("X [m]", color="white")
    ax.set_ylabel("Y [m]", color="white")
    ax.set_title(title, color="white", pad=10)
    legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.4,
                       labelcolor="white", facecolor="#333333")

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Evaluation PNG export                                                       #
# --------------------------------------------------------------------------- #

def save_eval_png(
    track: BaseTrack,
    trajectory: List[Tuple[float, float, float]],
    out_dir: Path,
    eval_index: int,
    lap_time: float,
    completion_pct: float,
    is_best: bool = False,
    mean_speed: float = 0.0,
    max_slip: float = 0.0,
    obs_mode: str = "",
    speed_max: float = 80.0,
) -> Path:
    """
    Save a racing-line PNG for one evaluation.

    Returns the path to the saved file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_tag = "_BEST" if is_best else ""
    filename = (
        f"eval_{eval_index:04d}"
        f"_laptime_{lap_time:.2f}"
        f"_completion_{completion_pct:.0f}"
        f"{best_tag}.png"
    )
    out_path = out_dir / filename

    title_parts = [
        f"Eval #{eval_index:04d}",
        f"Lap time: {lap_time:.2f} s" if completion_pct >= 99.9 else f"Completion: {completion_pct:.1f}%",
        f"Mean speed: {mean_speed:.1f} m/s",
        f"Max slip: {max_slip:.2f}",
    ]
    if obs_mode:
        title_parts.append(f"Obs: {obs_mode}")
    if is_best:
        title_parts.append("★ BEST")
    title = "   |   ".join(title_parts)

    fig = plot_racing_line(
        track=track,
        trajectory=trajectory,
        title=title,
        colour_by_speed=True,
        speed_max=speed_max,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return out_path


# --------------------------------------------------------------------------- #
#  Comparison plot across obs modes                                            #
# --------------------------------------------------------------------------- #

def plot_comparison(
    track: BaseTrack,
    trajectories: dict,          # {label: [(x, y, speed), ...]}
    title: str = "Racing Line Comparison",
    out_path: Optional[Path] = None,
    speed_max: float = 80.0,
) -> plt.Figure:
    """
    Plot multiple racing lines on the same track for visual comparison.

    Parameters
    ----------
    trajectories:
        Dict mapping label strings to trajectory lists.
    out_path:
        If given, the figure is saved and closed.  Otherwise returned open.
    """
    colours = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a8ff78", "#ff8b94"]
    fig, ax = plt.subplots(figsize=(12, 9), dpi=120)

    # Draw track background
    cl = track.centreline_xy
    lb = track.left_boundary_xy
    rb = track.right_boundary_xy
    cl_c = np.vstack([cl, cl[0]])
    lb_c = np.vstack([lb, lb[0]])
    rb_c = np.vstack([rb, rb[0]])

    ax.fill(
        np.concatenate([lb_c[:, 0], rb_c[::-1, 0]]),
        np.concatenate([lb_c[:, 1], rb_c[::-1, 1]]),
        color="#2a2a2a", alpha=0.85, zorder=1,
    )
    ax.plot(lb_c[:, 0], lb_c[:, 1], "w-", lw=1.0, zorder=2, alpha=0.6)
    ax.plot(rb_c[:, 0], rb_c[:, 1], "w-", lw=1.0, zorder=2, alpha=0.6)
    ax.plot(cl_c[:, 0], cl_c[:, 1], "--", color="#888888", lw=0.7, zorder=3, alpha=0.5)

    for i, (label, traj) in enumerate(trajectories.items()):
        if not traj:
            continue
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        col = colours[i % len(colours)]
        ax.plot(xs, ys, color=col, lw=2.0, zorder=5 + i, label=label, alpha=0.9)

    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_title(title, color="white", pad=10)
    ax.set_xlabel("X [m]", color="white")
    ax.set_ylabel("Y [m]", color="white")
    ax.tick_params(colors="white")
    legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.5,
                       labelcolor="white", facecolor="#333333")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    return fig
