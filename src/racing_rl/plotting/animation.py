"""
Animated replay of a racing lap.

Produces an MP4 (or GIF fallback) showing:
  - A zoomed camera following an F1-style car around the track
  - A steering wheel indicator on the side
  - Throttle and brake bar gauges
  - A smooth zoom-out at the end to reveal the full track + trajectory
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
from PIL import Image

from racing_rl.tracks.base import BaseTrack
from racing_rl.plotting.wheel_asset import get_wheel_path

# Trajectory tuple indices
_X, _Y, _SPD, _HDG, _STEER, _ACCEL = 0, 1, 2, 3, 4, 5

# Animation settings
_FPS = 30
_DT = 0.05  # physics timestep — one trajectory point per dt
_ZOOM_WINDOW = 120.0  # metres around car when zoomed in
_ZOOM_OUT_SECONDS = 3.0  # duration of end zoom-out
_HOLD_SECONDS = 2.0  # hold full-track view at end
_SPEED_CMAP = matplotlib.colormaps.get_cmap("plasma")


def save_lap_animation(
    track: BaseTrack,
    trajectory: List[Tuple],
    out_path: Path,
    speed_max: float = 80.0,
    title: str = "Best Lap",
) -> Path:
    """
    Render and save an animated replay of a lap.

    Parameters
    ----------
    track : BaseTrack
    trajectory : list of (x, y, speed, heading, steering_angle, accel_cmd)
    out_path : output file path (.mp4 or .gif)
    speed_max : max speed for colour normalisation
    title : displayed title

    Returns
    -------
    Path to saved animation file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    traj = np.array(trajectory)  # (N, 6)
    n_frames_lap = len(traj)

    # Build frame indices for real-time playback.
    # Total sim time = n_frames_lap * _DT seconds.
    # At _FPS, we need n_anim_frames = sim_time * _FPS animation frames.
    sim_time = n_frames_lap * _DT
    n_anim_frames = max(1, int(sim_time * _FPS))
    lap_frame_indices = [
        min(int(i * n_frames_lap / n_anim_frames), n_frames_lap - 1)
        for i in range(n_anim_frames)
    ]
    if lap_frame_indices[-1] != n_frames_lap - 1:
        lap_frame_indices.append(n_frames_lap - 1)

    n_zoom_out = int(_ZOOM_OUT_SECONDS * _FPS)
    n_hold = int(_HOLD_SECONDS * _FPS)
    total_frames = len(lap_frame_indices) + n_zoom_out + n_hold

    # Pre-compute full-track bounds for zoom-out
    cl = track.centreline_xy
    lb = track.left_boundary_xy
    rb = track.right_boundary_xy
    all_pts = np.vstack([lb, rb])
    x_min, x_max = all_pts[:, 0].min() - 20, all_pts[:, 0].max() + 20
    y_min, y_max = all_pts[:, 1].min() - 20, all_pts[:, 1].max() + 20
    # Make full-track view square-ish
    cx_full = (x_min + x_max) / 2
    cy_full = (y_min + y_max) / 2
    span_full = max(x_max - x_min, y_max - y_min) / 2

    # Close track loops
    cl_closed = np.vstack([cl, cl[0]])
    lb_closed = np.vstack([lb, lb[0]])
    rb_closed = np.vstack([rb, rb[0]])

    # ------------------------------------------------------------------ #
    #  Figure layout                                                      #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(14, 8), dpi=120, facecolor="#1a1a1a")
    # Main track view
    ax_track = fig.add_axes([0.02, 0.02, 0.68, 0.92])
    # Steering wheel
    ax_wheel = fig.add_axes([0.74, 0.50, 0.24, 0.44])
    # Throttle bar
    ax_thr = fig.add_axes([0.76, 0.06, 0.08, 0.36])
    # Brake bar
    ax_brk = fig.add_axes([0.88, 0.06, 0.08, 0.36])

    for ax in [ax_track, ax_wheel, ax_thr, ax_brk]:
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # Title
    title_text = fig.text(
        0.36, 0.97, title, color="white", fontsize=13,
        ha="center", va="top", fontweight="bold",
    )

    # Speed readout
    speed_text = fig.text(
        0.36, 0.93, "", color="#00ff88", fontsize=11,
        ha="center", va="top", fontfamily="monospace",
    )

    # ------------------------------------------------------------------ #
    #  Static track drawing                                               #
    # ------------------------------------------------------------------ #
    def draw_track(ax):
        ax.fill(
            np.concatenate([lb_closed[:, 0], rb_closed[::-1, 0]]),
            np.concatenate([lb_closed[:, 1], rb_closed[::-1, 1]]),
            color="#2a2a2a", alpha=0.85, zorder=1,
        )
        ax.plot(lb_closed[:, 0], lb_closed[:, 1], "w-", lw=1.0, zorder=2)
        ax.plot(rb_closed[:, 0], rb_closed[:, 1], "w-", lw=1.0, zorder=2)
        ax.plot(cl_closed[:, 0], cl_closed[:, 1], "--", color="#555555",
                lw=0.6, zorder=3, alpha=0.5)

    # ------------------------------------------------------------------ #
    #  Car shape — 2026 F1 regulations (top-down)                         #
    #  Key changes: narrower body, active aero front wing (simpler,       #
    #  cleaner), wider front tyres, slimmer sidepods, prominent floor     #
    #  edges, larger rear wing with active flap.                          #
    # ------------------------------------------------------------------ #
    def _draw_car(ax, x, y, heading, steer_angle, length=6.0):
        """Draw a top-down 2026-spec F1 car at (x, y) with given heading."""
        w = length  # reference length
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s], [s, c]])

        def _t(pts):
            """Transform local coords to world."""
            return (np.array(pts) @ rot.T) + np.array([x, y])

        hw = w * 0.16   # half-width of monocoque (narrower 2026 body)
        nw = w * 0.06   # half-width of nose tip (rounder nose)

        # --- Floor / underbody edges (visible as dark strips beside body) --- #
        floor_hw = hw * 1.25
        floor = _t([
            [w * 0.30, floor_hw],
            [w * 0.05, floor_hw * 1.1],     # floor widens ahead of sidepods
            [-w * 0.10, floor_hw * 1.15],    # floor edge widest
            [-w * 0.30, floor_hw * 1.05],    # tapers toward diffuser
            [-w * 0.42, floor_hw * 0.7],     # diffuser narrows
            [-w * 0.42, -floor_hw * 0.7],
            [-w * 0.30, -floor_hw * 1.05],
            [-w * 0.10, -floor_hw * 1.15],
            [w * 0.05, -floor_hw * 1.1],
            [w * 0.30, -floor_hw],
        ])
        ax.add_patch(patches.Polygon(
            floor, closed=True, facecolor="#1f1f1f", edgecolor="#333333",
            lw=0.6, zorder=10,
        ))

        # --- Main body (narrower, cleaner 2026 monocoque) --- #
        body = _t([
            [w * 0.48, 0],              # nose tip (rounder)
            [w * 0.44, nw * 0.8],       # nose cone
            [w * 0.38, nw * 1.5],       # nose widens gently
            [w * 0.30, hw * 0.65],      # front bulkhead
            [w * 0.20, hw * 0.9],       # ahead of cockpit
            [w * 0.10, hw],             # cockpit sides
            [-w * 0.02, hw * 0.95],     # slim sidepod inlet
            [-w * 0.15, hw * 0.90],     # sidepod (much slimmer 2026)
            [-w * 0.28, hw * 0.85],     # engine cover
            [-w * 0.38, hw * 0.65],     # rear taper
            [-w * 0.45, hw * 0.40],     # gearbox
            [-w * 0.45, -hw * 0.40],    # gearbox (other side)
            [-w * 0.38, -hw * 0.65],
            [-w * 0.28, -hw * 0.85],
            [-w * 0.15, -hw * 0.90],
            [-w * 0.02, -hw * 0.95],
            [w * 0.10, -hw],
            [w * 0.20, -hw * 0.9],
            [w * 0.30, -hw * 0.65],
            [w * 0.38, -nw * 1.5],
            [w * 0.44, -nw * 0.8],
        ])
        body_patch = patches.Polygon(
            body, closed=True, facecolor="#cc0000", edgecolor="#e83030",
            lw=0.8, zorder=11,
        )
        ax.add_patch(body_patch)

        # --- Engine cover spine (raised centre line) --- #
        spine = _t([
            [-w * 0.05, hw * 0.12],
            [-w * 0.42, hw * 0.08],
            [-w * 0.42, -hw * 0.08],
            [-w * 0.05, -hw * 0.12],
        ])
        ax.add_patch(patches.Polygon(
            spine, closed=True, facecolor="#aa0000", edgecolor="none",
            zorder=11.5,
        ))

        # --- Front wing (2026: simpler, active, 3-element) --- #
        fw_w = hw * 1.05  # slightly wider than body
        # Main plane
        fw_pts = _t([
            [w * 0.52, -fw_w],
            [w * 0.52,  fw_w],
            [w * 0.48,  fw_w],
            [w * 0.48, -fw_w],
        ])
        ax.add_patch(patches.Polygon(
            fw_pts, closed=True, facecolor="#282828", edgecolor="#444444",
            lw=0.6, zorder=12,
        ))
        # Active flap element (slightly swept)
        flap = _t([
            [w * 0.48, -fw_w * 0.95],
            [w * 0.48,  fw_w * 0.95],
            [w * 0.46,  fw_w * 0.90],
            [w * 0.46, -fw_w * 0.90],
        ])
        ax.add_patch(patches.Polygon(
            flap, closed=True, facecolor="#333333", edgecolor="#444444",
            lw=0.4, zorder=12,
        ))

        # Front wing endplates (smaller, cleaner for 2026)
        for side in [1, -1]:
            ep = _t([
                [w * 0.52, side * fw_w],
                [w * 0.52, side * (fw_w + w * 0.012)],
                [w * 0.46, side * (fw_w + w * 0.012)],
                [w * 0.46, side * fw_w],
            ])
            ax.add_patch(patches.Polygon(
                ep, closed=True, facecolor="#333333", edgecolor="#555555",
                lw=0.3, zorder=12,
            ))

        # --- Rear wing (2026: larger, with active DRS-less flap) --- #
        rw_w = hw * 1.0
        # Main plane
        rw_main = _t([
            [-w * 0.45, -rw_w],
            [-w * 0.45,  rw_w],
            [-w * 0.50,  rw_w],
            [-w * 0.50, -rw_w],
        ])
        ax.add_patch(patches.Polygon(
            rw_main, closed=True, facecolor="#282828", edgecolor="#444444",
            lw=0.6, zorder=12,
        ))
        # Active upper flap
        rw_flap = _t([
            [-w * 0.50, -rw_w * 0.92],
            [-w * 0.50,  rw_w * 0.92],
            [-w * 0.54,  rw_w * 0.85],
            [-w * 0.54, -rw_w * 0.85],
        ])
        ax.add_patch(patches.Polygon(
            rw_flap, closed=True, facecolor="#333333", edgecolor="#444444",
            lw=0.4, zorder=12,
        ))
        # Rear wing endplates (swan-neck mounted)
        for side in [1, -1]:
            rep = _t([
                [-w * 0.44, side * rw_w],
                [-w * 0.44, side * (rw_w + w * 0.015)],
                [-w * 0.54, side * (rw_w + w * 0.015)],
                [-w * 0.54, side * rw_w],
            ])
            ax.add_patch(patches.Polygon(
                rep, closed=True, facecolor="#333333", edgecolor="#555555",
                lw=0.3, zorder=12,
            ))

        # --- Cockpit opening (dark) --- #
        cockpit = _t([
            [w * 0.16, hw * 0.50],
            [w * 0.08, hw * 0.55],
            [-w * 0.03, hw * 0.50],
            [-w * 0.03, -hw * 0.50],
            [w * 0.08, -hw * 0.55],
            [w * 0.16, -hw * 0.50],
        ])
        ax.add_patch(patches.Polygon(
            cockpit, closed=True, facecolor="#0d0d0d", edgecolor="none",
            zorder=13,
        ))

        # --- Halo (titanium, mandatory) --- #
        halo = _t([
            [w * 0.15, hw * 0.12],
            [w * 0.15, -hw * 0.12],
            [-w * 0.01, -hw * 0.12],
            [-w * 0.01, hw * 0.12],
        ])
        ax.add_patch(patches.Polygon(
            halo, closed=True, facecolor="#555555", edgecolor="#777777",
            lw=0.5, zorder=14,
        ))

        # --- Tyres (2026: wider fronts, 18" wheels) --- #
        # Front tyres are wider for 2026 regs
        ft_l = w * 0.10   # front tyre length
        ft_w = w * 0.055  # front tyre width (wider than before)
        rt_l = w * 0.11   # rear tyre length
        rt_w = w * 0.055  # rear tyre width

        front_axle_x = w * 0.28
        front_y_offset = hw * 1.50
        rear_axle_x = -w * 0.36
        rear_y_offset = hw * 1.40
        fc, fs = np.cos(steer_angle), np.sin(steer_angle)

        for side in [1, -1]:
            # Front tyre (steered, wider 2026)
            local_tyre = np.array([
                [ ft_l,  ft_w],
                [ ft_l, -ft_w],
                [-ft_l, -ft_w],
                [-ft_l,  ft_w],
            ])
            steer_rot = np.array([[fc, -fs], [fs, fc]])
            local_tyre = local_tyre @ steer_rot.T
            local_tyre[:, 0] += front_axle_x
            local_tyre[:, 1] += side * front_y_offset
            tyre_pts = _t(local_tyre.tolist())
            ax.add_patch(patches.Polygon(
                tyre_pts, closed=True, facecolor="#1a1a1a",
                edgecolor="#333333", lw=0.5, zorder=10,
            ))

            # Rear tyre (fixed)
            rear_tyre = _t([
                [rear_axle_x + rt_l, side * (rear_y_offset + rt_w)],
                [rear_axle_x + rt_l, side * (rear_y_offset - rt_w)],
                [rear_axle_x - rt_l, side * (rear_y_offset - rt_w)],
                [rear_axle_x - rt_l, side * (rear_y_offset + rt_w)],
            ])
            ax.add_patch(patches.Polygon(
                rear_tyre, closed=True, facecolor="#1a1a1a",
                edgecolor="#333333", lw=0.5, zorder=10,
            ))

    # ------------------------------------------------------------------ #
    #  Steering wheel — PNG-based, rotated per frame                     #
    # ------------------------------------------------------------------ #
    # Visual rotation gain: map max_steering_angle → ±180° of wheel turn.
    _STEER_VISUAL_GAIN = np.pi / 0.45

    # Load the wheel image once
    _wheel_img = np.array(Image.open(str(get_wheel_path())))

    def draw_steering_wheel(ax, angle_rad):
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")

        # Amplify the rotation so full lock = 180° visual turn
        vis_deg = np.degrees(angle_rad * _STEER_VISUAL_GAIN)

        # Rotate the PIL image (high quality, keeps transparency)
        wheel_pil = Image.fromarray(_wheel_img)
        rotated = wheel_pil.rotate(vis_deg, resample=Image.BICUBIC, expand=False)
        ax.imshow(np.array(rotated))
        ax.set_title("STEERING", color="#888888", fontsize=9, pad=2)

    # ------------------------------------------------------------------ #
    #  Throttle / brake bars                                              #
    # ------------------------------------------------------------------ #
    def draw_bar(ax, value, color, label):
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Background
        bg = patches.FancyBboxPatch(
            (0.1, 0.05), 0.8, 0.85,
            boxstyle="round,pad=0.02", facecolor="#2a2a2a", edgecolor="#444444", lw=1,
        )
        ax.add_patch(bg)

        # Fill
        fill_h = max(0, min(value, 1.0)) * 0.85
        fill = patches.FancyBboxPatch(
            (0.1, 0.05), 0.8, fill_h,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor="none", alpha=0.9,
        )
        ax.add_patch(fill)

        # Label
        ax.text(0.5, 0.95, label, color="#888888", fontsize=9,
                ha="center", va="top", fontweight="bold")
        # Percentage
        ax.text(0.5, 0.0, f"{value * 100:.0f}%", color="white", fontsize=8,
                ha="center", va="bottom")

    # ------------------------------------------------------------------ #
    #  Trail line (speed-coloured trajectory so far)                      #
    # ------------------------------------------------------------------ #
    norm = mcolors.Normalize(vmin=0, vmax=speed_max)

    # ------------------------------------------------------------------ #
    #  Animation update function                                          #
    # ------------------------------------------------------------------ #
    def update(frame_idx):
        ax_track.clear()
        draw_track(ax_track)
        ax_track.set_aspect("equal")
        ax_track.axis("off")

        # Determine which trajectory index we're at
        if frame_idx < len(lap_frame_indices):
            # Driving phase
            ti = lap_frame_indices[frame_idx]
            phase = "driving"
        else:
            # Zoom-out or hold phase
            ti = n_frames_lap - 1
            phase = "zoom_out"

        # Current state
        cx, cy = traj[ti, _X], traj[ti, _Y]
        spd = traj[ti, _SPD]
        hdg = traj[ti, _HDG]
        steer = traj[ti, _STEER]
        accel = traj[ti, _ACCEL]

        # Draw trajectory trail (up to current point)
        trail_end = ti + 1
        if trail_end > 1:
            xs = traj[:trail_end, _X]
            ys = traj[:trail_end, _Y]
            speeds = traj[:trail_end, _SPD]
            # Draw in chunks for speed colouring
            step = max(1, trail_end // 500)  # limit segments for perf
            for i in range(0, trail_end - 1, step):
                c = _SPEED_CMAP(norm(speeds[i]))
                j = min(i + step + 1, trail_end)
                ax_track.plot(xs[i:j], ys[i:j], color=c, lw=1.5,
                              zorder=4, solid_capstyle="round")

        # Draw car
        _draw_car(ax_track, cx, cy, hdg, steer)

        # Camera window
        if phase == "driving":
            hw = _ZOOM_WINDOW / 2
            ax_track.set_xlim(cx - hw, cx + hw)
            ax_track.set_ylim(cy - hw, cy + hw)
        else:
            # Smooth zoom-out
            zoom_frame = frame_idx - len(lap_frame_indices)
            t = min(zoom_frame / max(n_zoom_out, 1), 1.0)
            # Ease-in-out
            t = t * t * (3 - 2 * t)
            # Interpolate from car-centred to full-track
            hw_start = _ZOOM_WINDOW / 2
            cx_interp = cx * (1 - t) + cx_full * t
            cy_interp = cy * (1 - t) + cy_full * t
            hw_interp = hw_start * (1 - t) + span_full * t
            ax_track.set_xlim(cx_interp - hw_interp, cx_interp + hw_interp)
            ax_track.set_ylim(cy_interp - hw_interp, cy_interp + hw_interp)

        # Speed readout
        spd_kmh = spd * 3.6
        speed_text.set_text(f"{spd_kmh:.0f} km/h  |  {spd:.1f} m/s")

        # Steering wheel
        draw_steering_wheel(ax_wheel, steer)

        # Throttle / brake from accel command
        throttle_pct = max(accel, 0.0)
        brake_pct = max(-accel, 0.0)
        draw_bar(ax_thr, throttle_pct, "#00cc44", "THR")
        draw_bar(ax_brk, brake_pct, "#cc2200", "BRK")

        return []

    anim = FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 / _FPS, blit=False,
    )

    # Save — try MP4 (ffmpeg), fall back to GIF
    suffix = out_path.suffix.lower()
    try:
        if suffix == ".mp4" or suffix not in (".gif",):
            out_path = out_path.with_suffix(".mp4")
            anim.save(str(out_path), writer="ffmpeg", fps=_FPS,
                       savefig_kwargs={"facecolor": "#1a1a1a"})
        else:
            anim.save(str(out_path), writer="pillow", fps=_FPS,
                       savefig_kwargs={"facecolor": "#1a1a1a"})
    except Exception:
        # ffmpeg not available — fall back to GIF
        out_path = out_path.with_suffix(".gif")
        anim.save(str(out_path), writer="pillow", fps=min(_FPS, 15),
                   savefig_kwargs={"facecolor": "#1a1a1a"})

    plt.close(fig)
    return out_path
