"""Generate a high-quality F1 steering wheel PNG asset.

Run once to create the asset; the animation code loads and rotates it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_ASSET_PATH = Path(__file__).parent / "f1_wheel.png"


def generate_wheel(size: int = 512, force: bool = False) -> Path:
    """Render an F1 butterfly steering wheel to a transparent PNG."""
    if _ASSET_PATH.exists() and not force:
        return _ASSET_PATH

    fig, ax = plt.subplots(figsize=(6, 6), dpi=size // 6)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    r = 1.05

    # --- Outer rim: butterfly/yoke shape --- #
    # Top arc
    top_angles = np.linspace(0.5, np.pi - 0.5, 40)
    top_x = r * np.cos(top_angles)
    top_y = r * np.sin(top_angles)

    # Right grip curve down
    right_angles = np.linspace(-0.5, -1.25, 15)
    right_x = r * np.cos(right_angles)
    right_y = r * np.sin(right_angles)

    # Bottom flat (butterfly cutout)
    bot_right = (r * np.cos(1.25), -r * np.sin(1.25))
    bot_left = (-r * np.cos(1.25), -r * np.sin(1.25))

    # Left grip curve up
    left_angles = np.linspace(np.pi + 1.25, np.pi + 0.5, 15)
    left_x = r * np.cos(left_angles)
    left_y = r * np.sin(left_angles)

    rim_x = np.concatenate([top_x, right_x, [bot_right[0], bot_left[0]], left_x])
    rim_y = np.concatenate([top_y, right_y, [bot_right[1], bot_left[1]], left_y])
    rim_pts = list(zip(rim_x, rim_y))

    # Outer rim fill (dark carbon)
    ax.add_patch(patches.Polygon(
        rim_pts, closed=True,
        facecolor="#222222", edgecolor="#666666", lw=3.0, zorder=2,
    ))

    # Inner cutout to make it look like a rim (not a filled disc)
    inner_r = 0.72
    # Same shape but smaller
    top_i = np.linspace(0.55, np.pi - 0.55, 30)
    right_i = np.linspace(-0.55, -1.15, 10)
    left_i = np.linspace(np.pi + 1.15, np.pi + 0.55, 10)
    bi_r = (inner_r * np.cos(1.15), -inner_r * np.sin(1.15))
    bi_l = (-inner_r * np.cos(1.15), -inner_r * np.sin(1.15))
    inner_x = np.concatenate([
        inner_r * np.cos(top_i), inner_r * np.cos(right_i),
        [bi_r[0], bi_l[0]], inner_r * np.cos(left_i),
    ])
    inner_y = np.concatenate([
        inner_r * np.sin(top_i), inner_r * np.sin(right_i),
        [bi_r[1], bi_l[1]], inner_r * np.sin(left_i),
    ])

    # --- Centre hub (where buttons/screen live) --- #
    hub = patches.FancyBboxPatch(
        (-0.62, -0.42), 1.24, 0.84,
        boxstyle="round,pad=0.08",
        facecolor="#1a1a1a", edgecolor="#444444", lw=2.0, zorder=5,
    )
    ax.add_patch(hub)

    # --- Grip sections (thicker at 3 and 9 o'clock) --- #
    for side in [1, -1]:
        grip = patches.FancyBboxPatch(
            (side * 0.76 - 0.18, -0.38), 0.36, 0.76,
            boxstyle="round,pad=0.04",
            facecolor="#333333", edgecolor="#555555", lw=1.5, zorder=4,
        )
        ax.add_patch(grip)
        # Grip ridges
        for gy in np.linspace(-0.28, 0.28, 6):
            ax.plot(
                [side * 0.65, side * 1.02], [gy, gy],
                color="#3d3d3d", lw=1.2, zorder=4.5,
            )

    # --- Display screen (centre) --- #
    screen = patches.FancyBboxPatch(
        (-0.42, -0.16), 0.84, 0.32,
        boxstyle="round,pad=0.03",
        facecolor="#001800", edgecolor="#004400", lw=1.5, zorder=6,
    )
    ax.add_patch(screen)

    # Screen segments (rev lights / data)
    for i, col in enumerate(["#003300", "#004400", "#003300", "#004400", "#003300"]):
        sx = -0.34 + i * 0.17
        seg = patches.Rectangle(
            (sx, -0.08), 0.14, 0.20,
            facecolor=col, edgecolor="none", zorder=6.5,
        )
        ax.add_patch(seg)

    # --- Buttons --- #
    button_specs = [
        # Row above screen
        (-0.28, 0.28, "#ee2222", 7),
        (0.00, 0.30, "#22cc22", 7),
        (0.28, 0.28, "#2255ee", 7),
        # Upper row
        (-0.28, 0.50, "#ee8800", 5.5),
        (0.00, 0.52, "#cc22cc", 5.5),
        (0.28, 0.50, "#cccc22", 5.5),
        # Top row (near rim)
        (-0.15, 0.68, "#cccccc", 4),
        (0.15, 0.68, "#cccccc", 4),
    ]
    for bx, by, col, sz in button_specs:
        ax.plot(bx, by, "o", color=col, markersize=sz, zorder=8,
                markeredgecolor="#111111", markeredgewidth=0.5)

    # --- Rotary encoders (left and right of screen) --- #
    for side in [1, -1]:
        cx_dial = side * 0.56
        dial = plt.Circle(
            (cx_dial, 0.0), 0.09,
            color="#444444", ec="#666666", lw=1.5, zorder=8,
        )
        ax.add_patch(dial)
        # Pointer
        ax.plot([cx_dial, cx_dial], [0.0, 0.07],
                color="#999999", lw=1.5, zorder=9)

    # --- Paddle shifters (behind wheel) --- #
    for side in [1, -1]:
        paddle = patches.FancyBboxPatch(
            (side * 0.55 - 0.22, -0.72), 0.44, 0.18,
            boxstyle="round,pad=0.03",
            facecolor="#555555", edgecolor="#777777", lw=1.0, zorder=1,
        )
        ax.add_patch(paddle)

    # --- Top centre marker (12 o'clock reference) --- #
    marker = patches.Rectangle(
        (-0.04, 0.90), 0.08, 0.18,
        facecolor="#ff2222", edgecolor="none", zorder=10,
    )
    ax.add_patch(marker)

    # --- "F1" text on hub --- #
    ax.text(0.0, -0.30, "F1", color="#666666", fontsize=12, fontweight="bold",
            ha="center", va="center", zorder=8)

    fig.savefig(str(_ASSET_PATH), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return _ASSET_PATH


def get_wheel_path() -> Path:
    """Return path to wheel asset, generating if needed."""
    if not _ASSET_PATH.exists():
        generate_wheel()
    return _ASSET_PATH


if __name__ == "__main__":
    p = generate_wheel(force=True)
    print(f"Steering wheel asset saved to: {p}")
