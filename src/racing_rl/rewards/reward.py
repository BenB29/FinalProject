"""
Reward calculator for the racing-line RL environment.

Design rationale — why this reward teaches a real racing line
-------------------------------------------------------------
The fundamental insight is that the racing line emerges from a *speed-
maximisation* objective constrained by *physics*.  We therefore:

  1. Reward PROGRESS  (delta arc-length per step).
     More forward distance per timestep → more reward.
     The agent is automatically incentivised to go faster.

  2. Penalise GRIP OVERUSE  (slip_ratio > threshold).
     When the car exceeds the traction circle, speed scrubs off and the
     car becomes unstable.  Penalty encourages smooth loading of tyres,
     which is exactly what the racing line does.

  3. Penalise CONTROL ROUGHNESS  (steering jerk, accel jerk).
     Smooth inputs preserve tyre load and allow higher average speed.
     This discourages oscillation and chatter.

  4. Add a tiny SPEED BONUS  to prevent the degenerate policy of stopping.

  5. Give ONE-TIME event rewards/penalties for lap completion / off-track.

What is deliberately ABSENT from the reward
--------------------------------------------
* No lateral offset penalty — the agent is FREE to use the full track width.
  A naive offset penalty would penalise wide entries and exits, which are
  exactly what makes a real racing line.  Without this term the agent must
  discover on its own that a wide entry → late apex → wide exit allows a
  higher minimum cornering speed.

* No target-speed term — there is no "go at this speed" directive.  Speed is
  purely emergent from the progress reward + physics constraints.

* No heading alignment term — the car may point anywhere relative to the
  centreline without penalty; this is necessary for late-apex entry angles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from racing_rl.config.schema import RewardConfig
from racing_rl.physics.vehicle import VehicleState
from racing_rl.tracks.base import TrackQuery


@dataclass
class RewardBreakdown:
    """Detailed decomposition of a single-step reward (useful for debugging)."""

    progress: float = 0.0
    speed_bonus: float = 0.0
    slip_penalty: float = 0.0
    steer_smooth_penalty: float = 0.0
    accel_smooth_penalty: float = 0.0
    event: float = 0.0          # lap completion bonus or off-track penalty
    total: float = 0.0


class RewardCalculator:
    """
    Computes per-step reward given old state, new state, and event flags.

    Parameters
    ----------
    cfg:
        Reward configuration from :class:`RewardConfig`.
    """

    # Slip threshold below which no penalty is applied.
    # Values above this indicate the car is working the tyres harder than
    # the limit, which carries a cost.
    SLIP_THRESHOLD = 0.85

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg

    def compute(
        self,
        state_prev: VehicleState,
        state_new: VehicleState,
        track_query_new: TrackQuery,
        delta_s: float,             # forward arc-length progress [m] this step
        action: np.ndarray,         # [steer_cmd, accel_cmd]
        action_prev: np.ndarray,    # previous action (for jerk penalty)
        off_track: bool = False,
        lap_complete: bool = False,
    ) -> RewardBreakdown:
        """
        Calculate one-step reward.

        Parameters
        ----------
        delta_s:
            Signed arc-length progress this step [m].  Positive = forward.
        action:
            Current action [steer_cmd, accel_cmd].
        action_prev:
            Previous action (same shape) for smoothness penalty.
        off_track:
            True if the vehicle is outside track boundaries.
        lap_complete:
            True if the vehicle just completed a lap.

        Returns
        -------
        RewardBreakdown with .total being the value to return to the agent.
        """
        cfg = self.cfg
        bd = RewardBreakdown()

        # ---- 1. Progress reward ------------------------------------------ #
        # Only reward genuine forward progress.
        # Negative delta_s (reversing) is simply zero progress reward.
        effective_progress = max(delta_s, 0.0)
        bd.progress = cfg.w_progress * effective_progress

        # ---- 2. Speed bonus  (anti-crawl) -------------------------------- #
        bd.speed_bonus = cfg.w_speed * state_new.speed

        # ---- 3. Grip overuse penalty -------------------------------------- #
        # slip_ratio ∈ [0, …].  Below SLIP_THRESHOLD no cost.
        # Above threshold, quadratic penalty grows rapidly.
        over_slip = max(state_new.slip_ratio - self.SLIP_THRESHOLD, 0.0)
        bd.slip_penalty = cfg.w_slip * (over_slip ** 2)

        # ---- 4. Steering smoothness penalty ------------------------------- #
        # Large steering-command changes chatter the tyres and indicate
        # sub-optimal control.  Penalise the *rate* of steer command change.
        delta_steer = action[0] - action_prev[0]
        bd.steer_smooth_penalty = cfg.w_steer_smooth * (delta_steer ** 2)

        # ---- 5. Acceleration smoothness penalty --------------------------- #
        delta_accel = action[1] - action_prev[1]
        bd.accel_smooth_penalty = cfg.w_accel_smooth * (delta_accel ** 2)

        # ---- 6. Event rewards / penalties -------------------------------- #
        if off_track:
            # Scale penalty with speed: crashing at full speed hurts more.
            # At speed=0 → 1x penalty, at speed=max → 2x penalty.
            speed_scale = 1.0 + state_new.speed / 80.0
            bd.event = cfg.off_track_penalty * speed_scale
        elif lap_complete:
            bd.event = cfg.lap_completion_bonus

        # ---- Sum up ------------------------------------------------------ #
        bd.total = (
            bd.progress
            + bd.speed_bonus
            + bd.slip_penalty
            + bd.steer_smooth_penalty
            + bd.accel_smooth_penalty
            + bd.event
        )

        return bd
