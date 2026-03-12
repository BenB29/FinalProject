"""
Simplified but physically credible single-track (bicycle) vehicle model.

Design philosophy
-----------------
This is NOT a full tyre-mechanics simulator.  The goal is *just enough*
physics to force real racing-line behaviour:

  * Corners must be taken at a speed compatible with the available lateral grip.
  * Taking a longer outer line allows higher minimum-corner-speed, which
    matters for exit velocity and therefore lap time.
  * Braking must happen before heavy corners; the car cannot magically
    decelerate mid-corner without losing lateral grip.

Traction circle
---------------
The combined grip budget is:

    sqrt(F_lon^2 + F_lat^2) <= mu * m * g

where F_lat comes from the centripetal demand (v^2 * kappa_path / wheelbase)
and F_lon is the net longitudinal drive/brake force minus drag/rolling.

If the combined demand exceeds the budget, both forces are scaled down
proportionally (grip-saturation).  This models the real trade-off: braking
into a corner steals lateral budget, and vice-versa.

Steering model
--------------
The path curvature is determined by the bicycle model:

    kappa_path = tan(steering_angle) / wheelbase

A first-order rate-limit on steering_angle prevents instantaneous slalom
inputs that are physically impossible for a high-speed car.

Action interface
----------------
Action = np.ndarray([steering_cmd, accel_cmd]) both in [-1, 1].

  * steering_cmd  → target steering-angle direction; applied as rate-limited
                    delta each step.
  * accel_cmd > 0 → throttle (normalised to max_throttle_accel).
  * accel_cmd < 0 → brake   (normalised to max_brake_decel).

Step pipeline (matches spec in the project README)
---------------------------------------------------
 1. Unpack action
 2. Clamp raw commands to [-1, 1]
 3. Apply steering rate limit → new steering_angle
 4. Clamp steering_angle to [-max_steering_angle, max_steering_angle]
 5. Compute commanded longitudinal acceleration (throttle or brake)
 6. Apply drag and rolling resistance → net longitudinal acceleration
 7. Compute path curvature from steering angle (bicycle model)
 8. Compute lateral acceleration demand  a_lat = v^2 * kappa_path
 9. Compute available grip budget  a_max = mu * g
10. Traction-circle scaling: if combined demand > budget, scale both
11. Compute slip metric (ratio to grip limit, clamped to [0, 1+])
12. Update speed (clamp to [0, max_speed])
13. Update heading  += omega * dt  where omega = v * kappa_path (post-scaling)
14. Update position
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from racing_rl.config.schema import VehicleConfig


# --------------------------------------------------------------------------- #
#  Vehicle State                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class VehicleState:
    """Mutable snapshot of the vehicle's physical state."""

    x: float = 0.0              # global x position  [m]
    y: float = 0.0              # global y position  [m]
    heading: float = 0.0        # global heading (yaw) [rad]
    speed: float = 0.0          # longitudinal speed  [m/s]
    steering_angle: float = 0.0 # current steering angle  [rad]
    throttle: float = 0.0       # last throttle command (normalised [0,1])
    brake: float = 0.0          # last brake command (normalised [0,1])

    # Derived quantities updated each step
    slip_ratio: float = 0.0     # how close we are to grip limit [0 = none, 1 = limit]
    lat_accel: float = 0.0      # lateral acceleration [m/s^2]
    lon_accel: float = 0.0      # net longitudinal acceleration [m/s^2]
    path_curvature: float = 0.0 # actual path curvature [1/m]

    # Running totals
    distance_travelled: float = 0.0  # total arc length driven [m]
    elapsed_time: float = 0.0        # total simulated time [s]
    steps: int = 0                   # number of steps taken

    def copy(self) -> "VehicleState":
        import copy
        return copy.copy(self)


# --------------------------------------------------------------------------- #
#  Vehicle Physics                                                             #
# --------------------------------------------------------------------------- #

class VehiclePhysics:
    """
    Stateless physics engine.  Accepts a VehicleState and an action, returns
    a new VehicleState.

    Keeping it stateless makes it easy to roll back, copy, or parallelise.
    """

    def __init__(self, cfg: VehicleConfig) -> None:
        self.cfg = cfg
        # Derived constants
        self._max_grip_accel = cfg.mu_grip * cfg.gravity   # m/s^2
        # Air density absorbed into drag_coeff (lumped)
        # F_drag = drag_coeff * v^2  →  a_drag = F_drag / mass
        self._drag_per_mass = cfg.drag_coeff / cfg.mass
        # Rolling resistance deceleration  (constant, independent of speed)
        self._roll_decel = cfg.roll_resist_coeff * cfg.gravity

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def step(
        self,
        state: VehicleState,
        action: np.ndarray,
    ) -> Tuple[VehicleState, float]:
        """
        Advance the vehicle by one timestep.

        Parameters
        ----------
        state:
            Current vehicle state (not mutated).
        action:
            np.ndarray([steering_cmd, accel_cmd]) in [-1, 1].

        Returns
        -------
        new_state:
            Updated vehicle state after one timestep.
        slip_ratio:
            Grip usage metric ∈ [0, …].  Values > 1.0 mean over-limit.
        """
        cfg = self.cfg
        dt = cfg.dt

        # ---- 1-2. Unpack and clamp action ------------------------------ #
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        accel_cmd = float(np.clip(action[1], -1.0, 1.0))

        # ---- 3-4. Steering rate limit ---------------------------------- #
        # steer_cmd drives steering angle toward ±max_steering_angle
        target_steer = steer_cmd * cfg.max_steering_angle
        delta_steer = target_steer - state.steering_angle
        max_delta = cfg.max_steering_rate * dt
        delta_steer = np.clip(delta_steer, -max_delta, max_delta)
        new_steering = np.clip(
            state.steering_angle + delta_steer,
            -cfg.max_steering_angle,
            cfg.max_steering_angle,
        )

        # ---- 5. Commanded longitudinal acceleration -------------------- #
        if accel_cmd >= 0.0:
            throttle_norm = accel_cmd
            brake_norm = 0.0
            cmd_lon_accel = throttle_norm * cfg.max_throttle_accel
        else:
            throttle_norm = 0.0
            brake_norm = -accel_cmd
            cmd_lon_accel = -brake_norm * cfg.max_brake_decel

        # ---- 6. Drag and rolling resistance ---------------------------- #
        v = state.speed
        drag_decel = self._drag_per_mass * v * v   # proportional to v^2
        roll_decel = self._roll_decel if v > 0.01 else 0.0

        # Net longitudinal acceleration (before traction-circle check)
        net_lon_accel = cmd_lon_accel - drag_decel - roll_decel

        # ---- 7. Path curvature from bicycle model ---------------------- #
        # kappa = tan(delta) / L  (small-angle OK for our delta range)
        kappa_path = math.tan(new_steering) / cfg.wheelbase

        # ---- 8. Lateral acceleration demand ---------------------------- #
        a_lat_demand = v * v * kappa_path   # centripetal [m/s^2]

        # ---- 9-10. Traction circle ------------------------------------- #
        # Combined demand vs budget
        a_lon = net_lon_accel
        a_lat = a_lat_demand
        combined = math.sqrt(a_lon * a_lon + a_lat * a_lat)
        budget = self._max_grip_accel

        if combined > budget and combined > 1e-9:
            scale = budget / combined
            a_lon *= scale
            a_lat *= scale
            # Actual path curvature after grip scaling
            kappa_actual = a_lat / (v * v) if v > 0.5 else kappa_path
        else:
            kappa_actual = kappa_path

        # ---- 11. Slip metric ------------------------------------------ #
        slip_ratio = combined / budget if budget > 0 else 0.0

        # ---- 12. Update speed ----------------------------------------- #
        new_speed = v + a_lon * dt
        new_speed = float(np.clip(new_speed, 0.0, cfg.max_speed))

        # Use average speed for position update (trapezoidal integration)
        avg_speed = 0.5 * (v + new_speed)

        # ---- 13. Update heading --------------------------------------- #
        # omega = v * kappa (yaw rate)
        omega = avg_speed * kappa_actual
        new_heading = state.heading + omega * dt

        # ---- 14. Update position -------------------------------------- #
        new_x = state.x + avg_speed * math.cos(new_heading) * dt
        new_y = state.y + avg_speed * math.sin(new_heading) * dt

        # ---- Build new state ----------------------------------------- #
        ns = VehicleState(
            x=new_x,
            y=new_y,
            heading=new_heading,
            speed=new_speed,
            steering_angle=float(new_steering),
            throttle=throttle_norm,
            brake=brake_norm,
            slip_ratio=float(np.clip(slip_ratio, 0.0, 5.0)),
            lat_accel=a_lat,
            lon_accel=a_lon,
            path_curvature=kappa_actual,
            distance_travelled=state.distance_travelled + avg_speed * dt,
            elapsed_time=state.elapsed_time + dt,
            steps=state.steps + 1,
        )

        return ns, float(slip_ratio)

    def make_initial_state(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float = 0.0,
    ) -> VehicleState:
        """Create a freshly-initialised vehicle state at a given pose."""
        return VehicleState(
            x=x,
            y=y,
            heading=heading,
            speed=speed,
            steering_angle=0.0,
            throttle=0.0,
            brake=0.0,
        )
