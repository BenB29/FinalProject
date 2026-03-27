"""
Non-learned baseline controllers for comparison with the RL agent.

Three controllers are available:
  1. pure-pursuit  — steers toward centreline + curvature-proportional speed
  2. constant      — steers toward centreline + fixed throttle (no speed mgmt)
  3. random        — uniform random actions (lower bound)

Usage
-----
    python scripts/baseline_centreline.py --config configs/silverstone.yaml --controller pure-pursuit
    python scripts/baseline_centreline.py --config configs/silverstone.yaml --controller constant
    python scripts/baseline_centreline.py --config configs/silverstone.yaml --controller random --episodes 10
    python scripts/baseline_centreline.py --config configs/silverstone.yaml --controller all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from racing_rl.config.loader import load_config
from racing_rl.env.racing_env import RacingEnv
from racing_rl.tracks.parametric import build_track
from racing_rl.utils.path_utils import _REPO_ROOT


# ---- Shared steering gains ----------------------------------------------- #
K_OFFSET = 0.08       # lateral offset correction
K_HEADING = 1.2        # heading error correction

# ---- Pure-pursuit speed gains -------------------------------------------- #
V_MAX = 80.0           # m/s
MU_G = 9.81 * 1.5      # mu * g
K_SPEED = 0.15         # proportional speed gain
CURVATURE_EPS = 1e-4

# ---- Constant-speed gains ------------------------------------------------ #
CONSTANT_THROTTLE = 0.3   # fixed throttle command (gentle cruise)


# ========================================================================== #
#  Controller functions                                                        #
# ========================================================================== #

def _steer_to_centreline(info: dict) -> float:
    """Proportional steering toward the centreline (shared by two controllers)."""
    lat_offset = info["lateral_offset"]
    heading_err = info["heading_error"]
    heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
    steer = -K_OFFSET * lat_offset - K_HEADING * heading_err
    return float(np.clip(steer, -1.0, 1.0))


def pure_pursuit_action(info: dict, lookahead_curvatures: np.ndarray,
                        current_speed: float) -> np.ndarray:
    """Steer to centreline + brake before corners using lookahead curvature."""
    steer_cmd = _steer_to_centreline(info)

    max_curv = max(float(np.max(np.abs(lookahead_curvatures))), CURVATURE_EPS)
    target_speed = min(V_MAX, math.sqrt(MU_G / max_curv))
    accel_cmd = float(np.clip(K_SPEED * (target_speed - current_speed), -1.0, 1.0))

    return np.array([steer_cmd, accel_cmd], dtype=np.float32)


def constant_speed_action(info: dict) -> np.ndarray:
    """Steer to centreline + fixed throttle. No speed management at all."""
    steer_cmd = _steer_to_centreline(info)
    return np.array([steer_cmd, CONSTANT_THROTTLE], dtype=np.float32)


def random_action(action_space) -> np.ndarray:
    """Uniform random action."""
    return action_space.sample()


# ========================================================================== #
#  Episode runner                                                              #
# ========================================================================== #

def run_episode(cfg, controller: str, episode_idx: int = 0) -> dict:
    """Run one episode with the given controller and return metrics."""
    cfg.obs_mode = "C"  # need curvature features for pure-pursuit
    env = RacingEnv(cfg, eval_mode=True)
    track = build_track(cfg.track.name, cfg.track.half_width)

    obs, _ = env.reset(seed=cfg.seed + episode_idx)

    info = {"lateral_offset": 0.0, "heading_error": 0.0, "speed": 80.0}
    x0, y0, _ = track.start_position()
    tq = track.query(x0, y0, cfg.track.lookahead_distances)
    lookahead_curvatures = tq.lookahead_curvatures

    done = False
    step_count = 0

    while not done:
        if controller == "pure-pursuit":
            action = pure_pursuit_action(info, lookahead_curvatures, info["speed"])
        elif controller == "constant":
            action = constant_speed_action(info)
        elif controller == "random":
            action = random_action(env.action_space)
        else:
            raise ValueError(f"Unknown controller: {controller}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        # Update lookahead curvatures for pure-pursuit
        if controller == "pure-pursuit":
            tq = track.query(
                env._state.x, env._state.y,
                cfg.track.lookahead_distances,
                hint_idx=env._hint_idx,
            )
            lookahead_curvatures = tq.lookahead_curvatures

    traj = env.get_trajectory()
    env.close()

    result = {
        "episode": episode_idx,
        "steps": step_count,
        "lap_complete": info.get("lap_complete", False),
        "lap_progress": info.get("episode_lap_progress", info.get("lap_progress", 0.0)),
        "lap_time": info.get("episode_lap_time", info.get("lap_time", 0.0)),
        "max_slip": info.get("episode_max_slip", 0.0),
        "trajectory_len": len(traj),
    }
    if traj:
        result["mean_speed"] = float(np.mean([t[2] for t in traj]))
    else:
        result["mean_speed"] = 0.0

    return result


# ========================================================================== #
#  Summary + saving                                                            #
# ========================================================================== #

def print_and_save(controller: str, results: list, track_name: str,
                   experiment: str | None) -> None:
    """Print summary table and save JSON."""
    completions = [r["lap_complete"] for r in results]
    progresses = [r["lap_progress"] for r in results]
    times = [r["lap_time"] for r in results if r["lap_complete"]]
    speeds = [r["mean_speed"] for r in results]
    slips = [r["max_slip"] for r in results]

    print(f"\n  {'='*55}")
    print(f"  {controller.upper()} — {track_name}  ({len(results)} episodes)")
    print(f"  {'='*55}")
    print(f"  Laps completed:   {sum(completions)} / {len(results)}")
    print(f"  Mean completion:  {np.mean(progresses)*100:.1f}%")
    print(f"  Best completion:  {np.max(progresses)*100:.1f}%")
    if times:
        print(f"  Best lap time:    {min(times):.2f}s")
        print(f"  Mean lap time:    {np.mean(times):.2f}s")
    print(f"  Mean speed:       {np.mean(speeds):.1f} m/s")
    print(f"  Mean max slip:    {np.mean(slips):.3f}")

    # Save
    if experiment:
        out_dir = _REPO_ROOT / "outputs" / experiment / "baseline"
    else:
        out_dir = _REPO_ROOT / "outputs" / f"experiment_{track_name}" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "track": track_name,
        "controller": controller,
        "episodes": len(results),
        "laps_completed": sum(completions),
        "mean_completion_pct": float(np.mean(progresses) * 100),
        "best_completion_pct": float(np.max(progresses) * 100),
        "best_lap_time": min(times) if times else None,
        "mean_lap_time": float(np.mean(times)) if times else None,
        "mean_speed": float(np.mean(speeds)),
        "mean_max_slip": float(np.mean(slips)),
        "per_episode": results,
    }

    out_path = out_dir / f"{controller}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")


# ========================================================================== #
#  Main                                                                        #
# ========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-learned baseline controllers")
    p.add_argument("--config", "-c", required=True, help="Config YAML file.")
    p.add_argument("--controller", choices=["pure-pursuit", "constant", "random", "all"],
                   default="all", help="Which controller to run.")
    p.add_argument("--episodes", "-n", type=int, default=1,
                   help="Episodes per controller (random benefits from more).")
    p.add_argument("--experiment", "-e", default=None,
                   help="Experiment folder name for saving results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    track_name = cfg.track.name

    if args.controller == "all":
        controllers = ["pure-pursuit", "constant", "random"]
    else:
        controllers = [args.controller]

    # Use more episodes for random by default
    episodes_map = {
        "pure-pursuit": args.episodes,
        "constant": args.episodes,
        "random": max(args.episodes, 10),  # random needs more to be meaningful
    }

    print(f"\nBaseline Evaluation — {track_name}")
    print(f"  Controllers: {', '.join(controllers)}")

    for ctrl in controllers:
        n_eps = episodes_map[ctrl]
        results = []
        for i in range(n_eps):
            r = run_episode(cfg, ctrl, episode_idx=i)
            results.append(r)
            status = "LAP" if r["lap_complete"] else f"{r['lap_progress']*100:.1f}%"
            print(f"  [{ctrl}] ep {i}: {status}  "
                  f"time={r['lap_time']:.2f}s  speed={r['mean_speed']:.1f}m/s  "
                  f"slip={r['max_slip']:.3f}")

        print_and_save(ctrl, results, track_name, args.experiment)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
