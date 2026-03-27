"""
Centreline-following baseline agent.

Uses a simple PD controller to steer toward the track centreline and
applies constant full throttle. This provides a lower-bound baseline
to compare against the RL agents.

Usage
-----
    python scripts/run_baseline.py --config configs/suzuka_full.yaml
    python scripts/run_baseline.py --config configs/silverstone.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from racing_rl.config.loader import load_config
from racing_rl.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centreline-following baseline")
    p.add_argument("--config", "-c", required=True, help="Track config YAML")
    p.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
    return p.parse_args()


class CentrelineAgent:
    """
    PD controller that steers toward the centreline at full throttle.

    Steering = -Kp * lateral_offset - Kd * heading_error
    Throttle = 1.0 (always full throttle)
    """

    def __init__(self, half_width: float):
        self.kp = 2.0 / half_width  # proportional gain (normalised by track width)
        self.kd = 1.5               # derivative gain on heading error

    def act(self, lateral_offset: float, heading_error: float) -> np.ndarray:
        steer = -self.kp * lateral_offset - self.kd * heading_error
        steer = np.clip(steer, -1.0, 1.0)
        throttle = 1.0
        return np.array([steer, throttle], dtype=np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.obs_mode = "B"  # need heading error in obs, but we use info dict directly

    env = RacingEnv(cfg, eval_mode=True)
    agent = CentrelineAgent(cfg.track.half_width)

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            tq = env.track.query(env._state.x, env._state.y,
                                 cfg.track.lookahead_distances,
                                 hint_idx=env._hint_idx)
            lat_off = tq.lateral_offset
            head_err = env._state.heading - tq.track_heading
            # Wrap heading error to [-pi, pi]
            head_err = (head_err + np.pi) % (2 * np.pi) - np.pi

            action = agent.act(lat_off, head_err)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        lap_complete = info.get("lap_complete", False)
        lap_time = info.get("episode_lap_time", 0.0)
        lap_progress = info.get("episode_lap_progress", 0.0)

        results.append({
            "episode": ep + 1,
            "lap_complete": lap_complete,
            "lap_time": f"{lap_time:.2f}" if lap_complete else "N/A",
            "completion_%": f"{lap_progress * 100:.1f}",
            "reward": f"{total_reward:.1f}",
        })

        status = f"LAP {lap_time:.2f}s" if lap_complete else f"{lap_progress*100:.1f}%"
        print(f"  Episode {ep+1}: {status}")

    # Summary
    completions = [r for r in results if r["lap_complete"]]
    print(f"\n{'='*50}")
    print(f"Baseline: Centreline follower (full throttle)")
    print(f"Track: {cfg.track.name}")
    print(f"Laps completed: {len(completions)}/{len(results)}")
    if completions:
        times = [float(r["lap_time"]) for r in completions]
        print(f"Best lap time: {min(times):.2f}s")
        print(f"Mean lap time: {np.mean(times):.2f}s")
    else:
        best_pct = max(float(r["completion_%"]) for r in results)
        print(f"Best completion: {best_pct:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
