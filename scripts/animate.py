"""
Generate an animated replay from a saved model.

Usage examples
--------------
    python scripts/animate.py --config configs/bahrain.yaml
    python scripts/animate.py --config configs/suzuka_full.yaml --model outputs/suzuka_full/models/best_model.zip
    python scripts/animate.py --config configs/barcelona.yaml --out my_lap.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from racing_rl.config.loader import load_config, load_config_for_mode
from racing_rl.env.racing_env import RacingEnv
from racing_rl.plotting.animation import save_lap_animation
from racing_rl.tracks.parametric import build_track
from racing_rl.utils.path_utils import get_run_dir
from stable_baselines3 import SAC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate animated lap replay")
    p.add_argument("--config", "-c", default=None, help="Config YAML file.")
    p.add_argument("--mode", "-m", choices=["A", "B", "C"], default="B",
                   help="Obs mode (ignored if --config provided).")
    p.add_argument("--model", default=None,
                   help="Path to model .zip. Defaults to best_model.zip in run dir.")
    p.add_argument("--out", "-o", default=None,
                   help="Output file path. Defaults to <run_dir>/animations/best_lap.mp4")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config_for_mode(args.mode)

    run_dir = get_run_dir(cfg)

    model_path = (
        Path(args.model) if args.model
        else run_dir / "models" / "best_model.zip"
    )
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    track = build_track(cfg.track.name, cfg.track.half_width)
    model = SAC.load(str(model_path))

    # Run one deterministic episode to collect trajectory
    env = RacingEnv(cfg, eval_mode=True)
    obs, _ = env.reset(seed=cfg.seed)
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    traj = env.get_trajectory()
    env.close()

    lap_complete = info.get("lap_complete", False)
    lap_time = info.get("episode_lap_time", info.get("lap_time", 0.0))
    completion_pct = info.get("episode_lap_progress", 0.0) * 100.0
    speeds = [p[2] for p in traj] if traj else [0.0]
    mean_speed = float(np.mean(speeds))

    title = (
        f"{cfg.track.name}  |  "
        f"{'Lap: {:.2f}s'.format(lap_time) if lap_complete else 'Completion: {:.1f}%'.format(completion_pct)}"
        f"  |  Avg speed: {mean_speed:.1f} m/s"
    )

    out_path = Path(args.out) if args.out else run_dir / "animations" / "best_lap.mp4"

    print(f"Rendering animation ({len(traj)} frames) ...")
    saved = save_lap_animation(
        track=track,
        trajectory=list(traj),
        out_path=out_path,
        speed_max=cfg.obs.speed_max,
        title=title,
    )
    print(f"Animation saved to {saved}")


if __name__ == "__main__":
    main()
