"""
Evaluate a saved model and save a racing-line PNG.

Usage examples
--------------
    python scripts/evaluate.py --mode B
    python scripts/evaluate.py --model outputs/obs_B/models/best_model.zip --mode B
    python scripts/evaluate.py --model outputs/obs_C/models/best_model.zip --mode C --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from racing_rl.config.loader import load_config, load_config_for_mode
from racing_rl.evaluation.evaluator import run_evaluation
from racing_rl.tracks.parametric import build_track
from racing_rl.utils.path_utils import get_run_dir
from stable_baselines3 import SAC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained racing-line model")
    p.add_argument("--mode", "-m", choices=["A", "B", "C"], default="B")
    p.add_argument("--config", "-c", default=None, help="Override config YAML.")
    p.add_argument(
        "--model",
        default=None,
        help="Path to model .zip.  Defaults to best_model.zip in run dir.",
    )
    p.add_argument("--episodes", "-e", type=int, default=3)
    p.add_argument("--out-dir", default=None, help="Directory for PNG output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config_for_mode(args.mode)

    run_dir = get_run_dir(cfg)

    model_path = (
        Path(args.model)
        if args.model
        else run_dir / "models" / "best_model.zip"
    )
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    track = build_track(cfg.track.name, cfg.track.half_width)
    model = SAC.load(str(model_path))

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval_pngs"

    print(f"Evaluating {model_path} ...")
    metrics, png_path = run_evaluation(
        model=model,
        cfg=cfg,
        track=track,
        out_dir=out_dir,
        eval_index=0,
        n_episodes=args.episodes,
        is_best=False,
    )

    print(f"\n{'='*50}")
    print(f"  Lap complete   : {metrics.lap_complete}")
    print(f"  Lap time       : {metrics.lap_time:.2f} s")
    print(f"  Completion     : {metrics.completion_pct:.1f}%")
    print(f"  Mean speed     : {metrics.mean_speed:.1f} m/s")
    print(f"  Max slip       : {metrics.max_slip:.3f}")
    print(f"  PNG saved to   : {png_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
