"""
Train a single observation mode.

Usage examples
--------------
    python scripts/train.py                          # default (mode B)
    python scripts/train.py --mode A
    python scripts/train.py --config configs/suzuka_full.yaml
    python scripts/train.py --config configs/suzuka_full.yaml --name my_run_v2
    python scripts/train.py --mode B --timesteps 500000 --name quick_test
    python scripts/train.py --resume outputs/my_run/models/latest_model.zip --config configs/suzuka_full.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src package importable when running scripts directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from racing_rl.config.loader import load_config, load_config_for_mode
from racing_rl.training.trainer import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC racing-line agent")
    p.add_argument(
        "--mode", "-m",
        choices=["A", "B", "C"],
        default=None,
        help="Observation mode (A/B/C).  Ignored if --config is provided.",
    )
    p.add_argument(
        "--config", "-c",
        default=None,
        help="Path to override YAML (merged on top of configs/base.yaml).",
    )
    p.add_argument(
        "--name", "-n",
        default=None,
        help="Experiment name (used as output folder name).  "
             "Overrides whatever is in the config.",
    )
    p.add_argument(
        "--timesteps", "-t",
        type=int,
        default=None,
        help="Override training timesteps.",
    )
    p.add_argument(
        "--resume", "-r",
        default=None,
        help="Path to a model .zip to resume from.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    if args.config:
        cfg = load_config(args.config)
    elif args.mode:
        cfg = load_config_for_mode(args.mode)
    else:
        cfg = load_config_for_mode("B")

    # Apply CLI overrides
    if args.name is not None:
        cfg.experiment_name = args.name
    if args.timesteps is not None:
        cfg.training.total_timesteps = args.timesteps
    if args.seed is not None:
        cfg.seed = args.seed

    print(f"\n{'='*60}")
    print(f"  Racing-Line RL Training")
    print(f"  Experiment : {cfg.experiment_name}")
    print(f"  Track      : {cfg.track.name}")
    print(f"  Obs mode   : {cfg.obs_mode}  (dim={cfg.obs_dim})")
    print(f"  Timesteps  : {cfg.training.total_timesteps:,}")
    print(f"  Seed       : {cfg.seed}")
    print(f"  Output     : outputs/{cfg.experiment_name}/")
    print(f"{'='*60}\n")

    resume = Path(args.resume) if args.resume else None
    model = train(cfg, resume_from=resume)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
