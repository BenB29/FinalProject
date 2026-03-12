"""
RacingLine — F1-style racing-line optimisation via RL.

This is the top-level entry point.  For full usage, see the scripts/ directory:

    python scripts/train.py --mode B          # train single mode
    python scripts/run_experiment.py          # train & compare A/B/C
    python scripts/evaluate.py --mode B       # evaluate best saved model
    python scripts/compare_runs.py            # plot training curves

Quick start (runs mode B for 50k steps, good for a smoke-test):
    python main.py --mode B --timesteps 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from racing_rl.config.loader import load_config_for_mode
from racing_rl.training.trainer import train


def main() -> None:
    p = argparse.ArgumentParser(description="RacingLine RL — quick-start entry point")
    p.add_argument("--mode", "-m", choices=["A", "B", "C"], default="B")
    p.add_argument("--timesteps", "-t", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config_for_mode(args.mode)
    if args.timesteps:
        cfg.training.total_timesteps = args.timesteps
    if args.seed is not None:
        cfg.seed = args.seed

    print(f"Starting training — obs_mode={cfg.obs_mode}, timesteps={cfg.training.total_timesteps:,}")
    train(cfg)


if __name__ == "__main__":
    main()
