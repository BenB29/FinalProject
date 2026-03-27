"""
Train a PPO agent for comparison with the SAC baseline.

Uses the same environment, eval callback, and config as the SAC trainer,
but swaps the algorithm to PPO.  This enables a direct algorithm comparison
(SAC vs PPO) on the same task with the same observation mode and reward.

Usage
-----
    python scripts/train_ppo.py --config configs/suzuka_full.yaml --mode C --timesteps 10000000 --name ppo_suzuka_C
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from racing_rl.config.loader import load_config, load_config_for_mode
from racing_rl.env.racing_env import RacingEnv
from racing_rl.tracks.parametric import build_track
from racing_rl.training.callbacks import RacingEvalCallback
from racing_rl.utils.path_utils import make_run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO racing-line agent")
    p.add_argument("--mode", "-m", choices=["A", "B", "C"], default="C")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--name", "-n", default=None,
                   help="Experiment name (output folder).")
    p.add_argument("--timesteps", "-t", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None,
                   help="Override number of parallel envs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Config ---------------------------------------------------------- #
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config_for_mode(args.mode)

    if args.mode:
        cfg.obs_mode = args.mode
    if args.name:
        cfg.experiment_name = args.name
    else:
        cfg.experiment_name = f"ppo_{cfg.track.name}/obs_{cfg.obs_mode}"
    cfg.training.total_timesteps = args.timesteps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_envs is not None:
        cfg.training.n_envs = args.n_envs

    # ---- Reproducibility ------------------------------------------------- #
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- Output ---------------------------------------------------------- #
    run_dir = make_run_dir(cfg)

    # ---- Track ----------------------------------------------------------- #
    track = build_track(cfg.track.name, cfg.track.half_width)

    # ---- Environment ----------------------------------------------------- #
    n_envs = cfg.training.n_envs

    def make_env(rank: int = 0):
        def _init():
            env = RacingEnv(cfg, eval_mode=False)
            env = Monitor(env)
            return env
        return _init

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0)])

    # ---- PPO Model ------------------------------------------------------- #
    tc = cfg.training
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=tc.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        device=cfg.device,
        seed=cfg.seed,
        verbose=1,
    )

    # ---- Callback -------------------------------------------------------- #
    callback = RacingEvalCallback(
        cfg=cfg,
        track=track,
        run_dir=run_dir,
        eval_freq=tc.eval_freq,
        n_eval_episodes=tc.n_eval_episodes,
        verbose=1,
    )

    # ---- Print info ------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  PPO Racing-Line Training")
    print(f"  Experiment : {cfg.experiment_name}")
    print(f"  Track      : {cfg.track.name}")
    print(f"  Obs mode   : {cfg.obs_mode}")
    print(f"  Timesteps  : {cfg.training.total_timesteps:,}")
    print(f"  Envs       : {n_envs}")
    print(f"  Seed       : {cfg.seed}")
    print(f"  Output     : {run_dir}")
    print(f"{'='*60}\n")

    # ---- Train ----------------------------------------------------------- #
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # ---- Save ------------------------------------------------------------ #
    final_path = run_dir / "models" / "final_model"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
