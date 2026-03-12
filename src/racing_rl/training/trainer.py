"""
Training entry point.

Creates the SAC model, wires up the eval callback, and runs training.
Returns the final trained model.

SAC choice justification
------------------------
Soft Actor-Critic is the canonical algorithm for this type of task:
  * Continuous action space (steering, throttle/brake)
  * High sample efficiency from off-policy replay buffer
  * Entropy regularisation prevents premature convergence
  * Robust to hyperparameter choice (relative to PPO in continuous domains)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from racing_rl.config.schema import RacingConfig
from racing_rl.env.racing_env import RacingEnv
from racing_rl.tracks.base import BaseTrack
from racing_rl.tracks.parametric import build_track
from racing_rl.training.callbacks import RacingEvalCallback
from racing_rl.utils.logging_utils import get_logger
from racing_rl.utils.path_utils import make_run_dir

log = get_logger(__name__)


def train(
    cfg: RacingConfig,
    resume_from: Optional[Path] = None,
) -> SAC:
    """
    Train a SAC agent according to *cfg*.

    Parameters
    ----------
    cfg:
        Full racing config (obs_mode, physics, reward, etc.).
    resume_from:
        If provided, load an existing model from this path and continue.

    Returns
    -------
    Trained SB3 SAC model.
    """
    # ---- Reproducibility ------------------------------------------------ #
    import numpy as np
    import random
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- Output directory ----------------------------------------------- #
    run_dir = make_run_dir(cfg)
    log.info("Run directory: %s", run_dir)

    # ---- Track ---------------------------------------------------------- #
    track = build_track(cfg.track.name, cfg.track.half_width)
    log.info(
        "Track '%s' loaded: length=%.1f m, half_width=%.1f m",
        cfg.track.name,
        track.length,
        track.half_width,
    )

    # ---- Training environment (wrapped in Monitor for SB3 logging) ------ #
    n_envs = cfg.training.n_envs

    def make_env(rank: int = 0):
        def _init():
            env = RacingEnv(cfg, eval_mode=False)
            env = Monitor(env)
            return env
        return _init

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        log.info("Using %d parallel environments (SubprocVecEnv)", n_envs)
    else:
        vec_env = DummyVecEnv([make_env(0)])

    # ---- Model ---------------------------------------------------------- #
    tc = cfg.training
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
    )

    if resume_from is not None and Path(resume_from).exists():
        log.info("Resuming from %s", resume_from)
        model = SAC.load(str(resume_from), env=vec_env, device=cfg.device)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=tc.learning_rate,
            buffer_size=tc.buffer_size,
            batch_size=tc.batch_size,
            tau=tc.tau,
            gamma=tc.gamma,
            ent_coef=tc.ent_coef,
            learning_starts=tc.learning_starts,
            gradient_steps=tc.gradient_steps,
            train_freq=tc.train_freq,
            policy_kwargs=policy_kwargs,
            device=cfg.device,
            seed=cfg.seed,
            verbose=1,
        )

    # ---- Callback ------------------------------------------------------- #
    callback = RacingEvalCallback(
        cfg=cfg,
        track=track,
        run_dir=run_dir,
        eval_freq=tc.eval_freq,
        n_eval_episodes=tc.n_eval_episodes,
        verbose=1,
    )

    # ---- Train ---------------------------------------------------------- #
    log.info(
        "Training obs_mode=%s for %d timesteps (eval every %d steps)",
        cfg.obs_mode,
        tc.total_timesteps,
        tc.eval_freq,
    )

    model.learn(
        total_timesteps=tc.total_timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=(resume_from is None),
    )

    # ---- Save final model ----------------------------------------------- #
    final_path = run_dir / "models" / "final_model"
    model.save(str(final_path))
    log.info("Final model saved to %s", final_path)

    return model
