# Racing Line RL

Reinforcement learning agent that learns optimal racing lines on F1 circuits using Soft Actor-Critic (SAC).

## Overview

A bicycle-model car navigates real F1 track layouts (loaded from GeoJSON), learning to maximise progress around the circuit while respecting traction limits. The agent receives observations including speed, track position, distances to edges, and lookahead curvature, then outputs continuous steering and throttle/brake commands.

## Features

- **SAC training** via Stable Baselines3 with parallel environments (SubprocVecEnv)
- **Three observation modes** (A/B/C) with increasing complexity — Mode C includes lookahead curvature
- **Bicycle model physics** with traction circle, slip dynamics, and realistic steering constraints
- **Animated replays** — best lap rendered as MP4 with F1 car visualisation, steering wheel, and throttle/brake inputs
- **Multiple tracks** — Suzuka, Bahrain, Hungary, Barcelona (GeoJSON format)
- **GPU support** — CUDA-accelerated training for the neural network

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Train on Suzuka
python scripts/train.py --config configs/suzuka_full.yaml

# Evaluate a trained model
python scripts/evaluate.py --config configs/suzuka_full.yaml

# Generate animation from a trained model
python scripts/animate.py --config configs/suzuka_full.yaml
```

## Project Structure

```
configs/          # YAML training configs per track
data/tracks/      # GeoJSON track layouts
scripts/          # CLI entry points (train, evaluate, animate)
src/racing_rl/
  config/         # Config schema and loader
  env/            # Gymnasium racing environment
  evaluation/     # Eval loop and best model tracking
  observations/   # Observation space builders (modes A/B/C)
  physics/        # Bicycle model vehicle dynamics
  plotting/       # Track plots and animated replays
  rewards/        # Reward function
  tracks/         # Track loading and parametric representation
  training/       # Trainer, callbacks, SubprocVecEnv setup
tests/            # Unit tests
```

## Tracks

| Track | Config |
|-------|--------|
| Suzuka | `configs/suzuka_full.yaml` |
| Bahrain | `configs/bahrain.yaml` |
| Hungary | `configs/hungary.yaml` |
| Barcelona | `configs/barcelona.yaml` |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
