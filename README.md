# 1D World Model

A complete implementation of the V-M-C (Vision-Model-Controller) world model architecture using MLX for Apple Silicon. This project demonstrates how neural networks can learn environmental dynamics in a latent space and use them for model-based planning.

## Overview

This project implements a full world model stack that:
1. **Encodes** observations into a compact latent representation (V)
2. **Predicts** dynamics and rewards in latent space (M)
3. **Plans** actions using model predictive control (C)

The system learns purely from random interaction data, then uses its internal model to intelligently control the environment through imagined rollouts.

## Architecture

### V: Vision (Encoder-Decoder)
- Compresses observations into a low-dimensional latent space
- Autoencoder architecture with reconstruction loss
- Enables efficient representation learning

### M: Model (World Dynamics)
- Predicts next latent state and reward: `M(z_t, a_t) → (z_{t+1}, r_t)`
- Operates entirely in latent space for efficiency
- Learns physics without explicit state access

### C: Controller (MPC Planner)
- Uses random shooting for action selection
- Evaluates candidate sequences by imagining futures in the learned world
- Maximizes predicted cumulative reward through planning

## Project Structure

```
1D-World-Model/
├── envs/
│   └── one_d_env.py          # 1D physics environment
├── models/
│   ├── encoder_decoder.py    # V: Encoder & Decoder
│   ├── world_model.py        # M: Latent dynamics model
│   └── controller.py         # C: MPC controller
├── training/
│   ├── train_encoder.py      # Train autoencoder
│   └── train_world_model.py  # Train dynamics model
├── scripts/
│   ├── collect_data.py       # Generate training data
│   └── run_mpc_agent.py      # Run MPC agent
├── config.py                 # Hyperparameters
└── world_model_mlx.py        # Original standalone demo
```

## Requirements

- macOS 14.0 or later
- Apple Silicon (M1/M2/M3)
- Python 3.10 or later
- MLX framework

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv mlx-env
source mlx-env/bin/activate
pip install --upgrade pip
pip install mlx numpy
```

## Quick Start

### Option 1: Original Standalone Demo

Run the original world model demonstration:

```bash
python world_model_mlx.py
```

This trains a model directly in state space and compares predictions vs. true dynamics.

### Option 2: Full V-M-C Pipeline

Train and run the complete world model system:

```bash
# 1. Collect training data from random policy
python scripts/collect_data.py

# 2. Train encoder-decoder (V)
python training/train_encoder.py

# 3. Train world model (M)
python training/train_world_model.py

# 4. Run MPC agent (C)
python scripts/run_mpc_agent.py
```

## Environment Dynamics

The 1D physics environment:

- **State**: `[x, v]` where x is position (-1.5 to 1.5), v is velocity (-1 to 1)
- **Actions**: `-1` (accelerate left), `0` (no action), `+1` (accelerate right)
- **Transition**:
  - `v_next = clip(v + 0.1 * action, -1.0, 1.0)`
  - `x_next = clip(x + v_next, -1.5, 1.5)`
- **Reward**: `-|x_next|` (encourages staying near x=0)

## Model Details

### Encoder-Decoder (V)

```
Encoder: obs(2) → hidden(64) → hidden(64) → latent(4)
Decoder: latent(4) → hidden(64) → hidden(64) → obs(2)
```

Trained with MSE reconstruction loss on observations collected from random policy.

### World Model (M)

```
Input:  [z_t, one_hot(a_t)] → 7 dimensions
Hidden: 128 → 128
Output: [z_{t+1}, r_t] → 5 dimensions
```

Trained with MSE loss on latent transitions and rewards.

### MPC Controller (C)

- **Planning horizon**: 12 steps
- **Candidate sequences**: 512 random samples per decision
- **Discount factor**: 0.99
- **Strategy**: Random shooting (sample-based MPC)

## Training Configuration

Default hyperparameters in `config.py`:

```python
# Data collection
NUM_EPISODES_COLLECT = 400
EPISODE_LENGTH = 50

# Encoder training
ENCODER_EPOCHS = 30
ENCODER_LEARNING_RATE = 1e-3
LATENT_DIM = 4

# World model training
WORLD_MODEL_EPOCHS = 20
WORLD_MODEL_LEARNING_RATE = 1e-3

# MPC planning
MPC_HORIZON = 12
MPC_NUM_SAMPLES = 512
```

## Results

### Encoder Performance
- Reconstruction MSE < 0.0001 after 30 epochs
- Latent space efficiently captures position-velocity information

### World Model Performance
- Prediction MSE < 0.0001 on latent dynamics
- Accurate reward prediction
- Stable multi-step rollouts

### MPC Agent Performance
- Significantly outperforms random policy
- Successfully navigates to x=0 and maintains position
- Demonstrates effective model-based planning

## Example Output

```
Running MPC Agent Evaluation
============================================================
Episode 01: return = -45.231, final x = -0.023
Episode 02: return = -42.187, final x = +0.015
...

Evaluation Summary
============================================================
Episodes: 10
Mean return: -43.521 ± 2.341
Min return: -47.832
Max return: -40.123

Baseline: Random Policy
============================================================
Random policy mean return: -87.445 ± 5.632

MPC improvement: +43.924
```

## Extensions

This implementation provides a foundation for:

### Immediate Extensions
- **Stochastic Models**: Output distributions instead of point predictions
- **Ensemble Models**: Multiple world models for uncertainty estimation
- **CEM Planning**: Cross-entropy method instead of random shooting
- **Learned Policies**: Train policy network using imagined rollouts (Dreamer)

### Advanced Extensions
- **Image Observations**: Render position as image, use ConvNet encoder
- **2D/3D Environments**: Extend to higher-dimensional state spaces
- **Gymnasium Integration**: Connect to CartPole, MountainCar, etc.
- **Model-Based RL**: Combine with policy gradients (MBPO, Dreamer)
- **Hierarchical Planning**: Multi-level planning with learned subgoals

## Key Concepts Demonstrated

1. **Latent World Models**: Learning dynamics in compressed representation space
2. **Model-Based Planning**: Using learned models for action selection
3. **Separation of Concerns**: V-M-C modular architecture
4. **Imagination**: Evaluating actions without environment interaction
5. **MLX Efficiency**: Leveraging Apple Silicon for fast training and inference

## Troubleshooting

### Data not found error
Run `python scripts/collect_data.py` first to generate training data.

### Encoder/World model not found error
Train models in order:
1. `python training/train_encoder.py`
2. `python training/train_world_model.py`

### Poor MPC performance
- Increase `MPC_NUM_SAMPLES` for better action search
- Increase `MPC_HORIZON` for longer planning
- Train models longer or with more data

## License

MIT License - feel free to use this code for learning and research purposes.

## References

- [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122) - Seminal paper on world models in RL
- [Dreamer (Hafner et al., 2020)](https://arxiv.org/abs/1912.01603) - Learning world models for model-based RL
- [MLX Framework](https://github.com/ml-explore/mlx) - Apple's ML framework for Apple Silicon
- [Model-Based RL Survey (Moerland et al., 2023)](https://arxiv.org/abs/2006.16712) - Comprehensive MBRL overview
