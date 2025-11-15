# 1D World Model

A minimal implementation of a world model using MLX for Apple Silicon, demonstrating how neural networks can learn to predict environmental dynamics.

## Overview

This project implements a simple yet complete world model that learns the physics of a 1D environment. The model takes the current state (position and velocity) along with an action, and predicts the next state and reward. Once trained, the model can "dream" or simulate future trajectories without interacting with the real environment.

## Features

- **Efficient Training**: Leverages MLX for optimized performance on Apple Silicon
- **Simple Physics**: 1D position-velocity dynamics with discrete actions
- **End-to-End Pipeline**: Data generation, model training, and rollout visualization
- **Accurate Predictions**: Achieves sub-0.001 MSE loss on dynamics prediction

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

## Usage

Run the world model training and demonstration:

```bash
python world_model_mlx.py
```

The script will:
1. Generate 20,000 state transitions from simulated physics
2. Train a neural network to predict state transitions
3. Display training progress over 20 epochs
4. Demonstrate a 15-step rollout comparing true physics vs. model predictions

### Example Output

```
Dataset size: 20000 transitions
Epoch 01 | loss = 0.035199
Epoch 02 | loss = 0.001836
...
Epoch 20 | loss = 0.000037

--- World Model Rollout Demo ---
t=00 a=+1 | TRUE x=+1.000, v=+0.000, r=-1.100  ||  MODEL x=+1.107, v=+0.103, r=-1.103
t=01 a=+1 | TRUE x=+1.100, v=+0.100, r=-1.300  ||  MODEL x=+1.319, v=+0.206, r=-1.314
...
```

## How It Works

### Environment Dynamics

The simulated environment follows these physics rules:

- **State**: `[x, v]` where x is position and v is velocity
- **Actions**: `-1` (accelerate left), `0` (no action), `+1` (accelerate right)
- **Transition**:
  - `v_next = clip(v + 0.1 * action, -1.0, 1.0)`
  - `x_next = clip(x + v_next, -1.5, 1.5)`
- **Reward**: `-|x_next|` (encourages staying near position 0)

### Model Architecture

The world model is a simple feedforward neural network:

- **Input**: State (2 dims) + one-hot encoded action (3 dims) = 5 dimensions
- **Hidden Layers**: Two layers of 128 units each with ReLU activation
- **Output**: Next state (2 dims) + reward (1 dim) = 3 dimensions

### Training

- **Dataset**: 400 episodes of 50 timesteps each (20,000 transitions)
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Mean squared error on predicted next state and reward
- **Batch Size**: 128
- **Epochs**: 20

## Key Results

The trained model achieves:
- Final training loss of ~0.00004 MSE
- Accurate prediction of position and velocity dynamics
- Correct learning of boundary constraints (position and velocity clipping)
- Stable long-horizon rollouts with minimal drift

## Extensions

This implementation serves as a foundation for exploring:

- **More Complex Environments**: Extend to 2D/3D spaces or Gymnasium environments
- **Latent World Models**: Add encoder/decoder for high-dimensional observations
- **Planning**: Implement model-based planning algorithms (e.g., CEM, MPPI)
- **Model-Based RL**: Use the world model for policy optimization
- **Uncertainty Quantification**: Add ensemble models or probabilistic predictions

## License

MIT License - feel free to use this code for learning and research purposes.

## References

- [MLX Framework](https://github.com/ml-explore/mlx) - Apple's ML framework for Apple Silicon
- [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122) - Seminal paper on world models in RL
