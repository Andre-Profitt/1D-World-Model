"""
Training script for the latent world model.

Learns to predict next latent state and reward from current latent and action.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Encoder, LatentWorldModel
from models.controller import actions_to_one_hot
import config


def make_batches(N, batch_size, rng):
    """Generate random batches."""
    indices = np.arange(N)
    rng.shuffle(indices)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        yield mx.array(indices[start:end])


def train_world_model():
    """Train world model in latent space."""

    # Load transition data
    print("Loading transition data...")
    obs = np.load(f"{config.DATA_DIR}/obs.npy")
    actions = np.load(f"{config.DATA_DIR}/actions.npy")
    next_obs = np.load(f"{config.DATA_DIR}/next_obs.npy")
    rewards = np.load(f"{config.DATA_DIR}/rewards.npy")
    N = obs.shape[0]
    print(f"Loaded {N} transitions")

    # Load encoder
    print("Loading encoder...")
    encoder = Encoder(
        obs_dim=config.OBS_DIM,
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM_ENCODER
    )
    encoder.load_weights(config.ENCODER_WEIGHTS_PATH)
    mx.eval(encoder.parameters())
    print(f"Encoder loaded from {config.ENCODER_WEIGHTS_PATH}")

    # Encode observations to latents
    print("Encoding observations to latents...")
    obs_mx = mx.array(obs, dtype=mx.float32)
    next_obs_mx = mx.array(next_obs, dtype=mx.float32)

    z = encoder(obs_mx)
    z_next = encoder(next_obs_mx)
    mx.eval(z, z_next)

    print(f"Encoded to latent space: z.shape = {z.shape}, z_next.shape = {z_next.shape}")

    # One-hot encode actions
    actions_oh = actions_to_one_hot(actions, num_actions=config.ACTION_DIM)
    actions_oh_mx = mx.array(actions_oh, dtype=mx.float32)

    # Build inputs and targets
    rewards_mx = mx.array(rewards.reshape(-1, 1), dtype=mx.float32)

    # Input: [z, one_hot(a)]
    x_in = mx.concatenate([z, actions_oh_mx], axis=-1)

    # Target: [z_next, reward]
    y_out = mx.concatenate([z_next, rewards_mx], axis=-1)

    print(f"Training data: x_in.shape = {x_in.shape}, y_out.shape = {y_out.shape}")

    # Create world model
    world_model = LatentWorldModel(
        latent_dim=config.LATENT_DIM,
        action_dim=config.ACTION_DIM,
        hidden_dim=config.HIDDEN_DIM_WORLD_MODEL
    )

    optimizer = optim.Adam(learning_rate=config.WORLD_MODEL_LEARNING_RATE)

    # Force initialization
    mx.eval(world_model.parameters())

    # Loss function
    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    # Get gradient function
    loss_and_grad_fn = nn.value_and_grad(world_model, loss_fn)

    # Training loop
    print(f"\nTraining world model for {config.WORLD_MODEL_EPOCHS} epochs...")
    rng = np.random.default_rng(456)

    for epoch in range(1, config.WORLD_MODEL_EPOCHS + 1):
        epoch_losses = []

        for batch_idx in make_batches(N, config.WORLD_MODEL_BATCH_SIZE, rng):
            xb = x_in[batch_idx]
            yb = y_out[batch_idx]

            loss, grads = loss_and_grad_fn(world_model, xb, yb)
            optimizer.update(world_model, grads)

            mx.eval(world_model.parameters(), optimizer.state)
            epoch_losses.append(float(loss.item()))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch:02d}/{config.WORLD_MODEL_EPOCHS} | loss = {mean_loss:.6f}")

    # Save weights
    print("\nSaving model weights...")
    world_model.save_weights(config.WORLD_MODEL_WEIGHTS_PATH)
    print(f"World model weights saved to {config.WORLD_MODEL_WEIGHTS_PATH}")

    # Test predictions
    print("\n--- Testing World Model Predictions ---")
    test_indices = [0, 100, 500, 1000]
    for idx in test_indices:
        if idx < N:
            x_test = x_in[idx:idx+1]
            y_true = y_out[idx:idx+1]
            y_pred = world_model(x_test)
            mx.eval(y_pred)

            y_true_np = np.array(y_true[0])
            y_pred_np = np.array(y_pred[0])
            error = np.abs(y_true_np - y_pred_np)

            print(f"Sample {idx}:")
            print(f"  True:  {y_true_np}")
            print(f"  Pred:  {y_pred_np}")
            print(f"  Error: {error}")


if __name__ == "__main__":
    # Check if data and encoder exist
    if not os.path.exists(f"{config.DATA_DIR}/obs.npy"):
        print(f"Error: Data not found. Please run 'python scripts/collect_data.py' first.")
        sys.exit(1)

    if not os.path.exists(config.ENCODER_WEIGHTS_PATH):
        print(f"Error: Encoder weights not found. Please run 'python training/train_encoder.py' first.")
        sys.exit(1)

    train_world_model()
