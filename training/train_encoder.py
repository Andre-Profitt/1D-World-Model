"""
Training script for the encoder-decoder autoencoder.

Learns to compress observations into a latent representation.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Encoder, Decoder
import config


def make_batches(N, batch_size, rng):
    """Generate random batches."""
    indices = np.arange(N)
    rng.shuffle(indices)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        yield mx.array(indices[start:end])


def train_autoencoder():
    """Train encoder-decoder to reconstruct observations."""

    # Load data
    print("Loading observations...")
    observations = np.load(f"{config.DATA_DIR}/observations.npy")
    N = observations.shape[0]
    print(f"Loaded {N} observations of shape {observations.shape}")

    # Convert to MLX array
    x_all = mx.array(observations, dtype=mx.float32)

    # Create models
    encoder = Encoder(
        obs_dim=config.OBS_DIM,
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM_ENCODER
    )
    decoder = Decoder(
        latent_dim=config.LATENT_DIM,
        obs_dim=config.OBS_DIM,
        hidden_dim=config.HIDDEN_DIM_ENCODER
    )

    # Combine parameters for joint optimization
    params = {"encoder": encoder.parameters(), "decoder": decoder.parameters()}

    # Optimizer
    optimizer = optim.Adam(learning_rate=config.ENCODER_LEARNING_RATE)

    # Force initialization
    mx.eval(encoder.parameters())
    mx.eval(decoder.parameters())

    # Loss function
    def loss_fn(params_dict, x):
        encoder_params = params_dict["encoder"]
        decoder_params = params_dict["decoder"]

        # Temporarily update model parameters
        encoder.update(encoder_params)
        decoder.update(decoder_params)

        z = encoder(x)
        x_recon = decoder(z)
        return mx.mean((x_recon - x) ** 2)

    # Get gradient function
    loss_and_grad_fn = nn.value_and_grad(params, loss_fn)

    # Training loop
    print(f"\nTraining autoencoder for {config.ENCODER_EPOCHS} epochs...")
    rng = np.random.default_rng(123)

    for epoch in range(1, config.ENCODER_EPOCHS + 1):
        epoch_losses = []

        for batch_idx in make_batches(N, config.ENCODER_BATCH_SIZE, rng):
            xb = x_all[batch_idx]

            loss, grads = loss_and_grad_fn(params, xb)
            optimizer.update(params, grads)

            mx.eval(params, optimizer.state)
            epoch_losses.append(float(loss.item()))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch:02d}/{config.ENCODER_EPOCHS} | loss = {mean_loss:.6f}")

    # Save weights
    print("\nSaving model weights...")
    os.makedirs("weights", exist_ok=True)

    encoder.save_weights(config.ENCODER_WEIGHTS_PATH)
    decoder.save_weights(config.DECODER_WEIGHTS_PATH)

    print(f"Encoder weights saved to {config.ENCODER_WEIGHTS_PATH}")
    print(f"Decoder weights saved to {config.DECODER_WEIGHTS_PATH}")

    # Test reconstruction
    print("\n--- Testing Reconstruction ---")
    test_indices = [0, 100, 500, 1000]
    for idx in test_indices:
        if idx < N:
            obs = x_all[idx:idx+1]
            z = encoder(obs)
            recon = decoder(z)
            mx.eval(recon)

            obs_np = np.array(obs[0])
            recon_np = np.array(recon[0])
            error = np.abs(obs_np - recon_np)

            print(f"Sample {idx}: obs={obs_np}, recon={recon_np}, error={error}")


if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists(f"{config.DATA_DIR}/observations.npy"):
        print(f"Error: Data not found. Please run 'python scripts/collect_data.py' first.")
        sys.exit(1)

    train_autoencoder()
