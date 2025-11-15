import mlx.core as mx
import mlx.nn as nn


class Encoder(nn.Module):
    """
    Encodes observations to latent representations.

    Maps obs_dim -> latent_dim through MLP layers.
    """

    def __init__(self, obs_dim: int = 2, latent_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.layers = [
            nn.Linear(obs_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


class Decoder(nn.Module):
    """
    Decodes latent representations back to observation space.

    Maps latent_dim -> obs_dim through MLP layers.
    """

    def __init__(self, latent_dim: int = 4, obs_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, obs_dim),
        ]

    def __call__(self, z):
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = nn.relu(z)
        return z
