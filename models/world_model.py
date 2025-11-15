import mlx.core as mx
import mlx.nn as nn


class LatentWorldModel(nn.Module):
    """
    Latent world dynamics model.

    Predicts next latent state and reward given current latent and action.
    Input: [z_t, one_hot(a_t)] -> latent_dim + action_dim
    Output: [z_{t+1}, r_t] -> latent_dim + 1
    """

    def __init__(self, latent_dim: int = 4, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        input_dim = latent_dim + action_dim
        output_dim = latent_dim + 1  # next_latent + reward

        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ]

    def __call__(self, z_and_a):
        """
        Args:
            z_and_a: concatenated [z, one_hot(a)] of shape [..., latent_dim + action_dim]

        Returns:
            output: [z_next, reward] of shape [..., latent_dim + 1]
        """
        x = z_and_a
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x
