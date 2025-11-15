import numpy as np
import mlx.core as mx


def actions_to_one_hot(actions, num_actions=3):
    """
    Convert actions to one-hot encoding.

    Args:
        actions: array of ints in {-1, 0, 1}, shape [N] or [N, H]
        num_actions: number of discrete actions (default 3)

    Returns:
        one_hot: array of shape [N, num_actions] or [N, H, num_actions]
    """
    original_shape = actions.shape
    actions_flat = actions.flatten()

    # Map {-1, 0, 1} to {0, 1, 2}
    idx = (actions_flat + 1).astype(np.int32)
    one_hot = np.zeros((actions_flat.shape[0], num_actions), dtype=np.float32)
    one_hot[np.arange(actions_flat.shape[0]), idx] = 1.0

    if len(original_shape) == 1:
        return one_hot
    else:
        # Reshape back to [..., num_actions]
        return one_hot.reshape(*original_shape, num_actions)


class MPCController:
    """
    Model Predictive Control using random shooting.

    Samples random action sequences, evaluates them in the learned world model,
    and selects the action sequence with highest predicted return.
    """

    def __init__(
        self,
        encoder,
        world_model,
        action_values,  # e.g., np.array([-1, 0, 1])
        horizon: int = 12,
        num_samples: int = 512,
        gamma: float = 0.99,
    ):
        self.encoder = encoder
        self.world_model = world_model
        self.action_values = action_values
        self.num_actions = len(action_values)
        self.horizon = horizon
        self.num_samples = num_samples
        self.gamma = gamma

    def select_action(self, obs: np.ndarray) -> int:
        """
        Select best action using MPC.

        Args:
            obs: observation array of shape [obs_dim]

        Returns:
            action: int from action_values
        """
        # Encode observation to latent
        z0 = self.encoder(mx.array(obs[None, :], dtype=mx.float32))  # [1, latent_dim]
        mx.eval(z0)

        # Sample candidate action sequences
        action_seqs = self._sample_action_sequences()  # [num_samples, horizon]

        # Evaluate each sequence
        returns = self._evaluate_sequences(z0, action_seqs)  # [num_samples]

        # Choose best sequence and return its first action
        best_idx = int(np.argmax(returns))
        best_first_action_idx = int(action_seqs[best_idx, 0])
        return int(self.action_values[best_first_action_idx])

    def _sample_action_sequences(self) -> np.ndarray:
        """
        Sample random action sequences.

        Returns:
            action_seqs: array of shape [num_samples, horizon] with indices into action_values
        """
        return np.random.randint(0, self.num_actions, size=(self.num_samples, self.horizon))

    def _evaluate_sequences(self, z0, action_seqs) -> np.ndarray:
        """
        Evaluate action sequences by rolling out in the world model.

        Args:
            z0: initial latent state [1, latent_dim]
            action_seqs: action sequences [num_samples, horizon]

        Returns:
            returns: discounted returns for each sequence [num_samples]
        """
        num_samples = action_seqs.shape[0]
        horizon = action_seqs.shape[1]

        # Replicate z0 for all samples
        z = mx.repeat(z0, num_samples, axis=0)  # [num_samples, latent_dim]

        returns = np.zeros(num_samples, dtype=np.float32)
        discount = 1.0

        for t in range(horizon):
            # Get actions for this timestep
            actions_t = action_seqs[:, t]  # [num_samples]

            # Map action indices to actual action values
            action_vals = self.action_values[actions_t]  # [num_samples]

            # One-hot encode
            actions_oh = actions_to_one_hot(action_vals, self.num_actions)  # [num_samples, num_actions]
            actions_oh_mx = mx.array(actions_oh, dtype=mx.float32)

            # Concatenate z and action
            z_and_a = mx.concatenate([z, actions_oh_mx], axis=-1)  # [num_samples, latent_dim + num_actions]

            # Predict next state and reward
            output = self.world_model(z_and_a)  # [num_samples, latent_dim + 1]
            mx.eval(output)

            # Split into next latent and reward
            z = output[:, :-1]  # [num_samples, latent_dim]
            r = output[:, -1]   # [num_samples]

            # Accumulate discounted rewards
            r_np = np.array(r)
            returns += discount * r_np
            discount *= self.gamma

        return returns
