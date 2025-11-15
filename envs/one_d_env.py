import numpy as np


class OneDWorldEnv:
    """
    Simple 1D physics environment with position and velocity.

    State: [x, v] where x is position, v is velocity
    Actions: -1 (left), 0 (none), +1 (right)
    Reward: -|x_next| (encourages staying near x=0)
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.action_space = [-1, 0, 1]
        self.state = None

    def reset(self):
        """Reset environment to random initial state."""
        x0 = self.rng.uniform(-1.0, 1.0)
        v0 = self.rng.uniform(-0.5, 0.5)
        self.state = np.array([x0, v0], dtype=np.float32)
        return self._state_to_obs(self.state)

    def step(self, action: int):
        """
        Take action in environment.

        Args:
            action: int in {-1, 0, 1}

        Returns:
            obs: observation (current: same as state)
            reward: float reward
            done: bool terminal flag
            info: dict with extra info
        """
        next_state, reward = self._physics_step(self.state, action)
        self.state = next_state
        obs = self._state_to_obs(next_state)
        done = False
        info = {"state": next_state.copy()}
        return obs, reward, done, info

    def _physics_step(self, state, action):
        """Apply physics dynamics."""
        x, v = state
        a = action

        v_next = v + 0.1 * a
        v_next = np.clip(v_next, -1.0, 1.0)

        x_next = x + v_next
        x_next = np.clip(x_next, -1.5, 1.5)

        reward = -abs(x_next)

        return np.array([x_next, v_next], dtype=np.float32), float(reward)

    def _state_to_obs(self, state):
        """
        Convert state to observation.
        Currently identity mapping, but could be extended to:
        - Add noise
        - Render as image
        - Project to higher dimensional space
        """
        return state.copy()
