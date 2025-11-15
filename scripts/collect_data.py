"""
Data collection script for the 1D World Model.

Runs random policy in the environment and saves observations and transitions.
"""

import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OneDWorldEnv
import config


def collect_observations(env, num_episodes, episode_len, seed=None):
    """
    Collect observations by running random policy.

    Args:
        env: environment instance
        num_episodes: number of episodes to run
        episode_len: length of each episode
        seed: random seed for action selection

    Returns:
        observations: np.array of shape [N, obs_dim]
    """
    rng = np.random.default_rng(seed)
    observations = []

    for ep in range(num_episodes):
        obs = env.reset()
        observations.append(obs)

        for t in range(episode_len):
            action = rng.choice(env.action_space)
            obs, reward, done, info = env.step(action)
            observations.append(obs)

            if done:
                break

    return np.stack(observations, axis=0)


def collect_transitions(env, num_episodes, episode_len, seed=None):
    """
    Collect full transition tuples (obs, action, next_obs, reward).

    Args:
        env: environment instance
        num_episodes: number of episodes to run
        episode_len: length of each episode
        seed: random seed for action selection

    Returns:
        obs: np.array [N, obs_dim]
        actions: np.array [N] of ints
        next_obs: np.array [N, obs_dim]
        rewards: np.array [N] of floats
    """
    rng = np.random.default_rng(seed)

    obs_list = []
    actions_list = []
    next_obs_list = []
    rewards_list = []

    for ep in range(num_episodes):
        obs = env.reset()

        for t in range(episode_len):
            action = rng.choice(env.action_space)
            next_obs, reward, done, info = env.step(action)

            obs_list.append(obs)
            actions_list.append(action)
            next_obs_list.append(next_obs)
            rewards_list.append(reward)

            obs = next_obs

            if done:
                break

    obs_array = np.stack(obs_list, axis=0)
    actions_array = np.array(actions_list, dtype=np.int32)
    next_obs_array = np.stack(next_obs_list, axis=0)
    rewards_array = np.array(rewards_list, dtype=np.float32)

    return obs_array, actions_array, next_obs_array, rewards_array


if __name__ == "__main__":
    # Create data directory
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Create environment
    env = OneDWorldEnv(seed=config.ENV_SEED)

    print(f"Collecting data from {config.NUM_EPISODES_COLLECT} episodes...")

    # Collect observations for encoder training
    observations = collect_observations(
        env, config.NUM_EPISODES_COLLECT, config.EPISODE_LENGTH, seed=config.ENV_SEED
    )
    np.save(f"{config.DATA_DIR}/observations.npy", observations)
    print(f"Saved {observations.shape[0]} observations to {config.DATA_DIR}/observations.npy")

    # Reset env with same seed for consistency
    env = OneDWorldEnv(seed=config.ENV_SEED)

    # Collect transitions for world model training
    obs, actions, next_obs, rewards = collect_transitions(
        env, config.NUM_EPISODES_COLLECT, config.EPISODE_LENGTH, seed=config.ENV_SEED
    )

    np.save(f"{config.DATA_DIR}/obs.npy", obs)
    np.save(f"{config.DATA_DIR}/actions.npy", actions)
    np.save(f"{config.DATA_DIR}/next_obs.npy", next_obs)
    np.save(f"{config.DATA_DIR}/rewards.npy", rewards)

    print(f"Saved {obs.shape[0]} transitions to {config.DATA_DIR}/")
    print(f"  obs.npy: {obs.shape}")
    print(f"  actions.npy: {actions.shape}")
    print(f"  next_obs.npy: {next_obs.shape}")
    print(f"  rewards.npy: {rewards.shape}")
