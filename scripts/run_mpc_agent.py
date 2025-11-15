"""
Run MPC agent using the learned world model.

The agent uses model-based planning (MPC with random shooting) to select
actions that maximize predicted future rewards.
"""

import numpy as np
import mlx.core as mx
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OneDWorldEnv
from models import Encoder, LatentWorldModel, MPCController
import config


def run_episode(env, controller, max_steps=200, render=False):
    """
    Run one episode with the MPC controller.

    Args:
        env: environment instance
        controller: MPC controller instance
        max_steps: maximum episode length
        render: whether to print step-by-step info

    Returns:
        total_reward: cumulative reward for the episode
        trajectory: list of (obs, action, reward) tuples
    """
    obs = env.reset()
    total_reward = 0.0
    trajectory = []

    for t in range(max_steps):
        action = controller.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        total_reward += reward
        trajectory.append((obs.copy(), action, reward))

        if render:
            state = info["state"]
            print(f"t={t:03d} | x={state[0]:+.3f}, v={state[1]:+.3f} | a={action:+d} | r={reward:+.3f}")

        obs = next_obs

        if done:
            break

    return total_reward, trajectory


def main():
    """Main entry point."""

    # Check if models exist
    if not os.path.exists(config.ENCODER_WEIGHTS_PATH):
        print(f"Error: Encoder weights not found at {config.ENCODER_WEIGHTS_PATH}")
        print("Please train the encoder first: python training/train_encoder.py")
        sys.exit(1)

    if not os.path.exists(config.WORLD_MODEL_WEIGHTS_PATH):
        print(f"Error: World model weights not found at {config.WORLD_MODEL_WEIGHTS_PATH}")
        print("Please train the world model first: python training/train_world_model.py")
        sys.exit(1)

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

    # Load world model
    print("Loading world model...")
    world_model = LatentWorldModel(
        latent_dim=config.LATENT_DIM,
        action_dim=config.ACTION_DIM,
        hidden_dim=config.HIDDEN_DIM_WORLD_MODEL
    )
    world_model.load_weights(config.WORLD_MODEL_WEIGHTS_PATH)
    mx.eval(world_model.parameters())
    print(f"World model loaded from {config.WORLD_MODEL_WEIGHTS_PATH}")

    # Create controller
    print(f"\nCreating MPC controller (horizon={config.MPC_HORIZON}, samples={config.MPC_NUM_SAMPLES})...")
    controller = MPCController(
        encoder=encoder,
        world_model=world_model,
        action_values=np.array(config.ACTION_SPACE, dtype=np.int32),
        horizon=config.MPC_HORIZON,
        num_samples=config.MPC_NUM_SAMPLES,
        gamma=config.MPC_GAMMA,
    )

    # Create environment
    env = OneDWorldEnv(seed=789)

    # Run evaluation episodes
    print("\n" + "="*60)
    print("Running MPC Agent Evaluation")
    print("="*60)

    num_episodes = 10
    max_steps = 100
    returns = []

    for ep in range(num_episodes):
        total_reward, trajectory = run_episode(env, controller, max_steps=max_steps, render=(ep == 0))

        returns.append(total_reward)
        final_state = trajectory[-1][0] if trajectory else env.state

        print(f"\nEpisode {ep+1:02d}: return = {total_reward:.3f}, final x = {final_state[0]:+.3f}")

    # Summary statistics
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean return: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"Min return: {np.min(returns):.3f}")
    print(f"Max return: {np.max(returns):.3f}")

    # Compare with random policy
    print("\n" + "="*60)
    print("Baseline: Random Policy")
    print("="*60)

    random_returns = []
    for ep in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            action = np.random.choice(config.ACTION_SPACE)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        random_returns.append(total_reward)

    print(f"Random policy mean return: {np.mean(random_returns):.3f} ± {np.std(random_returns):.3f}")
    print(f"\nMPC improvement: {np.mean(returns) - np.mean(random_returns):.3f}")


if __name__ == "__main__":
    main()
