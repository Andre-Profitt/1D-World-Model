import math
import random

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ----------  Environment (true world)  ----------

ACTIONS = [-1, 0, 1]  # accelerate left / none / right


def step_env(state, action):
    """
    state: np.array([x, v])
    action: int in {-1, 0, 1}
    returns: next_state (np.array), reward (float)
    """
    x, v = state
    a = action

    v_next = v + 0.1 * a
    v_next = np.clip(v_next, -1.0, 1.0)

    x_next = x + v_next
    x_next = np.clip(x_next, -1.5, 1.5)

    reward = -abs(x_next)

    return np.array([x_next, v_next], dtype=np.float32), float(reward)


def generate_dataset(num_episodes=200, episode_len=50, seed=0):
    """
    Generate (s, a, s', r) transitions.
    """
    rng = np.random.default_rng(seed)

    states = []
    actions = []
    next_states = []
    rewards = []

    for _ in range(num_episodes):
        # random initial state
        x0 = rng.uniform(-1.0, 1.0)
        v0 = rng.uniform(-0.5, 0.5)
        state = np.array([x0, v0], dtype=np.float32)

        for _ in range(episode_len):
            action = rng.choice(ACTIONS)
            next_state, reward = step_env(state, action)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)

            state = next_state

    states = np.stack(states, axis=0)          # [N, 2]
    actions = np.array(actions, dtype=np.int32)  # [N]
    next_states = np.stack(next_states, axis=0)  # [N, 2]
    rewards = np.array(rewards, dtype=np.float32)  # [N]

    return states, actions, next_states, rewards


def actions_to_one_hot(actions):
    """
    actions: np.array of ints in {-1, 0, 1}
    returns: np.array of shape [N, 3] one-hot encoded
    index 0 -> -1, 1 -> 0, 2 -> +1
    """
    idx = (actions + 1).astype(np.int64)  # -1->0, 0->1, 1->2
    one_hot = np.zeros((actions.shape[0], 3), dtype=np.float32)
    one_hot[np.arange(actions.shape[0]), idx] = 1.0
    return one_hot


# ----------  World Model (MLX MLP)  ----------

class WorldModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=3):
        """
        input: [x, v, one_hot(a)] -> 2 + 3 = 5
        output: [x_next, v_next, reward] -> 3
        """
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ]

    def __call__(self, x):
        # simple MLP with ReLU
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


def make_batches(N, batch_size, rng):
    indices = np.arange(N)
    rng.shuffle(indices)
    for start in range(0, N, batch_size):
        end = start + batch_size
        yield mx.array(indices[start:end])


# ----------  Training code in MLX  ----------

def main():
    # 1) Create dataset from true environment
    states, actions, next_states, rewards = generate_dataset(
        num_episodes=400, episode_len=50, seed=42
    )
    N = states.shape[0]
    print(f"Dataset size: {N} transitions")

    action_oh = actions_to_one_hot(actions)

    # inputs: concat [state, one_hot(action)] -> [N, 5]
    inputs_np = np.concatenate([states, action_oh], axis=-1)
    # targets: concat [next_state, reward] -> [N, 3]
    rewards_np = rewards.reshape(-1, 1)
    targets_np = np.concatenate([next_states, rewards_np], axis=-1)

    # Move to MLX arrays
    x_all = mx.array(inputs_np, dtype=mx.float32)
    y_all = mx.array(targets_np, dtype=mx.float32)

    # 2) Define model and optimizer
    model = WorldModel(input_dim=5, hidden_dim=128, output_dim=3)
    optimizer = optim.Adam(learning_rate=1e-3)

    # Force parameter initialization once (MLX is lazy)
    mx.eval(model.parameters())

    # Define loss function: simple MSE on all 3 outputs
    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    # Get loss + gradient function w.r.t. model params
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 3) Training loop
    num_epochs = 20
    batch_size = 128
    rng = np.random.default_rng(123)

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []

        for batch_idx in make_batches(N, batch_size, rng):
            xb = x_all[batch_idx]
            yb = y_all[batch_idx]

            loss, grads = loss_and_grad_fn(model, xb, yb)
            optimizer.update(model, grads)

            # Evaluate lazy arrays so updates actually happen
            mx.eval(model.parameters(), optimizer.state)

            epoch_losses.append(float(loss.item()))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch:02d} | loss = {mean_loss:.6f}")

    # 4) Test: roll out with the learned world model vs true env
    print("\n--- World Model Rollout Demo ---")
    test_state = np.array([1.0, 0.0], dtype=np.float32)  # start at x=1, v=0
    horizon = 15
    action_seq = [1] * horizon  # always accelerate right (+1)

    state_true = test_state.copy()
    state_model = mx.array(test_state.reshape(1, -1), dtype=mx.float32)

    for t, a in enumerate(action_seq):
        # True environment step
        next_true, r_true = step_env(state_true, a)

        # World model step
        a_oh = actions_to_one_hot(np.array([a], dtype=np.int32))  # shape [1,3]
        model_input = mx.array(
            np.concatenate([np.array(state_model), a_oh], axis=-1),
            dtype=mx.float32,
        )

        pred = model(model_input)[0]  # [3]
        mx.eval(pred)
        pred_np = np.array(pred)

        x_next_m, v_next_m, r_m = (
            float(pred_np[0]),
            float(pred_np[1]),
            float(pred_np[2]),
        )

        print(
            f"t={t:02d} a={a:+d} | "
            f"TRUE x={state_true[0]:+.3f}, v={state_true[1]:+.3f}, r={r_true:+.3f}  ||  "
            f"MODEL x={x_next_m:+.3f}, v={v_next_m:+.3f}, r={r_m:+.3f}"
        )

        # advance both worlds
        state_true = next_true
        state_model = mx.array(np.array([x_next_m, v_next_m]).reshape(1, -1), dtype=mx.float32)


if __name__ == "__main__":
    main()
