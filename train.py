"""
Train DQN and PPO on CartPole-v1, generate comparison charts.

Run: python train.py
Results saved to results/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent

RESULTS_DIR = "results"


# ─────────────────────────────────────────────────────────────────────
# Smoothing helper
# ─────────────────────────────────────────────────────────────────────
def smooth(x, window=20):
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ─────────────────────────────────────────────────────────────────────
# Train DQN
# ─────────────────────────────────────────────────────────────────────
def train_dqn(n_episodes=600, seed=42):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    state_dim  = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    agent      = DQNAgent(state_dim, action_dim)

    episode_rewards = []

    print("Training DQN on CartPole-v1...")
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            action     = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            state       = next_state
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {ep+1:4d} | Avg reward (last 100): {avg:.1f} | ε: {agent.epsilon:.3f}")

    env.close()
    return episode_rewards


# ─────────────────────────────────────────────────────────────────────
# Train PPO
# ─────────────────────────────────────────────────────────────────────
def train_ppo(n_updates=60, rollout_steps=2048, seed=42):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent      = PPOAgent(state_dim, action_dim)

    all_rewards = []

    print("\nTraining PPO on CartPole-v1...")
    state, _ = env.reset()
    ep_reward = 0

    for update in range(n_updates):
        states, actions, rewards, dones, log_probs, values = agent.collect_rollout(env, rollout_steps)
        advantages, returns = agent.compute_gae(rewards, dones, values)
        agent.update(states, actions, log_probs, advantages, returns)

        # Collect episode rewards from rollout for plotting
        ep_r = 0
        for r, d in zip(rewards, dones):
            ep_r += r
            if d:
                all_rewards.append(ep_r)
                ep_r = 0

        if (update + 1) % 10 == 0:
            avg = np.mean(all_rewards[-20:]) if all_rewards else 0
            print(f"  Update {update+1:3d} | Avg episode reward (last 20): {avg:.1f}")

    env.close()
    return all_rewards


# ─────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────
def plot_results(dqn_rewards, ppo_rewards):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DQN vs PPO on CartPole-v1", fontsize=14, fontweight="bold")

    # ── DQN ──
    ax = axes[0]
    ax.plot(dqn_rewards, alpha=0.3, color="#4C72B0", label="Raw")
    if len(dqn_rewards) >= 20:
        ax.plot(range(19, len(dqn_rewards)), smooth(dqn_rewards),
                color="#4C72B0", linewidth=2, label="Smoothed (w=20)")
    ax.axhline(y=195, color="red", linestyle="--", alpha=0.6, label="Solved (195)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN — Learning Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── PPO ──
    ax2 = axes[1]
    ax2.plot(ppo_rewards, alpha=0.3, color="#DD8452", label="Raw")
    if len(ppo_rewards) >= 20:
        ax2.plot(range(19, len(ppo_rewards)), smooth(ppo_rewards),
                 color="#DD8452", linewidth=2, label="Smoothed (w=20)")
    ax2.axhline(y=195, color="red", linestyle="--", alpha=0.6, label="Solved (195)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("PPO — Learning Curve")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Comparison: smoothed moving average ──
    ax3 = axes[2]
    window = 20
    if len(dqn_rewards) >= window:
        dqn_smooth = smooth(dqn_rewards, window)
        ax3.plot(range(window - 1, len(dqn_rewards)), dqn_smooth,
                 color="#4C72B0", linewidth=2, label="DQN")
    if len(ppo_rewards) >= window:
        ppo_smooth = smooth(ppo_rewards, window)
        ax3.plot(range(window - 1, len(ppo_rewards)), ppo_smooth,
                 color="#DD8452", linewidth=2, label="PPO")
    ax3.axhline(y=195, color="red", linestyle="--", alpha=0.6, label="Solved (195)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Smoothed Reward")
    ax3.set_title("DQN vs PPO — Comparison")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {RESULTS_DIR}/training_curves.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dqn_rewards = train_dqn(n_episodes=600)
    ppo_rewards = train_ppo(n_updates=60, rollout_steps=2048)

    dqn_solved = next((i for i, r in enumerate(
        [np.mean(dqn_rewards[max(0,i-99):i+1]) for i in range(len(dqn_rewards))]
    ) if r >= 195), None)

    print(f"\n{'─'*40}")
    print(f"DQN — Final avg reward (last 100 ep): {np.mean(dqn_rewards[-100:]):.1f}")
    print(f"PPO — Final avg reward (last 20 ep):  {np.mean(ppo_rewards[-20:]) if ppo_rewards else 0:.1f}")
    if dqn_solved:
        print(f"DQN solved at episode {dqn_solved} (avg ≥ 195 over 100 episodes)")
    print(f"{'─'*40}")

    plot_results(dqn_rewards, ppo_rewards)
