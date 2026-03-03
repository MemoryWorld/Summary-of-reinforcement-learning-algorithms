"""
Proximal Policy Optimization (PPO) — from scratch
Paper: Schulman et al., 2017

Key components:
  - Actor-Critic architecture: shared backbone, separate heads
  - GAE (Generalized Advantage Estimation): variance-reduced advantage
  - Clipped surrogate objective: prevent destructively large policy updates
  - Multiple epochs per rollout: sample efficiency over vanilla PG
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ─────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, action_dim)  # policy head → logits
        self.critic = nn.Linear(hidden, 1)            # value head → V(s)

    def forward(self, x: torch.Tensor):
        h     = self.shared(x)
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        return logits, value

    def get_action(self, state: torch.Tensor):
        logits, value = self(state)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self(states)
        dist    = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy


# ─────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(
        self,
        state_dim:      int,
        action_dim:     int,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        lam:            float = 0.95,   # GAE lambda
        clip_eps:       float = 0.2,    # PPO clip ratio
        epochs:         int   = 4,      # update epochs per rollout
        batch_size:     int   = 64,
        vf_coef:        float = 0.5,    # value loss coefficient
        ent_coef:       float = 0.01,   # entropy bonus coefficient
    ):
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.epochs     = epochs
        self.batch_size = batch_size
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net    = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt    = optim.Adam(self.net.parameters(), lr=lr)

    # ── Rollout collection ────────────────────────────────────────────
    def collect_rollout(self, env, n_steps: int = 2048):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state, _ = env.reset()

        for _ in range(n_steps):
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.net.get_action(s)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state if not done else env.reset()[0]

        return states, actions, rewards, dones, log_probs, values

    # ── GAE advantage estimation ──────────────────────────────────────
    def compute_gae(self, rewards, dones, values, last_value: float = 0.0):
        advantages = np.zeros(len(rewards))
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            delta    = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae      = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values)
        return advantages, returns

    # ── PPO update ────────────────────────────────────────────────────
    def update(self, states, actions, log_probs_old, advantages, returns):
        states       = torch.FloatTensor(np.array(states)).to(self.device)
        actions      = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        advantages   = torch.FloatTensor(advantages).to(self.device)
        returns      = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        total_loss = 0.0

        for _ in range(self.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                log_probs_new, values, entropy = self.net.evaluate(states[mb], actions[mb])

                # Clipped surrogate loss
                ratio      = (log_probs_new - log_probs_old[mb]).exp()
                surr1      = ratio * advantages[mb]
                surr2      = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb]
                actor_loss  = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.MSELoss()(values, returns[mb])

                # Total loss
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                total_loss += loss.item()

        return total_loss
