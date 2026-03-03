"""
Deep Q-Network (DQN) — from scratch
Paper: Mnih et al., 2015 (Nature)

Key components:
  - Experience Replay: break temporal correlations by sampling random minibatches
  - Target Network: stabilize training by fixing Q-targets for C steps
  - ε-greedy exploration: balance exploration vs exploitation
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_dim:      int,
        action_dim:     int,
        lr:             float = 1e-3,
        gamma:          float = 0.99,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.01,
        epsilon_decay:  float = 0.995,
        buffer_size:    int   = 10_000,
        batch_size:     int   = 64,
        target_update:  int   = 100,   # steps between target network syncs
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps         = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net      = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(s).argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = [x.to(self.device) for x in self.buffer.sample(self.batch_size)]

        # Current Q-values
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            next_q   = self.target_net(ns).max(1)[0]
            q_targets = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
