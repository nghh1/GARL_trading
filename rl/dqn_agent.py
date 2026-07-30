"""
DQN (Deep Q-Network) core -- shared by single-agent and multi-agent DQN
baselines.

Structurally the most different of the three algorithms here from A2C/PPO:
  - No actor-critic split -- one Q-network, no separate value head.
  - Off-policy: learns from a replay buffer of past experience rather than
    only the most recent rollout (A2C/PPO are on-policy).
  - Epsilon-greedy exploration instead of sampling from a learned
    stochastic policy.
  - A separate, periodically-synced TARGET network stabilizes the
    bootstrapped Q-learning target (standard DQN trick, Mnih et al. 2015).

This is also why DQN can't participate in GARL/DDAL's gradient-sharing
protocol: Algorithm 1 assumes each epoch's gradient reflects a fresh
on-policy rollout, which is close to the opposite of an off-policy,
replay-buffer-averaged algorithm -- there's no single well-defined
"this epoch's gradient" the way A2C has one.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (np.stack(obs).astype(np.float32), np.array(act),
                np.array(rew, dtype=np.float32), np.stack(next_obs).astype(np.float32),
                np.array(done, dtype=np.float32))


def epsilon_greedy_action(model: QNetwork, obs: np.ndarray, epsilon: float,
                           n_actions: int, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    device = next(model.parameters()).device
    with torch.no_grad():
        q = model(torch.from_numpy(obs).float().unsqueeze(0).to(device))
    return int(torch.argmax(q, dim=-1).item())


def dqn_update(model: QNetwork, target_model: QNetwork, optimizer: torch.optim.Optimizer,
               batch, gamma: float) -> float:
    device = next(model.parameters()).device
    obs, act, rew, next_obs, done = batch
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(act).long().to(device)
    rew_t = torch.from_numpy(rew).to(device)
    next_obs_t = torch.from_numpy(next_obs).to(device)
    done_t = torch.from_numpy(done).to(device)

    q_values = model(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_obs_t).max(dim=1)[0]
        target = rew_t + gamma * next_q * (1.0 - done_t)
    loss = Fnn.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def epsilon_at(epoch: int, total_epochs: int, eps_start: float, eps_end: float,
               decay_frac: float) -> float:
    """
    Linear decay from eps_start to eps_end over the first `decay_frac`
    fraction of training, held at eps_end afterward.
    """
    decay_epochs = max(1, int(total_epochs * decay_frac))
    if epoch >= decay_epochs:
        return eps_end
    frac = epoch / decay_epochs
    return eps_start + frac * (eps_end - eps_start)


if __name__ == "__main__":
    import config as C
    from envs.trading_env import SingleStockTradingEnv
    from data import synthetic
    from data import features as F

    torch.manual_seed(0)
    raw = synthetic.download_universe(["AAPL"], "2015-01-01", "2018-01-01")["AAPL"]
    feat = F.build_features(raw).dropna()
    env = SingleStockTradingEnv(feat, raw["close"], ticker="AAPL")
    obs, _ = env.reset()
    n_actions = env.action_space.n

    model = QNetwork(obs.shape[0], n_actions)
    target = QNetwork(obs.shape[0], n_actions)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    buffer = ReplayBuffer(2000, seed=0)
    rng = np.random.default_rng(0)

    losses = []
    cur_obs = obs
    for ep in range(60):
        eps = epsilon_at(ep, 60, 1.0, 0.05, 0.5)
        a = epsilon_greedy_action(model, cur_obs, eps, n_actions, rng)
        next_obs, reward, terminated, truncated, info = env.step(a)
        buffer.push(cur_obs, a, reward, next_obs, terminated or truncated)
        cur_obs = next_obs
        if terminated or truncated:
            cur_obs, _ = env.reset()
        if len(buffer) >= 32:
            batch = buffer.sample(32)
            loss = dqn_update(model, target, opt, batch, gamma=0.99)
            losses.append(loss)
        if ep % 10 == 0:
            target.load_state_dict(model.state_dict())
    print("losses (last 5):", [round(l, 4) for l in losses[-5:]])
    print("OK dqn core works")