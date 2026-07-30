"""
Shared Actor-Critic network + one-epoch A2C gradient computation.

This is deliberately factored so that fit-and-apply-immediately (plain
single-agent / independent multi-agent baselines) and fit-then-hand-the-
gradient-to-a-sharing-protocol (GARL/DDAL, garl/ddal.py) both build on the
exact same underlying agent and loss -- the ONLY difference between the
"multi-agent" baseline and "GARL" in this codebase is whether gradients are
shared (garl/ddal.py) or not (rl/multi_agent.py), which is precisely the
comparison the GARL paper is making.

`epoch_step()` mirrors "Algorithm 1, lines 2-4" from Wu & Zeng (2023):
generate k experiences, compute average loss, compute gradients -- and
stops there (populates .grad on every parameter but does NOT call
optimizer.step()), so the caller decides how those gradients get used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, obs: np.ndarray, rng: Optional[np.random.Generator] = None):
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            logits, value = self.forward(obs_t)
            probs = Fnn.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if rng is not None:
            action = rng.choice(len(probs), p=probs / probs.sum())
        else:
            action = int(np.argmax(probs))
        return int(action), float(value.item())


@dataclass
class RolloutState:
    """Carries the environment forward across epochs (episode may span many epochs)."""
    obs: np.ndarray
    env: object


def collect_rollout(state: RolloutState, model: ActorCritic, k: int,
                     rng: np.random.Generator):
    """Collect k steps of experience starting from state.obs, resetting the
    env if an episode ends mid-rollout. Returns experiences + updated state.
    """
    obs_buf, act_buf, rew_buf, val_buf, done_buf = [], [], [], [], []
    obs = state.obs
    for _ in range(k):
        action, value = model.act(obs, rng=rng)
        next_obs, reward, terminated, truncated, info = state.env.step(action)
        obs_buf.append(obs); act_buf.append(action); rew_buf.append(reward)
        val_buf.append(value); done_buf.append(terminated or truncated)
        if terminated or truncated:
            next_obs, _ = state.env.reset()
        obs = next_obs
    state.obs = obs
    return obs_buf, act_buf, rew_buf, val_buf, done_buf


def compute_a2c_loss(model: ActorCritic, obs_buf, act_buf, rew_buf, done_buf,
                      bootstrap_obs: np.ndarray, gamma: float, entropy_coef: float,
                      value_coef: float) -> torch.Tensor:
    device = next(model.parameters()).device
    obs_t = torch.from_numpy(np.stack(obs_buf)).float().to(device)
    act_t = torch.tensor(act_buf, dtype=torch.long, device=device)
    logits, values = model(obs_t)
    with torch.no_grad():
        _, bootstrap_value = model(torch.from_numpy(bootstrap_obs).float().unsqueeze(0).to(device))
        bootstrap_value = bootstrap_value.item()

    returns = np.zeros(len(rew_buf), dtype=np.float32)
    R = bootstrap_value
    for t in reversed(range(len(rew_buf))):
        R = rew_buf[t] + gamma * R * (0.0 if done_buf[t] else 1.0)
        returns[t] = R
    returns_t = torch.from_numpy(returns).float().to(device)
    advantages = returns_t - values

    log_probs_all = Fnn.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(1, act_t.unsqueeze(1)).squeeze(1)
    probs_all = Fnn.softmax(logits, dim=-1)
    entropy = -(probs_all * log_probs_all).sum(dim=1).mean()

    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = Fnn.mse_loss(values, returns_t)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss


def epoch_step(state: RolloutState, model: ActorCritic, k: int, gamma: float,
               entropy_coef: float, value_coef: float, rng: np.random.Generator) -> float:
    """One epoch: rollout k steps, compute loss, backward() -> populates
    .grad on model.parameters(). Optimizer step is the CALLER's decision.
    Returns the scalar loss value for logging.
    """
    obs_buf, act_buf, rew_buf, val_buf, done_buf = collect_rollout(state, model, k, rng)
    model.zero_grad()
    loss = compute_a2c_loss(model, obs_buf, act_buf, rew_buf, done_buf,
                             bootstrap_obs=state.obs, gamma=gamma,
                             entropy_coef=entropy_coef, value_coef=value_coef)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    return float(loss.item())


def grads_of(model: ActorCritic) -> List[torch.Tensor]:
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]


def apply_grads(model: ActorCritic, grads: List[torch.Tensor], optimizer: torch.optim.Optimizer):
    for p, g in zip(model.parameters(), grads):
        p.grad = g.clone()
    optimizer.step()


@torch.no_grad()
def greedy_action_series(model: ActorCritic, env) -> List[int]:
    """Deterministic (argmax) rollout over a full episode -- used at
    evaluation/backtest time so results are reproducible (no sampling).
    """
    device = next(model.parameters()).device
    obs, _ = env.reset()
    actions = []
    while True:
        logits, _ = model(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        a = int(torch.argmax(logits, dim=-1).item())
        actions.append(a)
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break
    return actions
