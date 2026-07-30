"""
PPO (Proximal Policy Optimization) core -- shared by single-agent and
multi-agent PPO baselines.

Reuses the exact same ActorCritic network as A2C (rl/a2c_agent.py). PPO
only differs in HOW the network is updated (clipped surrogate objective +
GAE advantages + multiple epochs of minibatch SGD per rollout), not in
architecture -- this keeps PPO directly comparable to the A2C baselines:
same policy/value network, same observation/action space, different
(generally more sample-efficient and stable) update rule.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as Fnn

from rl.a2c_agent import ActorCritic, RolloutState


def collect_rollout_ppo(state: RolloutState, model: ActorCritic, k: int,
                         rng: np.random.Generator):
    """Like a2c_agent.collect_rollout, but also records each action's
    log-probability under the CURRENT (soon-to-be-old) policy -- needed
    for PPO's probability ratio -- and the value estimate at each step,
    needed for GAE.
    """
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    obs = state.obs
    device = next(model.parameters()).device
    for _ in range(k):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            logits, value = model(obs_t)
            probs = Fnn.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        probs = probs / probs.sum()
        action = rng.choice(len(probs), p=probs)
        logp = float(np.log(probs[action] + 1e-8))
        next_obs, reward, terminated, truncated, info = state.env.step(action)
        obs_buf.append(obs); act_buf.append(action); rew_buf.append(reward)
        val_buf.append(value.item()); logp_buf.append(logp); done_buf.append(terminated or truncated)
        if terminated or truncated:
            next_obs, _ = state.env.reset()
        obs = next_obs
    state.obs = obs
    return obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf


def compute_gae(rewards: List[float], values: List[float], dones: List[bool],
                 next_value: float, gamma: float, lam: float):
    """Generalized Advantage Estimation (Schulman et al., 2016)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
                obs_buf, act_buf, old_logp_buf, advantages: np.ndarray, returns: np.ndarray,
                clip_eps: float, n_epochs: int, minibatch_size: int,
                entropy_coef: float, value_coef: float) -> float:
    device = next(model.parameters()).device
    obs_t = torch.from_numpy(np.stack(obs_buf)).float().to(device)
    act_t = torch.tensor(act_buf, dtype=torch.long, device=device)
    old_logp_t = torch.tensor(old_logp_buf, dtype=torch.float32, device=device)
    adv_t = torch.from_numpy(advantages).float().to(device)
    ret_t = torch.from_numpy(returns).float().to(device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)  # standard PPO stabilization

    n = len(obs_buf)
    idx_all = np.arange(n)
    last_loss = 0.0
    for _ in range(n_epochs):
        np.random.shuffle(idx_all)
        for start in range(0, n, minibatch_size):
            mb_idx = torch.from_numpy(idx_all[start:start + minibatch_size]).long().to(device)

            logits, values = model(obs_t[mb_idx])
            log_probs_all = Fnn.log_softmax(logits, dim=-1)
            new_logp = log_probs_all.gather(1, act_t[mb_idx].unsqueeze(1)).squeeze(1)
            probs_all = Fnn.softmax(logits, dim=-1)
            entropy = -(probs_all * log_probs_all).sum(dim=1).mean()

            ratio = torch.exp(new_logp - old_logp_t[mb_idx])
            surr1 = ratio * adv_t[mb_idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = Fnn.mse_loss(values, ret_t[mb_idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.item())
    return last_loss


def epoch_step_ppo(state: RolloutState, model: ActorCritic, optimizer: torch.optim.Optimizer,
                    k: int, gamma: float, gae_lambda: float, clip_eps: float,
                    n_epochs: int, minibatch_size: int, entropy_coef: float,
                    value_coef: float, rng: np.random.Generator) -> float:
    """
    One PPO 'epoch': collect a k-step rollout, compute GAE, run
    n_epochs of minibatch clipped-surrogate updates on it.

    This is not directly 1:1 comparable to an A2C 'epoch' (which does
    a single gradient step per k-step rollout) -- PPO deliberately reuses
    each rollout for several gradient steps. Both algorithms use the same
    RL_EPOCHS_TRAIN as "how many rollouts of environment experience to
    collect", which keeps the amount of ENVIRONMENT EXPERIENCE comparable
    across algorithms even though gradient steps per rollout differ.
    """
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = collect_rollout_ppo(state, model, k, rng)
    device = next(model.parameters()).device
    with torch.no_grad():
        _, next_value = model(torch.from_numpy(state.obs).float().unsqueeze(0).to(device))
    advantages, returns = compute_gae(rew_buf, val_buf, done_buf, next_value.item(), gamma, gae_lambda)
    return ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, advantages, returns,
                       clip_eps, n_epochs, minibatch_size, entropy_coef, value_coef)


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
    model = ActorCritic(obs.shape[0], env.action_space.n)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    state = RolloutState(obs=obs, env=env)
    rng = np.random.default_rng(0)
    losses = []
    for ep in range(10):
        loss = epoch_step_ppo(state, model, opt, k=32, gamma=0.99, gae_lambda=0.95,
                               clip_eps=0.2, n_epochs=4, minibatch_size=8,
                               entropy_coef=0.01, value_coef=0.5, rng=rng)
        losses.append(loss)
    print("losses:", [round(l, 3) for l in losses])
    print("OK ppo epoch_step_ppo works")