"""
Single-agent RL baseline.

One monolithic policy observes ALL tickers' feature vectors concatenated
together and outputs one discrete position-level action PER ticker (a
joint action, modeled as independent categorical/Q heads sharing one
trunk -- a standard "multi-discrete via factorized heads" design). This
is the natural single-agent contrast to the multi-agent / GARL setups:
same total capital, same tickers, but ONE policy that must learn a joint
representation instead of N specialists. Available in three algorithms
(A2C, PPO, DQN).
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import config as C
from envs.trading_env import SingleStockTradingEnv
from rl.dqn_agent import ReplayBuffer, epsilon_at

class MultiHeadActorCritic(nn.Module):
    def __init__(self, obs_dim_per_ticker: int, n_tickers: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.n_tickers, self.n_actions = n_tickers, n_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim_per_ticker * n_tickers, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_heads = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(n_tickers)])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        logits = torch.stack([head(h) for head in self.policy_heads], dim=1)  # (B, n_tickers, n_actions)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class MultiHeadQNetwork(nn.Module):
    """DQN analogue of MultiHeadActorCritic: shared trunk, one Q-value
    head per ticker, NO value head (DQN has no critic/actor split)."""
    def __init__(self, obs_dim_per_ticker: int, n_tickers: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.n_tickers, self.n_actions = n_tickers, n_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim_per_ticker * n_tickers, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.q_heads = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(n_tickers)])

    def forward(self, x):
        h = self.trunk(x)
        return torch.stack([head(h) for head in self.q_heads], dim=1)  # (B, n_tickers, n_actions)


class JointPortfolioEnvWrapper:
    """Steps N single-stock envs together as one joint environment; used only
    by the single-agent baselines (multi-agent/GARL keep envs fully separate).
    """
    def __init__(self, envs_by_ticker: Dict[str, SingleStockTradingEnv]):
        self.envs = envs_by_ticker
        self.tickers = list(envs_by_ticker.keys())
        self.n = min(e.n for e in self.envs.values())

    def reset(self):
        obs = [self.envs[t].reset()[0] for t in self.tickers]
        return np.stack(obs)  # (n_tickers, obs_dim)

    def step(self, actions: List[int]):
        obs_list, rew_list, term_list, info_list = [], [], [], []
        for t, a in zip(self.tickers, actions):
            o, r, term, trunc, info = self.envs[t].step(a)
            obs_list.append(o); rew_list.append(r); term_list.append(term or trunc); info_list.append(info)
        obs = np.stack(obs_list)
        reward = float(np.mean(rew_list))          # shared scalar reward = avg across tickers
        terminated = any(term_list)
        return obs, reward, terminated, False, info_list

# ---------------------------------------------------------------------------
# A2C
# ---------------------------------------------------------------------------
def train_single_agent_a2c(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                            epochs: int = C.RL_EPOCHS_TRAIN, seed: int = C.RANDOM_SEED,
                            rollout_len: int = C.RL_ROLLOUT_LEN, device: str = None) -> MultiHeadActorCritic:
    device = device or C.DEVICE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = {t: SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
            for t in features_by_ticker}
    joint = JointPortfolioEnvWrapper(envs)
    obs = joint.reset()
    obs_dim = obs.shape[1]
    model = MultiHeadActorCritic(obs_dim, len(joint.tickers), C.N_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.RL_LR)

    for ep in range(epochs):
        obs_buf, act_buf, rew_buf, val_buf, done_buf = [], [], [], [], []
        cur_obs = obs
        for _ in range(rollout_len):
            with torch.no_grad():
                logits, value = model(torch.from_numpy(cur_obs).float().reshape(1, -1).to(device))
                probs = Fnn.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (n_tickers, n_actions)
            actions = [rng.choice(C.N_ACTIONS, p=probs[i] / probs[i].sum()) for i in range(len(joint.tickers))]
            next_obs, reward, terminated, _, _ = joint.step(actions)
            obs_buf.append(cur_obs.copy()); act_buf.append(actions); rew_buf.append(reward)
            val_buf.append(value.item()); done_buf.append(terminated)
            if terminated:
                next_obs = joint.reset()
            cur_obs = next_obs
        obs = cur_obs

        obs_t = torch.from_numpy(np.stack(obs_buf)).float().reshape(len(obs_buf), -1).to(device)
        act_t = torch.tensor(act_buf, dtype=torch.long, device=device)  # (T, n_tickers)
        logits, values = model(obs_t)
        with torch.no_grad():
            _, bootstrap_value = model(torch.from_numpy(obs).float().reshape(1, -1).to(device))
        returns = np.zeros(len(rew_buf), dtype=np.float32)
        R = bootstrap_value.item()
        for t in reversed(range(len(rew_buf))):
            R = rew_buf[t] + C.RL_GAMMA * R * (0.0 if done_buf[t] else 1.0)
            returns[t] = R
        returns_t = torch.from_numpy(returns).float().to(device)
        advantages = returns_t - values

        log_probs_all = Fnn.log_softmax(logits, dim=-1)  # (T, n_tickers, n_actions)
        log_probs = log_probs_all.gather(2, act_t.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        probs_all = Fnn.softmax(logits, dim=-1)
        entropy = -(probs_all * log_probs_all).sum(dim=-1).mean()

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = Fnn.mse_loss(values, returns_t)
        loss = policy_loss + C.RL_VALUE_COEF * value_loss - C.RL_ENTROPY_COEF * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return model


@torch.no_grad()
def predict_positions_single_agent_a2c(model: MultiHeadActorCritic, features_by_ticker: Dict[str, pd.DataFrame],
                                        close_by_ticker: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    device = next(model.parameters()).device
    envs = {t: SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
            for t in features_by_ticker}
    joint = JointPortfolioEnvWrapper(envs)
    obs = joint.reset()
    tickers = joint.tickers
    positions = {t: [] for t in tickers}
    dates = {t: envs[t].features.index for t in tickers}
    while True:
        logits, _ = model(torch.from_numpy(obs).float().reshape(1, -1).to(device))
        actions = torch.argmax(logits.squeeze(0), dim=-1).cpu().numpy()
        for i, t in enumerate(tickers):
            positions[t].append(C.POSITION_LEVELS[actions[i]])
        obs, reward, terminated, _, _ = joint.step(list(actions))
        if terminated:
            break
    out = {}
    for t in tickers:
        n = len(positions[t])
        out[t] = pd.Series(positions[t], index=dates[t][:n])
    return out


# ---------------------------------------------------------------------------
# PPO -- same MultiHeadActorCritic network, clipped-surrogate + GAE update
# over the joint (factorized multi-discrete) action space instead of a
# single A2C gradient step per rollout.
# ---------------------------------------------------------------------------
def train_single_agent_ppo(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                            epochs: int = C.RL_EPOCHS_TRAIN, seed: int = C.RANDOM_SEED,
                            rollout_len: int = C.RL_ROLLOUT_LEN, clip_eps: float = C.PPO_CLIP_EPS, 
                            gae_lambda: float = C.PPO_GAE_LAMBDA, device: str = None) -> MultiHeadActorCritic:
    device = device or C.DEVICE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = {t: SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
            for t in features_by_ticker}
    joint = JointPortfolioEnvWrapper(envs)
    obs = joint.reset()
    obs_dim = obs.shape[1]
    model = MultiHeadActorCritic(obs_dim, len(joint.tickers), C.N_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.RL_LR)
    minibatch = min(C.PPO_MINIBATCH_SIZE, rollout_len)

    for ep in range(epochs):
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        cur_obs = obs
        for _ in range(rollout_len):
            with torch.no_grad():
                logits, value = model(torch.from_numpy(cur_obs).float().reshape(1, -1).to(device))
                probs = Fnn.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (n_tickers, n_actions)
            actions = [rng.choice(C.N_ACTIONS, p=probs[i] / probs[i].sum()) for i in range(len(joint.tickers))]
            joint_logp = float(sum(np.log(probs[i, actions[i]] + 1e-8) for i in range(len(joint.tickers))))
            next_obs, reward, terminated, _, _ = joint.step(actions)
            obs_buf.append(cur_obs.copy()); act_buf.append(actions); rew_buf.append(reward)
            val_buf.append(value.item()); logp_buf.append(joint_logp); done_buf.append(terminated)
            if terminated:
                next_obs = joint.reset()
            cur_obs = next_obs
        obs = cur_obs

        with torch.no_grad():
            _, next_value = model(torch.from_numpy(obs).float().reshape(1, -1).to(device))
        # GAE (scalar reward/value per timestep -- same interface as single-ticker PPO)
        T = len(rew_buf)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_v = next_value.item() if t == T - 1 else val_buf[t + 1]
            mask = 0.0 if done_buf[t] else 1.0
            delta = rew_buf[t] + C.RL_GAMMA * next_v * mask - val_buf[t]
            last_gae = delta + C.RL_GAMMA * gae_lambda * mask * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(val_buf, dtype=np.float32)

        obs_t = torch.from_numpy(np.stack(obs_buf)).float().reshape(T, -1).to(device)
        act_t = torch.tensor(act_buf, dtype=torch.long, device=device)  # (T, n_tickers)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.from_numpy(advantages).float().to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.from_numpy(returns).float().to(device)

        idx_all = np.arange(T)
        for _ in range(C.PPO_EPOCHS_PER_UPDATE):
            np.random.shuffle(idx_all)
            for start in range(0, T, minibatch):
                mb = torch.from_numpy(idx_all[start:start + minibatch]).long().to(device)
                logits, values = model(obs_t[mb])
                log_probs_all = Fnn.log_softmax(logits, dim=-1)
                new_logp = log_probs_all.gather(2, act_t[mb].unsqueeze(-1)).squeeze(-1).sum(dim=1)
                probs_all = Fnn.softmax(logits, dim=-1)
                entropy = -(probs_all * log_probs_all).sum(dim=-1).mean()

                ratio = torch.exp(new_logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = Fnn.mse_loss(values, ret_t[mb])
                loss = policy_loss + C.RL_VALUE_COEF * value_loss - C.RL_ENTROPY_COEF * entropy

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

    return model


# PPO shares MultiHeadActorCritic with A2C, so prediction is identical
predict_positions_single_agent_ppo = predict_positions_single_agent_a2c


# ---------------------------------------------------------------------------
# DQN -- MultiHeadQNetwork, joint replay buffer (shared scalar reward),
# per-head epsilon-greedy action selection, per-head TD loss summed.
# ---------------------------------------------------------------------------
def train_single_agent_dqn(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                            epochs: int = C.RL_EPOCHS_TRAIN, seed: int = C.RANDOM_SEED,
                            rollout_len: int = C.RL_ROLLOUT_LEN, 
                            epsilon_decay_frac: float=C.DQN_EPSILON_DECAY_FRAC, device: str = None) -> MultiHeadQNetwork:
    device = device or C.DEVICE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = {t: SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
            for t in features_by_ticker}
    joint = JointPortfolioEnvWrapper(envs)
    obs = joint.reset()
    n_tickers, obs_dim = obs.shape[0], obs.shape[1]

    model = MultiHeadQNetwork(obs_dim, n_tickers, C.N_ACTIONS).to(device)
    target = MultiHeadQNetwork(obs_dim, n_tickers, C.N_ACTIONS).to(device)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=C.RL_LR)
    buffer = ReplayBuffer(C.DQN_REPLAY_BUFFER_SIZE, seed=seed)

    cur_obs = obs
    for ep in range(epochs):
        eps = epsilon_at(ep, epochs, C.DQN_EPSILON_START, C.DQN_EPSILON_END, epsilon_decay_frac)
        for _ in range(rollout_len):
            if rng.random() < eps:
                actions = [int(rng.integers(0, C.N_ACTIONS)) for _ in range(n_tickers)]
            else:
                with torch.no_grad():
                    q = model(torch.from_numpy(cur_obs).float().reshape(1, -1).to(device))  # (1, n_tickers, n_actions)
                actions = torch.argmax(q.squeeze(0), dim=-1).cpu().numpy().tolist()
            next_obs, reward, terminated, _, _ = joint.step(actions)
            buffer.push(cur_obs.reshape(-1), actions, reward, next_obs.reshape(-1), terminated)
            cur_obs = next_obs
            if terminated:
                cur_obs = joint.reset()

            if len(buffer) >= C.DQN_MIN_REPLAY_SIZE:
                b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(C.DQN_BATCH_SIZE)
                b_obs_t = torch.from_numpy(b_obs).to(device)
                b_act_t = torch.from_numpy(b_act).long().to(device)          # (B, n_tickers)
                b_rew_t = torch.from_numpy(b_rew).to(device)                 # (B,) shared reward
                b_next_obs_t = torch.from_numpy(b_next_obs).to(device)
                b_done_t = torch.from_numpy(b_done).to(device)

                q_all = model(b_obs_t)  # (B, n_tickers, n_actions)
                q_taken = q_all.gather(2, b_act_t.unsqueeze(-1)).squeeze(-1)  # (B, n_tickers)
                with torch.no_grad():
                    next_q_all = target(b_next_obs_t).max(dim=-1)[0]  # (B, n_tickers)
                    tgt = b_rew_t.unsqueeze(-1) + C.RL_GAMMA * next_q_all * (1.0 - b_done_t.unsqueeze(-1))
                loss = Fnn.mse_loss(q_taken, tgt)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if ep % C.DQN_TARGET_SYNC_EVERY == 0:
            target.load_state_dict(model.state_dict())

    return model


@torch.no_grad()
def predict_positions_single_agent_dqn(model: MultiHeadQNetwork, features_by_ticker: Dict[str, pd.DataFrame],
                                        close_by_ticker: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    device = next(model.parameters()).device
    envs = {t: SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
            for t in features_by_ticker}
    joint = JointPortfolioEnvWrapper(envs)
    obs = joint.reset()
    tickers = joint.tickers
    positions = {t: [] for t in tickers}
    dates = {t: envs[t].features.index for t in tickers}
    while True:
        q = model(torch.from_numpy(obs).float().reshape(1, -1).to(device))
        actions = torch.argmax(q.squeeze(0), dim=-1).cpu().numpy()
        for i, t in enumerate(tickers):
            positions[t].append(C.POSITION_LEVELS[actions[i]])
        obs, reward, terminated, _, _ = joint.step(list(actions))
        if terminated:
            break
    out = {}
    for t in tickers:
        n = len(positions[t])
        out[t] = pd.Series(positions[t], index=dates[t][:n])
    return out


if __name__ == "__main__":
    from data import synthetic
    from data import features as F

    raw = synthetic.download_universe(["AAPL", "JPM"], "2015-01-01", "2018-01-01")
    feats = {t: F.build_features(df).dropna() for t, df in raw.items()}
    closes = {t: raw[t]["close"] for t in raw}

    for name, train_fn, predict_fn in [
        ("A2C", train_single_agent_a2c, predict_positions_single_agent_a2c),
        ("PPO", train_single_agent_ppo, predict_positions_single_agent_ppo),
        ("DQN", train_single_agent_dqn, predict_positions_single_agent_dqn),
    ]:
        model = train_fn(feats, closes, epochs=8, rollout_len=8)
        pos = predict_fn(model, feats, closes)
        print(f"--- {name} ---")
        for t, s in pos.items():
            print(t, "unique positions:", s.unique().tolist()[:5])
    print("OK single-agent A2C/PPO/DQN train + predict")
