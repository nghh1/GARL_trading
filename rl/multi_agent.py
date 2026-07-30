"""
Multi-agent RL baselines: N agents, each with its OWN private environment
(one stock), trained completely INDEPENDENTLY -- no communication, no
knowledge sharing. Available in three algorithms (A2C, PPO, DQN),
matching the single-agent lineup.

train_multi_agent_a2c() is deliberately the "GARL minus the group learning
mechanism" ablation: same agents, same environments, same A2C algorithm,
same training budget as garl/ddal.py, differing only in whether gradients
are ever shared between agents. That isolated difference is exactly what
the GARL paper's evaluation is about, so THIS specific function is what
makes the GARL comparison meaningful rather than confounded by algorithm
changes. GARL only exists in A2C form (see garl/ddal.py's docstring for
why), so PPO/DQN multi-agent aren't GARL ablations -- they're additional
context: do independently-trained agents using algorithms individually
stronger than A2C still fail to match GARL's sharing advantage (if any)?
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import torch
import config as C
from envs.trading_env import SingleStockTradingEnv
from rl.a2c_agent import ActorCritic, RolloutState, epoch_step, greedy_action_series
from rl.ppo_agent import epoch_step_ppo
from rl.dqn_agent import QNetwork, ReplayBuffer, epsilon_greedy_action, dqn_update, epsilon_at

# ---------------------------------------------------------------------------
# A2C (the GARL ablation -- see module docstring)
# ---------------------------------------------------------------------------
def train_multi_agent_a2c(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                           epochs: int = C.RL_EPOCHS_TRAIN, seed: int = C.RANDOM_SEED,
                           rollout_len: int = C.RL_ROLLOUT_LEN, device: str = None, 
                           return_history: bool = False):
    """Train one independent ActorCritic per ticker. Returns dict[ticker] -> model."""
    device = device or C.DEVICE
    models, optimizers, states, rngs = {}, {}, {}, {}
    for i, (t, feat) in enumerate(features_by_ticker.items()):
        torch.manual_seed(seed + i)
        env = SingleStockTradingEnv(feat, close_by_ticker[t], ticker=t)
        obs, _ = env.reset()
        model = ActorCritic(obs.shape[0], env.action_space.n).to(device)
        models[t] = model
        optimizers[t] = torch.optim.Adam(model.parameters(), lr=C.RL_LR)
        states[t] = RolloutState(obs=obs, env=env)
        rngs[t] = np.random.default_rng(seed + i)
    history = {t:[] for t in models}

    for ep in range(epochs):
        for t in models:
            loss_val = epoch_step(states[t], models[t], rollout_len, C.RL_GAMMA,
                       C.RL_ENTROPY_COEF, C.RL_VALUE_COEF, rngs[t])
            optimizers[t].step()  # <-- applies its OWN gradient only, no sharing
            history[t].append(loss_val)
    if return_history:
        return models, history
    return models

@torch.no_grad()
def predict_positions_multi_agent_a2c(models: Dict[str, ActorCritic],
                                       features_by_ticker: Dict[str, pd.DataFrame],
                                       close_by_ticker: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    out = {}
    for t, model in models.items():
        env = SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
        actions = greedy_action_series(model, env)
        levels = [C.POSITION_LEVELS[a] for a in actions]
        out[t] = pd.Series(levels, index=env.features.index[:len(levels)])
    return out

# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
def train_multi_agent_ppo(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                           epochs: int = C.RL_EPOCHS_TRAIN, seed: int = C.RANDOM_SEED,
                           rollout_len: int = C.RL_ROLLOUT_LEN, clip_eps: float = C.PPO_CLIP_EPS, 
                           gae_lambda: float = C.PPO_GAE_LAMBDA, device: str = None) -> Dict[str, ActorCritic]:
    """Same population structure as the A2C version, trained via PPO's
    clipped-surrogate update (rl/ppo_agent.py) instead of a single A2C
    gradient step per rollout.
    """
    device = device or C.DEVICE
    models, optimizers, states, rngs = {}, {}, {}, {}
    for i, (t, feat) in enumerate(features_by_ticker.items()):
        torch.manual_seed(seed + i)
        env = SingleStockTradingEnv(feat, close_by_ticker[t], ticker=t)
        obs, _ = env.reset()
        model = ActorCritic(obs.shape[0], env.action_space.n).to(device)
        models[t] = model
        optimizers[t] = torch.optim.Adam(model.parameters(), lr=C.RL_LR)
        states[t] = RolloutState(obs=obs, env=env)
        rngs[t] = np.random.default_rng(seed + i)

    minibatch = min(C.PPO_MINIBATCH_SIZE, rollout_len)
    for ep in range(epochs):
        for t in models:
            epoch_step_ppo(states[t], models[t], optimizers[t], rollout_len, C.RL_GAMMA,
                            gae_lambda, clip_eps, C.PPO_EPOCHS_PER_UPDATE,
                            minibatch, C.RL_ENTROPY_COEF, C.RL_VALUE_COEF, rngs[t])
    return models


# PPO shares ActorCritic with A2C, so prediction is identical
predict_positions_multi_agent_ppo = predict_positions_multi_agent_a2c


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------
def train_multi_agent_dqn(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                           epochs: int = C.RL_EPOCHS_TRAIN, seed: int = C.RANDOM_SEED,
                           rollout_len: int = C.RL_ROLLOUT_LEN, 
                           epsilon_decay_frac: float = C.DQN_EPSILON_DECAY_FRAC, device: str = None) -> Dict[str, QNetwork]:
    """N independent Q-networks, one per ticker, each with its own replay
    buffer and target network. rollout_len here is reused as "env steps
    collected between learning updates" for consistency with the other
    two algorithms' signatures, not a DQN-native concept.
    """
    device = device or C.DEVICE
    models, targets, optimizers, buffers, envs_, rngs = {}, {}, {}, {}, {}, {}
    cur_obs = {}
    for i, (t, feat) in enumerate(features_by_ticker.items()):
        torch.manual_seed(seed + i)
        env = SingleStockTradingEnv(feat, close_by_ticker[t], ticker=t)
        obs, _ = env.reset()
        n_actions = env.action_space.n
        model = QNetwork(obs.shape[0], n_actions).to(device)
        target = QNetwork(obs.shape[0], n_actions).to(device)
        target.load_state_dict(model.state_dict())
        models[t], targets[t] = model, target
        optimizers[t] = torch.optim.Adam(model.parameters(), lr=C.RL_LR)
        buffers[t] = ReplayBuffer(C.DQN_REPLAY_BUFFER_SIZE, seed=seed + i)
        envs_[t] = env
        rngs[t] = np.random.default_rng(seed + i)
        cur_obs[t] = obs

    n_actions = C.N_ACTIONS
    for ep in range(epochs):
        eps = epsilon_at(ep, epochs, C.DQN_EPSILON_START, C.DQN_EPSILON_END, epsilon_decay_frac)
        for t in models:
            for _ in range(rollout_len):
                a = epsilon_greedy_action(models[t], cur_obs[t], eps, n_actions, rngs[t])
                next_obs, reward, terminated, truncated, info = envs_[t].step(a)
                buffers[t].push(cur_obs[t], a, reward, next_obs, terminated or truncated)
                cur_obs[t] = next_obs
                if terminated or truncated:
                    cur_obs[t], _ = envs_[t].reset()
                if len(buffers[t]) >= C.DQN_MIN_REPLAY_SIZE:
                    batch = buffers[t].sample(C.DQN_BATCH_SIZE)
                    dqn_update(models[t], targets[t], optimizers[t], batch, C.RL_GAMMA)
            if ep % C.DQN_TARGET_SYNC_EVERY == 0:
                targets[t].load_state_dict(models[t].state_dict())
    return models

@torch.no_grad()
def predict_positions_multi_agent_dqn(models: Dict[str, QNetwork],
                                       features_by_ticker: Dict[str, pd.DataFrame],
                                       close_by_ticker: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    out = {}
    for t, model in models.items():
        device = next(model.parameters()).device
        env = SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
        obs, _ = env.reset()
        levels = []
        while True:
            q = model(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            a = int(torch.argmax(q, dim=-1).item())
            levels.append(C.POSITION_LEVELS[a])
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        out[t] = pd.Series(levels, index=env.features.index[:len(levels)])
    return out


if __name__ == "__main__":
    from data import synthetic
    from data import features as F

    raw = synthetic.download_universe(["AAPL", "JPM"], "2015-01-01", "2018-01-01")
    feats = {t: F.build_features(df).dropna() for t, df in raw.items()}
    closes = {t: raw[t]["close"] for t in raw}

    for name, train_fn, predict_fn in [
        ("A2C", train_multi_agent_a2c, predict_positions_multi_agent_a2c),
        ("PPO", train_multi_agent_ppo, predict_positions_multi_agent_ppo),
        ("DQN", train_multi_agent_dqn, predict_positions_multi_agent_dqn),
    ]:
        models = train_fn(feats, closes, epochs=8, rollout_len=8)
        pos = predict_fn(models, feats, closes)
        print(f"--- {name} ---")
        for t, s in pos.items():
            print(t, "unique positions:", s.unique().tolist()[:5])
    print("OK multi-agent A2C/PPO/DQN train + predict")