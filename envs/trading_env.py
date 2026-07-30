"""
Single-stock trading environment (Gymnasium API).
 
This is the "private environment" each GARL agent lives in (S_i, A_i, P_i,
R_i in the paper's group-MDP tuple). One instance = one ticker.
 
Look-ahead protection: observation at step t is built from feature data up
to and including bar t (see data/features.py, causal by construction). The
action chosen at step t is executed as the position held during bar t+1
(reward at step t uses close[t] -> close[t+1] return), i.e. identical
timing convention to backtest/engine.py's shift(1) guard, just expressed
inside the environment's step() instead of after the fact.
 
Temporal context (lookback): by default (lookback=1) each observation is
just the current bar's features + current position -- RL agents structurally
had NO raw sequence window at all, unlike LSTM/TCN/TFT's explicit 20-bar
window (baselines/*.py, config.SEQUENCE_LOOKBACK-equivalent pinned value).
Setting lookback>1 stacks the last `lookback` bars' features (flattened)
into one observation, giving RL agents the same kind of raw multi-day
context the sequence baselines get, closing that documented asymmetry.
At episode start, bars before t=0 don't exist -- the earliest bars pad by
repeating the first available row (never zeros, which would look like a
real all-zero feature reading rather than "no history yet").
 
Action space: Discrete(len(POSITION_LEVELS)) -- target position as a
fraction of the agent's allocated capital (config.POSITION_LEVELS),
supporting shorting, flat, and two long sizes. Chosen over pure
buy/hold/sell because position sizing is what real portfolio-style trading
actually does, while staying tractable for actor-critic training (matches
the DDAL paper's underlying A2C agent).
 
Reward: net-of-cost log return of the agent's own capital for that step,
plus a small drawdown penalty to discourage reckless leverage flips (RL
agents can otherwise learn a chattery position that looks fine in-sample
but drives huge turnover cost out-of-sample).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import config as C

class SingleStockTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, close: pd.Series,
                 cost_bps: float = C.TRANSACTION_COST_BPS,
                 slippage_bps: float = C.SLIPPAGE_BPS,
                 position_levels=None, ticker: str = "", lookback: int = C.RL_LOOKBACK):
        super().__init__()
        self.ticker = ticker
        self.position_levels = np.array(position_levels or C.POSITION_LEVELS, dtype=np.float32)
        self.cost_rate = (cost_bps + slippage_bps) / 1e4
        self.lookback = max(1, lookback)

        feat = features[C.FEATURE_COLUMNS].copy()
        valid = feat.notna().all(axis=1) & close.reindex(feat.index).notna()
        self.features = feat.loc[valid].astype(np.float32)
        self.close = close.reindex(self.features.index).astype(np.float32)
        self.features_arr = self.features.values
        self.n = len(self.features)
        if self.n < max(20, self.lookback):
            raise ValueError(f"Not enough valid bars for {ticker}: {self.n}")
        self.action_space = spaces.Discrete(len(self.position_levels))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(C.FEATURE_COLUMNS)*self.lookback+1, ), 
            dtype=np.float32
        )  # features + current position (agent needs to know its own state)
        self.t = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0

    def _obs(self):
        if self.lookback == 1:
            window = self.features_arr[self.t:self.t + 1]  # (1, n_features)
        else:
            start = self.t - self.lookback + 1
            if start >= 0:
                window = self.features_arr[start:self.t + 1]
            else:
                # pad by repeating the first available row -- never zeros,
                # which would look like a genuine (and misleading) reading
                pad = np.repeat(self.features_arr[0:1], -start, axis=0)
                window = np.concatenate([pad, self.features_arr[0:self.t + 1]], axis=0)
        return np.concatenate([window.reshape(-1), [self.position]]).astype(np.float32)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        return self._obs(), {}

    def step(self, action: int):
        target = float(self.position_levels[action])
        # cost incurred for moving into `target` from current position, charged now
        cost = abs(target - self.position) * self.cost_rate
        # this step's market return realized from close[t] -> close[t+1],
        # earned by the position we are moving INTO (i.e. the trade executes
        # now and is exposed to the next bar's move -- see module docstring)
        if self.t + 1 < self.n:
            step_ret = float(self.close.iloc[self.t + 1] / self.close.iloc[self.t] - 1.0)
        else:
            step_ret = 0.0

        pnl = target * step_ret - cost
        self.equity *= (1 + pnl)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = self.equity / self.peak_equity - 1.0
        reward = np.log(max(1 + pnl, 1e-6)) + 0.02 * drawdown  # small dd penalty (both <=0 terms combine)
        self.position = target
        self.t += 1
        terminated = self.t >= self.n - 1
        truncated = False
        info = {"pnl": pnl, "equity": self.equity, "position": target}
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info

if __name__ == "__main__":
    from data import synthetic
    from data import features as F

    raw = synthetic.download_universe(["AAPL"], "2018-01-01", "2020-01-01")["AAPL"]
    feat = F.build_features(raw)
    env = SingleStockTradingEnv(feat, raw["close"], ticker="AAPL")
    obs, _ = env.reset()
    total_r = 0
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.integers(0, env.action_space.n)
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term:
            break
    print("obs shape:", obs.shape, "steps run OK, total_reward:", round(total_r, 4))
    print("action_space:", env.action_space, "obs_space:", env.observation_space)
