"""
Nested walk-forward + Optuna tuning for the RL-family baselines
(single-agent, independent multi-agent, GARL/DDAL).

Same nested-CV discipline as tuning/optuna_utils.py (tune only inside an
outer-fold's TRAIN block, never touch the outer test fold), but RL training
is far more expensive per trial than the supervised baselines, so:
  - tuning uses a SHORT training budget (tune_epochs << final RL_EPOCHS_TRAIN)
  - tuning uses a SINGLE inner fold (the most recent one) rather than
    averaging over all inner folds, which is the standard practical
    compromise for RL hyperparameter search under a compute budget (still
    strictly separated from the outer test fold, so no leakage into the
    reported test-fold metric -- it just means less-averaged, noisier
    hyperparameter selection than the supervised baselines get, which we
    consider an acceptable, documented trade-off rather than skipping
    tuning altogether).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

from cv import walk_forward as WF
from backtest import engine as ENGINE
import config as C


def _slice_dict(d: Dict[str, pd.DataFrame], idx) -> Dict[str, pd.DataFrame]:
    return {t: df.iloc[idx] for t, df in d.items()}


def _eval_positions(positions: Dict[str, pd.Series], close_by_ticker: Dict[str, pd.Series]) -> float:
    """Mean single-asset Sharpe across tickers -- cheap, robust tuning objective."""
    scores = []
    for t, pos in positions.items():
        close = close_by_ticker[t].reindex(pos.index)
        res = ENGINE.single_asset_backtest(close, pos)
        s = res.summary["Sharpe"]
        scores.append(float(s) if np.isfinite(s) else -10.0)
    return float(np.mean(scores)) if scores else -10.0


def tune_rl_baseline(train_fn: Callable, predict_fn: Callable, param_space_fn: Callable,
                      features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                      n_trials: int = max(5, C.N_TRIALS // 3), tune_epochs: int = 20,
                      min_train_bars: int = C.MIN_TRAIN_BARS, embargo: int = C.EMBARGO_BARS,
                      seed: int = C.RANDOM_SEED):
    """`train_fn(features, closes, epochs, seed, **params) -> models`
       `predict_fn(models, features, closes) -> positions dict`
       `param_space_fn(trial) -> dict` (must NOT include epochs/seed)
    """
    any_ticker = next(iter(features_by_ticker))
    n = len(features_by_ticker[any_ticker])
    idx = np.arange(n)
    inner = WF.inner_splits(idx, n_folds=C.N_INNER_FOLDS, min_train_bars=min(min_train_bars, n // 3),
                             embargo=embargo)
    if not inner:
        raise ValueError("No inner fold available for RL tuning.")
    tr_idx, va_idx = inner[-1]  # most recent inner fold only (compute budget trade-off, see module docstring)

    feat_tr = _slice_dict(features_by_ticker, tr_idx)
    close_tr = {t: s.iloc[tr_idx] for t, s in close_by_ticker.items()}
    feat_va = _slice_dict(features_by_ticker, va_idx)
    close_va = {t: s.iloc[va_idx] for t, s in close_by_ticker.items()}

    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial)
        try:
            models = train_fn(feat_tr, close_tr, epochs=tune_epochs, seed=seed, **params)
            positions = predict_fn(models, feat_va, close_va)
            return _eval_positions(positions, close_va)
        except Exception:
            return -10.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study

def a2c_param_space(trial) -> dict:
    """Used by SingleAgentA2C, MultiAgentA2C, GARL_DDAL -- the core ablation.
    rollout_len is pinned to a single value (not tuned) so all three train
    under an identical regime; the only thing allowed to differ between
    them is whether gradients are shared. Same single-value-categorical
    trick already used to fix the LSTM/TCN/TFT lookback confound.
    """
    return {
        "rollout_len": trial.suggest_categorical("rollout_len", [C.RL_ROLLOUT_LEN]),
    }

def garl_ddal_param_space(trial) -> dict:
    """Extends the core A2C space with DDAL-specific hyperparameters,
    like gradient staleness, for dedicated GARL tuning runs.
    """
    return {
        "rollout_len": trial.suggest_categorical("rollout_len", [C.RL_ROLLOUT_LEN]),
        "staleness_epochs": trial.suggest_categorical("staleness_epochs", [0, 2, 5, 10]),
        "share_threshold_frac": trial.suggest_categorical("share_threshold_frac", [0.1, 0.3, 0.5]),
        "minibatch_epochs": trial.suggest_categorical("minibatch_epochs", [2, 4, 8]),
    }

def ppo_param_space(trial) -> dict:
    """SingleAgentPPO / MultiAgentPPO -- NOT part of the isolated ablation,
    free to tune PPO's own algorithm-specific hyperparameters."""
    return {
        "rollout_len": trial.suggest_categorical("rollout_len", [16, 32, 64]),
        "clip_eps": trial.suggest_float("clip_eps", 0.1, 0.3),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
    }

def dqn_param_space(trial) -> dict:
    """SingleAgentDQN / MultiAgentDQN."""
    return {
        "rollout_len": trial.suggest_categorical("rollout_len", [16, 32, 64]),
        "epsilon_decay_frac": trial.suggest_float("epsilon_decay_frac", 0.3, 0.7),
    }
