"""
Generic nested walk-forward + Optuna tuner shared by every supervised
baseline (ARIMAX, Random Forest, LSTM, TCN, TFT). RL/GARL agents use their
own tuner in rl/ and garl/ since they train via env rollouts, not (X, y).

The objective is the MEAN Sharpe ratio across the inner walk-forward
folds (computed via the exact same backtest engine used for final
reporting, so the number Optuna optimizes for is the number we actually
care about, not a proxy like MSE). Optuna never sees the outer test fold:
`X_train_full`/`y_train_full`/`close_train_full` passed in here must
already be sliced to a single outer-fold's TRAIN block only.
"""
from __future__ import annotations

import logging
from typing import Type

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

from cv import walk_forward as WF
from backtest import engine as ENGINE
import config as C


def tune_baseline(baseline_cls: Type, X_train_full: pd.DataFrame, y_train_full: pd.Series,
                   close_train_full: pd.Series, n_trials: int = C.N_TRIALS,
                   n_inner_folds: int = C.N_INNER_FOLDS, min_train_bars: int = C.MIN_TRAIN_BARS,
                   embargo: int = C.EMBARGO_BARS, seed: int = C.RANDOM_SEED):
    train_idx = np.arange(len(X_train_full))
    inner_min_train_bars = min(min_train_bars, len(train_idx) // 3)
    inner_folds = WF.inner_splits(train_idx, n_inner_folds, inner_min_train_bars, embargo)
    if not inner_folds:
        raise ValueError("No inner folds could be constructed -- outer-train block too small.")

    # Equalize inner-fold training length, same embargo-compensation logic as
    # the outer-fold fix (config.MIN_TRAIN_BARS/MAX_TRAIN_BARS): inner_splits()
    # produces a genuinely expanding sequence (e.g. 611/1025/1439 bars for a
    # 1864-bar outer block) with no capping of its own. Inner fold 0's train
    # can never exceed inner_min_train_bars-embargo, so that's the natural
    # equal-length target every fold gets capped/aligned to here -- makes all
    # n_inner_folds train on the SAME length, rather than averaging the Optuna
    # objective across folds with meaningfully different amounts of data.
    target_inner_train = inner_min_train_bars - embargo
    inner_folds = [(tr[-target_inner_train:] if len(tr) > target_inner_train else tr, va)
                   for tr, va in inner_folds]

    def objective(trial: optuna.Trial) -> float:
        params = baseline_cls.default_param_space(trial)
        scores = []
        for tr_idx, va_idx in inner_folds:
            X_tr, y_tr = X_train_full.iloc[tr_idx], y_train_full.iloc[tr_idx]
            X_va, close_va = X_train_full.iloc[va_idx], close_train_full.iloc[va_idx]
            try:
                model = baseline_cls(**params)
                model.fit(X_tr, y_tr)
                pos = model.predict_position(X_va)
                res = ENGINE.single_asset_backtest(close_va, pos)
                score = res.summary["Sharpe"]
                score = float(score) if np.isfinite(score) else -10.0
            except Exception as e:  # noqa: BLE001
                logger.debug("Trial failed on a fold: %s", e)
                score = -10.0
            scores.append(score)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study