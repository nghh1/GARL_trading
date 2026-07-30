"""
Common interface every baseline (ARIMAX, RF, LSTM, TCN, TFT, ...) implements,
so experiments/run_experiment.py can loop over baselines generically and so
each baseline lives in an isolated module/file (errors in one can't corrupt
another -- each is fit/tuned/evaluated independently and wrapped in
try/except at the orchestration layer).

Contract:
  - fit(X_train, y_train, **params) -> self
  - predict_position(X) -> pd.Series of target positions in [-1, 1], aligned
    to X's index. This is what backtest/engine.py consumes directly.
  - default_param_space(trial: optuna.Trial) -> dict, used by tuning/optuna_utils.py

All baselines predict a continuous next-bar return signal internally and
convert to a position via a tanh-squash of a z-scored prediction, so the
same "prediction -> position" translation logic isn't duplicated five times
with subtly different bugs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def signal_to_position(pred: pd.Series, lookback: int = 60, scale: float = 2.5) -> pd.Series:
    """Turn a raw predicted-return signal into a position in [-1, 1].

    Uses a ROLLING (causal, expanding-then-rolling) z-score so the
    normalization at time t never uses information beyond t -- avoids the
    subtle look-ahead bug of z-scoring with the full-sample mean/std.
    """
    roll_mean = pred.rolling(lookback, min_periods=10).mean()
    roll_std = pred.rolling(lookback, min_periods=10).std().replace(0, np.nan)
    z = ((pred - roll_mean) / roll_std).fillna(0.0)
    pos = np.tanh(z / scale)
    return pos.clip(-1, 1)


class BaseBaseline(ABC):
    name: str = "base"

    def __init__(self, **params):
        self.params = params
        self.model = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseBaseline":
        ...

    @abstractmethod
    def predict_returns(self, X: pd.DataFrame) -> pd.Series:
        """Predicted next-bar return signal (not yet a position)."""
        ...

    def predict_position(self, X: pd.DataFrame) -> pd.Series:
        pred = self.predict_returns(X)
        return signal_to_position(pred)

    @staticmethod
    def default_param_space(trial) -> dict:
        return {}
