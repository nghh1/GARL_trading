"""LSTM sequence baseline: rolling window of engineered features -> next-bar
return prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config as C
from .base import BaseBaseline
from .sequence_utils import train_torch_regressor, predict_torch_regressor

class LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.head(h[-1])

class LSTMBaseline(BaseBaseline):
    name = "LSTM"

    def __init__(self, lookback: int = 20, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.1, lr: float = 1e-3, epochs: int = 25, device: str = None, **kwargs):
        super().__init__(lookback=lookback, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, lr=lr, epochs=epochs, **kwargs)
        self.lookback = lookback
        self.hidden_size, self.num_layers, self.dropout = hidden_size, num_layers, dropout
        self.lr, self.epochs = lr, epochs
        self.device = device or C.DEVICE
        self.model = None
        self.history = None
        self.cols = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LSTMBaseline":
        torch.manual_seed(C.RANDOM_SEED) 
        self.cols = list(X_train.columns)
        self.model = LSTMNet(len(self.cols), self.hidden_size, self.num_layers, self.dropout)
        self.model = train_torch_regressor(self.model, X_train[self.cols], y_train,
                                            self.lookback, epochs=self.epochs, lr=self.lr, device=self.device)
        self.history = X_train[self.cols].tail(self.lookback - 1)
        return self

    def predict_returns(self, X: pd.DataFrame) -> pd.Series:
        pred = predict_torch_regressor(self.model, self.history, X[self.cols], self.lookback, self.device)
        return pred.fillna(0.0)

    @staticmethod
    def default_param_space(trial) -> dict:
        return {
            "lookback": trial.suggest_categorical("lookback", [20]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "epochs": trial.suggest_int("epochs", 10, 30),
        }

if __name__ == "__main__":
    from data import synthetic
    from data import features as F
    import config as C

    torch.manual_seed(0)
    raw = synthetic.download_universe(["AAPL"], "2015-01-01", "2021-01-01")["AAPL"]
    feat = F.build_features(raw).dropna()
    X, y = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"]
    cut = int(len(X) * 0.8)
    m = LSTMBaseline(epochs=8)
    m.fit(X.iloc[:cut], y.iloc[:cut])
    pos = m.predict_position(X.iloc[cut:])
    print(pos.describe())
    print("OK LSTM fit + predict")