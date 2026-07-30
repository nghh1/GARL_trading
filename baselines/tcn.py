"""Temporal Convolutional Network (TCN) baseline: dilated causal 1D
convolutions over the feature window -> next-bar return prediction.

Causality: every conv uses `padding` on the LEFT only (implemented via
symmetric padding + right-side trim), so the receptive field of the
prediction at the last timestep never includes future timesteps within the
window (and the window itself is already causal per sequence_utils).
"""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import config as C
from .base import BaseBaseline
from .sequence_utils import train_torch_regressor, predict_torch_regressor

class Chomp1d(nn.Module):
    """Removes the extra right-side padding introduced by causal conv."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNNet(nn.Module):
    def __init__(self, n_features: int, channels=(32, 32), kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        n_in = n_features
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(n_in, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            n_in = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        # x: (batch, seq, features) -> conv1d wants (batch, features, seq)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last = out[:, :, -1]
        return self.head(last)

class TCNBaseline(BaseBaseline):
    name = "TCN"

    def __init__(self, lookback: int = 20, n_channels: int = 32, n_layers: int = 2,
                 kernel_size: int = 3, dropout: float = 0.1, lr: float = 1e-3, epochs: int = 25, device: str = None, **kwargs):
        super().__init__(lookback=lookback, n_channels=n_channels, n_layers=n_layers,
                          kernel_size=kernel_size, dropout=dropout, lr=lr, epochs=epochs, **kwargs)
        self.lookback = lookback
        self.channels = tuple([n_channels] * n_layers)
        self.kernel_size, self.dropout = kernel_size, dropout
        self.lr, self.epochs = lr, epochs
        self.device = device or C.DEVICE
        self.model = None
        self.history = None
        self.cols = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TCNBaseline":
        torch.manual_seed(C.RANDOM_SEED) 
        self.cols = list(X_train.columns)
        self.model = TCNNet(len(self.cols), self.channels, self.kernel_size, self.dropout)
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
            "n_channels": trial.suggest_categorical("n_channels", [16, 32, 64]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
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
    m = TCNBaseline(epochs=8)
    m.fit(X.iloc[:cut], y.iloc[:cut])
    pos = m.predict_position(X.iloc[cut:])
    print(pos.describe())
    print("OK TCN fit + predict")