"""
Temporal Fusion Transformer -- lightweight custom implementation ("TFT-lite").

We implement TFT's core architectural ideas directly in PyTorch rather than
depending on a heavy external library (pytorch-forecasting + lightning),
which is both harder to control for a nested-CV/Optuna loop and overkill
for a single-series regression head:
  - Variable Selection Network (VSN): a small gating network that learns a
    softmax weighting over the input features at each timestep, so the
    model can down-weight uninformative indicators instead of forcing the
    encoder to learn that itself.
  - Gated Residual Network (GRN): the standard TFT building block
    (Linear -> ELU -> Linear -> GLU-gate -> residual + LayerNorm), used
    both inside the VSN and after attention.
  - LSTM encoder over the (variable-selected) sequence, standard for TFT's
    "locality enhancement" stage.
  - Interpretable multi-head self-attention over the encoded sequence,
    TFT's mechanism for long-range dependency weighting.
  - A GRN + linear head on the final attended representation produces the
    next-bar return point forecast (we use a point forecast rather than
    TFT's quantile outputs to keep the prediction -> position pipeline
    identical across all baselines).
"""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import config as C
from .base import BaseBaseline
from .sequence_utils import train_torch_regressor, predict_torch_regressor


class GRN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_out, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.ln = nn.LayerNorm(d_out)

    def forward(self, x):
        h = self.fc2(self.elu(self.fc1(x)))
        h = self.dropout(h)
        a, b = self.gate(h).chunk(2, dim=-1)
        gated = a * torch.sigmoid(b)  # GLU
        return self.ln(gated + self.skip(x))


class VariableSelection(nn.Module):
    """Per-timestep softmax gating over input features, GRN-processed then combined."""
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.feature_grns = nn.ModuleList([GRN(1, d_model, d_model, dropout) for _ in range(n_features)])
        self.weight_grn = GRN(n_features, d_model, n_features, dropout)

    def forward(self, x):
        # x: (batch, seq, n_features)
        b, s, f = x.shape
        flat = x.reshape(b * s, f)
        weights = torch.softmax(self.weight_grn(flat), dim=-1)  # (b*s, f)
        processed = torch.stack(
            [self.feature_grns[i](flat[:, i:i + 1]) for i in range(f)], dim=1
        )  # (b*s, f, d_model)
        combined = (processed * weights.unsqueeze(-1)).sum(dim=1)  # (b*s, d_model)
        return combined.reshape(b, s, -1)


class TFTLiteNet(nn.Module):
    def __init__(self, n_features: int, d_model: int = 24, n_heads: int = 4,
                 lstm_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.vsn = VariableSelection(n_features, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=lstm_layers, batch_first=True)
        self.post_lstm_grn = GRN(d_model, d_model, d_model, dropout)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_attn_grn = GRN(d_model, d_model, d_model, dropout)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        v = self.vsn(x) # (b, s, d)
        enc, _ = self.lstm(v)
        enc = self.post_lstm_grn(enc)
        # causal mask: position i can only attend to positions <= i
        s = enc.size(1)
        mask = torch.triu(torch.ones(s, s, device=enc.device), diagonal=1).bool()
        attn_out, _ = self.attn(enc, enc, enc, attn_mask=mask)
        out = self.post_attn_grn(attn_out)
        last = out[:, -1, :]
        return self.head(last)


class TFTBaseline(BaseBaseline):
    name = "TFT"

    def __init__(self, lookback: int = 20, d_model: int = 24, n_heads: int = 4,
                 lstm_layers: int = 1, dropout: float = 0.1, lr: float = 1e-3, epochs: int = 25, device: str = None, **kwargs):
        super().__init__(lookback=lookback, d_model=d_model, n_heads=n_heads, lstm_layers=lstm_layers,
                          dropout=dropout, lr=lr, epochs=epochs, **kwargs)
        self.lookback = lookback
        self.d_model, self.n_heads, self.lstm_layers, self.dropout = d_model, n_heads, lstm_layers, dropout
        self.lr, self.epochs = lr, epochs
        self.device = device or C.DEVICE
        self.model = None
        self.history = None
        self.cols = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TFTBaseline":
        torch.manual_seed(C.RANDOM_SEED) 
        self.cols = list(X_train.columns)
        self.model = TFTLiteNet(len(self.cols), self.d_model, self.n_heads,
                                 self.lstm_layers, self.dropout)
        self.model = train_torch_regressor(self.model, X_train[self.cols], y_train,
                                            self.lookback, epochs=self.epochs, lr=self.lr, batch_size=32, device=self.device)
        self.history = X_train[self.cols].tail(self.lookback - 1)
        return self

    def predict_returns(self, X: pd.DataFrame) -> pd.Series:
        pred = predict_torch_regressor(self.model, self.history, X[self.cols], self.lookback, self.device)
        return pred.fillna(0.0)

    @staticmethod
    def default_param_space(trial) -> dict:
        d_model = trial.suggest_categorical("d_model", [16, 24, 32])
        n_heads_choices = [h for h in (2, 4) if d_model % h == 0]
        return {
            "lookback": trial.suggest_categorical("lookback", [20]),
            "d_model": d_model,
            "n_heads": trial.suggest_categorical("n_heads", n_heads_choices),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "epochs": trial.suggest_int("epochs", 10, 25),
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
    m = TFTBaseline(epochs=6)
    m.fit(X.iloc[:cut], y.iloc[:cut])
    pos = m.predict_position(X.iloc[cut:])
    print(pos.describe())
    print("OK TFT-lite fit + predict")