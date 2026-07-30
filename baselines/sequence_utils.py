"""
Shared utilities for the PyTorch sequence baselines (LSTM, TCN, TFT-lite).

Windowing look-ahead protection: the sequence ending at index t (i.e.
covering rows [t-lookback+1, ..., t]) is used to predict y[t] = fwd_ret_h
at row t (return from close_t to close_{t+1}). No row after t is ever
included in a window that predicts y[t].
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X, self.y, self.lookback = X, y, lookback
        self.valid_starts = np.arange(lookback - 1, len(X))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, i):
        t = self.valid_starts[i]
        window = self.X[t - self.lookback + 1: t + 1]
        return torch.from_numpy(window).float(), torch.tensor(self.y[t], dtype=torch.float32)


def make_windows(X: pd.DataFrame, y: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    Xv, yv = X.values.astype(np.float32), y.values.astype(np.float32)
    n = len(Xv)
    if n < lookback:
        return np.empty((0, lookback, Xv.shape[1])), np.empty((0,)), X.index[:0]
    idxs = np.arange(lookback - 1, n)
    Xw = np.stack([Xv[t - lookback + 1: t + 1] for t in idxs])
    yw = yv[idxs]
    return Xw, yw, X.index[idxs]


def train_torch_regressor(model: torch.nn.Module, X: pd.DataFrame, y: pd.Series,
                           lookback: int, epochs: int = 25, lr: float = 1e-3,
                           batch_size: int = 64, weight_decay: float = 1e-5,
                           device: str = "cpu", verbose: bool = False) -> torch.nn.Module:
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]
    Xw, yw, _ = make_windows(X, y, lookback)
    if len(Xw) < 20:
        raise ValueError("Not enough windows to train sequence model.")

    Xt = torch.from_numpy(Xw).float().to(device)
    yt = torch.from_numpy(yw).float().to(device)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(idx)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep}: mse={total_loss / n:.6f}")
    model.eval()
    return model


@torch.no_grad()
def predict_torch_regressor(model: torch.nn.Module, X_full_history: pd.DataFrame,
                             X_target: pd.DataFrame, lookback: int, device: str = "cpu") -> pd.Series:
    """Predict for every timestamp in X_target, using X_full_history (which
    must contain X_target plus at least `lookback-1` prior rows) to build
    each window. Rows in X_target without enough history get NaN (later
    treated as flat/no position, never silently dropped-and-forward-filled
    in a way that could leak).
    """
    model.eval()
    combined = X_full_history.combine_first(X_target).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    preds = {}
    idx_list = list(combined.index)
    pos_of = {d: i for i, d in enumerate(idx_list)}
    Xv = combined.values.astype(np.float32)

    for ts in X_target.index:
        i = pos_of[ts]
        if i - lookback + 1 < 0:
            preds[ts] = np.nan
            continue
        window = Xv[i - lookback + 1: i + 1]
        if np.isnan(window).any():
            preds[ts] = np.nan
            continue
        xb = torch.from_numpy(window).float().unsqueeze(0).to(device)
        preds[ts] = model(xb).squeeze().item()
    return pd.Series(preds).reindex(X_target.index)
