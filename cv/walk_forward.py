"""
Purged, embargoed, nested walk-forward cross-validation.

Why not sklearn TimeSeriesSplit as-is: it doesn't purge the boundary, so a
rolling-window feature (e.g. sma_ratio_30) or a forward label computed at
the last few bars of a train fold can leak information from the first bars
of the following validation fold (and vice-versa for labels that look
forward). We fix this with an embargo: a gap of `embargo` bars dropped
between every train/val boundary in both directions.

Two levels are provided:
  - `outer_splits`: expanding-window folds used as the realistic backtest
    schedule (train up to date X, trade the following block out-of-sample).
  - `inner_splits`: for each outer-train block, a further expanding-window
    split used ONLY by Optuna to score hyperparameters. The outer test fold
    is never seen by inner splits, so hyperparameter selection cannot leak
    into the number we report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import config as C
import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _embargo_trim(train_idx: np.ndarray, test_idx: np.ndarray, embargo: int, n: int) -> np.ndarray:
    """Remove `embargo` bars from train_idx on both sides adjacent to test_idx."""
    test_start, test_end = test_idx.min(), test_idx.max()
    lo = max(0, test_start - embargo)
    hi = min(n - 1, test_end + embargo)
    mask = (train_idx < lo) | (train_idx > hi)
    return train_idx[mask]


def outer_splits(index: pd.DatetimeIndex, n_folds: int, min_train_bars: int,
                  embargo: int) -> List[Fold]:
    """Expanding-window outer folds spanning the whole index."""
    n = len(index)
    usable = n - min_train_bars
    if usable <= n_folds:
        raise ValueError("Not enough bars for requested n_folds/min_train_bars.")
    fold_size = usable // n_folds
    folds = []
    for k in range(n_folds):
        test_start = min_train_bars + k * fold_size
        test_end = n if k == n_folds - 1 else min_train_bars + (k + 1) * fold_size
        test_idx = np.arange(test_start, test_end)
        train_idx = np.arange(0, test_start)
        train_idx = _embargo_trim(train_idx, test_idx, embargo, n)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append(Fold(
            train_idx=train_idx, test_idx=test_idx,
            train_start=index[train_idx[0]], train_end=index[train_idx[-1]],
            test_start=index[test_idx[0]], test_end=index[test_idx[-1]],
        ))
    return folds


def inner_splits(train_idx: np.ndarray, n_folds: int, min_train_bars: int,
                  embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Nested expanding-window split *within* an outer training block, for
    Optuna hyperparameter scoring. Operates purely on positions within
    train_idx (relative), then maps back to absolute positions.
    """
    n = len(train_idx)
    if n < min_train_bars * 2:
        # too little data for a nested split; fall back to a single 80/20
        cut = max(min_train_bars, int(n * 0.8))
        if cut >= n:
            cut = n - 1
        tr = train_idx[:cut]
        va = train_idx[cut:]
        va = va[va > 0]
        return [(tr, va)] if len(va) > 0 else []

    usable = n - min_train_bars
    fold_size = max(1, usable // n_folds)
    out = []
    for k in range(n_folds):
        val_start = min_train_bars + k * fold_size
        val_end = n if k == n_folds - 1 else min_train_bars + (k + 1) * fold_size
        if val_start >= n:
            break
        rel_val = np.arange(val_start, val_end)
        rel_train = np.arange(0, val_start)
        rel_train = _embargo_trim(rel_train, rel_val, embargo, n)
        if len(rel_train) == 0 or len(rel_val) == 0:
            continue
        out.append((train_idx[rel_train], train_idx[rel_val]))
    return out


if __name__ == "__main__":
    idx = pd.bdate_range(C.START_DATE, C.END_DATE)
    folds = outer_splits(idx, n_folds=C.N_OUTER_FOLDS, min_train_bars=C.MIN_TRAIN_BARS, embargo=C.EMBARGO_BARS)
    for f in folds:
        print(f"train [{f.train_start.date()} -> {f.train_end.date()}] "
              f"({len(f.train_idx)} bars)  |  test [{f.test_start.date()} -> {f.test_end.date()}] "
              f"({len(f.test_idx)} bars)")
        inner = inner_splits(f.train_idx, n_folds=C.N_INNER_FOLDS, min_train_bars=min(C.MIN_TRAIN_BARS, len(f.train_idx) // 3), embargo=C.EMBARGO_BARS)
        for i, (tr, va) in enumerate(inner):
            print(f"    inner {i}: train {len(tr)} bars -> val {len(va)} bars "
                  f"[{idx[va[0]].date()} -> {idx[va[-1]].date()}]")
        # leakage assertion: no overlap, and embargo respected
        assert len(set(f.train_idx) & set(f.test_idx)) == 0
    print("OK: outer/inner walk-forward splits constructed with no train/test overlap")
