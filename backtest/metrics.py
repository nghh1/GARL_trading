"""Standard performance metrics computed from a daily equity curve / return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return np.nan
    n_years = len(equity) / TRADING_DAYS
    if n_years <= 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def ann_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return (excess.mean() / sd) * np.sqrt(TRADING_DAYS)


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std()
    if dd == 0 or np.isnan(dd):
        return 0.0
    return (excess.mean() / dd) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return dd.min()


def calmar(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return cagr(equity) / abs(mdd)


def win_rate(returns: pd.Series) -> float:
    nz = returns[returns != 0]
    if len(nz) == 0:
        return 0.0
    return (nz > 0).mean()


def turnover(positions: pd.Series) -> float:
    return positions.diff().abs().fillna(0).mean()


def summarize(equity: pd.Series, returns: pd.Series, positions: pd.Series = None) -> dict:
    out = {
        "CAGR": cagr(equity),
        "AnnVol": ann_vol(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "MaxDrawdown": max_drawdown(equity),
        "Calmar": calmar(equity),
        "WinRate": win_rate(returns),
        "FinalEquity": equity.iloc[-1],
        "TotalReturn": equity.iloc[-1] / equity.iloc[0] - 1,
    }
    if positions is not None:
        out["Turnover"] = turnover(positions)
    return out
