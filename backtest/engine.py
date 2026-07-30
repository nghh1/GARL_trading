"""
Backtest engine.

Look-ahead protection (decision side): callers pass a `target_position`
series where target_position[t] is whatever the model/agent decided using
information available AT bar t (features, or an RL action taken after
observing bar t's close). This function shifts it by one bar internally,
so the position actually *held* during bar t's return is
target_position[t-1] -- i.e. you decide at t, you're exposed starting t+1.
This is the single choke point enforcing "trade at t+1 on a decision made
at t" for every baseline in the codebase; individual baselines never need
to shift things themselves.

Costs: proportional to turnover (change in position), charged in both
transaction cost and slippage bps, applied against the notional traded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions_held: pd.Series   # already shifted (what was actually held during the bar)
    costs: pd.Series
    summary: dict
    gross_equity: pd.Series = None  
    gross_returns: pd.Series = None


def single_asset_backtest(
    close: pd.Series,
    target_position: pd.Series,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    close = close.astype(float)
    target_position = target_position.reindex(close.index).fillna(0.0).clip(-1.0, 1.0)

    held = target_position.shift(1).fillna(0.0)  # <-- the one-line look-ahead guard
    asset_ret = close.pct_change().fillna(0.0)

    gross_ret = held * asset_ret
    turnover = held.diff().abs().fillna(held.abs().iloc[0] if len(held) else 0.0)
    cost_rate = (cost_bps + slippage_bps) / 1e4
    cost = turnover * cost_rate

    net_ret = gross_ret - cost
    equity = initial_capital * (1 + net_ret).cumprod()
    gross_equity = initial_capital * (1 + gross_ret).cumprod() 

    summary = M.summarize(equity, net_ret, held)
    gross_summary = M.summarize(gross_equity, gross_ret, held)    
    for k, v in gross_summary.items():                            
        if k != "Turnover":                                       
            summary[f"Gross_{k}"] = v

    return BacktestResult(equity=equity, returns=net_ret, positions_held=held,
                           costs=cost, summary=summary, gross_equity=gross_equity, gross_returns=gross_ret)

def portfolio_backtest(
    close_by_ticker: dict,
    target_position_by_ticker: dict,
    capital_per_ticker: float,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    """Aggregate independent single-asset backtests (one per agent/ticker)
    into one portfolio equity curve -- this is how we score the
    multi-agent / GARL group (N agents, N private environments) as a whole.
    """
    results = {}
    common_index = None
    for t, close in close_by_ticker.items():
        r = single_asset_backtest(
            close, target_position_by_ticker[t],
            initial_capital=capital_per_ticker,
            cost_bps=cost_bps, slippage_bps=slippage_bps,
        )
        results[t] = r
        common_index = close.index if common_index is None else common_index.intersection(close.index)

    equity_sum = None
    gross_equity_sum = None
    for t, r in results.items():
        eq = r.equity.reindex(common_index).ffill()
        equity_sum = eq if equity_sum is None else equity_sum + eq
        geq = r.gross_equity.reindex(common_index).ffill()
        gross_equity_sum = geq if gross_equity_sum is None else gross_equity_sum + geq

    ret = equity_sum.pct_change().fillna(0.0)
    gross_ret = gross_equity_sum.pct_change().fillna(0.0)
    total_capital = capital_per_ticker * len(close_by_ticker)
    avg_position = pd.concat(
        [results[t].positions_held.reindex(common_index) for t in results], axis=1
    ).abs().mean(axis=1)

    summary = M.summarize(equity_sum, ret, avg_position)
    gross_summary = M.summarize(gross_equity_sum, gross_ret, avg_position)    
    for k, v in gross_summary.items():                                       
        if k != "Turnover":                                                  
            summary[f"Gross_{k}"] = v
    return BacktestResult(equity=equity_sum, returns=ret, positions_held=avg_position,
                           costs=pd.Series(0, index=common_index), summary=summary,
                           gross_equity=gross_equity_sum, gross_returns=gross_ret), results


if __name__ == "__main__":
    idx = pd.bdate_range("2023-01-01", periods=250)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx)
    signal = pd.Series(np.sign(rng.normal(0, 1, len(idx))), index=idx)  # random +-1 signal
    res = single_asset_backtest(close, signal)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.summary.items()})

    # Look-ahead guard check.
    # A buggy "cheat" signal that already knows bar t's own return before
    # applying it (target_position[t] = sign(pct[t])) is the classic
    # look-ahead bug and, if applied WITHOUT delay, produces suspiciously
    # perfect Sharpe (held*ret = |ret| >= 0 always). Because
    # single_asset_backtest ALWAYS shifts target_position by one bar before
    # applying it (held[t] = target_position[t-1]), this cheat signal is
    # forced to trade one bar later than the information it (mis)used,
    # which destroys the "perfection" and returns an unremarkable Sharpe.
    cheat_signal = np.sign(close.pct_change()).fillna(0)
    res_cheat = single_asset_backtest(close, cheat_signal)
    print("Cheat signal Sharpe (engine's internal shift neutralizes it):",
          round(res_cheat.summary["Sharpe"], 3),
          "<- unremarkable, not the near-perfect Sharpe you'd get if positions were applied with no delay")
