"""
Benchmark comparison tables, matching the style of Johnson (2024)'s
Table 2 (single-agent RL vs Buy & Hold) and Table 3 (GARL vs single-agent
vs ARIMA vs Buy & Hold) -- grouped tables of Metric x {baselines}, rather
than one unwieldy 13-column table.

Buy & Hold is computed here for the first time in this project -- it did
not exist anywhere in the pipeline before. Computed via the SAME
backtest.engine.single_asset_backtest() every other baseline uses (a
constant, always-fully-long position series), not a separate ad hoc
calculation -- this guarantees identical cost/look-ahead methodology to
every other row in these tables, which matters for the comparison to be
fair. A constant position means turnover is ~0 after the first bar (one
entry cost, then hold), exactly matching what "buy and hold" means.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C


def compute_buy_hold(close_by_ticker: Dict[str, pd.Series], test_start: str, test_end: str,
                      cost_bps: float = C.TRANSACTION_COST_BPS,
                      slippage_bps: float = C.SLIPPAGE_BPS) -> pd.DataFrame:
    """Buy & Hold metrics per ticker over [test_start, test_end], via the
    project's own backtest engine (constant full-long position). Returns
    a DataFrame indexed by ticker with the same metric columns as
    results_raw.csv/results_5fold.csv, so it can be concatenated directly
    with baseline rows for a table.
    """
    from backtest import engine as ENGINE

    rows = []
    for ticker, close in close_by_ticker.items():
        window = close.loc[test_start:test_end]
        if len(window) < 2:
            continue
        position = pd.Series(1.0, index=window.index)  # always fully long
        result = ENGINE.single_asset_backtest(window, position, cost_bps=cost_bps,
                                               slippage_bps=slippage_bps)
        rows.append({"ticker": ticker, **result.summary})
    return pd.DataFrame(rows).set_index("ticker")


def buy_hold_from_universe(start: str, end: str, test_start: str, test_end: str,
                            use_synthetic: bool = False) -> pd.DataFrame:
    """Convenience wrapper: downloads config.TICKERS and computes Buy &
    Hold over the given test window in one call.

    NOTE: this is a SINGLE fixed window. For comparison against 5-fold-mean
    baseline columns (build_comparison_table's default), use
    buy_hold_5fold() instead -- see its docstring for why a single window
    here is a real methodological mismatch, not just a simplification.
    """
    if use_synthetic:
        from data import synthetic as loader
        raw = loader.download_universe(C.TICKERS, start, end, seed=C.RANDOM_SEED)
    else:
        from data import loader
        raw = loader.download_universe(C.TICKERS, start, end)
    close_by_ticker = {t: df["close"] for t, df in raw.items()}
    return compute_buy_hold(close_by_ticker, test_start, test_end)


def buy_hold_5fold(use_synthetic: bool = False) -> pd.DataFrame:
    """Buy & Hold computed on EACH of the same 5 walk-forward folds
    results_5fold.csv was built from, then averaged the same way
    build_comparison_table() averages baseline metrics -- mean across
    folds, mean across tickers.

    This is the metric that should actually go in Tables A/B/C, not
    buy_hold_from_universe()'s single fixed window: the baseline columns
    next to it are 5-fold means spanning five genuinely different market
    regimes (including the 2008 GFC and COVID crash as separate test
    folds), so Buy & Hold needs to be averaged across those same five
    regimes to be a fair comparison -- not just the one (2021-2025, an
    unusually strong tech-led bull run) that happens to be the single-fold
    test window. Comparing 5-fold-mean strategies against single-window
    Buy & Hold silently stacks the deck in whichever direction that one
    window happens to favour.

    IMPORTANT: fold boundaries are computed on the TRUE common_index --
    the intersection of all 9 tickers' FEATURE indices (after
    data/features.py's ~200-bar sma_ratio_200 warmup drop), exactly
    matching experiments/run_experiment.py's own construction -- not a
    raw pd.bdate_range(). Using the raw date range here was tried first
    and produced fold-0 test dates off by ~5-6 weeks from
    results_5fold.csv's actual recorded test_start/test_end, for exactly
    the reason documented in load_reported_result()'s docstring: the
    feature-warmup-adjusted index and the raw calendar range are not the
    same thing, and outer_splits() is sensitive to which one it's given.

    Returns a DataFrame indexed by ticker (mean across the 5 folds each).
    """
    from cv import walk_forward as WF
    from data import features as F

    if use_synthetic:
        from data import synthetic as loader
        raw = loader.download_universe(C.TICKERS, C.START_DATE, C.END_DATE, seed=C.RANDOM_SEED)
    else:
        from data import loader
        raw = loader.download_universe(C.TICKERS, C.START_DATE, C.END_DATE)

    features_by_ticker = {t: F.build_features(df, label_horizon=C.LABEL_HORIZON) for t, df in raw.items()}
    common_index = None
    for feat in features_by_ticker.values():
        common_index = feat.index if common_index is None else common_index.intersection(feat.index)
    close_by_ticker = {t: df["close"].loc[common_index] for t, df in raw.items()}

    folds = WF.outer_splits(common_index, n_folds=C.N_OUTER_FOLDS, min_train_bars=C.MIN_TRAIN_BARS,
                             embargo=C.EMBARGO_BARS)

    per_fold = []
    for fold in folds:
        test_start = str(common_index[fold.test_idx[0]].date())
        test_end = str(common_index[fold.test_idx[-1]].date())
        fold_result = compute_buy_hold(close_by_ticker, test_start, test_end)
        per_fold.append(fold_result[["TotalReturn", "CAGR", "Sharpe", "MaxDrawdown"]])

    combined = pd.concat(per_fold, keys=range(len(per_fold)), names=["fold", "ticker"])
    return combined.groupby("ticker").mean()


def buy_hold_portfolio_5fold(use_synthetic: bool = False) -> dict:
    """Portfolio-LEVEL Buy & Hold, matching backtest.engine.portfolio_backtest()'s
    exact methodology (sum all 9 tickers' equity curves into ONE combined
    curve, then compute Sharpe/CAGR/MaxDrawdown from that), not a naive
    average of 9 independent per-ticker Sharpe ratios.

    This distinction matters and was previously missing: every RL/GARL
    "PORTFOLIO" row (GARL_DDAL, MultiAgentA2C, GARL_DDAL_SECTOR, ...) is
    computed via portfolio_backtest(), which captures diversification --
    the combined portfolio's volatility is generally lower than the simple
    average of its 9 components' individual volatilities, since the
    tickers aren't perfectly correlated. buy_hold_5fold()'s per-ticker-mean
    Sharpe does NOT capture this and is the wrong number to put next to
    GARL_DDAL/MultiAgentA2C/GARL_DDAL_SECTOR's PORTFOLIO Sharpe in Table B
    specifically (Tables A/C are fine -- supervised baselines are never
    portfolio-aggregated either, so the per-ticker-mean comparison there
    is apples-to-apples). Use THIS function for Table B's Buy & Hold row,
    buy_hold_5fold() for Tables A/C.

    Returns a single dict of {TotalReturn, CAGR, Sharpe, MaxDrawdown},
    averaged across the same 5 folds (each fold's OWN portfolio-level
    Buy & Hold computed via portfolio_backtest(), then averaged the same
    way every other PORTFOLIO row in results_5fold.csv is averaged across
    folds) -- not a single fixed window, for the same reason buy_hold_5fold()
    isn't.
    """
    from cv import walk_forward as WF
    from data import features as F
    from backtest import engine as ENGINE

    if use_synthetic:
        from data import synthetic as loader
        raw = loader.download_universe(C.TICKERS, C.START_DATE, C.END_DATE, seed=C.RANDOM_SEED)
    else:
        from data import loader
        raw = loader.download_universe(C.TICKERS, C.START_DATE, C.END_DATE)

    features_by_ticker = {t: F.build_features(df, label_horizon=C.LABEL_HORIZON) for t, df in raw.items()}
    common_index = None
    for feat in features_by_ticker.values():
        common_index = feat.index if common_index is None else common_index.intersection(feat.index)
    close_by_ticker = {t: df["close"].loc[common_index] for t, df in raw.items()}

    folds = WF.outer_splits(common_index, n_folds=C.N_OUTER_FOLDS, min_train_bars=C.MIN_TRAIN_BARS,
                             embargo=C.EMBARGO_BARS)

    per_fold_metrics = []
    for fold in folds:
        test_idx = common_index[fold.test_idx]
        close_test = {t: s.loc[test_idx] for t, s in close_by_ticker.items()}
        pos_test = {t: pd.Series(1.0, index=test_idx) for t in close_by_ticker}
        result, _ = ENGINE.portfolio_backtest(close_test, pos_test, capital_per_ticker=C.CAPITAL_PER_AGENT)
        per_fold_metrics.append(result.summary)

    df = pd.DataFrame(per_fold_metrics)
    return df[["TotalReturn", "CAGR", "Sharpe", "MaxDrawdown"]].mean().to_dict()


def _fmt_row(metrics: dict) -> dict:
    return {
        "Cumulative Return": f"{metrics['TotalReturn']*100:.2f}%",
        "CAGR%": f"{metrics['CAGR']*100:.2f}%",
        "Sharpe Ratio": f"{metrics['Sharpe']:.2f}",
        "Max Drawdown": f"{metrics['MaxDrawdown']*100:.2f}%",
    }


def build_comparison_table(results_5fold_path: str, baselines: List[str],
                            buy_hold_metrics: Optional[dict] = None,
                            fold: Optional[int] = None) -> pd.DataFrame:
    """Metric x {baselines [+ Buy & Hold]} table, Johnson-Table-2/3 style.
    Uses 5-fold MEAN by default (the project's primary, more reliable
    result -- see README's "verified findings"), or one specific fold if
    given. PORTFOLIO row used for RL baselines, per-ticker mean otherwise.

    buy_hold_metrics: a dict of {'TotalReturn':..., 'CAGR':..., 'Sharpe':...,
    'MaxDrawdown':...} -- e.g. compute_buy_hold(...)['some_ticker'] for a
    single-ticker table, or an already-averaged dict across tickers for a
    portfolio-level Buy & Hold row. Pass None to omit the row.
    """
    df = pd.read_csv(results_5fold_path)
    if fold is not None:
        df = df[df["fold"] == fold]

    columns = {}
    for b in baselines:
        g = df[df["baseline"] == b]
        if (g["ticker"] == "PORTFOLIO").any():
            row = g[g["ticker"] == "PORTFOLIO"][["TotalReturn", "CAGR", "Sharpe", "MaxDrawdown"]].mean()
        else:
            row = g[g["ticker"] != "PORTFOLIO"][["TotalReturn", "CAGR", "Sharpe", "MaxDrawdown"]].mean()
        columns[b] = _fmt_row(row)

    if buy_hold_metrics is not None:
        columns["Buy and Hold"] = _fmt_row(buy_hold_metrics)

    table = pd.DataFrame(columns)
    table.index.name = "Metric"
    return table


if __name__ == "__main__":
    # Self-test on synthetic data -- confirms the MECHANISM works
    # end-to-end (real numbers will differ once run on real data).
    from data import synthetic
    from backtest import engine as ENGINE

    raw = synthetic.download_universe(["AAPL", "JPM"], "2018-01-01", "2021-01-01", seed=42)
    close_by_ticker = {t: df["close"] for t, df in raw.items()}
    bh = compute_buy_hold(close_by_ticker, "2019-06-01", "2020-12-31")
    print("Buy & Hold self-test:")
    print(bh[["TotalReturn", "CAGR", "Sharpe", "MaxDrawdown"]])
    assert not bh.empty and bh["Sharpe"].notna().all()
    print("OK benchmark_table.compute_buy_hold")