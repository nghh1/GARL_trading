"""
Synthetic OHLCV generator, used ONLY because this sandbox cannot reach
Yahoo Finance (see data/loader.py docstring). It produces data with the
*exact same schema* as data/loader.download_universe(), so every baseline,
CV splitter, feature pipeline and backtester downstream is agnostic to
which loader produced the data.

Design goals (so bugs found here generalize to real data):
  - regime-switching drift/vol per ticker (bull / bear / choppy), via a
    Markov chain, so models actually have (imperfect, noisy) signal to find
    and CV can show over/under-fitting
  - a shared market factor + sector factors, so cross-sectional structure
    exists (useful for the multi-agent / GARL story: correlated but
    non-identical environments)
  - realistic OHLC construction (open gaps, intrabar high/low envelope)
    and volume that reacts to volatility
  - reproducible via config.RANDOM_SEED
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

SECTOR_MAP = {
    "AAPL": "tech", "MSFT": "tech", "JPM": "financials", "XOM": "energy",
    "JNJ": "healthcare", "PG": "staples", "CAT": "industrials",
    "NEE": "utilities", "DIS": "comm",
}

# regime transition matrix: bull, bear, choppy
REGIME_NAMES = ["bull", "bear", "choppy"]
TRANS = np.array([
    [0.985, 0.005, 0.010],
    [0.010, 0.960, 0.030],
    [0.020, 0.020, 0.960],
])
REGIME_DRIFT = {"bull": 0.00045, "bear": -0.00060, "choppy": 0.00005}
REGIME_VOL = {"bull": 0.010, "bear": 0.022, "choppy": 0.015}


def _simulate_regimes(n: int, rng: np.random.Generator) -> List[str]:
    state = 0  # start bull
    seq = []
    for _ in range(n):
        seq.append(REGIME_NAMES[state])
        state = rng.choice(3, p=TRANS[state])
    return seq


def _gen_ticker_prices(ticker: str, dates: pd.DatetimeIndex, market_factor: np.ndarray,
                        sector_factor: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    regimes = _simulate_regimes(n, rng)
    drift = np.array([REGIME_DRIFT[r] for r in regimes])
    vol = np.array([REGIME_VOL[r] for r in regimes])

    idio = rng.normal(0, 1, n)
    # beta exposure to market/sector factors + idiosyncratic noise
    beta_mkt, beta_sec = rng.uniform(0.6, 1.3), rng.uniform(0.3, 0.9)
    daily_ret = drift + vol * (beta_mkt * market_factor + beta_sec * sector_factor + idio) / np.sqrt(
        beta_mkt ** 2 + beta_sec ** 2 + 1
    )
    # mild mean-reversion / momentum microstructure noise
    daily_ret += 0.03 * pd.Series(daily_ret).shift(1).fillna(0).values

    close = 50 * np.exp(np.cumsum(daily_ret))
    open_gap = rng.normal(0, 0.15, n) * vol
    open_ = close * np.exp(open_gap - daily_ret)  # gap relative to prior close
    open_[0] = close[0] * (1 + rng.normal(0, 0.01))

    intrabar_range = np.abs(rng.normal(0, 1, n)) * vol * close
    high = np.maximum(open_, close) + intrabar_range * rng.uniform(0.2, 0.6, n)
    low = np.minimum(open_, close) - intrabar_range * rng.uniform(0.2, 0.6, n)
    low = np.maximum(low, 0.5)  # keep positive

    base_vol = rng.uniform(3e6, 2e7)
    vol_std = vol.std() if vol.std() > 1e-8 else 1.0
    vol_z = np.clip((vol - vol.mean()) / vol_std, -2.5, 2.5)
    volume = base_vol * np.exp(0.6 * vol_z) * rng.lognormal(0, 0.25, n)
    volume = np.nan_to_num(volume, nan=base_vol)
    volume = np.clip(volume, 5e4, None)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume.astype(np.int64),
    }, index=dates)
    df.index.name = "date"
    df["ticker"] = ticker
    df["regime"] = regimes  # kept for diagnostics only, NOT a feature (would be lookahead)
    return df


def download_universe(tickers: List[str], start: str, end: str, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Drop-in synthetic replacement for data.loader.download_universe."""
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    rng = np.random.default_rng(seed)

    market_factor = pd.Series(rng.normal(0, 1, n)).ewm(span=3).mean().values
    sectors = sorted(set(SECTOR_MAP.get(t, "other") for t in tickers))
    sector_factors = {s: pd.Series(rng.normal(0, 1, n)).ewm(span=3).mean().values for s in sectors}

    out = {}
    for t in tickers:
        sec = SECTOR_MAP.get(t, "other")
        out[t] = _gen_ticker_prices(t, dates, market_factor, sector_factors[sec], rng)
    return out


if __name__ == "__main__":
    d = download_universe(["AAPL", "JPM"], "2020-01-01", "2020-03-01")
    print(d["AAPL"].head())
    print(d["AAPL"][["open", "high", "low", "close", "volume"]].describe())
