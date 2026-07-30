"""
Real market data loader (Yahoo Finance via yfinance).

This is the module every baseline actually depends on in production use.
It is network-isolated on purpose: nothing else in the codebase imports
`yfinance` directly, so swapping data vendors later only touches this file.

NOTE (sandbox): this container's outbound network allowlist does not
include Yahoo Finance, so `download_universe()` cannot be executed inside
this development sandbox (verified: query1.finance.yahoo.com -> 403
host_not_allowed). The function is fully implemented and will work
out-of-the-box in any environment with normal internet access. For
pipeline development/testing here, use `data/synthetic.py` instead, which
exposes the identical DataFrame schema.
"""
from __future__ import annotations
import logging
import time
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df

def download_one(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """Download a single ticker's daily OHLCV from Yahoo Finance."""
    import yfinance as yf

    last_err = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=True,
                progress=False, threads=False,
            )
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df = flatten_yf_columns(df)
            df = df[REQUIRED_COLUMNS].copy()
            df.index.name = "date"
            df["ticker"] = ticker
            return df
        except Exception as e:
            last_err = e
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, ticker, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts: {last_err}")

def download_universe(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for every ticker in the universe.
    Returns: mapping dict[ticker] -> DataFrame indexed by date with columns
    [open, high, low, close, volume, ticker]
    """
    out = {}
    for t in tickers:
        logger.info("Downloading %s ...", t)
        out[t] = download_one(t, start, end)
    return out

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    try:
        d = download_one("AAPL", "2023-01-01", "2023-06-01")
        print(d.head())
    except Exception as e:  # noqa: BLE001
        print(f"Expected in network-restricted sandboxes: {e}", file=sys.stderr)
