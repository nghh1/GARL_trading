"""
Feature engineering with explicit look-ahead protection.

CONVENTION (enforced throughout the codebase):
  - Row t's FEATURE columns are only allowed to use information available
    up to and including the close of bar t (i.e. "what you could compute
    the moment bar t closes").
  - Row t's LABEL column `fwd_ret_h` is the return from close_t to
    close_{t+LABEL_HORIZON}, i.e. it is knowable only in the future relative
    to t. It must NEVER be used as a feature, and any row whose label
    reaches past the end of the available data is dropped.
  - Trading decisions for bar t+1 are made using the feature row at t (all
    envs/backtests shift signals by one bar before applying them) -- this
    file only guarantees the feature side is leak-free; envs/backtest.py
    guarantees the *decision* side is leak-free (trade at t+1 open/close
    using info known at t).
  - All rolling windows use pandas `.rolling()/.ewm()` which are causal by
    construction (only look backward), so no additional shifting is needed
    within a single ticker's feature computation. Cross-sectional features
    (if ever added) would need same-day-only aggregation to stay causal.

A small pytest-style self-check at the bottom verifies that perturbing a
*future* bar's OHLCV never changes a *past* bar's feature value.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss==0 (zero losses in the window -- a pure/near-pure uptrend)
    # makes rs undefined (division by the NaN-replaced zero), which the
    # blanket .fillna(50.0) below used to silently resolve to a "neutral"
    # 50 -- verified directly: a monotonic uptrend produced RSI=50 instead
    # of the mathematically correct ~100 (maximally overbought). The
    # opposite case (avg_gain==0, pure downtrend) was already correct
    # without this fix, since rs=0 there is a valid, well-defined number,
    # not a NaN -- this bug was one-sided.
    rsi = rsi.where(avg_loss != 0, 100.0)
    # the one genuinely undefined case: truly no price movement at all
    # (avg_gain==0 AND avg_loss==0) -- correctly neutral at 50
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi.fillna(50.0)

def build_features(df: pd.DataFrame, label_horizon: int = 1) -> pd.DataFrame:
    """Take a raw OHLCV frame (index=date) and return engineered features
    + forward label, all causal as documented above.
    """
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    sma10, sma30, sma200 = close.rolling(10).mean(), close.rolling(30).mean(), close.rolling(200).mean()
    out["sma_ratio_10"] = close / sma10 - 1
    out["sma_ratio_30"] = close / sma30 - 1
    out["sma_ratio_200"] = close / sma200 - 1

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    out["ema_ratio_12"] = close / ema12 - 1
    out["ema_ratio_26"] = close / ema26 - 1

    out["rsi_14"] = (_rsi(close, 14) - 50.0) / 50.0  # rescale 0..100 -> -1..1,
    # matching the scale of every other bounded feature here (ret_*,
    # sma/ema_ratio_*, macd/macd_signal, bb_zscore). Previously left as raw
    # 0..100 -- roughly 1000-2000x the magnitude of ret_1 -- which risked
    # dominating gradient-based updates for the RL/deep baselines despite
    # Adam's per-parameter adaptive scaling only partially compensating.

    out["macd"] = (ema12 - ema26) / close
    # macd_signal derived DIRECTLY from out["macd"] (the already-normalized
    # column), not a separately-computed EMA-of-raw-macd normalized
    # afterward. Previously: EMA(ema12-ema26) computed first in raw price
    # units, THEN divided by today's close -- a different, if defensible,
    # operation from EMA(macd/close) (smoothing happens on the un-normalized
    # scale in the old version, on the actual reported feature in this one).
    # This version means "macd_signal" is unambiguously and verifiably "the
    # 9-day EMA of the macd column you actually see," with no separate
    # hidden raw computation path to reason about.
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_zscore"] = (close - bb_mid) / (2 * bb_std.replace(0, np.nan))

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    out["atr_14_norm"] = atr14 / close

    out["volatility_10"] = out["ret_1"].rolling(10).std()
    out["volatility_30"] = out["ret_1"].rolling(30).std()

    vol_mean20, vol_std20 = vol.rolling(20).mean(), vol.rolling(20).std()
    out["volume_z_20"] = ((vol - vol_mean20) / vol_std20.replace(0, np.nan)).clip(-4, 4)
    # clipped, not rescaled -- it's already a z-score by construction (O(1)
    # in the typical case), but extreme volume days pushed it to +-5..10,

    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    out["obv_slope_10"] = (obv.diff(10) / vol.rolling(10).mean().replace(0, np.nan)) / 10.0
    # /10.0: naturally bounded at +-10 by construction (max when all 10 days
    # move the same direction at ~average volume) -- same category of issue
    # as rsi_14 above, caught in the same feature-scale review, rescaled the
    # same way to bring it in line with the other ~[-1,1] features rather
    # than sitting at ~280x ret_1's scale (std 3.74 vs 0.013, pre-rescale).

    out = out.replace([np.inf, -np.inf], np.nan)

    # Forward label: pure future info, kept in a clearly-named column and
    # excluded from config.FEATURE_COLUMNS so it can never accidentally be
    # used as a model input.
    out["fwd_ret_h"] = close.pct_change(label_horizon).shift(-label_horizon)
    out["close"] = close  # convenience passthrough for backtest engine

    return out

def build_features_for_universe(raw: dict, label_horizon: int = 1) -> dict:
    return {t: build_features(df, label_horizon) for t, df in raw.items()}

def _self_check_no_lookahead():
    """Perturbing bar t+5's price must not change bar t's feature row."""
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": rng.uniform(1e6, 5e6, n),
    }, index=idx)

    feat_a = build_features(df)

    df2 = df.copy()
    t_perturb = 60
    df2.loc[df2.index[t_perturb:], "close"] += 1000  # blow up all future prices
    df2.loc[df2.index[t_perturb:], "open"] += 1000
    df2.loc[df2.index[t_perturb:], "high"] += 1000
    df2.loc[df2.index[t_perturb:], "low"] += 1000
    feat_b = build_features(df2)

    cols = [c for c in feat_a.columns if c not in ("fwd_ret_h",)]
    before = feat_a.iloc[:t_perturb][cols]
    after = feat_b.iloc[:t_perturb][cols]
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-8)
    print("OK: no look-ahead leakage detected in build_features()")

if __name__ == "__main__":
    _self_check_no_lookahead()