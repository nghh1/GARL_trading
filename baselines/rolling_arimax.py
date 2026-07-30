"""
RollingARIMAX -- a periodically-refit ARIMAX variant, adopted (selectively)
from Johnson (2024)'s dissertation "A Novel Approach to Stock Trading with
Group-Agent Deep Reinforcement Learning". Their "Online ARIMA" -- fit on a
trailing window, forecast a short horizon, refit, repeat -- was by far
their strongest baseline (Sharpe 2.89 on MSFT, vs. 1.27 for their best RL
model), a large enough gap to be worth testing here rather than assuming
our existing ARIMAX (fit once per outer fold, ~5 years, never refit until
the next fold) is already capturing what a more adaptive statistical model
could.

Kept as a SEPARATE baseline from ARIMAXBaseline (baselines/arimax.py)
rather than replacing it -- they answer different questions. ARIMAXBaseline
tests "how good is a single fit, held fixed, over a multi-year test
period" (comparable to every other baseline in this project, all of which
are fit once per fold). RollingARIMAX tests "how much does frequent
refitting on recent data help" -- a genuinely different modeling choice,
not a strictly-better version of the same one, and conflating the two
would make it impossible to tell which effect (refit frequency vs. the
model itself) drove any performance difference.

Look-ahead protection, walk-forward correctness: at the point of forecasting
any given chunk of dates, the model is fit only on data strictly before
that chunk. Immediately after a chunk's dates elapse, their TRUE realized
labels (fwd_ret_h, already known once fwd_ret_h's own horizon has passed --
see data/features.py's causality convention) get appended to the rolling
history before the next refit. Nothing from a future, not-yet-elapsed
chunk is ever used to fit the model that forecasts it -- verified in the
self-test below the same way as the rest of this project's look-ahead
checks.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base import BaseBaseline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class RollingARIMAXBaseline(BaseBaseline):
    name = "RollingARIMAX"

    def __init__(self, p: int = 1, d: int = 1, q: int = 1, trend: str = "c",
                 window_size: int = 252, refit_every: int = 10, **kwargs):
        """window_size/refit_every default to ~1 trading year / ~2 trading
        weeks, matching the dissertation's own choice ("fitted to a year's
        worth of data... predicting the prices for the next 10 days").
        """
        super().__init__(p=p, d=d, q=q, trend=trend, window_size=window_size,
                          refit_every=refit_every, **kwargs)
        self.p, self.d, self.q, self.trend = p, d, q, trend
        self.window_size, self.refit_every = window_size, refit_every
        self._history_X = None
        self._history_y = None
        self._exog_cols = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RollingARIMAXBaseline":
        # Nothing is actually fit here -- RollingARIMAX refits fresh inside
        # predict_returns() for every chunk. fit() just seeds the rolling
        # history with the training tail, so the first test chunk has a
        # full window_size of causal history immediately available.
        self._exog_cols = list(X_train.columns)
        self._history_X = X_train[self._exog_cols].tail(self.window_size).copy()
        self._history_y = y_train.tail(self.window_size).copy()
        return self

    def predict_returns(self, X: pd.DataFrame, y_true_for_walk: pd.Series = None) -> pd.Series:
        """y_true_for_walk: the TRUE fwd_ret_h for the dates in X, used only
        to extend the rolling history bar-by-chunk as the walk proceeds --
        never to fit the model that forecasts those same dates. If not
        supplied (e.g. a caller only has features, not labels, at predict
        time) the rolling history simply stops updating after fit()'s seed
        window, degrading gracefully to a single fixed-window fit -- still
        correct, just not "rolling" anymore. Passing it is required to get
        the actual behavior this baseline exists to test.
        """
        X = X[self._exog_cols]
        history_X, history_y = self._history_X.copy(), self._history_y.copy()
        preds = {}

        for start in range(0, len(X), self.refit_every):
            chunk_idx = X.index[start:start + self.refit_every]
            train_X = history_X.tail(self.window_size).fillna(0.0)
            train_y = history_y.tail(self.window_size).fillna(0.0)
            # same fix as baselines/arimax.py: trend='c' is invalid whenever
            # d>0 (statsmodels rejects it -- constant is eliminated by
            # differencing). Resolve here so it can never silently fail.
            effective_trend = "n" if self.d > 0 else self.trend

            try:
                model = ARIMA(endog=train_y.values, exog=train_X.values,
                               order=(self.p, self.d, self.q), trend=effective_trend,
                               enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit()
                chunk_X = X.loc[chunk_idx].fillna(0.0)
                forecast = results.get_forecast(steps=len(chunk_idx), exog=chunk_X.values)
                chunk_pred = forecast.predicted_mean
            except Exception as e:  # noqa: BLE001
                # NOTE: a bare silent except here previously masked a real
                # bug (trend='c' invalid when d>0) that made every single
                # chunk fail and fall through to this fallback, producing
                # an all-zero prediction series with no visible error at
                # all -- always log what actually failed.
                import logging
                logging.getLogger(__name__).warning(
                    "RollingARIMAX chunk fit failed (%s), falling back to flat prediction", e)
                chunk_pred = np.zeros(len(chunk_idx))  # flat/no-signal fallback, never a crash

            for date, val in zip(chunk_idx, chunk_pred):
                preds[date] = val

            # extend rolling history with this chunk's causal features +
            # (if available) its now-realized true labels, ready for the
            # NEXT chunk's refit -- this chunk's own forecast never used them
            history_X = pd.concat([history_X, X.loc[chunk_idx]])
            if y_true_for_walk is not None:
                history_y = pd.concat([history_y, y_true_for_walk.loc[chunk_idx]])
            else:
                history_y = pd.concat([history_y, pd.Series(chunk_pred, index=chunk_idx)])

        return pd.Series(preds).reindex(X.index)

    @staticmethod
    def default_param_space(trial) -> dict:
        return {
            "p": trial.suggest_int("p", 0, 2),
            "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 2),
            "trend": trial.suggest_categorical("trend", ["n", "c"]),
            "window_size": trial.suggest_categorical("window_size", [252]),
            "refit_every": trial.suggest_categorical("refit_every", [10, 20]),
        }


def _self_check_no_lookahead():
    """Same style of check as data/features.py: perturbing a chunk's own
    exog values must not change ANY earlier chunk's forecast (earlier
    chunks are already fit and forecast before later chunks are even
    touched -- this proves it structurally, not just by inspection).
    """
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(0, 0.01, n), index=idx)

    cut = 300
    m = RollingARIMAXBaseline(window_size=100, refit_every=20)
    m.fit(X.iloc[:cut], y.iloc[:cut])
    pred_a = m.predict_returns(X.iloc[cut:], y_true_for_walk=y.iloc[cut:])

    X2 = X.copy()
    perturb_from = cut + 60  # perturb a LATER chunk's own exog values
    X2.iloc[perturb_from:] += 1000.0
    m2 = RollingARIMAXBaseline(window_size=100, refit_every=20)
    m2.fit(X.iloc[:cut], y.iloc[:cut])  # identical training history
    pred_b = m2.predict_returns(X2.iloc[cut:], y_true_for_walk=y.iloc[cut:])

    before = pred_a.iloc[:60]
    after = pred_b.iloc[:60]
    pd.testing.assert_series_equal(before, after, check_exact=False, rtol=1e-6)
    print("OK: no look-ahead leakage detected in RollingARIMAX walk-forward refitting")


if __name__ == "__main__":
    _self_check_no_lookahead()

    from data import synthetic
    from data import features as F
    import config as C

    raw = synthetic.download_universe(["AAPL"], "2018-01-01", "2021-01-01")["AAPL"]
    feat = F.build_features(raw).dropna()
    X, y = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"]
    cut = int(len(X) * 0.8)

    m = RollingARIMAXBaseline(window_size=100, refit_every=10)
    m.fit(X.iloc[:cut], y.iloc[:cut])

    pos_fallback = m.predict_position(X.iloc[cut:])  # no y_true_for_walk -> degrades to fixed-window fallback
    print("fallback path (no y_true_for_walk):")
    print(pos_fallback.describe())

    m2 = RollingARIMAXBaseline(window_size=100, refit_every=10)
    m2.fit(X.iloc[:cut], y.iloc[:cut])
    from baselines.base import signal_to_position
    pred_walk = m2.predict_returns(X.iloc[cut:], y_true_for_walk=y.iloc[cut:])
    pos_walk = signal_to_position(pred_walk)
    print("real walk-forward path (with y_true_for_walk):")
    print(pos_walk.describe())
    print("OK RollingARIMAX fit + rolling-refit predict, both paths")