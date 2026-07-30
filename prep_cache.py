"""Precompute and cache data/features/fold once, so each chunked run doesn't redo it."""
import pandas as pd
import pickle
import config as C
from data import loader
from data import features as F
from cv import walk_forward as WF

raw = loader.download_universe(C.TICKERS, C.START_DATE, C.END_DATE)
features_by_ticker = {t: F.build_features(df, label_horizon=C.LABEL_HORIZON) for t, df in raw.items()}
close_by_ticker = {t: df["close"] for t, df in raw.items()}

common_index = None
for feat in features_by_ticker.values():
    idx = feat.index
    common_index = idx if common_index is None else common_index.intersection(idx)
features_by_ticker = {t: df.loc[common_index] for t, df in features_by_ticker.items()}
close_by_ticker = {t: s.loc[common_index] for t, s in close_by_ticker.items()}
min_train_bars = int(common_index.searchsorted(pd.Timestamp(C.TRAIN_VAL_END), side="right"))
outer_folds = WF.outer_splits(common_index, n_folds=1, min_train_bars=min_train_bars, embargo=C.EMBARGO_BARS)
fold = outer_folds[0]
print("Fold: train", fold.train_start.date(), "->", fold.train_end.date(),
      "test", fold.test_start.date(), "->", fold.test_end.date(),
      "| train bars", len(fold.train_idx), "| test bars", len(fold.test_idx))

with open("outputs/cache.pkl", "wb") as f:
    pickle.dump({"features_by_ticker": features_by_ticker, "close_by_ticker": close_by_ticker,
                 "fold": fold}, f)
print("Cached OK")
