
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseBaseline

"""
Random Forest regressor baseline predicting next-bar return from the
engineered technical-indicators set.

Assess whether nonlinear feature interactions outperform the linear ARIMAX model. 
Additionally, evaluate if Random Forest, which does not capture temporal structure 
beyond the rolling window, performs more competitively than sequential modelling methods.
"""
class RandomForestBaseline(BaseBaseline):
    name = "RandomForest"

    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 min_samples_leaf: int = 20, max_features: str = "sqrt",
                 random_state: int = 42, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                          min_samples_leaf=min_samples_leaf, max_features=max_features, **kwargs)
        # 4 feature per split
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            random_state=random_state, n_jobs=-1,
        )
        self._cols = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestBaseline":
        mask = y_train.notna() & X_train.notna().all(axis=1)
        self._cols = list(X_train.columns)
        self.model.fit(X_train.loc[mask], y_train.loc[mask])
        return self

    def predict_returns(self, X: pd.DataFrame) -> pd.Series:
        X = X[self._cols].fillna(0.0)
        return pd.Series(self.model.predict(X), index=X.index)

    @staticmethod
    def default_param_space(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

if __name__ == "__main__":
    # Quick smoke test (optional)
    from data import synthetic
    from data import features as F
    import config as C

    raw = synthetic.download_universe(["AAPL"], "2015-01-01", "2021-01-01")["AAPL"]
    feat = F.build_features(raw).dropna()
    X, y = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"]
    cut = int(len(X) * 0.8)
    m = RandomForestBaseline()
    m.fit(X.iloc[:cut], y.iloc[:cut])
    pos = m.predict_position(X.iloc[cut:])
    print(pos.describe())
    print("OK RandomForest fit + predict")
