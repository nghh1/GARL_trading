import os
import pickle
import sys
import time
import traceback
import pandas as pd
import config as C
from backtest import engine as ENGINE

"""
Runs exactly ONE baseline (across the full ticker universe) to completion
and appends its results to outputs/results_raw.csv. Intended to be invoked
once per baseline (separate tool calls / separate processes), 
e.g.:
    python3 run_baseline.py ARIMAX
    python3 run_baseline.py RandomForest
    python3 run_baseline.py LSTM
    python3 run_baseline.py TCN
    python3 run_baseline.py TFT
    python3 run_baseline.py SingleAgentRL
    python3 run_baseline.py MultiAgentRL
    python3 run_baseline.py GARL_DDAL

Each call runs its baseline through to completion on every ticker before returning. 
Uses the pre-cached data/features/fold from outputs/cache.pkl (by prep_cache.py) 
so repeated baseline invocations don't redo data prep. 
Note: already-completed (baseline, ticker) rows in results_raw.csv are skipped, 
so a call can be safely re-run if interrupted.
"""

RESULTS_CSV = "outputs/results_raw.csv"

with open("outputs/cache.pkl", "rb") as f:
    cache = pickle.load(f)
features_by_ticker, close_by_ticker, fold = cache["features_by_ticker"], cache["close_by_ticker"], cache["fold"]

MAX_TRAIN_BARS = C.MAX_TRAIN_BARS
N_TRIALS = C.N_TRIALS
RL_N_TRIALS = C.RL_TUNE_N_TRIALS
RL_TUNE_EPOCHS = C.RL_TUNE_EPOCHS
RL_TRAIN_EPOCHS = C.RL_EPOCHS_TRAIN

SUPERVISED_NAMES = ["ARIMAX", "RandomForest", "LSTM", "TCN", "TFT"]
RL_NAMES = ["SingleAgentA2C", "SingleAgentPPO", "SingleAgentDQN", 
            "MultiAgentA2C", "MultiAgentPPO", "MultiAgentDQN", "GARL_DDAL", "GARL_DDAL_SECTOR", "GARL_DDAL_TUNED"]

def get_supervised_cls(name):
    if name == "ARIMAX":
        from baselines.rolling_arimax import RollingARIMAXBaseline
        return RollingARIMAXBaseline
    if name == "RandomForest":
        from baselines.random_forest import RandomForestBaseline
        return RandomForestBaseline
    if name == "LSTM":
        from baselines.lstm import LSTMBaseline
        return LSTMBaseline
    if name == "TCN":
        from baselines.tcn import TCNBaseline
        return TCNBaseline
    if name == "TFT":
        from baselines.tft import TFTBaseline
        return TFTBaseline
    raise ValueError(name)

def load_done():
    if not os.path.exists(RESULTS_CSV):
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df["baseline"], df["ticker"]))

def append_rows(rows):
    df_new = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        df_new.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULTS_CSV, index=False)

def run_one_supervised(name, ticker):
    from tuning.optuna_utils import tune_baseline
    cls = get_supervised_cls(name)
    feat = features_by_ticker[ticker]
    close = close_by_ticker[ticker]
    X, y = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"]
    train_idx = fold.train_idx[-MAX_TRAIN_BARS:]
    X_train, y_train, close_train = X.iloc[train_idx], y.iloc[train_idx], close.iloc[train_idx]
    X_test, close_test = X.iloc[fold.test_idx], close.iloc[fold.test_idx]

    # ARIMAX now dispatches to RollingARIMAXBaseline -- see
    # config.N_TRIALS_ROLLING_ARIMAX's comment for why this needs a much
    # smaller trial budget than the other supervised baselines.
    trial = C.N_ARIMAX_TRIALS if name=="ARIMAX" else N_TRIALS
    best_params, _ = tune_baseline(cls, X_train, y_train, close_train, n_trials=trial)
    model = cls(**best_params)
    model.fit(X_train, y_train)
    # ARIMAX-family models accept an optional y_true_for_walk to use REAL,
    # already-realized test-period returns for AR-term state extension
    # instead of a dummy zero placeholder -- see experiments/run_experiment.py
    # for the full explanation and verification. Every other supervised
    # baseline's predict_returns() doesn't have this parameter, so this
    # transparently falls through to the normal call for them.
    import inspect
    from baselines.base import signal_to_position
    if "y_true_for_walk" in inspect.signature(model.predict_returns).parameters:
        y_test = y.iloc[fold.test_idx]
        pred = model.predict_returns(X_test, y_true_for_walk=y_test)
        pos_test = signal_to_position(pred)
    else:
        pos_test = model.predict_position(X_test)
    result = ENGINE.single_asset_backtest(close_test, pos_test)
    return {"baseline": name, "fold": 0, "ticker": ticker, **result.summary,
            "params": str(best_params)}


def run_one_rl(name):
    from tuning.rl_tuning import tune_rl_baseline, a2c_param_space, ppo_param_space, dqn_param_space, garl_ddal_param_space
    dispatch = {
        "SingleAgentA2C": "rl.single_agent:train_single_agent_a2c:predict_positions_single_agent_a2c",
        "SingleAgentPPO": "rl.single_agent:train_single_agent_ppo:predict_positions_single_agent_ppo",
        "SingleAgentDQN": "rl.single_agent:train_single_agent_dqn:predict_positions_single_agent_dqn",
        "MultiAgentA2C": "rl.multi_agent:train_multi_agent_a2c:predict_positions_multi_agent_a2c",
        "MultiAgentPPO": "rl.multi_agent:train_multi_agent_ppo:predict_positions_multi_agent_ppo",
        "MultiAgentDQN": "rl.multi_agent:train_multi_agent_dqn:predict_positions_multi_agent_dqn",
        "GARL_DDAL": "garl.ddal:run_ddal:predict_positions_garl",
        "GARL_DDAL_SECTOR": "garl.ddal:run_ddal_sector:predict_positions_garl",
        "GARL_DDAL_TUNED": "garl.ddal:run_ddal:predict_positions_garl"
    }
    param_space_by_name = {
        "SingleAgentA2C": a2c_param_space, "MultiAgentA2C": a2c_param_space, "GARL_DDAL": a2c_param_space,
        "GARL_DDAL_SECTOR": a2c_param_space, "SingleAgentPPO": ppo_param_space, "MultiAgentPPO": ppo_param_space,
        "SingleAgentDQN": dqn_param_space, "MultiAgentDQN": dqn_param_space,
        "GARL_DDAL_TUNED": garl_ddal_param_space
    }
    module_path, train_name, predict_name = dispatch[name].split(":")
    import importlib
    mod = importlib.import_module(module_path)
    train_fn, predict_fn = getattr(mod, train_name), getattr(mod, predict_name)

    train_idx = fold.train_idx[-MAX_TRAIN_BARS:]
    feat_train = {t: df.iloc[train_idx] for t, df in features_by_ticker.items()}
    close_train = {t: s.iloc[train_idx] for t, s in close_by_ticker.items()}
    feat_test = {t: df.iloc[fold.test_idx] for t, df in features_by_ticker.items()}
    close_test = {t: s.iloc[fold.test_idx] for t, s in close_by_ticker.items()}

    kind_n_trials = C.RL_TUNE_N_TRIALS_GARL_TUNED if name == "GARL_DDAL_TUNED" else RL_N_TRIALS
    best_params, _ = tune_rl_baseline(train_fn, predict_fn, param_space_by_name[name], feat_train, close_train,
                                       n_trials=kind_n_trials, tune_epochs=RL_TUNE_EPOCHS)
    models = train_fn(feat_train, close_train, epochs=RL_TRAIN_EPOCHS, seed=C.RANDOM_SEED, **best_params)
    positions = predict_fn(models, feat_test, close_test)
    portfolio_res, per_ticker_res = ENGINE.portfolio_backtest(
        close_test, positions, capital_per_ticker=C.CAPITAL_PER_AGENT
    )

    rows = [{"baseline": name, "fold": 0, "ticker": "PORTFOLIO", **portfolio_res.summary,
             "params": str(best_params)}]
    for t, r in per_ticker_res.items():
        rows.append({"baseline": name, "fold": 0, "ticker": t, **r.summary, "params": ""})
    with open(f"outputs/rl_positions_{name}.pkl", "wb") as f:
        pickle.dump(positions, f)
    return rows


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 run_baseline.py <name>  (one of "
              f"{SUPERVISED_NAMES + RL_NAMES})")
        sys.exit(1)
    name = sys.argv[1]
    t0 = time.time()
    done = load_done()

    if name in SUPERVISED_NAMES:
        for ticker in C.TICKERS:
            if (name, ticker) in done:
                print(f"skip {name}/{ticker} (already done)")
                continue
            t1 = time.time()
            try:
                row = run_one_supervised(name, ticker)
                append_rows([row])
                print(f"{name}/{ticker} done in {time.time()-t1:.1f}s "
                      f"Sharpe={row['Sharpe']:.3f} CAGR={row['CAGR']:.3f}", flush=True)
            except Exception as e:
                print(f"FAILED {name}/{ticker}: {e}")
                traceback.print_exc()
                append_rows([{"baseline": name, "fold": 0, "ticker": ticker, "error": str(e)}])
    elif name in RL_NAMES:
        if (name, "PORTFOLIO") in done:
            print(f"skip {name} (already done)")
        else:
            t1 = time.time()
            try:
                rows = run_one_rl(name)
                append_rows(rows)
                port = rows[0]
                print(f"{name} done in {time.time()-t1:.1f}s "
                      f"Portfolio Sharpe={port['Sharpe']:.3f} CAGR={port['CAGR']:.3f}", flush=True)
            except Exception as e:
                print(f"FAILED {name}: {e}")
                traceback.print_exc()
                append_rows([{"baseline": name, "fold": 0, "ticker": "PORTFOLIO", "error": str(e)}])
    else:
        print(f"Unknown baseline name: {name}")
        sys.exit(1)

    print(f"=== {name} TOTAL TIME: {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()