"""
Generate every dissertation figure in one run, in dependency order:
cheapest/no-computation first, most expensive last. The 6 "live" chart
functions are each called ONCE, on ONE deliberately chosen (baseline,
ticker) pair -- not looped across combinations -- specifically to avoid
the repetitive-re-execution problem. Each choice below is picked because
it illustrates a finding already established earlier in this project, not
arbitrarily -- see the comment above each call.

Run once, real data already in outputs/results_raw.csv / results_5fold.csv:
    python3 generate_all_figures.py
"""
import pandas as pd
import plot as P
import config as C

USE_SYNTHETIC = False  # real Yahoo data -- set True only if testing without network

# Deliberately NOT config.START_DATE/END_DATE (2001-2025) for the live charts
# below: every one of them internally shows only a short recent window
# (last 20% of whatever range is passed, or the last N bars) anyway, so
# fitting on the full 25-year research-scale range is pure wasted
# computation working against the whole point of keeping this bounded --
# confirmed directly: TFT alone went from >4 minutes (didn't even finish)
# on the full range to 24 seconds on this 3-year window. The displayed
# test window still lands inside the project's real single-fold test
# period (2021-2025), so it's representative, not an arbitrary substitute.
ILLUSTRATIVE_START, ILLUSTRATIVE_END = "2020-01-01", "2023-01-01"

print("=== 1. CSV-based charts (zero computation) ===")
df = pd.read_csv("outputs/results_raw.csv")
df5 = pd.read_csv("outputs/results_5fold.csv")
df_old = pd.read_csv("outputs/results_raw_pre_lookback.csv")
df5_old = pd.read_csv("outputs/results_5fold_pre_lookback.csv")

P.plot_baseline_comparison_bar(df, metric="Sharpe")
P.plot_gross_vs_net_bar(df)
P.plot_ticker_consensus_heatmap(df, baselines=["ARIMAX", "RandomForest", "LSTM", "TCN", "TFT"])
P.plot_full_fold_heatmap(df5, metric="Sharpe")
P.plot_fold_comparison_line(df5, baselines=["MultiAgentA2C", "GARL_DDAL", "GARL_DDAL_SECTOR", "GARL_DDAL_TUNED"])
P.plot_turnover_vs_sharpe_scatter(df, baselines=["ARIMAX", "RandomForest", "LSTM", "TCN", "TFT"])

# the lookback=1 -> lookback=20 ablation, the project's other major reversal finding
# ARIMAX deliberately excluded here: "after" now reflects BOTH the
# lookback=20 change AND the later, unrelated ARIMAX->RollingARIMAX
# implementation swap -- including it would confound two different
# changes in what's supposed to be an isolated lookback comparison.
affected = ["SingleAgentA2C", "SingleAgentPPO", "SingleAgentDQN",
            "MultiAgentA2C", "MultiAgentPPO", "MultiAgentDQN", "GARL_DDAL", "GARL_DDAL_SECTOR"]
P.plot_before_after_comparison(df_old, df, affected, before_label="lookback=1", after_label="lookback=20")
P.plot_before_after_comparison(df5_old, df5, affected, before_label="lookback=1 (5-fold)",
                                after_label="lookback=20 (5-fold)")

print("\n=== 2. Methodology chart (zero computation, no data needed) ===")
P.plot_cv_fold_schedule()

print("\n=== 3. EDA charts (single ticker, no model fitting) ===")
# NVDA: the ticker with the most dramatic architecture-dependent story in
# this whole project (single-agent shared-trunk collapse, resolved by lookback)
P.plot_price_overview("NVDA", ILLUSTRATIVE_START, ILLUSTRATIVE_END, use_synthetic=USE_SYNTHETIC)
P.plot_feature_correlation_heatmap("NVDA", ILLUSTRATIVE_START, ILLUSTRATIVE_END, use_synthetic=USE_SYNTHETIC)

print("\n=== 4. Live single-fit charts (one baseline x one ticker each) ===")
# RandomForest/LSTM/TCN are fast even on the FULL real training window
# (5-6s each, confirmed), so these reproduce the ACTUAL reported result
# exactly (deterministic RandomForest) or closely (LSTM, same architecture/
# data/hyperparameters, but PyTorch training isn't seeded -- see plot.py's
# build_and_fit_supervised docstring) rather than an unrelated illustrative
# fit -- via load_reported_result(). ARIMAX now dispatches to
# RollingARIMAXBaseline (periodic refit, not fit-once) -- still fully
# deterministic (ARIMA's MLE fit has no randomness), but slower: confirmed
# ~35s for a full-range reproduction, not the old plain ARIMAX's ~5-6s --
# still comfortably affordable, well short of TFT's cost below.
#
# TFT is the one exception: fitting it on the full ~20yr real window took
# >4 minutes (didn't even finish) vs 24s on a 3yr window, so it stays on
# ILLUSTRATIVE_START/END with default params -- plot_position_exposure's
# title automatically labels this "(illustrative, default params)" so the
# figure is honest about what it is, rather than silently looking like a
# reproduced result it isn't.

# LSTM on JPM: the best cross-model-corroborated finding -- reproduces the
# actual reported Sharpe=0.93 result, not an unrelated illustrative fit
lstm_params, lstm_start, lstm_test_start, lstm_end = P.load_reported_result(
    "outputs/results_raw.csv", "LSTM", "JPM")
P.plot_equity_curve("LSTM", "JPM", lstm_start, lstm_end, use_synthetic=USE_SYNTHETIC,
                     params=lstm_params, test_start=lstm_test_start)
P.plot_predicted_vs_actual("LSTM", "JPM", lstm_start, lstm_end, use_synthetic=USE_SYNTHETIC,
                            params=lstm_params, test_start=lstm_test_start)

# TFT on AAPL vs ARIMAX on AAPL: same ticker, opposite ends of the cost-drag
# spectrum -- position_exposure makes WHY visually obvious (smooth vs choppy).
# Both now reproduce the ACTUAL reported result on the SAME real window --
# this is a correctness fix, not just a nicety: an earlier version had TFT
# on the bounded ILLUSTRATIVE_START/END window (for speed) while ARIMAX used
# the real ~2001-2025 window, which meant the pair wasn't actually isolating
# architecture anymore, it was also comparing two different market periods.
# TFT takes several minutes on the full real window (~20yr train, confirmed
# earlier) vs. ARIMAX's few seconds -- worth paying once here specifically
# because this comparison is central to the cost-drag argument; every other
# TFT-adjacent chart in this script stays on the bounded window deliberately.
tft_params, tft_start, tft_test_start, tft_end = P.load_reported_result(
    "outputs/results_raw.csv", "TFT", "AAPL")
P.plot_position_exposure("TFT", "AAPL", tft_start, tft_end, use_synthetic=USE_SYNTHETIC,
                          params=tft_params, test_start=tft_test_start)
arimax_params, arimax_start, arimax_test_start, arimax_end = P.load_reported_result(
    "outputs/results_raw.csv", "ARIMAX", "AAPL")
P.plot_position_exposure("ARIMAX", "AAPL", arimax_start, arimax_end, use_synthetic=USE_SYNTHETIC,
                          params=arimax_params, test_start=arimax_test_start)

# BAC under ARIMAX: the single worst result in the whole project -- reproduces
# the actual reported drawdown, not an illustrative approximation of it
bac_params, bac_start, bac_test_start, bac_end = P.load_reported_result(
    "outputs/results_raw.csv", "ARIMAX", "BAC")
P.plot_drawdown("ARIMAX", "BAC", bac_start, bac_end, use_synthetic=USE_SYNTHETIC,
                 params=bac_params, test_start=bac_test_start)

print("\n=== 5. Optuna tuning-convergence chart (one baseline x one ticker) ===")
# Illustrative by necessity: re-running the actual tuning search itself is
# the point of this chart (it shows convergence, not a fixed endpoint), so
# there's no "reported result" to reproduce here the way sections 4 above
# could. n_trials matches config.N_TRIALS so the search WIDTH is at least
# representative of the real tuning budget, on the bounded illustrative
# window for speed.
P.plot_optuna_history("LSTM", "JPM", ILLUSTRATIVE_START, ILLUSTRATIVE_END, n_trials=C.N_TRIALS,
                       use_synthetic=USE_SYNTHETIC)

print("\n=== 6. GARL training curve (heaviest single call -- reduced epochs, illustrative only) ===")
from data import loader as real_loader
from data import synthetic as synth_loader
from data import features as F

_loader = synth_loader if USE_SYNTHETIC else real_loader
raw = _loader.download_universe(C.TICKERS, ILLUSTRATIVE_START, ILLUSTRATIVE_END) if not USE_SYNTHETIC \
      else _loader.download_universe(C.TICKERS, ILLUSTRATIVE_START, ILLUSTRATIVE_END, seed=C.RANDOM_SEED)
feats = {t: F.build_features(df).dropna() for t, df in raw.items()}
closes = {t: raw[t]["close"] for t in raw}
# 100 epochs, not the full RL_EPOCHS_TRAIN=300 -- the independent-then-shared
# pattern is already clearly visible well before 300, no need to pay for the
# full training budget just to illustrate the mechanism
P.plot_training_curve(feats, closes, epochs=100, rollout_len=C.RL_ROLLOUT_LEN)

print("\nAll dissertation figures generated in outputs/figures/")