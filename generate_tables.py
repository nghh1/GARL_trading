"""
Produce every benchmark comparison table in one run: the three grouped
tables (single-agent RL, GARL ablation, supervised baselines) plus a real
Buy & Hold row computed from actual price data via the same backtest
engine every other baseline uses.

Run:
    python3 generate_tables.py

Output: printed to console AND saved to outputs/tables.md (ready to paste
into the dissertation).
"""
import pandas as pd

import config as C
from benchmark_table import build_comparison_table, buy_hold_5fold, buy_hold_portfolio_5fold

USE_SYNTHETIC = False  # real Yahoo data -- set True only if testing without network

RESULTS_5FOLD = "outputs/results_5fold.csv"

print("=== Computing Buy & Hold benchmark, averaged across the SAME 5 walk-forward ===")
print("=== folds/regimes results_5fold.csv's baseline columns are averaged over    ===")
bh_per_ticker = buy_hold_5fold(use_synthetic=USE_SYNTHETIC)
bh_naive_mean = bh_per_ticker.mean().to_dict()          # for Table C only (per-ticker basis)
bh_portfolio = buy_hold_portfolio_5fold(use_synthetic=USE_SYNTHETIC)  # for Tables A/B (portfolio basis)
print(bh_per_ticker)
print()
print("naive per-ticker-mean:", {k: round(v, 3) for k, v in bh_naive_mean.items()})
print("true portfolio-aggregated (captures diversification):", {k: round(v, 3) for k, v in bh_portfolio.items()})
print()

print("=== Table A: Single-agent RL comparison (5-fold mean) ===")
# SingleAgentA2C/PPO/DQN each report their PORTFOLIO row (true equity-curve
# aggregation across all 9 tickers), so Buy & Hold needs the matching
# portfolio-aggregated figure here too -- not the naive per-ticker mean,
# which ignores diversification and understates Buy & Hold's true Sharpe
table_a = build_comparison_table(RESULTS_5FOLD, ["SingleAgentA2C", "SingleAgentPPO", "SingleAgentDQN"],
                                  buy_hold_metrics=bh_portfolio)
print(table_a)
print()

print("=== Table B: GARL ablation + classical baseline (5-fold mean) ===")
# GARL_DDAL/MultiAgentA2C/GARL_DDAL_SECTOR/GARL_DDAL_TUNED are all PORTFOLIO
# rows -- same portfolio-aggregated Buy & Hold as Table A. ARIMAX is the one
# column here that's per-ticker-averaged, not portfolio-aggregated -- a
# genuine, unavoidable asymmetry within this table, worth a footnote rather
# than something fixable here.
table_b = build_comparison_table(RESULTS_5FOLD, ["GARL_DDAL", "MultiAgentA2C", "GARL_DDAL_SECTOR",
                                                   "GARL_DDAL_TUNED", "ARIMAX"],
                                  buy_hold_metrics=bh_portfolio)
print(table_b)
print()

print("=== Table C: Supervised baselines (5-fold mean) ===")
# ARIMAX/RandomForest/LSTM/TCN/TFT never get a portfolio-aggregated row --
# the naive per-ticker-mean Buy & Hold IS the correct, consistent basis here
table_c = build_comparison_table(RESULTS_5FOLD, ["ARIMAX", "RandomForest", "LSTM", "TCN", "TFT"],
                                  buy_hold_metrics=bh_naive_mean)
print(table_c)

with open("outputs/tables.md", "w") as f:
    f.write("# Benchmark comparison tables\n\n")
    f.write("## Table A: Single-agent RL comparison (5-fold mean)\n\n")
    f.write(table_a.to_markdown())
    f.write("\n\n## Table B: GARL ablation + classical baseline (5-fold mean)\n\n")
    f.write(table_b.to_markdown())
    f.write("\n\n## Table C: Supervised baselines (5-fold mean)\n\n")
    f.write(table_c.to_markdown())
    f.write("\n\n## Buy & Hold, per ticker\n\n")
    f.write(bh_per_ticker.to_markdown())
    f.write("\n")

print("\nSaved to outputs/tables.md")