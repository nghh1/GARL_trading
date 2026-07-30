from __future__ import annotations
import os
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config as C
 
FIG_DIR = "outputs/figures"

# Consistent colour-by-family scheme used across every chart in this module,
# so the same baseline always reads as the same colour in every figure in
# the dissertation, not just within one plot.
FAMILY_COLORS = {
    "ARIMAX": "#4C72B0", "RollingARIMAX": "#4C72B0", "RandomForest": "#55A868",
    "LSTM": "#C44E52", "TCN": "#8172B2", "TFT": "#CCB974",
    "SingleAgentA2C": "#64B5CD", "SingleAgentPPO": "#4C9BB0", "SingleAgentDQN": "#357A8C",
    "MultiAgentA2C": "#DD8452", "MultiAgentPPO": "#C1652E", "MultiAgentDQN": "#A34E1A",
    "GARL_DDAL": "#8B0000", "GARL_DDAL_SECTOR": "#B22222", "GARL_DDAL_TUNED": "#FF8C00",
}
 
 
def color_for(baseline: str) -> str:
    return FAMILY_COLORS.get(baseline, "#777777")
 
 
def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)
 
 
def save(fig, name: str, save_path: Optional[str]):
    ensure_fig_dir()
    path = save_path or os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    return path
 
 
# ---------------------------------------------------------------------------
# 1a. Cross-baseline comparison -- bar chart
# ---------------------------------------------------------------------------
def plot_baseline_comparison_bar(results_df: pd.DataFrame, metric: str = "Sharpe",
                                  ticker: str = "PORTFOLIO",
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None):
    """One bar per baseline. For RL baselines, uses the PORTFOLIO row; for
    supervised baselines (no portfolio row), uses the mean across tickers.
    If `results_df` has a `fold` column with >1 distinct value, averages
    across folds first.
    """
    df = results_df.copy()
    if "fold" in df.columns and df["fold"].nunique() > 1:
        df = df.groupby(["baseline", "ticker"], as_index=False)[metric].mean()
 
    rows = []
    for baseline, g in df.groupby("baseline"):
        if (g["ticker"] == "PORTFOLIO").any():
            val = g.loc[g["ticker"] == "PORTFOLIO", metric].iloc[0]
        else:
            val = g[metric].mean()
        rows.append((baseline, val))
    rows.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*rows)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [color_for(n) for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by baseline" + (" (mean across folds)" if "fold" in results_df.columns and results_df["fold"].nunique() > 1 else ""))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 1 if v >= 0 else -10),
                    ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, f"baseline_comparison_{metric}.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 1b. Gross vs net -- grouped bar chart (the cost-drag finding)
# ---------------------------------------------------------------------------
def plot_gross_vs_net_bar(results_df: pd.DataFrame, save_path: Optional[str] = None):
    df = results_df.copy()
    if "fold" in df.columns and df["fold"].nunique() > 1:
        df = df.groupby(["baseline", "ticker"], as_index=False)[["Sharpe", "Gross_Sharpe"]].mean()
 
    rows = []
    for baseline, g in df.groupby("baseline"):
        if (g["ticker"] == "PORTFOLIO").any():
            r = g[g["ticker"] == "PORTFOLIO"].iloc[0]
        else:
            r = g[["Sharpe", "Gross_Sharpe"]].mean()
        rows.append((baseline, r["Sharpe"], r["Gross_Sharpe"]))
    rows.sort(key=lambda x: x[2], reverse=True)  # sort by gross, the "signal quality" axis
    names = [r[0] for r in rows]
    net = [r[1] for r in rows]
    gross = [r[2] for r in rows]
 
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, gross, width, label="Gross (no cost)", color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, net, width, label="Net (with cost)", color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Gross vs. net Sharpe by baseline (the gap is transaction-cost drag)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "gross_vs_net_sharpe.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 1c. Fold-by-fold comparison -- line chart (for the ablation / fold-dependence story)
# ---------------------------------------------------------------------------
def plot_fold_comparison_line(results_5fold_df: pd.DataFrame,
                               baselines: List[str] = None,
                               metric: str = "Sharpe",
                               ticker: str = "PORTFOLIO",
                               save_path: Optional[str] = None):
    """
    One line per baseline, fold index (or test period, if test_start is
    present) on the x-axis. This is the chart to use for showing whether an
    ablation's winner is stable across folds/regimes or not.
    """
    baselines = baselines or ["MultiAgentA2C", "GARL_DDAL", "GARL_DDAL_SECTOR"]
    df = results_5fold_df[results_5fold_df["ticker"] == ticker]
 
    # compute x-axis labels once, from the full (fold-indexed) frame -- not
    # inside the per-baseline loop, which was order-/missing-data-fragile
    # (previously undefined if the first baseline in the list had no rows)
    fold_order = sorted(df["fold"].unique())
    if "test_start" in df.columns:
        label_lookup = df.drop_duplicates("fold").set_index("fold")["test_start"].astype(str).str[:7]
        xlabels = [label_lookup.get(f, str(f)) for f in fold_order]
    else:
        xlabels = [str(f) for f in fold_order]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in baselines:
        g = df[df["baseline"] == b].set_index("fold").reindex(fold_order)
        if g[metric].isna().all():
            continue
        ax.plot(range(len(fold_order)), g[metric], marker="o", label=b, color=color_for(b), linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_xlabel("Fold (test period start)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across walk-forward folds: {', '.join(baselines)}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"fold_comparison_{metric}.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 1d. Cross-ticker / cross-baseline consensus -- heatmap
# ---------------------------------------------------------------------------
def plot_ticker_consensus_heatmap(results_df: pd.DataFrame, baselines: List[str],
                                   metric: str = "Gross_Sharpe",
                                   save_path: Optional[str] = None):
    df = results_df.copy()
    if "fold" in df.columns and df["fold"].nunique() > 1:
        df = df.groupby(["baseline", "ticker"], as_index=False)[metric].mean()
    df = df[df["baseline"].isin(baselines) & (df["ticker"] != "PORTFOLIO")]
    pivot = df.pivot(index="ticker", columns="baseline", values=metric)[baselines]
 
    fig, ax = plt.subplots(figsize=(1.1 * len(baselines) + 2, 0.55 * len(pivot) + 2))
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric, shrink=0.8)
    ax.set_title(f"{metric} by ticker x baseline")
    fig.tight_layout()
    save(fig, "ticker_consensus_heatmap.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 2a. LIVE equity curve: strategy vs. buy-and-hold benchmark
# (re-fits/backtests -- see module docstring)
# ---------------------------------------------------------------------------
def _cut_index(index: pd.DatetimeIndex, test_start: Optional[str] = None) -> int:
    """Where to split train/test. If test_start is given (e.g. from
    load_reported_result()), uses that EXACT date -- matching a real
    reported result's actual train/test boundary. Otherwise falls back to
    an approximate 80/20 split, fine for a purely illustrative chart but
    NOT the same boundary any reported result actually used.
    """
    if test_start is not None:
        return int(index.searchsorted(pd.Timestamp(test_start)))
    return int(len(index) * 0.8)


def plot_equity_curve(baseline_name: str, ticker: str, start: str, end: str,
                       use_synthetic: bool = False, params: dict = None,
                       test_start: Optional[str] = None, save_path: Optional[str] = None):
    """
    Matches the classic "cumulative returns vs benchmark" chart style
    (dissertation Figures 31/32/34): strategy equity vs. buy-and-hold,
    both normalized to start at 0% return.

    params/test_start default to None -- an illustrative fit with default
    hyperparameters on an approximate 80/20 split, NOT necessarily the same
    number as any row in results_raw.csv/results_5fold.csv. To reproduce a
    specific reported result exactly, pass params/test_start from
    load_reported_result(csv_path, baseline_name, ticker[, fold]).
    """
    from backtest import engine as ENGINE
    from data import features as F

    raw = load_one_ticker(ticker, start, end, use_synthetic)
    feat = F.build_features(raw).dropna()
    X, y, close = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"], raw["close"].reindex(feat.index)
    cut = _cut_index(X.index, test_start)

    model = build_and_fit_supervised(baseline_name, X.iloc[:cut], y.iloc[:cut], params=params)
    pos = predict_position_reported(model, X.iloc[cut:], y.iloc[cut:])
    result = ENGINE.single_asset_backtest(close.iloc[cut:], pos)

    strategy_ret = (result.equity / result.equity.iloc[0] - 1) * 100
    benchmark_ret = (close.iloc[cut:] / close.iloc[cut] - 1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(benchmark_ret.index, benchmark_ret.values, label="Buy & Hold", color="#CCB974", linewidth=1.5)
    ax.plot(strategy_ret.index, strategy_ret.values, label=f"{baseline_name} strategy",
            color=color_for(baseline_name), linewidth=2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(f"{baseline_name} on {ticker}: cumulative returns vs. buy & hold"
                 + (" (reproduces reported result)" if params else " (illustrative, default params)"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"equity_curve_{baseline_name}_{ticker}.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 2b. LIVE predicted vs actual returns
# ---------------------------------------------------------------------------
def plot_predicted_vs_actual(baseline_name: str, ticker: str, start: str, end: str,
                              use_synthetic: bool = True, n_bars: int = 120,
                              params: dict = None, test_start: Optional[str] = None,
                              save_path: Optional[str] = None):
    """Predicted next-bar return signal vs. the true realized return, for
    the last `n_bars` of the test period (a full multi-year series is too
    dense to read visually). Good for showing directional-accuracy /
    calibration rather than cumulative P&L.

    params/test_start: see plot_equity_curve's docstring -- None means
    illustrative default-hyperparameter fit, not a reported result.
    """
    from data import features as F

    raw = load_one_ticker(ticker, start, end, use_synthetic)
    feat = F.build_features(raw).dropna()
    X, y = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"]
    cut = _cut_index(X.index, test_start)

    model = build_and_fit_supervised(baseline_name, X.iloc[:cut], y.iloc[:cut], params=params)
    import inspect
    if "y_true_for_walk" in inspect.signature(model.predict_returns).parameters:
        pred = model.predict_returns(X.iloc[cut:], y_true_for_walk=y.iloc[cut:]).tail(n_bars)
    else:
        pred = model.predict_returns(X.iloc[cut:]).tail(n_bars)
    actual = y.iloc[cut:].tail(n_bars)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual.index, actual.values, label="Actual next-bar return", color="#333333", linewidth=1.5)
    ax.plot(pred.index, pred.values, label="Predicted return", color=color_for(baseline_name),
            linewidth=1.5, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Return")
    ax.set_title(f"{baseline_name} on {ticker}: predicted vs. actual next-bar return (last {n_bars} bars)"
                 + (" (reproduces reported result)" if params else " (illustrative, default params)"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"predicted_vs_actual_{baseline_name}_{ticker}.png", save_path)
    return fig
 
# ---------------------------------------------------------------------------
# 1e. Full baseline x fold heatmap (stability/robustness view, all 13 at once)
# ---------------------------------------------------------------------------
def plot_full_fold_heatmap(results_5fold_df: pd.DataFrame, metric: str = "Sharpe",
                            ticker: str = "PORTFOLIO",
                            save_path: Optional[str] = None):
    """
    Every baseline (rows) x every fold (columns) in one grid -- the
    "is this baseline stable across regimes, or fold-dependent" view, for
    all 13 baselines at once rather than 2-3 picked out by hand.
    """
    df = results_5fold_df.copy()
    rows = []
    for (baseline, fold), g in df.groupby(["baseline", "fold"]):
        if (g["ticker"] == "PORTFOLIO").any():
            val = g.loc[g["ticker"] == "PORTFOLIO", metric].iloc[0]
        else:
            val = g[metric].mean()
        rows.append((baseline, fold, val))
    pivot = pd.DataFrame(rows, columns=["baseline", "fold", metric]).pivot(
        index="baseline", columns="fold", values=metric)
    # order rows by mean value, best at top -- easier to read than alphabetical
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
 
    fig, ax = plt.subplots(figsize=(2 + 1.1 * pivot.shape[1], 0.5 * pivot.shape[0] + 2))
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"fold {f}" for f in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric, shrink=0.8)
    ax.set_title(f"{metric} by baseline x fold (all 13 baselines, rows sorted by mean)")
    fig.tight_layout()
    save(fig, f"full_fold_heatmap_{metric}.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 1f. General-purpose before/after comparison (lookback ablation, bug fixes, etc.)
# ---------------------------------------------------------------------------
def plot_before_after_comparison(before_df: pd.DataFrame, after_df: pd.DataFrame,
                                  baselines: List[str], metric: str = "Sharpe",
                                  before_label: str = "Before", after_label: str = "After",
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None):
    """Grouped bar chart, one pair of bars per baseline, comparing the same
    metric across two experimental conditions -- e.g. lookback=1 vs
    lookback=20, or pre/post the ARIMAX trend-vs-d bug fix. Each df should
    have the standard baseline/ticker/metric columns; PORTFOLIO row used
    for RL baselines, mean across tickers otherwise (same convention as
    plot_baseline_comparison_bar).
    """
    def value_for(df, b):
        g = df[df["baseline"] == b]
        if "fold" in g.columns and g["fold"].nunique() > 1:
            g = g.groupby("ticker", as_index=False)[metric].mean()
        if (g["ticker"] == "PORTFOLIO").any():
            return g.loc[g["ticker"] == "PORTFOLIO", metric].iloc[0]
        return g[metric].mean()
 
    before_vals = [value_for(before_df, b) for b in baselines]
    after_vals = [value_for(after_df, b) for b in baselines]
 
    x = np.arange(len(baselines))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(baselines)), 5))
    ax.bar(x - width / 2, before_vals, width, label=before_label, color="#999999", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, after_vals, width, label=after_label, color="#DD8452", edgecolor="black", linewidth=0.5)
    for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
        delta = av - bv
        y = max(bv, av) + 0.03 * (max(abs(min(before_vals + after_vals)), abs(max(before_vals + after_vals))) or 1)
        ax.annotate(f"{delta:+.2f}", (x[i], y), ha="center", fontsize=8,
                    color="#2E7D32" if delta > 0 else "#C62828")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric}: {before_label} vs {after_label}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    slug = lambda s: s.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    save(fig, f"before_after_{metric}_{slug(before_label)}_vs_{slug(after_label)}.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 1g. Turnover vs. net Sharpe -- scatter (cost-drag mechanism, per-ticker granularity)
# ---------------------------------------------------------------------------
def plot_turnover_vs_sharpe_scatter(results_df: pd.DataFrame, baselines: List[str] = None,
                                     save_path: Optional[str] = None):
    """One point per (baseline, ticker), turnover on x, net Sharpe on y,
    colored by baseline. Shows the cost-drag relationship at full per-ticker
    granularity rather than the baseline-level averages in
    plot_gross_vs_net_bar -- makes clear it's a consistent, not just
    average, relationship.
    """
    df = results_df.copy()
    if "fold" in df.columns and df["fold"].nunique() > 1:
        df = df.groupby(["baseline", "ticker"], as_index=False)[["Sharpe", "Turnover"]].mean()
    df = df[df["ticker"] != "PORTFOLIO"]
    if baselines:
        df = df[df["baseline"].isin(baselines)]
 
    fig, ax = plt.subplots(figsize=(8, 6))
    for b, g in df.groupby("baseline"):
        ax.scatter(g["Turnover"], g["Sharpe"], label=b, color=color_for(b),
                   s=50, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Turnover")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Turnover vs. net Sharpe (per baseline x ticker)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "turnover_vs_sharpe_scatter.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# 1h. LIVE training curve: loss vs epoch, sharing-start marker
# (re-trains -- see module docstring; this is the only chart type that
# genuinely can't be built from the results CSVs at all, since epoch-level
# loss was never persisted anywhere before this function's underlying
# return_history=True support was added to garl/ddal.py and rl/multi_agent.py)
# ---------------------------------------------------------------------------
def plot_training_curve(features_by_ticker: Dict[str, pd.DataFrame],
                         close_by_ticker: Dict[str, pd.Series],
                         epochs: int = C.RL_EPOCHS_TRAIN, rollout_len: int = C.RL_ROLLOUT_LEN,
                         tickers_to_show: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
    """Per-epoch training loss for GARL_DDAL, with a vertical line marking
    when gradient-sharing starts (config.DDAL_SHARE_THRESHOLD_FRAC). This
    is the chart to use for demonstrating the independent-learning-then-
    group-learning mechanism itself, not just its downstream effect on
    Sharpe -- nothing else in this module shows what happens DURING
    training.
    """
    from garl.ddal import run_ddal
 
    _, history = run_ddal(features_by_ticker, close_by_ticker, epochs=epochs,
                           rollout_len=rollout_len, return_history=True)
    tickers_to_show = tickers_to_show or list(history.keys())
    threshold_epoch = int(epochs * C.DDAL_SHARE_THRESHOLD_FRAC)
 
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in tickers_to_show:
        ax.plot(range(len(history[t])), history[t], label=t, linewidth=1.3, alpha=0.85)
    ax.axvline(threshold_epoch, color="black", linestyle="--", linewidth=1.2,
               label=f"sharing starts (epoch {threshold_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("A2C loss (policy + value - entropy)")
    ax.set_title("GARL_DDAL training curve: independent learning -> gradient sharing")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "training_curve_garl_ddal.png", save_path)
    return fig

# ===========================================================================
# EDA / DATA PREPROCESSING
# ===========================================================================
def plot_price_overview(ticker: str, start: str, end: str, use_synthetic: bool = True,
                         save_path: Optional[str] = None):
    """Price + volume overview, the standard first-figure-in-an-EDA-section
    chart. Two panels sharing the x-axis: close price on top, volume below.
    """
    raw = load_one_ticker(ticker, start, end, use_synthetic)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(raw.index, raw["close"], color="#4C72B0", linewidth=1)
    ax1.set_ylabel("Close price")
    ax1.set_title(f"{ticker}: price and volume, {start} to {end}")
    ax1.grid(alpha=0.3)
 
    ax2.bar(raw.index, raw["volume"], color="#999999", width=1.0)
    ax2.set_ylabel("Volume")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"price_overview_{ticker}.png", save_path)
    return fig
 
 
def plot_feature_correlation_heatmap(ticker: str, start: str, end: str, use_synthetic: bool = True,
                                      save_path: Optional[str] = None):
    """Correlation matrix of the 17 engineered features -- motivates any
    discussion of multicollinearity (e.g. sma_ratio_10/30/200 are related
    by construction) and generally justifies the feature set design.
    """
    from data import features as F
 
    raw = load_one_ticker(ticker, start, end, use_synthetic)
    feat = F.build_features(raw).dropna()
    corr = feat[C.FEATURE_COLUMNS].corr()
 
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.8)
    ax.set_title(f"Feature correlation matrix ({ticker})")
    fig.tight_layout()
    save(fig, f"feature_correlation_{ticker}.png", save_path)
    return fig
 
 
def plot_cv_fold_schedule(save_path: Optional[str] = None):
    """Gantt-style visualization of the walk-forward CV schedule: one row
    per outer fold, train block and test block as horizontal bars spanning
    their actual calendar date range. Uses config.py's real settings
    directly (cv/walk_forward.py::outer_splits), so this always reflects
    whatever N_OUTER_FOLDS/MIN_TRAIN_BARS/MAX_TRAIN_BARS/EMBARGO_BARS
    currently are -- not a hand-drawn illustration. Makes fold 0's shorter,
    uncapped training window (discussed at length earlier -- it's the only
    fold not yet hitting MAX_TRAIN_BARS) immediately visible rather than
    something that has to be explained in prose.
    """
    from cv import walk_forward as WF
 
    idx = pd.bdate_range(C.START_DATE, C.END_DATE)
    folds = WF.outer_splits(idx, n_folds=C.N_OUTER_FOLDS, min_train_bars=C.MIN_TRAIN_BARS,
                             embargo=C.EMBARGO_BARS)
 
    fig, ax = plt.subplots(figsize=(11, 1.2 * len(folds) + 1))
    for i, f in enumerate(folds):
        train_bars = len(f.train_idx)
        capped = train_bars > C.MAX_TRAIN_BARS
        train_start = idx[f.train_idx[-1] - min(train_bars, C.MAX_TRAIN_BARS) + 1]
        ax.barh(i, (f.train_end - train_start).days, left=train_start, height=0.6,
                color="#4C72B0", edgecolor="black", linewidth=0.5,
                label="Train (capped)" if capped and i == 1 else ("Train" if not capped and i == 0 else None))
        ax.barh(i, (f.test_end - f.test_start).days, left=f.test_start, height=0.6,
                color="#DD8452", edgecolor="black", linewidth=0.5,
                label="Test" if i == 0 else None)
        note = f"fold {i}" + ("  (uncapped, {} bars)".format(train_bars) if not capped else "")
        ax.text(train_start, i + 0.42, note, fontsize=8, va="bottom")
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"fold {i}" for i in range(len(folds))])
    ax.invert_yaxis()
    ax.set_xlabel("Date")
    ax.set_title(f"Walk-forward CV schedule ({C.N_OUTER_FOLDS} outer folds, "
                 f"MAX_TRAIN_BARS={C.MAX_TRAIN_BARS}, EMBARGO_BARS={C.EMBARGO_BARS})")
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "cv_fold_schedule.png", save_path)
    return fig
 
 
# ===========================================================================
# MODEL TRAINING / OPTIMIZATION DYNAMICS
# ===========================================================================
def plot_optuna_history(baseline_name: str, ticker: str, start: str, end: str,
                         n_trials: int = 20, use_synthetic: bool = True,
                         save_path: Optional[str] = None):
    """Optuna objective (validation Sharpe) vs. trial number, with the
    running best overlaid -- the tuning-convergence chart. Reuses
    tuning/optuna_utils.py::tune_baseline() directly rather than
    reimplementing a tuning loop, so this always reflects the actual
    nested-CV objective every real run optimizes, not an approximation.
    """
    from tuning.optuna_utils import tune_baseline
    from data import features as F
 
    registry = {
        "ARIMAX": "baselines.arimax.ARIMAXBaseline", "RollingARIMAX": "baselines.rolling_arimax.RollingARIMAXBaseline",
        "RandomForest": "baselines.random_forest.RandomForestBaseline",
        "LSTM": "baselines.lstm.LSTMBaseline", "TCN": "baselines.tcn.TCNBaseline", "TFT": "baselines.tft.TFTBaseline",
    }
    module_path, cls_name = registry[baseline_name].rsplit(".", 1)
    import importlib
    cls = getattr(importlib.import_module(module_path), cls_name)
 
    raw = load_one_ticker(ticker, start, end, use_synthetic)
    feat = F.build_features(raw).dropna()
    X, y, close = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"], raw["close"].reindex(feat.index)
 
    _, study = tune_baseline(cls, X, y, close, n_trials=n_trials)
    values = [t.value for t in study.trials]
    running_best = np.maximum.accumulate(values)
 
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(range(len(values)), values, color="#999999", s=25, label="Trial objective (Sharpe)")
    ax.plot(range(len(values)), running_best, color="#C44E52", linewidth=2, label="Running best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective (mean inner-fold Sharpe)")
    ax.set_title(f"Optuna hyperparameter search: {baseline_name} on {ticker}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"optuna_history_{baseline_name}_{ticker}.png", save_path)
    return fig
 
 
# ===========================================================================
# FINANCIAL PERFORMANCE / BACKTESTING EVALUATION
# ===========================================================================
def plot_drawdown(baseline_name: str, ticker: str, start: str, end: str,
                   use_synthetic: bool = True, params: dict = None,
                   test_start: Optional[str] = None, save_path: Optional[str] = None):
    """Underwater equity plot: running drawdown from the prior peak, filled.
    The classic risk-visualization companion to the equity-curve chart --
    shows HOW BAD and HOW LONG the worst drawdown was, which a Sharpe/MDD
    number alone doesn't communicate as viscerally.

    params/test_start: see plot_equity_curve's docstring.
    """
    from backtest import engine as ENGINE
    from data import features as F

    raw = load_one_ticker(ticker, start, end, use_synthetic)
    feat = F.build_features(raw).dropna()
    X, y, close = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"], raw["close"].reindex(feat.index)
    cut = _cut_index(X.index, test_start)

    model = build_and_fit_supervised(baseline_name, X.iloc[:cut], y.iloc[:cut], params=params)
    pos = predict_position_reported(model, X.iloc[cut:], y.iloc[cut:])
    result = ENGINE.single_asset_backtest(close.iloc[cut:], pos)

    drawdown = (result.equity / result.equity.cummax() - 1) * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#C44E52", alpha=0.6)
    ax.plot(drawdown.index, drawdown.values, color="#8B0000", linewidth=1)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"{baseline_name} on {ticker}: drawdown from prior peak "
                 f"(max: {drawdown.min():.1f}%)"
                 + (" (reproduces reported result)" if params else " (illustrative, default params)"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"drawdown_{baseline_name}_{ticker}.png", save_path)
    return fig
 
 
def plot_position_exposure(baseline_name: str, ticker: str, start: str, end: str,
                            use_synthetic: bool = True, params: dict = None,
                            test_start: Optional[str] = None, save_path: Optional[str] = None):
    """Held position over time (step plot) -- shows the actual trading
    behavior a strategy produces (how often it flips, how extreme its bets
    are), the natural companion to the equity-curve and predicted-vs-actual
    charts. Directly relevant given how much this project's findings turned
    on turnover/position-sizing behavior (cost drag, RL's discrete
    POSITION_LEVELS vs. supervised baselines' continuous sizing).

    params/test_start: see plot_equity_curve's docstring.
    """
    from data import features as F

    raw = load_one_ticker(ticker, start, end, use_synthetic)
    feat = F.build_features(raw).dropna()
    X, y = feat[C.FEATURE_COLUMNS], feat["fwd_ret_h"]
    cut = _cut_index(X.index, test_start)

    model = build_and_fit_supervised(baseline_name, X.iloc[:cut], y.iloc[:cut], params=params)
    pos = predict_position_reported(model, X.iloc[cut:], y.iloc[cut:])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(pos.index, pos.values, where="post", color=color_for(baseline_name), linewidth=1.2)
    ax.fill_between(pos.index, pos.values, 0, step="post", color=color_for(baseline_name), alpha=0.25)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Position (fraction of capital)")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(f"{baseline_name} on {ticker}: held position over time"
                 + (" (reproduces reported result)" if params else " (illustrative, default params)"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"position_exposure_{baseline_name}_{ticker}.png", save_path)
    return fig
 
 
# ---------------------------------------------------------------------------
# helpers shared by the live-chart functions
# ---------------------------------------------------------------------------
def load_one_ticker(ticker: str, start: str, end: str, use_synthetic: bool) -> pd.DataFrame:
    if use_synthetic:
        from data import synthetic as loader
        return loader.download_universe([ticker], start, end, seed=C.RANDOM_SEED)[ticker]
    from data import loader
    return loader.download_one(ticker, start, end)
 
 
def build_and_fit_supervised(baseline_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                              params: dict = None):
    """params=None (default) fits with the baseline's DEFAULT constructor
    arguments -- fine for a purely illustrative chart, but NOT the same
    model/hyperparameters as any reported result in results_raw.csv /
    results_5fold.csv (those were tuned per-fold by Optuna). Pass params=
    (e.g. from load_reported_result() below) to reproduce a specific
    reported number exactly.
    """
    registry = {
        "ARIMAX": ("baselines.rolling_arimax", "RollingARIMAXBaseline"),  # matches the
        # swap in experiments/run_experiment.py and run_baseline.py -- was
        # ("baselines.arimax", "ARIMAXBaseline"). Left unfixed here, this
        # class-mapping bug wouldn't crash (ARIMAXBaseline.__init__ accepts
        # **kwargs, silently absorbing window_size/refit_every) -- it would
        # construct the WRONG model, discard those two params, and still
        # label the chart "(reproduces reported result)" when it isn't.
        "RollingARIMAX": ("baselines.rolling_arimax", "RollingARIMAXBaseline"),
        "RandomForest": ("baselines.random_forest", "RandomForestBaseline"),
        "LSTM": ("baselines.lstm", "LSTMBaseline"),
        "TCN": ("baselines.tcn", "TCNBaseline"),
        "TFT": ("baselines.tft", "TFTBaseline"),
    }
    if baseline_name not in registry:
        raise ValueError(f"plot_equity_curve/plot_predicted_vs_actual currently support "
                          f"the supervised baselines only: {list(registry)}. "
                          f"For RL/GARL live equity curves, backtest the PORTFOLIO result "
                          f"from run_rl_family_baseline() directly and plot its .equity series.")
    module_path, cls_name = registry[baseline_name]
    import importlib
    cls = getattr(importlib.import_module(module_path), cls_name)
    model = cls(**(params or {}))
    model.fit(X_train, y_train)
    return model


def predict_position_reported(model, X_test: pd.DataFrame, y_test: pd.Series = None) -> pd.Series:
    """Same inspect.signature-based check as run_supervised_baseline() /
    run_baseline.py's run_one_supervised(): ARIMAX-family models
    (RollingARIMAXBaseline, and ARIMAXBaseline if ever used directly) use
    REAL, already-realized y_test for their AR-term state extension instead
    of a dummy zero placeholder when it's available -- causally valid (see
    experiments/run_experiment.py for the full explanation and the direct
    verification that this measurably changes predictions vs. dummy zeros).
    Every other supervised baseline's predict_returns() doesn't have this
    parameter, so this transparently falls through to predict_position()
    for them. Centralized here since 4 live-chart functions need the same
    logic, instead of duplicating the signature check 4 times.
    """
    import inspect
    from baselines.base import signal_to_position
    if y_test is not None and "y_true_for_walk" in inspect.signature(model.predict_returns).parameters:
        pred = model.predict_returns(X_test, y_true_for_walk=y_test)
        return signal_to_position(pred)
    return model.predict_position(X_test)


def load_reported_result(results_csv_path: str, baseline: str, ticker: str, fold: int = None):
    """Pulls the EXACT tuned hyperparameters (parsed from the params column)
    and the EXACT train/test date range a reported result actually used,
    from results_raw.csv (single-fold, uses config.TRAIN_VAL_END/dates) or
    results_5fold.csv (uses that row's own test_start/test_end columns,
    written by experiments/run_experiment.py).

    Returns (params: dict, start, test_start, end) -- feed straight into
    the live chart functions' `start=`/`end=`/`params=`/`test_start=`
    arguments to make them reproduce the reported number, not an
    unrelated illustrative fit.

    KNOWN, EXPLAINED small discrepancy: the original reported numbers were
    computed via prep_cache.py, which downloads ALL C.TICKERS together and
    builds a single common_index = the INTERSECTION of every ticker's valid
    feature dates -- so the exact test-start position for e.g. BAC is
    determined jointly by all 9 tickers' availability, not BAC's alone.
    This function instead fetches and builds features for the ONE requested
    ticker in isolation, so its test-split cut point (via X.index.searchsorted)
    can land a few bars away from the original multi-ticker common_index's
    cut point if that ticker's own valid-date set differs even slightly from
    the intersection. Observed effect: ~0.5 percentage points on a reported
    MaxDrawdown of -62.7% (i.e. small, not substantive) -- fine for
    dissertation figures illustrating a finding, but if you need bit-perfect
    reproduction, rebuild the exact common_index across all C.TICKERS first
    (see prep_cache.py) rather than fetching this one ticker alone.
    """
    import ast
    df = pd.read_csv(results_csv_path)
    row = df[(df["baseline"] == baseline) & (df["ticker"] == ticker)]
    if fold is not None:
        row = row[row["fold"] == fold]
    if row.empty:
        raise ValueError(f"No row found for baseline={baseline}, ticker={ticker}, fold={fold} "
                          f"in {results_csv_path}")
    row = row.iloc[0]
    params_str = row["params"]
    params = ast.literal_eval(params_str) if isinstance(params_str, str) and params_str.strip() else {}

    if "test_start" in row and pd.notna(row.get("test_start")):
        fetch_start, test_start = C.START_DATE, str(row["test_start"])[:10]
        test_end = str(row["test_end"])[:10]
    else:
        # single-fold results_raw.csv: no test_start/test_end column --
        # matches prep_cache.py's TRAIN_VAL_END-derived split exactly
        fetch_start, test_start, test_end = C.START_DATE, C.TEST_START, C.END_DATE
    return params, fetch_start, test_start, test_end
 
if __name__ == "__main__":
    # Demonstration using synthetic data (this sandbox has no real Yahoo
    # access) and the existing merged results CSVs if present.
    plot_equity_curve("LSTM", "AAPL", "2018-01-01", "2021-01-01", use_synthetic=True)
    plot_predicted_vs_actual("LSTM", "AAPL", "2018-01-01", "2021-01-01", use_synthetic=True)
 
    results_path = "outputs/results_raw.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        plot_baseline_comparison_bar(df, metric="Sharpe")
        plot_gross_vs_net_bar(df)
        plot_ticker_consensus_heatmap(df, baselines=["ARIMAX", "RandomForest", "LSTM", "TCN", "TFT"])
    else:
        print(f"(skipping summary-stat charts: {results_path} not found in this sandbox)")
 
    fold5_path = "outputs/results_5fold.csv"
    if os.path.exists(fold5_path):
        df5 = pd.read_csv(fold5_path)
        plot_fold_comparison_line(df5)
    else:
        print(f"(skipping fold-comparison chart: {fold5_path} not found in this sandbox)")
 
    print("OK analysis.plots demo complete")