"""
Global configuration for the GARL trading experiment framework.

Everything that defines *what* the experiment is (universe, dates, costs,
CV layout, action space) lives here so every baseline module reads from a
single source of truth. This also makes it trivial to scale the framework
up (more tickers, more history, more Optuna trials) by editing one file.
"""
import torch
from dataclasses import dataclass, field
from typing import List

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                                   ("cuda" if torch.cuda.is_available() else "cpu"))

# ---------------------------------------------------------------------------
# Trading universe: stocks spread across major sectors e.g. tech, financial, industrial
# In GARL each ticker = one agent's private environment (its own MDP).
# Sector spread keeps the agents' neighbourhood meaningfully different,
# which is exactly the goal of GARL.
# ---------------------------------------------------------------------------
TICKERS: List[str] = [
    "NVDA",   # Tech
    "AAPL",   # Tech
    "MSFT",   # Tech
    "JPM",    # Financials
    "BAC",    # Financials
    "MS",     # Financials
    "CAT",    # Industrials
    "RTX",
    "BA"
]

SECTOR_MAP = {
    "NVDA": "tech", "AAPL": "tech", "MSFT": "tech",
    "JPM": "financials", "BAC": "financials", "MS": "financials",
    "CAT": "industrials", "RTX": "industrials", "BA": "industrials"
}

START_DATE = "2001-01-01"
END_DATE = "2025-12-31"

# Walk-forward split boundaries (outer, held-out test fold).
# Everything before TEST_START is available for nested CV / tuning.
TRAIN_VAL_END = "2020-12-31"
TEST_START = "2021-01-01"

# ---------------------------------------------------------------------------
# Look-ahead protection
# ---------------------------------------------------------------------------
# Number of bars a feature/label is deliberately lagged/shifted by so that,
# at decision time t, nothing computed using information from t (e.g. close_t)
# leaks into the feature vector used to trade AT t. See data/features.py.
LABEL_HORIZON = 1          # predict next-bar return
EMBARGO_BARS = 10           # gap enforced between train and val/test folds in CV
                            # (kills leakage from rolling-window features whose
                            # window straddles the split boundary)

# ---------------------------------------------------------------------------
# Walk-forward CV (nested): outer loop = expanding-window folds used both for
# realistic backtesting AND as the outer loop of nested CV; inner loop =
# further expanding-window split of each outer-train block, used by Optuna
# to score hyperparameters without ever touching the outer test fold.
# ---------------------------------------------------------------------------
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
MIN_TRAIN_BARS = 1874  # = 1864 (target train length) + EMBARGO_BARS (10) --
# fold 0's train is defined as [0, MIN_TRAIN_BARS) then embargo-trimmed, so
# it can never reach MIN_TRAIN_BARS itself, always MIN_TRAIN_BARS-EMBARGO_BARS.
# Compensating here makes fold 0 land on the SAME final length (1864) that
# folds 1-4 hit via MAX_TRAIN_BARS slicing an already-longer array -- verified
# directly: all 5 folds now train on exactly 1864 bars, not ~740 vs ~1250
# as before. Previously 750 (no embargo compensation, hence fold 0's real
# asymmetry against the other folds' 1250-bar cap).

# Cap on how much trailing history any single fit actually trains on. Turns
# the outer walk-forward from a pure "ever-expanding" window into a capped
# rolling-expanding window: this is standard practice in real quant
# pipelines (recent regimes matter more, and it keeps compute roughly
# constant across later folds instead of growing unboundedly). Applied
# uniformly to every baseline (supervised + RL) right after slicing an
# outer-fold's train block, and therefore also bounds inner-fold tuning
# cost automatically.
MAX_TRAIN_BARS = 1864  # ~2x the resulting test-block size (~929 bars/fold) --
# see MIN_TRAIN_BARS above for why these two are no longer numerically
# close the way they were before (750/1250); they now serve genuinely
# different roles (MIN_TRAIN_BARS embargo-compensates fold 0's boundary,
# MAX_TRAIN_BARS is the actual trailing-window cap every fold trains on).

# ---------------------------------------------------------------------------
# Optuna tuning budget (per baseline, per outer fold). Kept modest by default
# for sandbox runtime; bump N_TRIALS up for a real research run.
# ---------------------------------------------------------------------------
N_TRIALS = 15
N_ARIMAX_TRIALS = 10
OPTUNA_TIMEOUT_SEC = None
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Backtest / market microstructure assumptions
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_BPS = 5.0     # 5 bps per unit turnover, per side
SLIPPAGE_BPS = 2.0             # additional bps modeled as slippage
CAPITAL_PER_AGENT = INITIAL_CAPITAL / len(TICKERS)

# ---------------------------------------------------------------------------
# RL / GARL action space: discrete position sizing (not just buy/hold/sell).
# Each agent picks a TARGET position as a fraction of its allocated capital,
# realistic for portfolio-style trading while staying tractable for
# actor-critic training and consistent with the DDAL paper's A2C agent.
# ---------------------------------------------------------------------------
POSITION_LEVELS: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0] # full short..flat..full long
N_ACTIONS = len(POSITION_LEVELS)

RL_LOOKBACK = 20
# RL training budget (epochs = episodes over the training window, matching
# the paper's "epoch" unit in Algorithm 1).
RL_EPOCHS_TRAIN = 300
RL_GAMMA = 0.95 # discount factor
RL_LR = 3e-4 # Karpathy's learning rate for Adam, often yields better baseline stability
RL_ENTROPY_COEF = 0.01 # Mnih et al.'s A2C/A3C entropy coef delays early convergence
RL_VALUE_COEF = 0.5 # and value coef scales MSE between predicted V and actural return R
RL_ROLLOUT_LEN = 20 # Mnih et al.'s A3C's steps per gradient update (k in the paper)
RL_TUNE_N_TRIALS = 5 # Optuna trials per RL baseline fold
# GARL_DDAL_TUNED's search space (tuning/rl_tuning.py::garl_ddal_param_space)
# has 4 x 3 x 3 = 36 combinations -- RL_TUNE_N_TRIALS=5 alone covers only
# ~14% of it. Every other RL-family baseline either pins a single-value
# space (a2c_param_space -- extra trials there are pure waste, since every
# trial samples the identical point) or searches a small/continuous space
# already reasonably served by 5 TPE trials, so this is a baseline-specific
# override, not a global RL_TUNE_N_TRIALS increase that would waste compute
# everywhere else. 15 trials covers ~42% of the 36-combination space.
RL_TUNE_N_TRIALS_GARL_TUNED = 15
RL_TUNE_EPOCHS = 20 # training epochs per trial

# DDAL-specific hyperparameters (see garl/ddal.py, mirrors Algorithm 1)
DDAL_SHARE_THRESHOLD_FRAC = 0.3 # fraction of RL_EPOCHS_TRAIN before sharing starts
DDAL_MINIBATCH_EPOCHS = 4 # apply sharing/averaging gradient every N epochs
DDAL_GRADIENT_POOL_SIZE = len(TICKERS)-1 # m: how many gradient pieces to average
DDAL_GRADIENT_POOL_SIZE_SECTOR = 2

# ---------------------------------------------------------------------------
# PPO-specific hyperparameters.
# ---------------------------------------------------------------------------
PPO_CLIP_EPS = 0.2         # standard clipped-surrogate epsilon (Schulman et al. 2017)
PPO_GAE_LAMBDA = 0.95      # GAE bias/variance tradeoff (Schulman et al. 2016)
PPO_EPOCHS_PER_UPDATE = 4  # how many passes over each rollout before discarding it
PPO_MINIBATCH_SIZE = 8     # must divide evenly into RL_ROLLOUT_LEN-scale rollouts reasonably

# ---------------------------------------------------------------------------
# DQN-specific hyperparameters.
# ---------------------------------------------------------------------------
DQN_EPSILON_START = 1.0        # fully random actions at the start of training
DQN_EPSILON_END = 0.05         # small residual exploration once trained
DQN_EPSILON_DECAY_FRAC = 0.5   # linear decay over the first 50% of RL_EPOCHS_TRAIN
DQN_REPLAY_BUFFER_SIZE = 5000
DQN_BATCH_SIZE = 32
DQN_MIN_REPLAY_SIZE = 200      # don't start learning until the buffer has this many transitions
DQN_TARGET_SYNC_EVERY = 10     # epochs between target-network syncs

# Momentum(ret), trend(sma, ema, and macd), oscillators(rsi and bb_zscore), 
# volatility(atr and realised vol) and volumn(z-score and obv slope)
FEATURE_COLUMNS: List[str] = [
    "ret_1", "ret_5", "ret_10",
    "sma_ratio_10", "sma_ratio_30", "sma_ratio_200",
    "ema_ratio_12", "ema_ratio_26",
    "rsi_14", "macd", "macd_signal",
    "bb_zscore", "atr_14_norm", "volatility_10", "volatility_30",
    "volume_z_20", "obv_slope_10",
]

OUTPUT_DIR = "outputs"