"""
GARL: Group-Agent Reinforcement Learning, via DDAL (Decentralised
Distributed Asynchronous Learning), following Wu & Zeng (2023),
"Group-Agent Reinforcement Learning" (arXiv:2202.05135), Algorithm 1.

Mapping paper -> code:
  - "study group" of N agents, each in its own separate/stationary
    environment    -> one SingleStockTradingEnv per ticker (envs/trading_env.py)
  - agent's "brain" can be any single-agent algorithm; paper demonstrates
    A2C (-> DDA3C)  -> rl.a2c_agent.ActorCritic, same net/loss as the
                       multi_agent.py ablation baseline
  - knowledge K_i, K_{-i}                -> self.own_pool / self.inbox per agent
  - training experience T_j, relevance R_j -> epoch index at send time (T),
                                              and 1.0 for all pairs (R) since
                                              every agent works the same task
                                              type (as the paper does for its
                                              homogeneous-task CartPole eval)
  - weighted gradient average ḡ           -> weighted_average() below,
                                              identical formula to the paper:
                                              ḡ = 1/2 * (Σ T_j/ΣT_j * g_j + Σ R_j/ΣR_j * g_j)
  - Algorithm 1 lines 1-16                -> run_ddal() main loop

Faithful vs. simplified:
  - FAITHFUL: decentralised control (each agent only touches its own model
    + its own inbox), asynchronous-STYLE message passing (an agent's
    gradient becomes visible to peers as soon as it's sent, not gated by a
    global barrier), independent-learning-until-threshold, weighted
    gradient averaging exactly per the paper's formula, periodic
    (every DDAL_MINIBATCH_EPOCHS) application of the averaged gradient.
  - SIMPLIFIED: this sandbox runs everything in one Python process
    (single-threaded round-robin over agents per epoch) rather than real
    OS processes/network sockets exchanging messages via multiprocessing
    queues as in the paper's implementation. This affects WALL-CLOCK
    parallelism only, not the learning algorithm: the data structures
    (per-agent inbox "queues"), the order-independent pooling of gradients,
    and the update math are unchanged, so swapping in real
    multiprocessing.Queue + separate processes is a systems change, not an
    algorithm change (see `run_ddal_multiprocess_note` below for exactly
    what would move). This also means this implementation is SYNCHRONOUS,
    whereas the paper's is asynchronous; this file can simulate "studied"
    gradient staleness by passing `staleness_epochs > 0` to `run_ddal()`,
    which deliberately holds gradients in a buffer for a fixed number of
    epochs before making them available to peers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import config as C
from envs.trading_env import SingleStockTradingEnv
from rl.a2c_agent import ActorCritic, RolloutState, epoch_step, grads_of, apply_grads, greedy_action_series

@dataclass
class GradPiece:
    grads: List[torch.Tensor]
    T: float   # training experience (epoch index when generated)
    R: float   # relevance to the receiving agent

@dataclass
class AgentSlot:
    ticker: str
    model: ActorCritic
    optimizer: torch.optim.Optimizer
    state: RolloutState
    rng: np.random.Generator
    own_pool: List[GradPiece] = field(default_factory=list)
    inbox: List[GradPiece] = field(default_factory=list)
    device: str = "cpu"

def weighted_average(pieces: List[GradPiece]) -> List[torch.Tensor]:
    """ḡ = 1/2 * ( Σ (T_j / ΣT) g_j  +  Σ (R_j / ΣR) g_j ), Algorithm 1's
    weighted-gradient-average formula, computed parameter-tensor-wise.
    """
    sumT = sum(p.T for p in pieces) or 1.0
    sumR = sum(p.R for p in pieces) or 1.0
    n_params = len(pieces[0].grads)
    avg = []
    for pi in range(n_params):
        term_T = sum((p.T / sumT) * p.grads[pi] for p in pieces)
        term_R = sum((p.R / sumR) * p.grads[pi] for p in pieces)
        avg.append(0.5 * (term_T + term_R))
    return avg

def run_ddal(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
             epochs: int = C.RL_EPOCHS_TRAIN, rollout_len: int = C.RL_ROLLOUT_LEN,
             share_threshold_frac: float = C.DDAL_SHARE_THRESHOLD_FRAC,
             minibatch_epochs: int = C.DDAL_MINIBATCH_EPOCHS,
             pool_size: int = C.DDAL_GRADIENT_POOL_SIZE,
             staleness_epochs: int = 0, # New add
             peer_group_map: Dict[str, str] = None,
             seed: int = C.RANDOM_SEED, device: str = None,
             return_history: bool = False):
    """New add: staleness_epochs: if > 0, simulates asynchronous updates by holding
    a generated gradient for this many epochs before it becomes visible to
    any peer agent, as a way of studying the effect of gradient staleness."""
    device = device or C.DEVICE
    threshold = int(epochs * share_threshold_frac)
    agents: Dict[str, AgentSlot] = {}
    for i, (t, feat) in enumerate(features_by_ticker.items()):
        torch.manual_seed(seed + i)
        env = SingleStockTradingEnv(feat, close_by_ticker[t], ticker=t)
        obs, _ = env.reset()
        model = ActorCritic(obs.shape[0], env.action_space.n).to(device)
        agents[t] = AgentSlot(
            ticker=t, model=model, optimizer=torch.optim.Adam(model.parameters(), lr=C.RL_LR),
            state=RolloutState(obs=obs, env=env), rng=np.random.default_rng(seed + i), device=device
        )
    tickers = list(agents.keys())
    history: Dict[str, List[float]] = {t: [] for t in tickers}
    pending_broadcasts: List[Tuple[int, str, GradPiece]] = [] # New add

    def peers_of(t: str) -> List[str]:
        if peer_group_map is None:
            return [o for o in tickers if o!=t]
        g = peer_group_map.get(t)
        return [o for o in tickers if o!=t and peer_group_map.get(o)==g]

    for ep in range(epochs):
        # --- lines 2-4: every agent generates k experiences, computes its
        # own gradient for this epoch (independent-learning phase never stops
        # generating its own gradient; sharing only changes what's APPLIED) ---
        epoch_grads: Dict[str, List[torch.Tensor]] = {}
        for t in tickers:
            a = agents[t]
            loss_val = epoch_step(a.state, a.model, rollout_len, C.RL_GAMMA, C.RL_ENTROPY_COEF,
                       C.RL_VALUE_COEF, a.rng)
            epoch_grads[t] = grads_of(a.model)
            history[t].append(loss_val)

        if ep < threshold:
            # --- line 6: independent learning, update with own gradient ---
            for t in tickers:
                apply_grads(agents[t].model, epoch_grads[t], agents[t].optimizer)
        else:
            # --- lines 8-10: store own gradient, broadcast a copy to every
            # peer agent's inbox (with optional staleness) ---
            for t in tickers:
                piece = GradPiece(grads=epoch_grads[t], T=float(ep + 1), R=1.0)
                agents[t].own_pool.append(piece)
                availability_epoch = ep + staleness_epochs # New add
                for other in peers_of(t):
                    # Originally: agents[other].inbox.append(piece)
                    pending_broadcasts.append((availability_epoch, other, piece)) # New add

            # New add code block--- Process newly available gradients from the pending buffer ---
            # This is where the "asynchronous" part of DDAL is simulated:
            # gradients aren't available until their staleness delay has passed.
            remaining_broadcasts = []
            for availability_ep, target_ticker, piece in pending_broadcasts:
                if ep >= availability_ep:
                    agents[target_ticker].inbox.append(piece)
                else:
                    remaining_broadcasts.append((availability_ep, target_ticker, piece))
            pending_broadcasts = remaining_broadcasts

            # --- lines 11-14: every `minibatch_epochs` epochs, pull m
            # gradient pieces from own_pool ∪ inbox, weighted-average, and
            # update with the averaged gradient instead of the raw own one ---
            if (ep + 1) % minibatch_epochs == 0:
                for t in tickers:
                    a = agents[t]
                    pool = a.own_pool + a.inbox
                    if not pool:
                        apply_grads(a.model, epoch_grads[t], a.optimizer)
                        continue
                    m = min(pool_size, len(pool))
                    chosen = pool[-m:]  # most recent m pieces (own_pool ∪ inbox)
                    avg_grad = weighted_average(chosen)
                    apply_grads(a.model, avg_grad, a.optimizer)
                    # "get (and remove)" per Algorithm 1 line 12
                    a.own_pool, a.inbox = [], []
            else:
                # between minibatch epochs, still make progress on own gradient
                for t in tickers:
                    apply_grads(agents[t].model, epoch_grads[t], agents[t].optimizer)
    if return_history:
        return {t: agents[t].model for t in tickers}, history
    return {t: agents[t].model for t in tickers}

def run_ddal_sector(features_by_ticker: Dict[str, pd.DataFrame], close_by_ticker: Dict[str, pd.Series],
                     epochs: int = C.RL_EPOCHS_TRAIN, rollout_len: int = C.RL_ROLLOUT_LEN,
                     seed: int = C.RANDOM_SEED, device: str = None,
                     return_history: bool = False):
    """Thin wrapper: same DDAL algorithm as run_ddal(), restricted to
    same-sector gradient sharing via config.SECTOR_MAP and a pool size
    sized for 2 same-sector peers (config.DDAL_GRADIENT_POOL_SIZE_SECTOR)
    instead of all 8. Kept as a separate dispatch entry ("GARL_DDAL_SECTOR")
    rather than changing run_ddal()'s defaults, so the original
    all-to-all "GARL_DDAL" result is never overwritten -- both are
    genuinely different experiments worth comparing side by side, not
    a fix replacing a bug.

    return_history=False by default, matching run_ddal()'s own default --
    this was previously hardcoded to True here regardless of what the
    caller wanted, so EVERY call to run_ddal_sector() (including the real
    experimental dispatch in experiments/run_experiment.py and
    run_baseline.py, not just plot.py's training-curve chart) silently
    got back a (dict, history) tuple instead of the plain dict its own
    return-type annotation promises -- causing 'tuple' object has no
    attribute 'items()' downstream wherever the result was unpacked as
    a dict. Fixed by making it a real pass-through parameter instead.
    """
    return run_ddal(features_by_ticker, close_by_ticker, epochs=epochs, rollout_len=rollout_len,
                     pool_size=C.DDAL_GRADIENT_POOL_SIZE_SECTOR, peer_group_map=C.SECTOR_MAP,
                     seed=seed, device=device, return_history=return_history)

@torch.no_grad()
def predict_positions_garl(models: Dict[str, ActorCritic], features_by_ticker: Dict[str, pd.DataFrame],
                            close_by_ticker: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    out = {}
    for t, model in models.items():
        env = SingleStockTradingEnv(features_by_ticker[t], close_by_ticker[t], ticker=t)
        actions = greedy_action_series(model, env)
        levels = [C.POSITION_LEVELS[a] for a in actions]
        out[t] = pd.Series(levels, index=env.features.index[:len(levels)])
    return out


def run_ddal_multiprocess_note():
    """Not executed -- documents the systems-level change needed to go from
    this sandbox's single-process simulation to the paper's real
    multiprocessing deployment:
      1. Replace `AgentSlot` with a class run inside its own
         `multiprocessing.Process`.
      2. Replace `inbox: List[GradPiece]` with a `multiprocessing.Queue`
         per agent (paper: "every agent has its own queue ... shared among
         all agents so each agent is free to send its knowledge to any
         other agent's queue").
      3. `for other in tickers: agents[other].inbox.append(piece)` becomes
         `queues[other].put(piece)` (non-blocking put -- true asynchrony).
      4. Each process drains its own queue at minibatch-epoch boundaries
         instead of the main loop draining `a.inbox` directly.
      5. `weighted_average()` and the update rule are UNCHANGED -- the only
         code that moves is the transport layer.
    """


if __name__ == "__main__":
    from data import synthetic
    from data import features as F

    raw = synthetic.download_universe(["AAPL", "JPM", "XOM"], "2015-01-01", "2018-01-01")
    feats = {t: F.build_features(df).dropna() for t, df in raw.items()}
    closes = {t: raw[t]["close"] for t in raw}
    models = run_ddal(feats, closes, epochs=20, rollout_len=8)
    pos = predict_positions_garl(models, feats, closes)
    for t, s in pos.items():
        print(t, s.describe())
    print("OK GARL/DDAL train + predict")