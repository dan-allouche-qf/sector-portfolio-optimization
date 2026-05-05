"""Backtesting engine.

REFACTOR (correctness): the previous ``compute_portfolio_cumulative_return``
forward-filled rebalance weights and computed ``Σ wᵢ·rᵢ`` every day. That is
mathematically equivalent to **rebalancing daily back to the target weights**
(volatility pumping). The advertised behavior was buy-and-hold between
rebalances, where weights drift with prices. This bias systematically
inflates returns. We now implement the correct semantics:

    1. At rebalance ``d_k``: convert NAV into per-asset notionals
       ``v_i = w_i · NAV(d_k-)``.
    2. Between ``d_k`` and ``d_{k+1}``: ``v_i(t) = v_i(t-1)·(1+r_i(t))``,
       NAV(t) = Σ v_i(t).
    3. At ``d_{k+1}``: rebalance ``v_i = w_i_new · NAV(d_{k+1}-)``.

REFACTOR (survivorship): the rebalance loop no longer calls ``dropna(axis=1)``
on the full universe. Instead, at each rebalance date we keep tickers that
have *complete* data over the **lookback window** only — past delistings or
late entrants neither inflate nor deflate other dates' results.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .covariance import CovEstimator, ledoit_wolf_cov
from .data import DEFAULT_WINDOW_DAYS, MIN_HISTORY_DAYS, TRADING_DAYS
from .optimizers import Strategy, optimize


# --- Weight schedule ---------------------------------------------------------


def compute_rebalance_weights(
    returns: pd.DataFrame,
    rebalance_dates: Iterable[pd.Timestamp],
    strategy: Strategy = "min_variance",
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_history: int = MIN_HISTORY_DAYS,
    cov_estimator: CovEstimator = ledoit_wolf_cov,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Optimal weights at each rebalance date using a rolling window.

    The estimation window is ``[date - window_days, date)`` — strictly before
    the rebalance date — to keep the backtest free of look-ahead bias.

    A ticker is eligible at date ``d`` only if it has **no NaN** in the window
    AND at least ``min_history`` observations. This is the per-window
    eligibility filter that replaces the global ``dropna``.
    """
    rebalance_dates = sorted(pd.DatetimeIndex(rebalance_dates))
    rows = []
    valid_dates = []

    for date in rebalance_dates:
        window = returns.loc[
            (returns.index >= date - pd.Timedelta(days=window_days))
            & (returns.index < date)
        ]
        if len(window) < min_history:
            continue
        eligible = window.dropna(axis=1, how="any")
        if eligible.shape[1] == 0:
            continue
        try:
            w = optimize(
                eligible,
                strategy=strategy,
                cov_estimator=cov_estimator,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        except Exception:
            continue
        rows.append(pd.Series(w, index=eligible.columns))
        valid_dates.append(date)

    weights_df = pd.DataFrame(rows, index=valid_dates).reindex(columns=returns.columns)
    weights_df.index.name = "Rebalance_Date"
    return weights_df.fillna(0.0)


# --- Buy-and-hold simulator --------------------------------------------------


def run_backtest(
    returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Buy-and-hold-between-rebalances simulation.

    Parameters
    ----------
    returns : Date × Ticker matrix of per-period returns (NaN allowed for
        unavailable assets — treated as 0 between rebalances).
    weights_df : weights at rebalance dates only (one row per rebalance, columns
        are tickers). Need not span every date.
    initial_value : NAV at the first rebalance date.

    Returns
    -------
    DataFrame with columns:
        nav        – portfolio NAV
        ret        – daily portfolio return
        cum_return – cumulative return = nav / initial_value - 1
        turnover   – L1 weight change at each rebalance date (0 elsewhere)
    """
    if weights_df.empty:
        return pd.DataFrame(columns=["nav", "ret", "cum_return", "turnover"])

    weights_df = weights_df.sort_index().fillna(0.0)
    rebalance_dates = list(weights_df.index)
    start = rebalance_dates[0]
    sim_returns = returns.loc[start:].copy()
    if sim_returns.empty:
        return pd.DataFrame(columns=["nav", "ret", "cum_return", "turnover"])

    cols = sim_returns.columns
    weights_aligned = weights_df.reindex(columns=cols).fillna(0.0)
    ret_arr = sim_returns.to_numpy(dtype=float)
    np.nan_to_num(ret_arr, copy=False, nan=0.0)
    dates = sim_returns.index

    rebalance_set = set(rebalance_dates)
    n = len(cols)
    nav = np.empty(len(dates))
    daily_ret = np.zeros(len(dates))
    turnover_arr = np.zeros(len(dates))

    holdings = np.zeros(n)
    prev_nav = initial_value

    for t, date in enumerate(dates):
        if date in rebalance_set:
            target = weights_aligned.loc[date].to_numpy(dtype=float)
            current_weights = holdings / prev_nav if prev_nav > 0 else np.zeros(n)
            turnover_arr[t] = float(np.abs(target - current_weights).sum())
            holdings = target * prev_nav
        holdings = holdings * (1.0 + ret_arr[t])
        current_nav = float(holdings.sum())
        nav[t] = current_nav
        daily_ret[t] = (current_nav / prev_nav) - 1.0 if prev_nav > 0 else 0.0
        prev_nav = current_nav

    out = pd.DataFrame(
        {
            "nav": nav,
            "ret": daily_ret,
            "cum_return": nav / initial_value - 1.0,
            "turnover": turnover_arr,
        },
        index=dates,
    )
    return out


# --- Convenience: run multiple strategies at once ----------------------------


def run_strategies(
    returns: pd.DataFrame,
    rebalance_dates: Iterable[pd.Timestamp],
    strategies: Iterable[Strategy] = ("min_variance", "max_sharpe", "max_diversification", "hrp"),
    **kwargs,
) -> Mapping[str, pd.DataFrame]:
    """Run several strategies on the same universe and return their backtests.

    Returns a dict ``{strategy_name: backtest_df}`` (each ``backtest_df`` has
    the columns described in :func:`run_backtest`).
    """
    rebalance_dates = list(rebalance_dates)
    out: dict[str, pd.DataFrame] = {}
    for s in strategies:
        weights = compute_rebalance_weights(
            returns, rebalance_dates, strategy=s, **kwargs
        )
        out[s] = run_backtest(returns, weights)
    return out
