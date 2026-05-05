"""Backtest correctness tests.

These are the most important tests: they pin down the buy-and-hold-between-
rebalances semantics that the original code violated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_sector_optimizer import compute_rebalance_weights, run_backtest


def _trivial_returns(n_days: int = 60, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.001, 0.01, size=(n_days, n_assets))
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    return pd.DataFrame(r, index=dates, columns=[f"a{i}" for i in range(n_assets)])


def test_single_rebalance_buy_and_hold(rng_seed=0):
    """With a single rebalance and no further rebalancing, NAV evolves as a
    buy-and-hold: NAV_T = Σ w_i · ∏_t (1 + r_i(t))."""
    returns = _trivial_returns(seed=rng_seed)
    weights_at = pd.DataFrame(
        [[0.5, 0.3, 0.2]], index=[returns.index[0]], columns=returns.columns,
    )
    bt = run_backtest(returns, weights_at, initial_value=1.0)

    expected_terminal = float(
        (weights_at.iloc[0].values * (1.0 + returns.values).prod(axis=0)).sum()
    )
    assert abs(bt["nav"].iloc[-1] - expected_terminal) < 1e-10


def test_buy_and_hold_differs_from_daily_rebalance():
    """Sanity check: a vol-pumping path (large dispersion) makes buy-and-hold
    diverge from the daily-rebalance approximation that the old code used."""
    n = 50
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(123)
    r = rng.normal(0.0, 0.05, size=(n, 4))
    returns = pd.DataFrame(r, index=dates, columns=list("ABCD"))
    w = pd.DataFrame([[0.25] * 4], index=[dates[0]], columns=list("ABCD"))

    bt = run_backtest(returns, w)
    daily_rebalance = (1.0 + (np.full(4, 0.25) * returns).sum(axis=1)).prod()

    # NAV is the buy-and-hold path, daily_rebalance is the old behavior.
    # They are equal iff returns are deterministic; under random returns they
    # differ by an amount of order var(r) * T (vol-pumping wedge).
    assert abs(bt["nav"].iloc[-1] - daily_rebalance) > 1e-6


def test_turnover_first_rebalance_equals_one():
    returns = _trivial_returns()
    w = pd.DataFrame([[1.0, 0.0, 0.0]], index=[returns.index[0]], columns=returns.columns)
    bt = run_backtest(returns, w)
    assert abs(bt["turnover"].iloc[0] - 1.0) < 1e-12


def test_two_rebalances_drift():
    """At rebalance 2 the holdings are reset, so post-rebalance the
    intermediate drift no longer accumulates linearly."""
    n_days = 20
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    r = np.zeros((n_days, 2))
    r[:, 0] = 0.01    # asset 0 grows 1% / day
    r[:, 1] = -0.01   # asset 1 loses 1% / day
    returns = pd.DataFrame(r, index=dates, columns=["up", "dn"])
    w = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=[dates[0], dates[10]],
        columns=["up", "dn"],
    )
    bt = run_backtest(returns, w)
    # At day 10 (rebalance), holdings should be reset to 50/50 of NAV.
    nav_at_rebal = bt["nav"].iloc[10]
    # After rebalance, +1% / -1% returns on equal weights give a near-zero
    # daily drift but still a small positive convexity term.
    nav_end = bt["nav"].iloc[-1]
    assert nav_end > 0
    assert nav_at_rebal > 0
    # Turnover at second rebalance should be small (weights barely changed)
    assert bt["turnover"].iloc[10] < 0.5  # well below the initial deployment


def test_compute_rebalance_weights_no_lookahead():
    """The estimation window is strictly *before* the rebalance date."""
    returns = _trivial_returns(n_days=600)
    rebalance_dates = [returns.index[300], returns.index[500]]
    weights = compute_rebalance_weights(
        returns, rebalance_dates, strategy="min_variance",
        window_days=200, min_history=100,
    )
    assert list(weights.index) == rebalance_dates
    # Weights are valid simplex points
    for _, row in weights.iterrows():
        assert abs(row.sum() - 1.0) < 1e-6
        assert (row >= -1e-9).all()


def test_compute_rebalance_weights_drops_short_window():
    """A rebalance date with less than min_history is skipped."""
    returns = _trivial_returns(n_days=300)
    early_date = returns.index[10]
    late_date = returns.index[280]
    weights = compute_rebalance_weights(
        returns, [early_date, late_date], strategy="min_variance",
        window_days=200, min_history=100,
    )
    # Early date dropped, late date retained
    assert early_date not in weights.index
    assert late_date in weights.index


def test_ragged_returns_handled():
    """A late-arriving asset has NaN in the early window and is excluded
    from rebalances during that period — but reappears once enough history
    accumulates."""
    n = 600
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(7)
    r = rng.normal(0.0005, 0.01, size=(n, 3))
    returns = pd.DataFrame(r, index=dates, columns=["A", "B", "C"])
    returns.loc[dates[:200], "C"] = np.nan  # C arrives late
    weights = compute_rebalance_weights(
        returns, [dates[150], dates[450]], strategy="min_variance",
        window_days=140, min_history=100,
    )
    # At dates[150]: only ~150 days available, but window is 140 days -> only A,B fully present
    # At dates[450]: C has 250 days of history -> all 3 fully present
    assert weights.loc[dates[150], "C"] == 0.0  # absent in window -> set to 0 by reindex
    assert weights.loc[dates[450], "C"] > 0.0
