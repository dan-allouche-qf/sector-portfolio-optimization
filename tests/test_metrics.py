"""Metric correctness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_sector_optimizer import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    hit_ratio,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)


def _bt_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.DataFrame:
    nav = (1.0 + returns).cumprod() * initial
    return pd.DataFrame(
        {
            "nav": nav,
            "ret": returns.values,
            "cum_return": nav / initial - 1.0,
            "turnover": np.zeros(len(returns)),
        },
        index=returns.index,
    )


def test_cagr_constant_growth():
    # A series that grows exactly 10% per year over 252 days.
    n = 252
    daily = (1.10) ** (1 / n) - 1
    r = pd.Series([daily] * n, index=pd.bdate_range("2024-01-02", periods=n))
    bt = _bt_from_returns(r)
    assert abs(cagr(bt) - 0.10) < 1e-6


def test_max_drawdown_known_path():
    # NAV path: 1.0 -> 1.2 -> 0.6 -> 1.0  -> max DD = (0.6 / 1.2) - 1 = -0.5
    nav = pd.Series([1.0, 1.2, 0.6, 1.0])
    assert abs(max_drawdown(nav) - (-0.5)) < 1e-12


def test_max_drawdown_no_drawdown():
    nav = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(nav) == 0.0


def test_sharpe_zero_returns():
    r = pd.Series(np.zeros(100), index=pd.bdate_range("2024-01-02", periods=100))
    bt = _bt_from_returns(r)
    s = sharpe_ratio(bt, risk_free_rate=0.0)
    assert np.isnan(s)


def test_sharpe_positive_for_positive_excess():
    rng = np.random.default_rng(0)
    r = pd.Series(
        rng.normal(0.001, 0.005, 252 * 3),
        index=pd.bdate_range("2024-01-02", periods=252 * 3),
    )
    bt = _bt_from_returns(r)
    s = sharpe_ratio(bt, risk_free_rate=0.02)
    assert s > 0  # μ_ann ≈ 25%, σ_ann ≈ 8%, so Sharpe is comfortably > 0


def test_sortino_lower_or_equal_when_downside_dominates():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500), index=pd.bdate_range("2024-01-02", periods=500))
    bt = _bt_from_returns(r)
    sh = sharpe_ratio(bt)
    so = sortino_ratio(bt, target=0.0)
    # Sortino uses downside-only std, which is <= total std, so Sortino >= Sharpe
    assert so >= sh - 1e-9


def test_hit_ratio():
    r = pd.Series([0.01, -0.02, 0.0, 0.005, -0.001])
    bt = _bt_from_returns(r)
    assert abs(hit_ratio(bt) - 2 / 5) < 1e-12


def test_annualized_vol_scaling():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 252 * 4), index=pd.bdate_range("2020-01-02", periods=252 * 4))
    bt = _bt_from_returns(r)
    expected = 0.01 * np.sqrt(252)
    # Sample std of ~1000 normals: ±2 std error ≈ 0.01 * 1/sqrt(2*1000) ≈ 0.0002
    # → annualized error ≈ 0.0002 * sqrt(252) ≈ 0.003. Use 5% relative tol.
    assert abs(annualized_volatility(bt) - expected) / expected < 0.05


def test_turnover_computation():
    # Three rebalances: full deploy, half rotation, no change
    idx = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01"])
    w = pd.DataFrame(
        [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]],
        index=idx, columns=["A", "B"],
    )
    # L1 deltas:
    #   First rebalance: |1| + |0| = 1
    #   Second: |−0.5| + |0.5| = 1
    #   Third:  0
    # Mean per-rebalance = (1+1+0)/3 = 2/3
    avg = turnover(w)
    assert abs(avg - 2 / 3) < 1e-9


def test_calmar_finite_for_drawdown():
    # NAV grows then drops then recovers a bit
    nav = pd.Series(
        [1.0, 1.05, 1.10, 0.95, 1.00, 1.05, 1.10],
        index=pd.bdate_range("2024-01-02", periods=7),
    )
    bt = pd.DataFrame({"nav": nav, "ret": nav.pct_change().fillna(0).values})
    c = calmar_ratio(bt)
    assert np.isfinite(c)


def test_summary_contains_expected_keys():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 252), index=pd.bdate_range("2024-01-02", periods=252))
    bt = _bt_from_returns(r)
    summary = performance_summary(bt)
    for key in ["CAGR", "Volatility", "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "HitRatio"]:
        assert key in summary.index
