"""Optimizer invariants and analytic checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_sector_optimizer import (
    hrp,
    ledoit_wolf_cov,
    max_diversification,
    max_sharpe,
    min_variance,
    optimize,
    sample_cov,
)


# --- Long-only / sum-to-one invariants --------------------------------------


def _check_simplex(w: np.ndarray, tol: float = 1e-6) -> None:
    assert w.shape[0] > 0
    assert np.all(w >= -tol), f"weights have a negative entry: {w}"
    assert abs(w.sum() - 1.0) < tol, f"weights don't sum to one: {w.sum()}"


def test_min_variance_simplex(synthetic_returns):
    cov = ledoit_wolf_cov(synthetic_returns)
    w = min_variance(cov)
    _check_simplex(w)


def test_max_sharpe_simplex(synthetic_returns):
    cov = ledoit_wolf_cov(synthetic_returns)
    w = max_sharpe(synthetic_returns.mean().values, cov, risk_free_rate=0.02)
    _check_simplex(w)


def test_max_diversification_simplex(synthetic_returns):
    cov = ledoit_wolf_cov(synthetic_returns)
    w = max_diversification(cov)
    _check_simplex(w)


def test_hrp_simplex(synthetic_returns):
    w = hrp(synthetic_returns)
    _check_simplex(w)


# --- Min variance: analytic 2-asset benchmark -------------------------------


def test_min_variance_two_asset_analytic(two_asset_returns):
    """For two assets with cov [[a,c],[c,b]], the min-variance long-only
    weight on X is clipped to [0,1] of (b-c)/(a+b-2c)."""
    cov = sample_cov(two_asset_returns)
    a, b, c = cov[0, 0], cov[1, 1], cov[0, 1]
    w_x = (b - c) / (a + b - 2 * c)
    w_x = float(np.clip(w_x, 0.0, 1.0))
    w = min_variance(cov)
    assert abs(w[0] - w_x) < 1e-4
    assert abs(w[1] - (1 - w_x)) < 1e-4


# --- Max Sharpe: optimum beats equal weight ---------------------------------


def test_max_sharpe_beats_equal_weight(synthetic_returns):
    cov = ledoit_wolf_cov(synthetic_returns)
    mu = synthetic_returns.mean().values
    w_opt = max_sharpe(mu, cov, risk_free_rate=0.02)
    n = len(mu)
    w_eq = np.full(n, 1.0 / n)

    def sharpe(w):
        port_mu = float(w @ mu) * 252 - 0.02
        port_vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(252)
        return port_mu / port_vol

    assert sharpe(w_opt) >= sharpe(w_eq) - 1e-9


def test_max_sharpe_degenerate_falls_back(synthetic_returns):
    """If no asset has positive excess return, fall back to min-variance silently."""
    cov = ledoit_wolf_cov(synthetic_returns)
    mu = np.full(synthetic_returns.shape[1], -0.01)  # all negative
    w = max_sharpe(mu, cov, risk_free_rate=0.02)
    w_mv = min_variance(cov)
    np.testing.assert_allclose(w, w_mv, atol=1e-6)


# --- Max diversification: D >= 1 -------------------------------------------


def test_max_diversification_ratio_at_least_one(synthetic_returns):
    cov = sample_cov(synthetic_returns)
    sigma = np.sqrt(np.diag(cov))
    w = max_diversification(cov)
    d = float(w @ sigma) / np.sqrt(float(w @ cov @ w))
    assert d >= 1.0 - 1e-6  # diversification ratio is always >= 1


# --- HRP: dispersion ---------------------------------------------------------


def test_hrp_assigns_all_assets_positive_weight(synthetic_returns):
    """HRP never zeroes out an asset (unlike min-variance under noise)."""
    w = hrp(synthetic_returns)
    assert np.all(w > 0)


def test_hrp_single_asset():
    s = pd.DataFrame(np.random.normal(size=(100, 1)) * 0.01, columns=["only"])
    w = hrp(s)
    np.testing.assert_allclose(w, [1.0])


# --- Dispatcher --------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["min_variance", "max_sharpe", "max_diversification", "hrp"])
def test_optimize_dispatcher(synthetic_returns, strategy):
    w = optimize(synthetic_returns, strategy=strategy, risk_free_rate=0.02)
    _check_simplex(np.asarray(w))
