"""Shared pytest fixtures: synthetic returns with known statistical properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_returns(rng):
    """Five-asset Gaussian returns, 2-year history, mild positive drift.

    Designed so that:
      - the covariance is well-conditioned (no synthetic singular case),
      - ``mu - rf > 0`` for at least one asset (max-Sharpe is well-defined).
    """
    n_days, n_assets = 2 * 252, 5
    mu_daily = np.array([0.0008, 0.0006, 0.0010, 0.0004, 0.0005])
    base_vol = np.array([0.012, 0.014, 0.020, 0.010, 0.013])
    L = np.array(
        [
            [1.00, 0.30, 0.20, 0.10, 0.05],
            [0.30, 1.00, 0.25, 0.15, 0.10],
            [0.20, 0.25, 1.00, 0.05, 0.05],
            [0.10, 0.15, 0.05, 1.00, 0.20],
            [0.05, 0.10, 0.05, 0.20, 1.00],
        ]
    )
    chol = np.linalg.cholesky(L)
    z = rng.standard_normal((n_days, n_assets))
    eps = z @ chol.T
    returns = mu_daily + base_vol * eps
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, index=dates, columns=cols)


@pytest.fixture
def two_asset_returns(rng):
    """Two-asset returns with hand-tunable means and known covariance."""
    n = 1000
    mu = np.array([0.001, 0.0005])
    sd = np.array([0.01, 0.02])
    rho = 0.3
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])
    z = rng.standard_normal((n, 2))
    eps = z @ L.T
    returns = mu + sd * eps
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(returns, index=dates, columns=["X", "Y"])
