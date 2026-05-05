"""Sector-based portfolio analysis and optimization toolkit.

The package is organized in single-responsibility modules:

    data        – I/O, daily-return computation, outlier handling
    covariance  – sample and Ledoit-Wolf shrinkage estimators
    optimizers  – long-only Markowitz (min-var, max-Sharpe convex, max-div) + HRP
    backtest    – ragged-data, buy-and-hold-between-rebalances simulator
    metrics     – CAGR, Sharpe, Sortino, max drawdown, Calmar, hit ratio, turnover
    clustering  – correlation-distance K-Means and HRP cluster labels
    plotting    – Plotly figures (price, returns, performance, weights)

Default constants (TRADING_DAYS, DEFAULT_RISK_FREE_RATE) live in ``data``.
"""

from __future__ import annotations

from .data import (
    TRADING_DAYS,
    DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_WINDOW_DAYS,
    MIN_HISTORY_DAYS,
    load_panel,
    save_panel,
    prepare_returns,
    pivot_returns,
    winsorize_returns,
    clean_ohlc,
    fix_missing_date,
)
from .covariance import sample_cov, ledoit_wolf_cov, CovEstimator
from .optimizers import (
    min_variance,
    max_sharpe,
    max_diversification,
    hrp,
    optimize,
)
from .backtest import (
    compute_rebalance_weights,
    run_backtest,
)
from .metrics import (
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    hit_ratio,
    turnover,
    performance_summary,
)
from .clustering import cluster_by_correlation, hrp_cluster_labels

__all__ = [
    # constants
    "TRADING_DAYS",
    "DAYS_PER_YEAR",
    "DEFAULT_RISK_FREE_RATE",
    "DEFAULT_WINDOW_DAYS",
    "MIN_HISTORY_DAYS",
    # data
    "load_panel",
    "save_panel",
    "prepare_returns",
    "pivot_returns",
    "winsorize_returns",
    "clean_ohlc",
    "fix_missing_date",
    # covariance
    "sample_cov",
    "ledoit_wolf_cov",
    "CovEstimator",
    # optimizers
    "min_variance",
    "max_sharpe",
    "max_diversification",
    "hrp",
    "optimize",
    # backtest
    "compute_rebalance_weights",
    "run_backtest",
    # metrics
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_ratio",
    "turnover",
    "performance_summary",
    # clustering
    "cluster_by_correlation",
    "hrp_cluster_labels",
]

__version__ = "0.2.0"
