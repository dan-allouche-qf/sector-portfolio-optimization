"""Long-only portfolio optimizers.

REFACTOR: optimizers now operate on a *precomputed* covariance matrix (and
mean vector for max-Sharpe). This decouples covariance estimation from
optimization — the same optimizer can be run with sample, Ledoit-Wolf, or any
custom estimator.

REFACTOR (max_sharpe): the previous direct SLSQP minimization of -Sharpe is
non-convex (fractional programming) and gets stuck in local optima from a
single equal-weight start. We now solve the equivalent **convex QP** via the
Schur transform (Cornuéjols & Tütüncü):

    minimize    y' Σ y
    subject to  (μ - rf · 1)' y = 1
                y >= 0
    return      w = y / sum(y)

This is mathematically equivalent and *globally* solvable. When no asset has
positive excess return, max-Sharpe is degenerate and we fall back to
min-variance.

ADDED: ``hrp`` (Hierarchical Risk Parity, López de Prado 2016) — robust when
N is large relative to T because it never inverts Σ.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .covariance import CovEstimator, ledoit_wolf_cov

_SLSQP_OPTIONS = {"ftol": 1e-10, "maxiter": 500}
_FLOOR = 1e-12


# --- Helpers -----------------------------------------------------------------


def _equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _long_only_constraints(n: int, w_max: float = 1.0):
    bounds = [(0.0, w_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def _as_array(cov) -> np.ndarray:
    if isinstance(cov, pd.DataFrame):
        return cov.to_numpy(dtype=float)
    return np.asarray(cov, dtype=float)


# --- Min variance ------------------------------------------------------------


def min_variance(cov, w_max: float = 1.0) -> np.ndarray:
    """Long-only weights minimizing ``w' Σ w``, sum-to-one.

    Min-variance with linear constraints is a convex QP — SLSQP from
    equal-weight reaches the global optimum.
    """
    cov = _as_array(cov)
    n = cov.shape[0]
    bounds, constraints = _long_only_constraints(n, w_max=w_max)
    res = minimize(
        lambda w: float(w @ cov @ w),
        _equal_weights(n),
        jac=lambda w: 2.0 * cov @ w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SLSQP_OPTIONS,
    )
    return res.x


# --- Max Sharpe (convex Schur reformulation) ---------------------------------


def max_sharpe(
    mu: np.ndarray | pd.Series,
    cov,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Long-only weights maximizing the annualized Sharpe ratio.

    Parameters
    ----------
    mu : per-period mean returns (e.g. daily). Annualized internally.
    cov : per-period covariance matrix. Annualized internally.
    risk_free_rate : annualized risk-free rate.

    Returns
    -------
    Long-only weights summing to one. If no asset has positive annualized
    excess return, returns the min-variance solution and emits no error
    (max-Sharpe is undefined in that case).
    """
    cov = _as_array(cov)
    mu = np.asarray(mu, dtype=float).ravel()
    n = cov.shape[0]
    if mu.shape[0] != n:
        raise ValueError(f"mu length {mu.shape[0]} != cov size {n}")

    excess = mu * periods_per_year - risk_free_rate
    cov_ann = cov * periods_per_year

    if not np.any(excess > _FLOOR):
        return min_variance(cov)

    # Schur trick: minimize y' Σ y s.t. excess'·y = 1, y >= 0; w = y/sum(y).
    bounds = [(0.0, None)] * n
    constraints = [{"type": "eq", "fun": lambda y, e=excess: float(e @ y) - 1.0}]
    # Initial point: scale equal-weights so excess'·y0 = 1 (when feasible).
    e_dot_eq = float(excess.mean())
    y0 = (np.ones(n) / n) / max(e_dot_eq, _FLOOR) if e_dot_eq > 0 else np.full(n, 1.0 / max(excess.max(), _FLOOR) / n)
    y0 = np.maximum(y0, 0.0)

    res = minimize(
        lambda y: float(y @ cov_ann @ y),
        y0,
        jac=lambda y: 2.0 * cov_ann @ y,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SLSQP_OPTIONS,
    )
    y = np.maximum(res.x, 0.0)
    s = y.sum()
    if s < _FLOOR:
        return min_variance(cov)
    return y / s


# --- Max diversification -----------------------------------------------------


def max_diversification(cov) -> np.ndarray:
    """Long-only weights maximizing Choueifaty's diversification ratio
    ``D(w) = (w' σ) / sqrt(w' Σ w)``.

    The numerator is linear in w and the denominator is convex; the ratio is
    *quasi-concave*, so SLSQP converges reliably from equal-weight.
    """
    cov = _as_array(cov)
    n = cov.shape[0]
    sigma = np.sqrt(np.diag(cov))
    bounds, constraints = _long_only_constraints(n)

    def neg_div(w):
        port_var = float(w @ cov @ w)
        if port_var < _FLOOR:
            return 1e6
        return -float(w @ sigma) / np.sqrt(port_var)

    res = minimize(
        neg_div,
        _equal_weights(n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SLSQP_OPTIONS,
    )
    return res.x


# --- Hierarchical Risk Parity ------------------------------------------------


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))


def _quasi_diagonal_order(link: np.ndarray, n_items: int) -> list[int]:
    """López de Prado's quasi-diag: tree order from a SciPy ``linkage`` matrix."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= n_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n_items]
        i = df0.index
        j = df0.values - n_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    inv_diag = 1.0 / np.diag(sub)
    w = inv_diag / inv_diag.sum()
    return float(w @ sub @ w)


def hrp(returns: pd.DataFrame) -> np.ndarray:
    """Hierarchical Risk Parity weights (López de Prado 2016).

    Pipeline:
      1. correlation matrix from returns,
      2. distance ``d_ij = sqrt(0.5·(1 - ρ_ij))``,
      3. single-linkage hierarchical clustering,
      4. quasi-diagonalize,
      5. recursive bisection: split, allocate inversely to cluster variance.

    HRP never inverts Σ, so it is stable when T ≤ N or Σ is near-singular.
    """
    if returns.shape[1] == 0:
        return np.array([])
    if returns.shape[1] == 1:
        return np.array([1.0])

    cov = returns.cov().values
    corr = returns.corr().values
    n = corr.shape[0]
    dist = _correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)
    cond = squareform(dist, checks=False)
    link = linkage(cond, method="single")
    order = _quasi_diagonal_order(link, n)

    weights = np.ones(n)
    clusters = [order]
    while clusters:
        new_clusters = []
        for items in clusters:
            if len(items) <= 1:
                continue
            mid = len(items) // 2
            c1, c2 = items[:mid], items[mid:]
            v1 = _cluster_variance(cov, c1)
            v2 = _cluster_variance(cov, c2)
            alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
            for k in c1:
                weights[k] *= alpha
            for k in c2:
                weights[k] *= 1.0 - alpha
            new_clusters.extend([c1, c2])
        clusters = new_clusters

    return weights / weights.sum()


# --- Dispatcher --------------------------------------------------------------


Strategy = Literal["min_variance", "max_sharpe", "max_diversification", "hrp"]


def optimize(
    returns: pd.DataFrame,
    strategy: Strategy = "min_variance",
    cov_estimator: CovEstimator = ledoit_wolf_cov,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Run a single strategy and return weights aligned with ``returns.columns``.

    For ``hrp`` the covariance estimator is ignored (HRP uses sample correlation
    and cluster variance directly).
    """
    if strategy == "hrp":
        return hrp(returns)
    cov = cov_estimator(returns)
    if strategy == "min_variance":
        return min_variance(cov)
    if strategy == "max_sharpe":
        return max_sharpe(returns.mean().values, cov, risk_free_rate, periods_per_year)
    if strategy == "max_diversification":
        return max_diversification(cov)
    raise ValueError(f"Unknown strategy: {strategy!r}")
