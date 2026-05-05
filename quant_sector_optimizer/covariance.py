"""Covariance estimators.

ADDED: Ledoit-Wolf shrinkage (sklearn) — the default. The plain sample
covariance is rank-deficient when T < N and dangerously noisy when T ≈ N,
which is the regime of these notebooks (T ≈ 1260 days, N up to ~80 in
intra-sector windows).

A ``CovEstimator`` callable type lets callers swap estimators without changing
the optimizer signature.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

CovEstimator = Callable[[pd.DataFrame], np.ndarray]


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    """Plain sample covariance (no shrinkage). Use only for tests / diagnostics."""
    return returns.cov().values


def ledoit_wolf_cov(returns: pd.DataFrame, assume_centered: bool = False) -> np.ndarray:
    """Ledoit-Wolf 2004 shrinkage to ``(tr(S)/N)·I``.

    Returns the shrunk covariance as a NumPy array (assets in column order).
    The shrinkage intensity is computed analytically and is reproducible.
    """
    arr = returns.to_numpy(dtype=float, na_value=np.nan)
    if np.isnan(arr).any():
        raise ValueError("ledoit_wolf_cov requires a clean returns matrix (no NaN)")
    lw = LedoitWolf(assume_centered=assume_centered).fit(arr)
    return lw.covariance_


def get_estimator(name: Literal["ledoit_wolf", "sample"] | CovEstimator) -> CovEstimator:
    """Resolve a string alias or callable into a covariance estimator."""
    if callable(name):
        return name
    if name == "ledoit_wolf":
        return ledoit_wolf_cov
    if name == "sample":
        return sample_cov
    raise ValueError(f"Unknown covariance estimator: {name!r}")
