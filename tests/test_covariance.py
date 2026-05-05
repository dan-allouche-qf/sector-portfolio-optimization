"""Covariance estimator tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_sector_optimizer import ledoit_wolf_cov, sample_cov
from quant_sector_optimizer.covariance import get_estimator


def test_sample_cov_matches_pandas(synthetic_returns):
    cov = sample_cov(synthetic_returns)
    np.testing.assert_allclose(cov, synthetic_returns.cov().values)


def test_ledoit_wolf_psd(synthetic_returns):
    cov = ledoit_wolf_cov(synthetic_returns)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-12)


def test_ledoit_wolf_singular_when_T_lt_N():
    """If T < N, the sample covariance is singular (rank deficient).
    Ledoit-Wolf should still produce a strictly positive-definite estimate."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n_assets, n_days = 30, 20
    r = rng.normal(0.0, 0.01, size=(n_days, n_assets))
    df = pd.DataFrame(r, columns=[f"a{i}" for i in range(n_assets)])
    sample = sample_cov(df)
    assert np.linalg.matrix_rank(sample) < n_assets  # singular
    lw = ledoit_wolf_cov(df)
    eigvals = np.linalg.eigvalsh(lw)
    assert eigvals.min() > 0  # strictly PD


def test_get_estimator_aliases():
    assert get_estimator("ledoit_wolf") is ledoit_wolf_cov
    assert get_estimator("sample") is sample_cov

    f = lambda r: r.cov().values  # noqa: E731
    assert get_estimator(f) is f

    with pytest.raises(ValueError):
        get_estimator("not_a_real_estimator")
