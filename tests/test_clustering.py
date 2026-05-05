"""Clustering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_sector_optimizer import cluster_by_correlation, hrp_cluster_labels


def _two_block_returns(rng_seed: int = 0, n: int = 500):
    """Six tickers in two clear correlation blocks of three."""
    rng = np.random.default_rng(rng_seed)
    factor1 = rng.normal(0.0, 0.01, n)
    factor2 = rng.normal(0.0, 0.01, n)
    noise = rng.normal(0.0, 0.005, (n, 6))
    block1 = factor1[:, None] * 1.0 + noise[:, :3]
    block2 = factor2[:, None] * 1.0 + noise[:, 3:]
    r = np.hstack([block1, block2])
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame(r, index=dates, columns=[f"a{i}" for i in range(6)])


def test_correlation_kmeans_recovers_blocks():
    returns = _two_block_returns()
    labels = cluster_by_correlation(returns, n_clusters=2)
    block1_labels = set(labels.iloc[:3].tolist())
    block2_labels = set(labels.iloc[3:].tolist())
    assert len(block1_labels) == 1
    assert len(block2_labels) == 1
    assert block1_labels != block2_labels


def test_hrp_cluster_labels_recovers_blocks():
    returns = _two_block_returns(rng_seed=1)
    labels = hrp_cluster_labels(returns, n_clusters=2)
    assert labels.iloc[0] == labels.iloc[1] == labels.iloc[2]
    assert labels.iloc[3] == labels.iloc[4] == labels.iloc[5]
    assert labels.iloc[0] != labels.iloc[3]


def test_clustering_too_many_clusters_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("ABC"))
    with pytest.raises(ValueError):
        cluster_by_correlation(df, n_clusters=5)
