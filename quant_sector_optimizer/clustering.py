"""Clustering helpers.

Tickers are clustered in **correlation-distance** space
``d_ij = sqrt(0.5·(1 - ρ_ij))`` — a true metric on [0, 1] that treats two
assets as close when their co-movement pattern is similar. This is also the
distance HRP uses internally.

Standardizing the raw return series with ``StandardScaler`` and clustering on
those vectors is *not* used here: it destroys the volatility information that
should drive the grouping and produces a distance with little economic
meaning.

``hrp_cluster_labels`` exposes the cluster assignment derived from the same
single-linkage tree used by the HRP optimizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from .optimizers import _correlation_distance


def cluster_by_correlation(
    returns: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> pd.Series:
    """K-Means in correlation-distance space.

    Each ticker is represented by its row of correlation distances against
    every other ticker (an N-dim vector that already encodes all pairwise
    relationships). K-Means then partitions the universe.

    Returns a Series of cluster ids indexed by ticker.
    """
    returns = returns.dropna(axis=1)
    if returns.shape[1] < n_clusters:
        raise ValueError(
            f"Need at least n_clusters={n_clusters} tickers, got {returns.shape[1]}"
        )
    corr = returns.corr().values
    dist = _correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(dist)
    return pd.Series(labels, index=returns.columns, name="Cluster")


def hrp_cluster_labels(returns: pd.DataFrame, n_clusters: int) -> pd.Series:
    """Cluster ids from cutting the HRP single-linkage dendrogram.

    Useful when one wants the *same* tree HRP uses internally for allocation,
    but as a labeling for inspection or post-hoc grouping.
    """
    returns = returns.dropna(axis=1)
    if returns.shape[1] < n_clusters:
        raise ValueError(
            f"Need at least n_clusters={n_clusters} tickers, got {returns.shape[1]}"
        )
    corr = returns.corr().values
    dist = _correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)
    cond = squareform(dist, checks=False)
    link = linkage(cond, method="single")
    labels = fcluster(link, t=n_clusters, criterion="maxclust")
    return pd.Series(labels, index=returns.columns, name="Cluster")
