"""Data prep tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_sector_optimizer import (
    clean_ohlc,
    pivot_returns,
    prepare_returns,
    winsorize_returns,
)


def test_prepare_returns_pct_change():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "Close": [100.0, 110.0, 99.0, 50.0, 55.0, 60.0],
            "Ticker": ["X", "X", "X", "Y", "Y", "Y"],
        }
    )
    out = prepare_returns(df)
    # First date per-ticker dropped after pct_change
    assert len(out) == 4
    x = out[out["Ticker"] == "X"]["Daily_Return"].tolist()
    assert abs(x[0] - 0.10) < 1e-12
    assert abs(x[1] - (-0.10)) < 1e-12


def test_pivot_returns_preserves_nans():
    """Critical: the pivot must NOT drop columns with partial history.
    Dropping them would produce survivorship bias."""
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "Daily_Return": [0.01, 0.02, np.nan, 0.0, np.nan, np.nan],
            "Ticker": ["X", "X", "X", "Y", "Y", "Y"],
        }
    )
    wide = pivot_returns(df)
    assert "X" in wide.columns
    assert "Y" in wide.columns
    assert wide["Y"].isna().sum() == 2


def test_winsorize_caps_extremes():
    s = pd.Series([0.0] * 99 + [10.0])
    capped = winsorize_returns(s, n_sigma=3.0)
    assert capped.iloc[-1] < 10.0
    # Values within 3 sigma should be unchanged
    assert (capped.iloc[:99] == 0.0).all()


def test_winsorize_constant_series_unchanged():
    s = pd.Series([0.01] * 100)
    capped = winsorize_returns(s)
    pd.testing.assert_series_equal(capped, s)


def test_clean_ohlc_audit():
    df = pd.DataFrame(
        {
            "Open": [10.0, 20.0, -1.0, 30.0],
            "High": [12.0, 18.0, 0.0, 35.0],   # row 1: High < max(O,C); row 3: legit
            "Low":  [9.0, 17.0, -2.0, 28.0],
            "Close":[11.0, 19.0, -3.0, 33.0],
            "Volume":[1, 2, 3, 4],
            "Ticker":["A","A","A","A"],
            "Date": pd.to_datetime(["2024-01-02","2024-01-03","2024-01-04","2024-01-05"]),
        }
    )
    cleaned, audit = clean_ohlc(df)
    assert audit["removed_non_positive"] == 1
    # Row index 1: High was 18 but max(20,18,19) = 20, so it's a fix
    assert audit["high_fixes"] >= 1
    assert (cleaned["High"] >= cleaned[["Open", "Close"]].max(axis=1)).all()
    assert (cleaned["Low"] <= cleaned[["Open", "Close"]].min(axis=1)).all()
