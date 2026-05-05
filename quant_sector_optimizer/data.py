"""Data I/O and preparation.

REFACTOR: split out from the original ``portfolio_utils`` to isolate I/O from
modeling. Pickle is replaced by Parquet (smaller, faster, schema-aware) but
``load_panel`` still falls back to ``.pkl`` so legacy artifacts keep working.

REMOVED: the silent High/Low rewrite in ``clean_ohlc`` — corrections are now
returned alongside the cleaned frame so callers can audit them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25
DEFAULT_RISK_FREE_RATE = 0.02
MIN_HISTORY_DAYS = TRADING_DAYS
DEFAULT_WINDOW_DAYS = 5 * 365  # rolling estimation window
PRICE_COLUMNS = ("Open", "High", "Low", "Close")


# --- I/O ---------------------------------------------------------------------


def load_panel(path: str | Path) -> pd.DataFrame:
    """Load a long-format OHLCV panel from Parquet (preferred) or Pickle.

    The expected schema is::

        Date (datetime64[ns]) | Open | High | Low | Close | Volume |
        Ticker (str) | Sector (str) | Category (str)
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".pkl":
        df = pd.read_pickle(path)
    elif path.exists():
        df = pd.read_parquet(path)
    else:
        # Fall back: try parquet first, then pickle, with the same stem
        for ext in (".parquet", ".pkl"):
            candidate = path.with_suffix(ext)
            if candidate.exists():
                return load_panel(candidate)
        raise FileNotFoundError(f"No panel found at {path}")
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def save_panel(df: pd.DataFrame, path: str | Path) -> None:
    """Save a panel as Parquet (Snappy by default, ~5–10x smaller than pickle)."""
    path = Path(path)
    if path.suffix not in (".parquet", ".pkl"):
        path = path.with_suffix(".parquet")
    if path.suffix == ".parquet":
        df.to_parquet(path, compression="snappy", index=False)
    else:
        df.to_pickle(path)


# --- Cleaning ----------------------------------------------------------------


def clean_ohlc(
    df: pd.DataFrame,
    price_columns: Iterable[str] = PRICE_COLUMNS,
    extreme_threshold: float = 1e5,
) -> tuple[pd.DataFrame, dict]:
    """Clean an OHLC panel and return ``(cleaned_df, audit)``.

    Steps:
      1. Drop rows with non-positive prices.
      2. Drop rows above ``extreme_threshold`` (data-error guard).
      3. Repair OHLC hierarchy: enforce ``High = max(O,H,C)`` and
         ``Low = min(O,L,C)``. The corrections are *logged* and reported in
         ``audit``, no longer applied silently.
    """
    price_columns = list(price_columns)
    audit = {"removed_non_positive": 0, "removed_extreme": 0, "high_fixes": 0, "low_fixes": 0}

    mask_neg = (df[price_columns] <= 0).any(axis=1)
    if mask_neg.any():
        audit["removed_non_positive"] = int(mask_neg.sum())
        df = df.loc[~mask_neg].copy()
        logger.info("clean_ohlc: dropped %d rows with non-positive prices", audit["removed_non_positive"])

    mask_extreme = (df[price_columns] > extreme_threshold).any(axis=1)
    if mask_extreme.any():
        audit["removed_extreme"] = int(mask_extreme.sum())
        df = df.loc[~mask_extreme].copy()
        logger.info("clean_ohlc: dropped %d rows above %g", audit["removed_extreme"], extreme_threshold)

    actual_high = df[["Open", "High", "Close"]].max(axis=1)
    actual_low = df[["Open", "Low", "Close"]].min(axis=1)
    audit["high_fixes"] = int((df["High"] != actual_high).sum())
    audit["low_fixes"] = int((df["Low"] != actual_low).sum())
    if audit["high_fixes"] or audit["low_fixes"]:
        logger.info(
            "clean_ohlc: repaired %d High and %d Low values (logged, not silent)",
            audit["high_fixes"], audit["low_fixes"],
        )
    df = df.copy()
    df["High"] = actual_high
    df["Low"] = actual_low
    return df, audit


# --- Returns -----------------------------------------------------------------


def prepare_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add a ``Daily_Return`` column to a long-format panel.

    REFACTOR: identical contract to the original ``prepare_data`` but explicit
    about which price column is used (was implicitly ``Close``).
    """
    out = df.sort_values(["Ticker", "Date"]).copy()
    out["Daily_Return"] = out.groupby("Ticker", observed=True)[price_col].pct_change()
    return out.dropna(subset=["Daily_Return"]).reset_index(drop=True)


def pivot_returns(df: pd.DataFrame, value: str = "Daily_Return") -> pd.DataFrame:
    """Long → wide returns matrix. NaNs are preserved (no global ``dropna``).

    REFACTOR: the previous code called ``.dropna(axis=1)`` immediately, which
    silently introduced survivorship bias (only tickers with full history
    survived). We keep the ragged matrix and let the backtest engine decide
    asset eligibility per rolling window.
    """
    return df.pivot(index="Date", columns="Ticker", values=value).sort_index()


def fix_missing_date(
    df: pd.DataFrame,
    ticker: str,
    missing_date,
    copy_from_date,
) -> pd.DataFrame:
    """Fill a single missing day for a ticker by copying another day's row.

    Documented data correction. Returns a new frame; does not modify ``df``.
    """
    missing_date = pd.to_datetime(missing_date)
    copy_from_date = pd.to_datetime(copy_from_date)
    ticker_df = df[df["Ticker"] == ticker]
    if (ticker_df["Date"] == missing_date).any():
        return df
    source_row = ticker_df[ticker_df["Date"] == copy_from_date]
    if source_row.empty:
        raise KeyError(f"{ticker}: source date {copy_from_date.date()} not found")
    new_row = source_row.copy()
    new_row["Date"] = missing_date
    return (
        pd.concat([df, new_row], ignore_index=True)
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )


def winsorize_returns(returns: pd.DataFrame | pd.Series, n_sigma: float = 6.0) -> pd.DataFrame | pd.Series:
    """Per-column ``n_sigma`` winsorization (caps extreme values at μ ± n·σ).

    Works on a Series (single asset) or a wide DataFrame (one column per asset).
    Constant or all-NaN columns are returned untouched.
    """
    if isinstance(returns, pd.Series):
        mu, sigma = returns.mean(), returns.std()
        if not np.isfinite(sigma) or sigma == 0:
            return returns
        return returns.clip(mu - n_sigma * sigma, mu + n_sigma * sigma)

    mu = returns.mean()
    sigma = returns.std()
    sigma = sigma.replace(0, np.nan)
    lower = mu - n_sigma * sigma
    upper = mu + n_sigma * sigma
    return returns.clip(lower=lower, upper=upper, axis=1)
