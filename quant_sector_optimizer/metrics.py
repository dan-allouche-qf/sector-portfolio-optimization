"""Performance and risk metrics.

Includes Sharpe, Sortino, max drawdown, Calmar, hit ratio, turnover, and a
``performance_summary`` that packages them.

A single ``periods_per_year`` argument controls annualization everywhere,
and CAGR uses geometric compounding consistently — the same convention used
by the optimizers, so optimized and displayed Sharpe agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_returns(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if "ret" in x.columns:
            return x["ret"]
        if "Daily_Return" in x.columns:
            return x["Daily_Return"]
        raise ValueError("DataFrame has no 'ret' or 'Daily_Return' column")
    return x


def _to_nav(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if "nav" in x.columns:
            return x["nav"]
        raise ValueError("DataFrame has no 'nav' column")
    return x


def cagr(nav: pd.Series | pd.DataFrame, periods_per_year: int = 252) -> float:
    """Compound annual growth rate, computed from the NAV path.

    ``CAGR = (nav_T / nav_0)^(periods_per_year / T) - 1``.
    """
    nav = _to_nav(nav).dropna()
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return float("nan")
    total = nav.iloc[-1] / nav.iloc[0]
    n = len(nav) - 1
    return float(total ** (periods_per_year / n) - 1.0)


def annualized_volatility(returns, periods_per_year: int = 252) -> float:
    r = _to_returns(returns).dropna()
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio: ``(CAGR-equivalent − rf) / σ_ann``.

    Uses the geometric mean (consistent with CAGR) and ddof=1 std.
    """
    r = _to_returns(returns).dropna()
    if r.empty:
        return float("nan")
    geom_mean = float((1.0 + r).prod() ** (1.0 / len(r)) - 1.0)
    mu_ann = (1.0 + geom_mean) ** periods_per_year - 1.0
    sigma_ann = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    if sigma_ann == 0:
        return float("nan")
    return (mu_ann - risk_free_rate) / sigma_ann


def sortino_ratio(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target: float = 0.0,
) -> float:
    """Sortino ratio: same numerator as Sharpe, but only **downside** vol.

    ``downside_dev = sqrt(E[(min(r-target,0))²] · periods_per_year)``.
    """
    r = _to_returns(returns).dropna()
    if r.empty:
        return float("nan")
    geom_mean = float((1.0 + r).prod() ** (1.0 / len(r)) - 1.0)
    mu_ann = (1.0 + geom_mean) ** periods_per_year - 1.0
    downside = np.minimum(r - target, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2) * periods_per_year))
    if dd == 0:
        return float("nan")
    return (mu_ann - risk_free_rate) / dd


def max_drawdown(nav) -> float:
    """Maximum drawdown of the NAV path, returned as a *negative* number.

    -0.25 means a 25% peak-to-trough drawdown.
    """
    nav = _to_nav(nav).dropna()
    if nav.empty:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def calmar_ratio(nav, periods_per_year: int = 252) -> float:
    """Calmar ratio = CAGR / |max drawdown|."""
    mdd = max_drawdown(nav)
    if mdd == 0 or not np.isfinite(mdd):
        return float("nan")
    return cagr(nav, periods_per_year) / abs(mdd)


def hit_ratio(returns) -> float:
    """Fraction of periods with strictly positive return."""
    r = _to_returns(returns).dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def turnover(weights_df: pd.DataFrame, annualize_to: int | None = None) -> float:
    """Average per-rebalance L1 turnover.

    ``turnover_t = Σ |w_t - w_{t-1}|`` — the absolute change in target weights.
    The first rebalance contributes ``Σ|w_1| = 1`` (full deployment from cash).
    Pass ``annualize_to`` (e.g. 4 for quarterly) to scale the average to a
    yearly figure (e.g. ``annual_turnover = 2.0`` ≈ portfolio rotated twice
    per year).
    """
    if weights_df.empty:
        return float("nan")
    w = weights_df.fillna(0.0)
    diffs = w.diff().abs().sum(axis=1)
    # ``diff()`` is NaN on the first row and ``sum(axis=1)`` collapses NaN to
    # zero — we must explicitly seed the first rebalance as deployment from cash.
    diffs.iloc[0] = float(w.iloc[0].abs().sum())
    avg = float(diffs.mean())
    if annualize_to is not None:
        avg *= annualize_to
    return avg


# --- Summary ----------------------------------------------------------------


def performance_summary(
    backtest: pd.DataFrame,
    weights_df: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Return a Series with all standard metrics for a single backtest result."""
    out: dict[str, float] = {}
    out["CAGR"] = cagr(backtest, periods_per_year)
    out["Volatility"] = annualized_volatility(backtest, periods_per_year)
    out["Sharpe"] = sharpe_ratio(backtest, risk_free_rate, periods_per_year)
    out["Sortino"] = sortino_ratio(backtest, risk_free_rate, periods_per_year)
    out["MaxDrawdown"] = max_drawdown(backtest)
    out["Calmar"] = calmar_ratio(backtest, periods_per_year)
    out["HitRatio"] = hit_ratio(backtest)
    if weights_df is not None and not weights_df.empty:
        # Total L1 weight changes across all rebalances (including the initial
        # deployment from cash), divided by the span of the backtest in years.
        # This works for both quarterly-rebalanced strategies and single-rebalance
        # static portfolios (which then report a small but finite annual turnover).
        w = weights_df.fillna(0.0)
        diffs = w.diff().abs().sum(axis=1)
        diffs.iloc[0] = float(w.iloc[0].abs().sum())
        bt_years = (backtest.index[-1] - backtest.index[0]).days / 365.25
        out["AnnualTurnover"] = float(diffs.sum() / bt_years) if bt_years > 0 else float("nan")
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(backtest.index).dropna()
        common = backtest.loc[bench.index]
        excess = common["ret"] - bench
        out["TrackingError"] = float(excess.std(ddof=1) * np.sqrt(periods_per_year))
        out["InfoRatio"] = (
            float(excess.mean() * periods_per_year / out["TrackingError"])
            if out["TrackingError"] > 0 else float("nan")
        )
    return pd.Series(out)
