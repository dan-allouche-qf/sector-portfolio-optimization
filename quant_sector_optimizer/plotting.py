"""Plotly figures.

REMOVED: the ipywidgets dropdowns in the original module. They never worked
when the notebook renderer was set to ``"png"`` (the static export used to
keep figures inline on GitHub). Static plots are clearer for static notebooks
and the data slicing now lives where it belongs (in the notebook, before the
plot call).

REFACTOR: every helper now returns the Plotly ``Figure`` instead of calling
``fig.show()`` itself. Returning the figure makes the helpers composable and
testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import DAYS_PER_YEAR


# --- Sector / universe overview ---------------------------------------------


def plot_sector_category_distribution(df: pd.DataFrame) -> go.Figure:
    dist = df.groupby(["Sector", "Category"]).Ticker.nunique().reset_index(name="count")
    return px.sunburst(
        dist,
        path=["Sector", "Category"],
        values="count",
        title="Distribution of Instruments by Sector and Category",
        color="count",
        color_continuous_scale="RdBu",
    )


def plot_sector_average_close(df: pd.DataFrame) -> go.Figure:
    sector_avg = df.groupby(["Sector", "Date"])["Close"].mean().reset_index()
    return px.line(
        sector_avg, x="Date", y="Close", color="Sector",
        title="Average Close Price by Sector",
        labels={"Close": "Average Close Price"},
    )


def plot_close_boxplot_by_sector(df: pd.DataFrame, max_points: int = 50_000) -> go.Figure:
    plot_df = df
    sampled = len(df) > max_points
    if sampled:
        plot_df = df.sample(max_points, random_state=42)
    title = "Distribution of Close Prices by Sector" + (" (sampled)" if sampled else "")
    return px.box(plot_df, x="Sector", y="Close", color="Sector", title=title, log_y=True)


def plot_universe_growth(coverage: pd.DataFrame, mode: str = "net") -> go.Figure:
    dates = pd.date_range(
        start=coverage.first_date.min(),
        end=coverage.last_date.max(),
        freq="ME",
    )
    base_counts = (
        coverage[coverage.first_date <= dates[0]].groupby("Sector").size().to_dict()
    )
    rows = []
    for d in dates:
        counts = (
            coverage[coverage.first_date <= d]
            .groupby("Sector").size().reset_index(name="count")
        )
        counts["Date"] = d
        if mode == "net":
            counts["count"] = counts.apply(
                lambda r: r["count"] - base_counts.get(r["Sector"], 0), axis=1
            )
        rows.append(counts)
    df_growth = pd.concat(rows)
    title = "Net Tickers Added Since Start" if mode == "net" else "Total Tickers Available"
    return px.line(
        df_growth, x="Date", y="count", color="Sector",
        title=f"Evolution of the Investable Universe ({title})",
        labels={"count": "Ticker Count"},
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )


def plot_coverage_summary(coverage: pd.DataFrame) -> go.Figure:
    coverage = coverage.copy()
    coverage["years_of_history"] = (
        (coverage["last_date"] - coverage["first_date"]).dt.days / DAYS_PER_YEAR
    )
    return px.box(
        coverage, x="Sector", y="years_of_history", color="Sector",
        title="Distribution of Available History (Years) by Sector",
        labels={"years_of_history": "Years of Data"},
        points="all",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )


# --- Returns / cumulative perf ----------------------------------------------


def plot_sector_cumulative(df: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    df = df.copy()
    df["Cumulative_Return"] = df.groupby("Ticker", observed=True)["Daily_Return"].transform(
        lambda x: (1 + x).cumprod() - 1
    )
    avg = df.groupby(["Sector", "Date"])["Cumulative_Return"].mean().reset_index()
    avg["Log_Cum_Return"] = np.log1p(avg["Cumulative_Return"])
    return px.line(
        avg, x="Date", y="Log_Cum_Return", color="Sector",
        title=f"Average Cumulative Returns by Sector (log scale) {title_suffix}".rstrip(),
        labels={"Log_Cum_Return": "Log(1 + Avg Cumulative Return)"},
    )


def plot_strategy_comparison(
    backtests: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    title: str = "Strategy Comparison",
) -> go.Figure:
    """Overlay cumulative returns of multiple backtests (and optional benchmark)."""
    rows = []
    for name, bt in backtests.items():
        if bt.empty:
            continue
        s = bt["cum_return"]
        rows.append(pd.DataFrame({"Date": s.index, "Cumulative_Return": s.values, "Strategy": name}))
    if benchmark is not None and not benchmark.empty:
        rows.append(pd.DataFrame({"Date": benchmark.index, "Cumulative_Return": benchmark.values, "Strategy": "Benchmark"}))
    if not rows:
        return go.Figure()
    long = pd.concat(rows, ignore_index=True)
    return px.line(
        long, x="Date", y="Cumulative_Return", color="Strategy",
        title=title, labels={"Cumulative_Return": "Cumulative Return"},
    )


def plot_top_weights(weights_df: pd.DataFrame, top_n: int = 10, title: str = "Top Weights") -> go.Figure:
    """Stacked area chart of the ``top_n`` average holdings."""
    w = weights_df.fillna(0.0)
    row_sums = w.sum(axis=1).replace(0.0, np.nan)
    pct = w.div(row_sums, axis=0) * 100
    top = pct.mean().sort_values(ascending=False).head(top_n).index
    long = pct[top].reset_index().melt(
        id_vars=pct.index.name or "index",
        var_name="Ticker",
        value_name="Weight (%)",
    ).rename(columns={pct.index.name or "index": "Date"})
    return px.area(long, x="Date", y="Weight (%)", color="Ticker", title=title)


def plot_drawdown(nav: pd.Series | pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    if isinstance(nav, pd.DataFrame):
        nav = nav["nav"]
    dd = nav / nav.cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown"))
    fig.update_layout(title=title, yaxis_tickformat=".0%")
    return fig
