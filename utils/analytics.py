"""KPIs, aggregations, product trend analysis and AI insight text.

Pure pandas/numpy: no Streamlit imports, so every function can be tested on its own.
"""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

RISING_THRESHOLD = 10.0      # % change above which a product is "Rising"
DECLINING_THRESHOLD = -10.0  # % change below which a product is "Declining"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------------------------------------------------ headline numbers
def compute_kpis(df: pd.DataFrame) -> dict:
    """Headline numbers shown on the KPI cards."""
    kpis = {
        "total_sales": df["sales"].sum(),
        "avg_sales": df["sales"].mean(),
        "max_sales": df["sales"].max(),
        # Optional columns give None, so the UI can show "N/A" instead of crashing.
        "total_quantity": df["quantity"].sum() if "quantity" in df else None,
        "num_products": df["product"].nunique() if "product" in df else None,
        "growth": sales_growth(df),
    }
    return kpis


def sales_growth(df: pd.DataFrame) -> float | None:
    """% change between the second half and first half of the selected period."""
    if df.empty:
        return None
    start, end = df["date"].min(), df["date"].max()
    if (end - start).days < 2:  # too short to split into two halves
        return None
    mid = start + (end - start) / 2
    first = df.loc[df["date"] <= mid, "sales"].sum()
    second = df.loc[df["date"] > mid, "sales"].sum()
    if first <= 0:  # avoid dividing by zero (or by a negative base)
        return None
    return (second - first) / first * 100


# ------------------------------------------------------------------ aggregations
def sales_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate sales by D / W / M (month start)."""
    rule = {"D": "D", "W": "W", "M": "MS"}[freq]
    s = df.set_index("date")["sales"].resample(rule).sum().reset_index()
    return s


def group_sales(df: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    """Sales (and quantity if present) by a categorical column, biggest first."""
    agg = {"sales": "sum"}
    if "quantity" in df:
        agg["quantity"] = "sum"
    g = df.groupby(col).agg(agg).sort_values("sales", ascending=False).reset_index()
    return g.head(top) if top else g


def weekday_profile(df: pd.DataFrame) -> pd.Series:
    """Average sales per calendar day for Mon..Sun (always 7 values).

    Days with no records count as zero, so a quiet weekday is not hidden.
    """
    daily = df.set_index("date")["sales"].resample("D").sum()
    by_day = daily.groupby(daily.index.dayofweek).mean()
    return by_day.reindex(range(7), fill_value=0.0)


def product_sparkline(df: pd.DataFrame, product: str, points: int = 16) -> list[float]:
    """Recent sales of one product as a short list of numbers for a sparkline.

    Weekly totals when the period is long enough, daily totals otherwise.
    """
    sub = df[df["product"] == product]
    if sub.empty:
        return []
    span_days = (df["date"].max() - df["date"].min()).days
    rule = "W" if span_days >= 60 else "D"
    # Resample over the whole filtered range so every product is on the same time axis.
    s = sub.set_index("date")["sales"].resample(rule).sum()
    return [float(v) for v in s.tail(points).values]


# ------------------------------------------------------------------ product trends
def product_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each product's recent sales with its earlier sales.

    The date range is split in half: recent = latter half, earlier = first half.
    Change % = (recent - earlier) / earlier * 100, then labelled
    Rising (> +10%), Declining (< -10%) or Stable.
    """
    if "product" not in df or df.empty:
        return pd.DataFrame()
    start, end = df["date"].min(), df["date"].max()
    if (end - start).days < 2:
        return pd.DataFrame()
    mid = start + (end - start) / 2

    early = df[df["date"] <= mid].groupby("product")["sales"].sum()
    recent = df[df["date"] > mid].groupby("product")["sales"].sum()
    total = df.groupby("product")["sales"].sum()

    # A product missing from one half gets NaN here, which fillna turns into 0 sales.
    t = pd.DataFrame({"total_sales": total, "earlier_sales": early, "recent_sales": recent}).fillna(0)
    if "quantity" in df:
        t["total_quantity"] = df.groupby("product")["quantity"].sum()

    def pct(row):
        if row["earlier_sales"] > 0:
            return (row["recent_sales"] - row["earlier_sales"]) / row["earlier_sales"] * 100
        return 100.0 if row["recent_sales"] > 0 else 0.0   # brand-new product

    t["change_pct"] = t.apply(pct, axis=1)
    t["trend"] = np.select(
        [t["change_pct"] > RISING_THRESHOLD, t["change_pct"] < DECLINING_THRESHOLD],
        ["Rising", "Declining"], default="Stable")
    # FIX: name the index explicitly. The old rename({"index": "product"}) only worked
    # when pandas happened to keep the index name; rename_axis makes it certain.
    return t.rename_axis("product").reset_index().sort_values("total_sales", ascending=False)


def needs_attention(trends: pd.DataFrame) -> pd.DataFrame:
    """Declining products that still matter (above-median revenue) or fell sharply."""
    if trends.empty:
        return trends
    median = trends["total_sales"].median()
    mask = (trends["trend"] == "Declining") & (
        (trends["total_sales"] >= median) | (trends["change_pct"] <= -30))
    return trends[mask].sort_values("change_pct")


# ------------------------------------------------------------------ dashboard panels
def driver_items(df: pd.DataFrame, trends: pd.DataFrame) -> list[dict]:
    """Rows for the "What's driving sales" panel.

    Each row is a dict: icon, color, title, sub (plain text), value (text) and
    tone ("up", "down" or "flat") that decides the colour of the value.
    Rows are skipped when the data they need was not mapped.
    """
    rows = []

    # Share of total sales held by the biggest product / category / region.
    for field, title, icon, color in (("product", "Top product", "trophy", "#F5A524"),
                                      ("category", "Top category", "tag", "#8B5CF6"),
                                      ("region", "Top region", "pin", "#EF5B6B")):
        if field in df:
            top = group_sales(df, field, 1).iloc[0]
            total = df["sales"].sum()
            share = top["sales"] / total * 100 if total else 0
            rows.append({"icon": icon, "color": color, "title": title,
                         "sub": f"{top[field]}, {share:.0f}% of sales",
                         "value": f"{share:.0f}%", "tone": "flat"})

    # Weekend lift: average weekend day versus average weekday.
    prof = weekday_profile(df)
    weekday_avg, weekend_avg = prof.iloc[:5].mean(), prof.iloc[5:].mean()
    if weekday_avg > 0:
        lift = (weekend_avg - weekday_avg) / weekday_avg * 100
        rows.append({"icon": "calendar", "color": "#38BDF8", "title": "Weekend effect",
                     "sub": "Weekend days vs weekdays",
                     "value": f"{lift:+.0f}%", "tone": "up" if lift > 0 else "down" if lift < 0 else "flat"})

    # Momentum: second half of the period versus the first half.
    g = sales_growth(df)
    if g is not None:
        rows.append({"icon": "trend", "color": "#22C77A", "title": "Momentum",
                     "sub": "Second half vs first half",
                     "value": f"{g:+.0f}%", "tone": "up" if g > 0 else "down" if g < 0 else "flat"})
    return rows


def insight_items(df, trends, forecast_total=None, horizon=None) -> list[tuple[str, str, str]]:
    """Structured insights for the card UI: (icon, title, html text).

    Product names come from the user's file, so they are HTML-escaped here.
    """
    items = []
    if trends is not None and not trends.empty:
        best = trends.sort_values("total_sales", ascending=False).iloc[0]
        items.append(("🏆", "Best-selling item",
                      f"<b>{escape(str(best['product']))}</b> leads with {best['total_sales']:,.0f} in total sales."))
        rising = trends[trends["trend"] == "Rising"].sort_values("change_pct", ascending=False)
        if len(rising):
            r = rising.iloc[0]
            items.append(("🚀", "Most trending item",
                          f"<b>{escape(str(r['product']))}</b> is up {r['change_pct']:.0f}% compared with the earlier period."))
        else:
            items.append(("🚀", "Most trending item", "No product is clearly rising right now."))
        declining = trends[trends["trend"] == "Declining"].sort_values("change_pct")
        if len(declining):
            d = declining.iloc[0]
            items.append(("⚠️", "Most declining item",
                          f"<b>{escape(str(d['product']))}</b> is down {abs(d['change_pct']):.0f}%. "
                          "Consider a promotion or a review."))
        else:
            items.append(("⚠️", "Most declining item", "No product is clearly declining."))
    else:
        items.append(("ℹ️", "Product insights", "Unavailable because no product column was mapped."))

    g = sales_growth(df)
    if g is not None:
        word = "grew" if g >= 0 else "fell"
        items.append(("📈", "Overall trend",
                      f"Sales {word} by <b>{abs(g):.1f}%</b> between the first and second half of the period."))
    if forecast_total is not None:
        items.append(("🔮", f"Expected sales, next {horizon} days",
                      f"About <b>{forecast_total:,.0f}</b> in total, based on your sales history."))
    return items

# NOTE: the old build_insights() helper was removed. Nothing called it, and it
# duplicated insight_items() with slightly different wording, which invites drift.
