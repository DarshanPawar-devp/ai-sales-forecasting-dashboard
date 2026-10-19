"""AI Sales Forecasting Dashboard - a Streamlit app.

Run with:  streamlit run app.py

No database is used: uploaded data lives only in memory for the current session.
Styling comes from assets/style.css and small HTML helpers in utils/ui.py.

Flow of main():  sidebar (upload, map columns, filters, horizon)
              -> clean the data -> train the forecast -> draw six tabs.
"""
from __future__ import annotations

import inspect
import io
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import analytics as an
from utils import data_processing as dp
from utils import forecasting as fc
from utils import ui

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_sales.csv"

# Chart palette (kept in step with the CSS variables in assets/style.css).
PRIMARY, SECONDARY, VIOLET, GOOD, BAD, NEUTRAL = "#6366F1", "#38BDF8", "#8B5CF6", "#22C77A", "#F4506E", "#F5A524"
NONE = "— not available —"  # label for "this optional column is not in my file"

# st.set_page_config must be the first Streamlit call in the script.
st.set_page_config(page_title="AI Sales Forecasting Dashboard", page_icon="📈", layout="wide")
ui.load_css()


# ------------------------------------------------------------------ helpers
def fmt(n, decimals: int = 0) -> str:
    """Compact number format: 12.3K, 4.5M ..."""
    if n is None or pd.isna(n):
        return "N/A"
    # FIX: the old code used (1e4, "K") and divided by the same 1e4, so 98,000 was shown
    # as "9.8K" (10x too small). Each unit now has a separate threshold and divisor:
    # values from 10,000 up are shortened, and "K" always means thousands.
    for threshold, divisor, suffix in ((1e9, 1e9, "B"), (1e6, 1e6, "M"), (1e4, 1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / divisor:,.1f}{suffix}"
    return f"{n:,.{decimals}f}"


# Streamlit caches results by their inputs, so changing a widget that does not
# affect the data (like the active tab) never retrains the models.
@st.cache_data(show_spinner=False)
def read_upload(raw: bytes) -> pd.DataFrame:
    return dp.load_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def cached_forecast(df: pd.DataFrame, horizon: int) -> dict:
    return fc.forecast_sales(df, horizon)


@st.cache_data(show_spinner=False)
def cached_product_forecast(df: pd.DataFrame, horizon: int, products: tuple) -> dict:
    # `products` is a tuple (not a list) because cache keys must be hashable.
    return fc.forecast_products(df, horizon, list(products))


# Streamlit renamed the Plotly sizing argument (use_container_width -> width="stretch")
# and old names are being removed. Look at what this installed version supports.
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    """Draw a Plotly figure at full container width on any supported Streamlit version."""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def style_fig(fig, height=380):
    """Consistent dark chart look: transparent background, Inter font, soft grid."""
    fig.update_layout(
        height=height, margin=dict(l=12, r=12, t=56, b=12),
        legend=dict(orientation="h", y=1.12, x=0), template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#A3AABD"),
        title=dict(font=dict(size=16, color="#F2F4FA")),
        colorway=[PRIMARY, SECONDARY, GOOD, NEUTRAL, BAD],
        hoverlabel=dict(font_size=13, bgcolor="#161A27", bordercolor="rgba(255,255,255,.12)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def blend(c1: str, c2: str, t: float) -> str:
    """Colour at position t (0..1) between two #RRGGBB colours."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


# ------------------------------------------------------------------ sidebar
def sidebar_load_and_map():
    """Upload and column mapping. Returns (raw_df, mapping) or (None, None)."""
    ui.brand()
    st.sidebar.caption("Upload any sales CSV, map your columns, explore and forecast.")

    st.sidebar.header("1 · Data")
    upload = st.sidebar.file_uploader("Upload a sales CSV", type=["csv"],
                                      help="Any CSV with at least a date column and a sales/revenue column.")
    use_sample = st.sidebar.checkbox("Use the built-in sample data", value=False,
                                     help="Try the dashboard without uploading anything.")

    raw = None
    try:
        if upload is not None:                 # an uploaded file wins over the sample
            raw = read_upload(upload.getvalue())
        elif use_sample and SAMPLE_PATH.exists():
            raw = read_upload(SAMPLE_PATH.read_bytes())
    except Exception as e:
        st.sidebar.error(f"Could not read this file: {e}")
        return None, None
    if raw is None:
        return None, None

    st.sidebar.header("2 · Map columns")
    cols = list(raw.columns)

    def pick(label, field, required=False, numeric=False, help_text=None):
        """One selectbox, pre-selected with the best guess. Returns the column or None."""
        options = cols if required else [NONE] + cols  # optional fields may be left empty
        guess = dp.guess_column(raw, field, numeric)
        idx = options.index(guess) if guess in options else 0
        choice = st.sidebar.selectbox(label + (" *" if required else ""), options, index=idx, help=help_text)
        return None if choice == NONE else choice

    mapping = {
        "date": pick("Date", "date", True, help_text="Column with the sale date."),
        "sales": pick("Sales / Revenue", "sales", True, numeric=True, help_text="Numeric column with money amounts."),
        "quantity": pick("Quantity", "quantity", numeric=True, help_text="Units sold (optional)."),
        "product": pick("Product / Item", "product", help_text="Enables product-level analysis."),
        "category": pick("Category", "category"),
        "region": pick("Region", "region"),
    }
    return raw, mapping


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Date / product / category / region filters. Returns the filtered frame."""
    st.sidebar.header("3 · Filters")
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    picked = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    # While the user is still choosing, date_input returns a single date or a 1-item
    # tuple instead of a (start, end) pair. Treat that as a one-day range.
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    else:
        start = end = picked[0] if isinstance(picked, (tuple, list)) else picked
    out = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]

    for field, label in (("product", "Product"), ("category", "Category"), ("region", "Region")):
        if field in df:  # only offer filters for columns that were mapped
            options = sorted(df[field].dropna().unique())
            chosen = st.sidebar.multiselect(label, options, default=[], placeholder=f"All {label.lower()}s")
            if chosen:  # an empty selection means "no filter"
                out = out[out[field].isin(chosen)]
    return out


# ------------------------------------------------------------------ tabs
def tab_overview(df):
    """KPI cards, sales over time, and category / region / product breakdowns."""
    k = an.compute_kpis(df)
    g = k["growth"]
    ui.section("Key numbers", f"{df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}  ·  {len(df):,} records")
    ui.kpi_grid([
        ui.kpi_card("money", "Total sales", fmt(k["total_sales"]), "Sum of all sales", color=PRIMARY),
        ui.kpi_card("box", "Quantity sold", fmt(k["total_quantity"]) if k["total_quantity"] is not None else "N/A",
                    "Units sold" if k["total_quantity"] is not None else "No quantity column", color=SECONDARY),
        ui.kpi_card("calc", "Average sale", fmt(k["avg_sales"], 2), "Per record", color=VIOLET),
        ui.kpi_card("trophy", "Highest sale", fmt(k["max_sales"]), "Largest single record", color=NEUTRAL),
        ui.kpi_card("trend", "Sales growth", f"{g:+.1f}%" if g is not None else "N/A",
                    "2nd half vs 1st half of period", delta=g, color=GOOD),
        ui.kpi_card("layers", "Products", f"{k['num_products']:,}" if k["num_products"] is not None else "N/A",
                    "Distinct items" if k["num_products"] is not None else "No product column", color="#F472B6"),
    ])

    ui.section("Sales over time")
    gran = st.radio("Time view", ["Daily", "Weekly", "Monthly"], index=2, horizontal=True,
                    label_visibility="collapsed")
    code = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[gran]
    ts = an.sales_over_time(df, code)
    fig = px.area(ts, x="date", y="sales", title=f"{gran} sales", color_discrete_sequence=[PRIMARY])
    fig.update_traces(line=dict(width=2.5), fillcolor="rgba(99,102,241,0.18)",
                      hovertemplate="%{x|%d %b %Y}<br>Sales: %{y:,.0f}<extra></extra>")
    show_chart(style_fig(fig))

    ui.section("Where sales come from")
    cols = st.columns(2)
    for slot, field, title in ((cols[0], "category", "Category performance"),
                               (cols[1], "region", "Region performance")):
        with slot:
            if field in df:
                g_df = an.group_sales(df, field)
                f = px.bar(g_df, x=field, y="sales", title=title, color="sales",
                           color_continuous_scale=["#2B3170", PRIMARY])
                f.update_layout(coloraxis_showscale=False)
                f.update_traces(hovertemplate="%{x}<br>Sales: %{y:,.0f}<extra></extra>")
                show_chart(style_fig(f, 340))
            else:
                ui.notice(f"No <b>{field}</b> column was mapped, so {field} performance is not shown.")

    if "product" in df:
        top_n = st.slider("Products to show", 5, 30, 10, help="Top products by total sales.")
        p = an.group_sales(df, "product", top_n).sort_values("sales")  # ascending so the biggest bar is on top
        f = px.bar(p, x="sales", y="product", orientation="h", title=f"Top {top_n} products by sales",
                   color_discrete_sequence=[SECONDARY])
        f.update_traces(hovertemplate="%{y}<br>Sales: %{x:,.0f}<extra></extra>")
        show_chart(style_fig(f, max(340, 28 * len(p) + 100)))
    else:
        ui.notice("No <b>product</b> column was mapped, so product performance is not shown.")


def tab_trending(df, trends):
    """Top sellers, plus products that are rising, declining or need attention."""
    if "product" not in df:
        ui.empty_state("🛍️", "Product-level analysis can't be shown",
                       "No product or item column was mapped. Choose one in the sidebar (step 2) if your file "
                       "has it. Overall sales analysis still works in the other tabs.")
        return
    if trends.empty:
        ui.notice("The selected period is too short to compare recent and earlier sales. "
                  "Widen the date range.", "warn")
        return

    ui.section("Top sellers and trends",
               f"Each product's latter-half sales are compared with its first half. Above +{an.RISING_THRESHOLD:.0f}% "
               f"is Rising, below {an.DECLINING_THRESHOLD:.0f}% is Declining, otherwise Stable.")

    counts = trends["trend"].value_counts()
    ui.kpi_grid([
        ui.kpi_card("trend", "Rising", str(int(counts.get("Rising", 0))), "products growing", color=GOOD),
        ui.kpi_card("pulse", "Stable", str(int(counts.get("Stable", 0))), "products steady", color=NEUTRAL),
        ui.kpi_card("alert", "Declining", str(int(counts.get("Declining", 0))), "products falling", color=BAD),
    ])

    n = st.slider("Items to list", 3, 15, 5)
    c1, c2 = st.columns(2)
    top_rev = trends.nlargest(n, "total_sales").sort_values("total_sales")
    f = px.bar(top_rev, x="total_sales", y="product", orientation="h", title="Top sellers by revenue",
               color_discrete_sequence=[PRIMARY])
    with c1:
        show_chart(style_fig(f, 340))
    if "total_quantity" in trends:
        top_q = trends.nlargest(n, "total_quantity").sort_values("total_quantity")
        f = px.bar(top_q, x="total_quantity", y="product", orientation="h", title="Top sellers by quantity",
                   color_discrete_sequence=[SECONDARY])
        with c2:
            show_chart(style_fig(f, 340))
    else:
        with c2:
            ui.notice("Map a <b>quantity</b> column to see top sellers by quantity.")

    ui.section("🚀 Fastest-growing products")
    rising = trends[trends["trend"] == "Rising"].nlargest(n, "change_pct")
    if len(rising):
        ui.trend_table(rising)
    else:
        ui.notice("No products are rising by more than the threshold.")

    ui.section("📉 Declining products")
    dec = trends[trends["trend"] == "Declining"].nsmallest(n, "change_pct")
    if len(dec):
        ui.trend_table(dec)
    else:
        ui.notice("No products are declining by more than the threshold.", "good")

    ui.section("⚠️ Products needing attention",
               "Declining products with above-median revenue, or any product down 30% or more.")
    attn = an.needs_attention(trends)
    if len(attn):
        ui.trend_table(attn)
    else:
        ui.notice("No products need urgent attention.", "good")

    ui.section("All products and trend labels")
    colors = {"Rising": GOOD, "Stable": NEUTRAL, "Declining": BAD}
    fig = px.bar(trends.nlargest(25, "total_sales").sort_values("change_pct"), x="change_pct", y="product",
                 orientation="h", color="trend", color_discrete_map=colors,
                 title="Sales change % (top 25 products by revenue)")
    fig.update_traces(hovertemplate="%{y}<br>Change: %{x:+.1f}%<extra></extra>")
    show_chart(style_fig(fig, 620))
    with st.expander("Show full table"):
        ui.trend_table(trends)


def forecast_figure(result: dict, title: str) -> go.Figure:
    """Bar chart: recent history in grey, the forecast in indigo-to-violet bars.

    Thin whiskers on each forecast bar show the approximate 95% range.
    """
    hist, f = result["history"], result["forecast"]
    hist = hist.tail(max(len(f) * 2, 30))  # show a window of history, not years of it
    pred = f["predicted_sales"]
    span = (pred.max() - pred.min()) or 1.0
    # Taller bars get a more violet colour, like the reference design.
    colors = [blend("#4F46E5", "#A78BFA", float((v - pred.min()) / span)) for v in pred]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist["ds"], y=hist["y"], name="Recent sales", marker_color="rgba(148,163,184,0.30)",
                         hovertemplate="%{x|%d %b %Y}<br>Sales: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        x=f["date"], y=pred, name="Forecast", marker_color=colors,
        error_y=dict(type="data", symmetric=False, array=(f["upper_bound"] - pred).tolist(),
                     arrayminus=(pred - f["lower_bound"]).tolist(), color="rgba(255,255,255,0.28)",
                     thickness=1, width=0),
        hovertemplate="%{x|%d %b %Y}<br>Predicted: %{y:,.0f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=None, yaxis_title=None, barmode="overlay", bargap=0.25)
    return style_fig(fig, 430)


def product_change_cards(df, trends, n: int = 4):
    """The biggest movers among the top sellers, as small cards with a sparkline."""
    top = trends.nlargest(8, "total_sales")
    movers = top.assign(abs_change=top["change_pct"].abs()).nlargest(n, "abs_change")
    cards = []
    for r in movers.itertuples(index=False):
        is_new = r.earlier_sales <= 0  # no sales in the first half
        tone = "flat" if is_new else {"Rising": "up", "Declining": "down"}.get(r.trend, "flat")
        color = {"up": GOOD, "down": BAD, "flat": NEUTRAL}[tone]
        cards.append({
            "name": str(r.product), "tone": tone, "color": color,
            "tag": "New" if is_new else r.trend,
            "pct": "New" if is_new else f"{r.change_pct:+.0f}%",
            "values": an.product_sparkline(df, r.product),
            "note": f"{fmt(r.recent_sales)} in the second half, against {fmt(r.earlier_sales)} in the first half.",
        })
    ui.change_cards(cards)


def show_forecast(result: dict, title: str, key: str, horizon: int, side: dict | None = None):
    """Hero card + chart (+ side panel) + model details + table for one forecast result.

    `side` (overall forecast only) holds the data for the right-hand panel; product
    forecasts pass None and use the full width.
    """
    if not result["ok"]:
        ui.notice(f"<b>No forecast produced for {escape(title)}.</b> {escape(result['message'])}", "warn", "🚫")
        return
    f = result["forecast"]
    metrics = result["metrics"]
    s = fc.summarize_forecast(result)

    # Wording that matches the data's granularity (daily / weekly / monthly).
    plural = {"daily": "days", "weekly": "weeks", "monthly": "months"}[s["unit"]]
    singular = plural[:-1]
    date_fmt = "%b %Y" if s["unit"] == "monthly" else "%a %d %b"
    tone = {"High": "good", "Medium": "warn", "Low": "bad"}.get(s["accuracy_label"], "flat")

    hero = ui.forecast_hero(
        title=f"Next {horizon} days forecast", status="Forecast ready",
        subtitle=f"{result['model'].split(' (')[0]} model trained on {len(result['history'])} {s['unit']} data points",
        big=fmt(s["total"]), big_label="expected sales",
        change=s["change_pct"], change_label=f"vs previous {s['periods']} {plural}",
        acc=f"{s['accuracy']:.0f}%" if s["accuracy"] is not None else "N/A",
        acc_label=s["accuracy_label"] or "Not enough history", acc_tone=tone,
        range_text=f"Likely range {fmt(s['lower'])} to {fmt(s['upper'])}",
        minis=[("trend", NEUTRAL, fmt(s["peak_value"]), f"Peak {singular}", s["peak_date"].strftime(date_fmt)),
               ("calc", VIOLET, fmt(s["average"]), f"Average per {singular}", "Across the forecast"),
               ("calendar", SECONDARY, fmt(s["prev_total"]), f"Previous {s['periods']} {plural}", "Actual sales before the forecast")])

    # The forecast table doubles as the CSV download.
    table = f.copy()
    table["date"] = table["date"].dt.date
    table = table.round(2)

    def download_button():
        st.download_button("Download forecast CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{key}.csv", mime="text/csv", key=f"dl_{key}")

    if side:
        left, right = st.columns([2.2, 1], gap="medium")
    else:
        left, right = st.container(), None
    with left:
        ui.html(hero)
        show_chart(forecast_figure(result, f"Expected sales: {title}"), key=f"chart_{key}")
        if right is None:
            download_button()
    if right is not None:
        with right:
            ui.html(ui.drivers_panel(side["drivers"]))
            ui.html(ui.week_card(fmt(side["week"].mean()), [float(v) for v in side["week"]], an.WEEKDAYS))
            download_button()

    ui.section("Model details")
    ui.kpi_grid([
        ui.kpi_card("cpu", "Model", result["model"].split(" (")[0], f"{fc.FREQ[result['freq']]['label'].capitalize()} data", color=SECONDARY),
        ui.kpi_card("target", "MAE", fmt(metrics["mae"], 2) if metrics else "N/A",
                    "Avg. miss per period (lower = better)" if metrics else "Not enough history", color=GOOD),
        ui.kpi_card("bars", "RMSE", fmt(metrics["rmse"], 2) if metrics else "N/A",
                    "Penalises big misses" if metrics else "Not enough history", color=NEUTRAL),
    ])
    st.caption(f"Model: {result['model']}. MAE and RMSE come from forecasting the most recent part of your "
               "history and comparing it with what really happened. The thin lines on the forecast bars show "
               "an approximate 95% range, and the likely range adds those bounds up over the whole period.")
    with st.expander("📋 Forecast table"):
        st.dataframe(table, hide_index=True)


def tab_forecast(df, horizon, overall, trends):
    """Overall forecast (with drivers and movers) or per-product forecasts."""
    ui.section("AI sales forecast", f"Predicting the next {horizon} days from your sales history.")
    if not fc.HAS_PROPHET:
        ui.notice("Prophet is not installed, so a lighter fallback model is used. "
                  "Install <code>prophet</code> for the full model (see README).")

    scope = "Overall"
    if "product" in df:  # product-level forecasts need a product column
        scope = st.radio("Forecast level", ["Overall", "Product-level"], horizontal=True)

    if scope == "Overall":
        side = {"drivers": an.driver_items(df, trends), "week": an.weekday_profile(df)}
        show_forecast(overall, "all sales", "overall", horizon, side)
        if "product" in df and not trends.empty:
            ui.section("Sales changes by product",
                       "The biggest movers among your top sellers: second half of the period against the first half.")
            product_change_cards(df, trends)
        return

    top = an.group_sales(df, "product", 20)["product"].tolist()
    chosen = st.multiselect("Products to forecast (up to 5)", sorted(df["product"].unique()),
                            default=top[:3], max_selections=5,
                            help="Defaults to the top sellers. Each product needs enough history by itself.")
    if not chosen:
        ui.notice("Pick at least one product to forecast.")
        return
    with st.spinner("Training product-level models..."):
        results = cached_product_forecast(df, horizon, tuple(chosen))
    tabs = st.tabs([str(c) for c in chosen])
    for t, p in zip(tabs, chosen):
        with t:
            show_forecast(results[p], str(p), f"product_{p}", horizon)


def tab_insights(df, trends, overall, horizon):
    """Plain-language summary cards."""
    ui.section("AI Insights", "A plain-language summary of what your data says.")
    total = overall["forecast"]["predicted_sales"].sum() if overall["ok"] else None
    for icon_char, title, text in an.insight_items(df, trends, total, horizon):
        ui.insight_card(icon_char, title, text)
    if not overall["ok"]:
        ui.notice(f"<b>Future-sales insight unavailable.</b> {escape(overall['message'])}", "warn")
    st.caption("Insights come from rules and statistical models applied to your data, not from a language model. "
               "Treat forecasts as estimates.")


def tab_data(raw, clean_log, mapping):
    """Raw-file preview, quality warnings and the cleaning log."""
    ui.section("Data preview and quality")
    ui.kpi_grid([
        ui.kpi_card("db", "Rows", f"{len(raw):,}", "In the uploaded file", color=PRIMARY),
        ui.kpi_card("layers", "Columns", str(len(raw.columns)), "Detected", color=SECONDARY),
        ui.kpi_card("copy", "Duplicates", f"{int(raw.duplicated().sum()):,}", "Removed during cleaning", color=NEUTRAL),
        ui.kpi_card("alert", "Missing cells", f"{int(raw.isnull().sum().sum()):,}", "Across all columns", color=BAD),
    ])

    warnings = dp.quality_warnings(raw)
    if warnings:
        for w in warnings:
            ui.notice(escape(w), "warn")
    else:
        ui.notice("No data-quality problems detected.", "good")

    st.markdown("**Preview (first 50 rows)**")
    st.dataframe(raw.head(50))
    st.markdown("**Column types and missing values**")
    st.dataframe(dp.column_overview(raw))
    with st.expander("🧹 What was cleaned?", expanded=True):
        for line in clean_log:
            st.write("•", line)
        st.caption("Mapped columns: " + ", ".join(f"{k} → {v}" for k, v in mapping.items() if v))


def tab_about():
    ui.section("About this project")
    st.markdown("""
**AI Sales Forecasting Dashboard** turns any sales CSV into an interactive analytics and forecasting tool.
You map your own columns, explore KPIs and charts, spot trending products and get a data-driven forecast.

**Technologies**: Python, Streamlit, Pandas, Plotly, scikit-learn, with optional Prophet and statsmodels,
styled with custom HTML and CSS. No database is needed.

**How the forecasting works**
1. Sales are summed per period (daily, weekly or monthly, detected from your dates).
2. The app picks the best available model: **Prophet** if installed, otherwise **Holt-Winters exponential
   smoothing**, otherwise a **Ridge regression** on time and seasonality features (scikit-learn).
3. The most recent part of your history is held out, forecast and compared with reality to compute
   **MAE** and **RMSE**. Then the model is retrained on all data to predict the future.
4. If the history is too short, the app **refuses to forecast** and tells you what is missing.

**Trend labels** compare the latter half of the selected period with the first half for every product.

*Evolved from a student LinearRegression demand-prediction script into a general-purpose dashboard.*
""")


# ------------------------------------------------------------------ main
def landing():
    """Shown before any data is loaded: how to start, and what a CSV should look like."""
    ui.empty_state("📂", "Upload a sales CSV to get started",
                   "Use the sidebar on the left to upload your file, or tick “Use the built-in sample data” "
                   "to explore the dashboard instantly.")
    ui.steps([("Upload", "Drop in any sales CSV. Column names can be anything."),
              ("Map columns", "Tell the app which columns hold the date, sales, product and so on."),
              ("Explore", "KPIs, charts, filters, top sellers and rising or declining products."),
              ("Forecast", "Predict the next 7, 30, 60 or 90 days and download the result.")])
    ui.section("Expected CSV structure", "Only a date column and a sales/revenue column are required.")
    st.dataframe(pd.DataFrame({
        "Order_Date": ["2025-01-03", "2025-01-03"], "Product_Name": ["Wireless Mouse", "Desk Lamp"],
        "Category": ["Electronics", "Home"], "Region": ["North", "West"],
        "Units_Sold": [12, 5], "Revenue": [240.0, 130.0]}), hide_index=True)
    with st.expander("About this project"):
        tab_about()


def main():
    # The header lives in a placeholder so it can be redrawn with the date range
    # once the data is known, while every early `return` below still shows a header.
    header = st.empty()
    title, subtitle = "Sales Forecast", "Plan smarter. Stock better."
    header.markdown(ui.header_html(title, subtitle), unsafe_allow_html=True)

    privacy = ("Private by design", "Your file is processed in memory only. Nothing is saved to disk or a database.")

    raw, mapping = sidebar_load_and_map()
    if raw is None:
        ui.sidebar_note(*privacy)
        landing()
        return

    # Date and sales must be different columns, otherwise "sales" would be parsed as dates.
    if mapping["date"] == mapping["sales"]:
        ui.notice("The date and sales columns must be different. Please change the mapping in the sidebar.", "warn")
        return

    try:
        clean, log = dp.clean_data(raw, mapping)
    except Exception as e:
        ui.notice(f"Something went wrong while cleaning the data: {escape(str(e))}", "warn")
        return
    if clean.empty:
        ui.notice("No valid rows remain after cleaning. Check that the date column holds dates and the "
                  "sales column holds numbers.", "warn")
        return

    filtered = sidebar_filters(clean)
    if filtered.empty:
        ui.notice("No records match the current filters. Widen the date range or clear some filters.", "warn")
        return

    st.sidebar.header("4 · Forecast")
    horizon = st.sidebar.select_slider("Forecast period (days)", options=[7, 30, 60, 90], value=30,
                                       help="How far into the future to predict.")
    ui.sidebar_note(*privacy)

    # Now that the filtered range is known, redraw the header with it.
    header.markdown(ui.header_html(title, subtitle, f"{filtered['date'].min():%d %b %Y} to {filtered['date'].max():%d %b %Y}"),
                    unsafe_allow_html=True)

    trends = an.product_trends(filtered)
    with st.spinner("Training forecasting model..."):
        overall = cached_forecast(filtered, horizon)

    t = st.tabs(["📊 Overview", "🔥 Top & Trending", "🔮 Forecast", "🤖 AI Insights", "🧾 Data & Quality", "ℹ️ About"])
    with t[0]:
        tab_overview(filtered)
    with t[1]:
        tab_trending(filtered, trends)
    with t[2]:
        tab_forecast(filtered, horizon, overall, trends)
    with t[3]:
        tab_insights(filtered, trends, overall, horizon)
    with t[4]:
        tab_data(raw, log, mapping)  # raw (unfiltered) data on purpose: this tab is about the file itself
    with t[5]:
        tab_about()

    ui.html('<div class="footer">AI Sales Forecasting Dashboard · Built with Streamlit, Plotly and scikit-learn</div>')


main()
