"""Small HTML/CSS helpers for the dashboard UI.

All user-supplied text (product names, column names ...) is escaped with
`html.escape` before being placed in HTML, so a weird CSV cannot break the page.

Rules for the HTML strings below:
  * keep each string on ONE line (no blank lines), because Streamlit runs the
    text through a Markdown parser first and a blank line would end the HTML block;
  * styling lives in assets/style.css, referenced here only by class name.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"

# Colours shared with app.py's charts so HTML cards and Plotly charts match.
GOOD, BAD, WARN, PRIMARY, VIOLET, SKY = "#22C77A", "#F4506E", "#F5A524", "#6366F1", "#8B5CF6", "#38BDF8"

TREND_BADGE = {
    "Rising": '<span class="badge rising">▲ Rising</span>',
    "Stable": '<span class="badge stable">● Stable</span>',
    "Declining": '<span class="badge declining">▼ Declining</span>',
}

# Line icons (24x24, stroke only). They are drawn as tiny SVG images so they stay
# crisp and take any colour; no icon font or network request is needed.
ICON_PATHS = {
    "spark": '<path d="M12 3l1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2z"/><path d="M19 3v4M17 5h4"/>',
    "money": '<circle cx="12" cy="12" r="10"/><path d="M16 8h-6a2 2 0 1 0 0 4h4a2 2 0 1 1 0 4H8"/><path d="M12 18V6"/>',
    "box": '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="m3.3 7 8.7 5 8.7-5M12 22V12"/>',
    "calc": '<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M8 6h8M16 14v4M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M8 18h.01M12 18h.01"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6M18 9h1.5a2.5 2.5 0 0 0 0-5H18M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0z"/>',
    "trend": '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 12 10 5 10-5M2 17l10 5 10-5"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 0 1 16 0z"/><circle cx="12" cy="10" r="3"/>',
    "tag": '<path d="M12.59 2.59A2 2 0 0 0 11.17 2H4a2 2 0 0 0-2 2v7.17a2 2 0 0 0 .59 1.42l8.7 8.7a2.43 2.43 0 0 0 3.42 0l6.58-6.58a2.43 2.43 0 0 0 0-3.42z"/><circle cx="7.5" cy="7.5" r="1"/>',
    "pulse": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "bars": '<path d="M12 20V10M18 20V4M6 20v-4"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5M3 12a9 3 0 0 0 18 0"/>',
    "copy": '<rect x="8" y="8" width="14" height="14" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "lock": '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
}


@lru_cache(maxsize=256)
def icon(name: str, color: str = "#8C94A8", size: int = 20) -> str:
    """An <img> tag holding one line icon as an inline (base64) SVG."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
           f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
           f'{ICON_PATHS[name]}</svg>')
    data = base64.b64encode(svg.encode()).decode()
    return f'<img class="ico" src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt="">'


def tile(name: str, color: str) -> str:
    """A rounded square holding an icon; its background is a faint tint of `color`."""
    return f'<span class="tile" style="--c:{color}">{icon(name, color)}</span>'


def sparkline(values: list[float], color: str, width: int = 150, height: int = 44) -> str:
    """A tiny line + soft area chart as an inline SVG image (no Plotly needed)."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0  # a flat line must not divide by zero
    pad = 3
    pts = [(pad + i * (width - 2 * pad) / (len(values) - 1),
            pad + (1 - (v - lo) / span) * (height - 2 * pad)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pad},{height} {line} {width - pad},{height}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
           f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".32"/>'
           f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
           f'<polygon points="{area}" fill="url(#g)"/>'
           f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    data = base64.b64encode(svg.encode()).decode()
    return f'<img class="spark" src="data:image/svg+xml;base64,{data}" width="{width}" height="{height}" alt="">'


# ------------------------------------------------------------------ page plumbing
def load_css() -> None:
    """Inject assets/style.css into the page."""
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
    except OSError:  # a missing stylesheet should not crash the app
        css = ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render a trusted HTML string."""
    st.markdown(markup, unsafe_allow_html=True)


def brand() -> None:
    """Logo and product name at the top of the sidebar."""
    st.sidebar.markdown(
        f'<div class="brand"><span class="logo">{icon("spark", "#FFFFFF", 22)}</span>'
        '<div><b>SalesFlow</b><small>AI sales forecasting</small></div></div>', unsafe_allow_html=True)


def sidebar_note(title: str, text: str) -> None:
    """Small info card at the bottom of the sidebar."""
    st.sidebar.markdown(
        f'<div class="side-note">{icon("lock", "#A5B4FC", 18)}<b>{escape(title)}</b><p>{escape(text)}</p></div>',
        unsafe_allow_html=True)


def header_html(title: str, subtitle: str, pill: str = "") -> str:
    """Page title on the left and an optional date-range pill on the right."""
    pill_html = f'<span class="pill">{icon("calendar", "#C7CCDB", 16)}{escape(pill)}</span>' if pill else ""
    return (f'<div class="page-head"><div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>'
            f'<div class="head-right">{pill_html}</div></div>')


def section(title: str, subtitle: str = "") -> None:
    sub = f'<p class="section-sub">{escape(subtitle)}</p>' if subtitle else ""
    html(f'<div class="section-title"><h3>{escape(title)}</h3></div>{sub}')


# ------------------------------------------------------------------ KPI cards
def kpi_card(icon_name: str, label: str, value: str, sub: str = "", delta: float | None = None,
             color: str = PRIMARY) -> str:
    """Return the HTML for one KPI card (cards are placed in a grid by kpi_grid)."""
    delta_html = ""
    if delta is not None:
        # Changes smaller than 0.05% count as "flat" so we never show a misleading arrow.
        cls = "up" if delta > 0.05 else "down" if delta < -0.05 else "flat"
        arrow = "▲" if cls == "up" else "▼" if cls == "down" else "■"
        delta_html = f'<span class="delta {cls}">{arrow} {abs(delta):.1f}%</span>'
    return (f'<div class="kpi" style="--c:{color}" title="{escape(sub)}">'
            f'<div class="kpi-top">{tile(icon_name, color)}{delta_html}</div>'
            f'<div class="value">{escape(value)}</div>'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="sub">{escape(sub)}</div></div>')


def kpi_grid(cards: list[str]) -> None:
    html(f'<div class="kpi-grid">{"".join(cards)}</div>')


# ------------------------------------------------------------------ forecast hero
def forecast_hero(title: str, status: str, subtitle: str, big: str, big_label: str,
                  change: float | None, change_label: str, acc: str, acc_label: str, acc_tone: str,
                  range_text: str, minis: list[tuple]) -> str:
    """The large "next N days" card: big number, accuracy box and three mini stats.

    `minis` is a list of (icon_name, colour, value, label, sub) tuples.
    """
    chip = ""
    if change is not None:
        cls = "up" if change > 0.05 else "down" if change < -0.05 else "flat"
        arrow = "▲" if cls == "up" else "▼" if cls == "down" else "■"
        chip = f'<span class="delta {cls}">{arrow} {abs(change):.0f}%</span>'
    mini_html = "".join(
        f'<div class="mini">{tile(i, c)}<div class="mini-val">{escape(v)}</div>'
        f'<div class="mini-label">{escape(lab)}</div><div class="mini-sub">{escape(s)}</div></div>'
        for i, c, v, lab, s in minis)
    return (
        '<div class="hero-card">'
        f'<div class="hero-top">{tile("spark", VIOLET)}<div class="hero-heading"><h3>{escape(title)}</h3>'
        f'<p>{escape(subtitle)}</p></div><span class="status"><i></i>{escape(status)}</span></div>'
        '<div class="hero-main">'
        f'<div class="big"><span class="num">{escape(big)}</span><div class="big-side"><span>{escape(big_label)}</span>'
        f'<div>{chip}<em>{escape(change_label if chip else "")}</em></div></div></div>'
        f'<div class="conf"><small>Backtest accuracy</small><div class="conf-row"><b>{escape(acc)}</b>'
        f'{icon("bars", "#6B7CFF", 18)}<span class="tag {acc_tone}">{escape(acc_label)}</span></div>'
        f'<small>{escape(range_text)}</small></div></div>'
        f'<div class="mini-grid">{mini_html}</div></div>')


# ------------------------------------------------------------------ side panel and cards
def drivers_panel(rows: list[dict]) -> str:
    """"What's driving sales": a list of icon rows with a coloured value on the right."""
    items = "".join(
        f'<div class="drv">{tile(r["icon"], r["color"])}<div class="drv-text"><b>{escape(r["title"])}</b>'
        f'<small>{escape(r["sub"])}</small></div><span class="val {r["tone"]}">{escape(r["value"])}</span></div>'
        for r in rows)
    return (f'<div class="side-card"><div class="side-head">{tile("pulse", PRIMARY)}<b>What\u2019s driving sales</b></div>'
            f'<div class="drv-list">{items}</div></div>')


def week_card(avg_text: str, values: list[float], labels: list[str]) -> str:
    """"A typical week": the average day plus seven mini bars (Mon..Sun)."""
    peak = max(values) or 1.0
    bars = "".join(
        f'<div class="wk-col"><i class="{"hot" if v == peak else ""}" style="height:{max(v / peak * 100, 4):.0f}%"></i>'
        f'<span>{escape(lab[0])}</span></div>' for v, lab in zip(values, labels))
    return (f'<div class="side-card week"><div class="wk-left"><small>A typical week</small><b>{escape(avg_text)}</b>'
            f'<small>average per day</small></div><div class="wk-bars">{bars}</div></div>')


def change_cards(cards: list[dict]) -> None:
    """Row of cards, one per product: name, trend tag, % change, sparkline, note.

    Each dict: name, tag, tone (up/down/flat), pct (text), values (sparkline), note, color.
    """
    out = []
    for c in cards:
        out.append(
            '<div class="chg">'
            f'<div class="chg-top"><span class="thumb" style="--c:{c["color"]}">{escape(c["name"][:1].upper())}</span>'
            f'<b>{escape(c["name"])}</b><span class="tag {c["tone"]}">{escape(c["tag"])}</span></div>'
            f'<div class="chg-mid"><span class="pct {c["tone"]}">{escape(c["pct"])}</span>{sparkline(c["values"], c["color"])}</div>'
            f'<p>{escape(c["note"])}</p></div>')
    html(f'<div class="chg-grid">{"".join(out)}</div>')


# ------------------------------------------------------------------ messages and tables
def notice(text: str, kind: str = "info", icon_char: str | None = None) -> None:
    """kind: info | warn | good. `text` may contain trusted markup (<b>)."""
    icon_char = icon_char or {"info": "ℹ️", "warn": "⚠️", "good": "✅"}[kind]
    html(f'<div class="notice {kind}"><span>{icon_char}</span><div>{text}</div></div>')


def empty_state(icon_char: str, title: str, text: str) -> None:
    html(f'<div class="empty"><div class="big">{icon_char}</div><h3>{escape(title)}</h3><p>{escape(text)}</p></div>')


def insight_card(icon_char: str, title: str, text_html: str) -> None:
    html(f'<div class="insight"><div class="ico-box">{icon_char}</div><div><div class="ttl">{escape(title)}</div>'
         f'<div class="txt">{text_html}</div></div></div>')


def steps(items: list[tuple[str, str]]) -> None:
    cards = "".join(f'<div class="step"><div class="n">{i}</div><b>{escape(t)}</b><span>{escape(d)}</span></div>'
                    for i, (t, d) in enumerate(items, 1))
    html(f'<div class="steps">{cards}</div>')


def trend_table(data: pd.DataFrame, number_fmt: str = "{:,.0f}") -> None:
    """Render a product trend table as styled HTML with colour badges."""
    if data.empty:
        return
    rows = []
    for r in data.itertuples(index=False):
        pct = float(r.change_pct)
        cls = "pos" if pct > 0 else "neg" if pct < 0 else ""
        rows.append(
            f'<tr><td><b>{escape(str(r.product))}</b></td><td>{TREND_BADGE[r.trend]}</td>'
            f'<td class="num {cls}">{pct:+.1f}%</td>'
            f'<td class="num">{number_fmt.format(r.total_sales)}</td>'
            f'<td class="num">{number_fmt.format(r.earlier_sales)}</td>'
            f'<td class="num">{number_fmt.format(r.recent_sales)}</td></tr>')
    html('<div class="panel tbl-wrap"><table class="tbl"><thead><tr><th>Product</th><th>Trend</th>'
         '<th class="num">Change</th><th class="num">Total sales</th><th class="num">Earlier</th>'
         f'<th class="num">Recent</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
