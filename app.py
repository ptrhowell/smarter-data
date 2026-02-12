"""
app.py
======
BTC Sequence Analysis Dashboard  –  Streamlit frontend.

A BrighterData-inspired P1/P2 statistical analysis tool for Bitcoin
that identifies intraday probabilities based on time and distance.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone

from data_fetcher import (
    fetch_historical_ohlcv,
    fetch_today_candles,
    find_working_exchange,
)
from analysis_engine import (
    SESSIONS,
    identify_p1_p2,
    identify_weekly_p1_p2,
    classify_volatility,
    build_macro_map,
    build_heatmap_matrix,
    get_summary_stats,
    analyze_today,
    build_session_breakdown,
    get_pivot_zones,
)

# ═════════════════════════════════════════════════════════════
# Page config
# ═════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BTC Sequence Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════
# Colour palette & Plotly defaults
# ═════════════════════════════════════════════════════════════

GREEN  = "#00d26a"
RED    = "#ff4b4b"
BLUE   = "#58a6ff"
GOLD   = "#ffd700"
CYAN   = "#79dae8"
PURPLE = "#bc8cff"
DIM    = "#484f58"


def _hour_to_session(h: int) -> str:
    """Map hour (0-23) to broad session name."""
    if 0 <= h < 8:
        return "Asian (00-08)"
    elif 8 <= h < 13:
        return "London (08-13)"
    elif 13 <= h < 17:
        return "NY AM (13-17)"
    elif 17 <= h < 22:
        return "NY PM (17-22)"
    return "Late (22-00)"


_PLOTLY = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="SF Mono, Menlo, monospace", color="#c9d1d9", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
)

# ═════════════════════════════════════════════════════════════
# Custom CSS
# ═════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* metric cards */
    [data-testid="stMetricValue"]  { font-size: 1.45rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem; }

    /* tighter page padding */
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }

    /* tab styling */
    button[data-baseweb="tab"] { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# Sidebar  –  configuration
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ Sequence Analyzer")
    st.caption("P1 / P2 Statistical Analysis for BTC")
    st.divider()

    exchange_id = st.selectbox(
        "Exchange", ["auto", "okx", "kraken", "bybit", "coinbase", "binance"], index=0
    )
    if exchange_id == "binance":
        st.caption("⚠️ Binance may not work from cloud hosts")
    symbol = st.text_input("Symbol", "BTC/USDT")
    timeframe = st.selectbox("Candle Timeframe", ["15m", "5m", "1h"], index=0)
    days = st.slider("History (days)", 30, 730, 365, step=30)

    st.divider()
    session_name = st.selectbox("Session", list(SESSIONS.keys()), index=0)
    direction = st.selectbox(
        "P1 Direction",
        ["Both", "Bullish (P1=Low)", "Bearish (P1=High)"],
    )

    st.divider()
    vol_filter = st.toggle("Volatility Filter (ATR)", value=False)
    atr_period = 14
    vol_class = "All"
    if vol_filter:
        atr_period = st.slider("ATR Period", 7, 30, 14)
        vol_class = st.radio("Show Days", ["High", "Normal"], horizontal=True)

    st.divider()
    hour_bin = st.select_slider(
        "Macro Map Hour Bins", options=[1, 2, 3, 4, 6], value=2
    )

    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ═════════════════════════════════════════════════════════════
# Data loading & analysis
# ═════════════════════════════════════════════════════════════

# --- Step 1: Find a working exchange ---
if exchange_id == "auto":
    with st.status("Finding a reachable exchange...", expanded=True) as status:
        st.write("Testing exchanges: okx → kraken → bybit → coinbase → binance")
        active_exchange = find_working_exchange("okx", symbol)
        if active_exchange:
            status.update(label=f"Connected to {active_exchange.upper()}", state="complete")
            st.write(f"✓ Using **{active_exchange}**")
        else:
            status.update(label="No exchange reachable", state="error")
            st.error(
                "Could not reach any exchange from this server. "
                "All major crypto exchanges may be blocking this cloud IP. "
                "Try running the app locally instead."
            )
            st.stop()
else:
    active_exchange = exchange_id

# --- Step 2: Fetch historical data ---
with st.status(f"Downloading {days} days of {timeframe} candles from {active_exchange.upper()}...", expanded=True) as status:
    st.write("This takes 1-3 min on the first run (cached for 1 hour after).")
    raw = fetch_historical_ohlcv(active_exchange, symbol, timeframe, days)
    if raw.empty:
        status.update(label="No data received", state="error")
        st.error(
            f"No data returned from {active_exchange}. "
            "Try a different exchange or check the symbol."
        )
        st.stop()
    status.update(label=f"Loaded {len(raw):,} candles from {active_exchange.upper()}", state="complete")

p1p2_all = identify_p1_p2(raw, session_name)

if p1p2_all.empty:
    st.error(
        "Could not identify any P1/P2 sequences. "
        "Try a different session, timeframe, or longer history."
    )
    st.stop()

p1p2_all = classify_volatility(p1p2_all, atr_period)

# Volatility filter
if vol_filter and vol_class != "All":
    p1p2 = p1p2_all[p1p2_all["volatility"] == vol_class].copy()
    if p1p2.empty:
        st.warning("No data after volatility filter – showing all data.")
        p1p2 = p1p2_all.copy()
else:
    p1p2 = p1p2_all.copy()

stats = get_summary_stats(p1p2)

# Session Breakdown  (always Full Day, independent of sidebar session)
p1p2_fullday = identify_p1_p2(raw, "Full Day (00:00-23:59)")
if not p1p2_fullday.empty:
    p1p2_fullday = classify_volatility(p1p2_fullday, atr_period)
    if vol_filter and vol_class != "All":
        fd_filtered = p1p2_fullday[p1p2_fullday["volatility"] == vol_class].copy()
        if fd_filtered.empty:
            fd_filtered = p1p2_fullday.copy()
    else:
        fd_filtered = p1p2_fullday.copy()
    sess_brk = build_session_breakdown(fd_filtered)
else:
    sess_brk = None

# Weekly P1/P2
weekly_p1p2 = identify_weekly_p1_p2(raw)

# Pivot Zones  (previous day + previous week P1/P2 prices)
pivots = get_pivot_zones(p1p2_fullday, weekly_p1p2)

# Macro map  (pre-computed – used in Summary + Macro Map tab)
macro = build_macro_map(p1p2, hour_bin, direction)

# Today
today_candles = fetch_today_candles(active_exchange, symbol, timeframe)
today = analyze_today(today_candles, stats, session_name)


# ═════════════════════════════════════════════════════════════
# Header row
# ═════════════════════════════════════════════════════════════

h1, h2, h3, h4, h5 = st.columns([3, 1, 1, 1, 1])

with h1:
    st.markdown(f"### {symbol}  ·  Sequence Analysis")
    st.caption(
        f"{session_name}  ·  {stats['total_days']} days  ·  {timeframe} candles"
        + (f"  ·  **{vol_class} vol**" if vol_filter else "")
    )
with h2:
    if today:
        st.metric("Price", f"${today['price']:,.0f}")
with h3:
    st.metric("Bullish %", f"{stats.get('bullish_pct', 0):.1f}%")
with h4:
    st.metric("Avg Exp", f"{stats.get('avg_expansion', 0):.2f}%")
with h5:
    st.metric("Med Exp", f"{stats.get('med_expansion', 0):.2f}%")

st.divider()


# ═════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════

(
    tab_live,
    tab_macro,
    tab_sess,
    tab_time,
    tab_exp,
    tab_data,
) = st.tabs(
    [
        "📡  Live",
        "🗺️  Macro Map",
        "📊  Sessions",
        "🕐  Time",
        "📈  Expansion",
        "📋  Data",
    ]
)


# ─────────────────────────────────────────────
# TAB 1 – Live Dashboard
# ─────────────────────────────────────────────

with tab_live:
    if not today:
        st.info("Today's session data is not yet available.")
    else:
        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        bull_bear = "Bullish" if today["p1_type"] == "Low" else "Bearish"
        m1.metric(
            "Today's P1",
            f"{today['p1_type']}  @ ${today['p1_price']:,.0f}",
            delta=bull_bear,
            delta_color="normal" if today["p1_type"] == "Low" else "inverse",
        )
        m2.metric(
            "Current Expansion",
            f"{today['expansion']:.2f}%",
            delta=f"Day range: {today.get('max_expansion', today['expansion']):.2f}%",
        )
        m3.metric(
            "Remaining (est.)",
            f"{today['remaining']:.2f}%",
            delta="to avg P2",
        )
        m4.metric(
            "Progress",
            f"{today['progress']:.0f}%",
            delta="of avg daily range",
        )

        # ── Plain-English Summary ──
        st.divider()
        st.markdown("#### 📋 Today's Summary")

        p1_h = today.get("p1_hour", 0)
        p1_sess = _hour_to_session(p1_h)
        p1_time_str = (
            today["p1_time"].strftime("%H:%M UTC")
            if hasattr(today["p1_time"], "strftime")
            else f"{p1_h:02d}:00 UTC"
        )
        direction_word = "bullish" if today["p1_type"] == "Low" else "bearish"
        progress = today["progress"]

        # Macro map context for this P1 time window
        macro_line = ""
        if not macro.empty:
            for _, mrow in macro.iterrows():
                window = mrow["Time Window"]
                try:
                    start_h = int(window.split("-")[0].split(":")[0])
                    end_h = int(window.split("-")[1].split(":")[0])
                    if start_h <= p1_h < end_h:
                        thresh_cols = [c for c in macro.columns if c.startswith(">")]
                        probs = []
                        for tc in thresh_cols[:4]:
                            val = mrow[tc]
                            if val > 0:
                                probs.append(f"**{val:.0f}%** chance of {tc}")
                        if probs:
                            macro_line = (
                                f"📊 Historical odds from the **{window}** window: "
                                + ", ".join(probs[:3]) + "."
                            )
                        break
                except Exception:
                    pass

        # P2 session prediction from cross-tab
        p2_line = ""
        if sess_brk is not None and "cross" in sess_brk:
            cross_tbl = sess_brk["cross"]
            if p1_sess in cross_tbl.index:
                crow = cross_tbl.loc[p1_sess].drop("All", errors="ignore")
                if not crow.empty and crow.sum() > 0:
                    top_p2 = crow.idxmax()
                    top_pct = crow.max() / crow.sum() * 100
                    p2_line = (
                        f"🎯 When P1 is set in **{p1_sess}**, P2 most often "
                        f"forms during **{top_p2}** ({top_pct:.0f}% of the time)."
                    )

        # Average P1→P2 duration
        avg_hours = stats.get("avg_p1p2_hours", 0)
        duration_line = (
            f"⏱️ P1→P2 typically takes **{avg_hours:.1f} hours** on average."
            if avg_hours > 0 else ""
        )

        # Weekly context
        weekly_line = ""
        if not weekly_p1p2.empty:
            cur_week = weekly_p1p2.iloc[-1]
            w_dir = "bullish" if cur_week["p1_type"] == "Low" else "bearish"
            # Most common P2 day-of-week historically
            p2_dow_mode = weekly_p1p2["p2_day"].mode()
            p2_day_hint = p2_dow_mode.iloc[0] if not p2_dow_mode.empty else "mid-week"
            weekly_line = (
                f"📅 This week's P1 ({cur_week['p1_type']}) was set on "
                f"**{cur_week['p1_day']}** at **${cur_week['p1_price']:,.0f}** "
                f"({w_dir} week so far) — "
                f"weekly P2 most often lands on **{p2_day_hint}**."
            )

        # Progress-based outlook
        if progress >= 100:
            outlook = (
                "✅ Today's expansion **exceeds the historical average**. "
                "The typical daily move may be complete."
            )
        elif progress >= 80:
            outlook = "⏳ Most of the typical daily expansion appears **near completion**."
        elif progress >= 50:
            outlook = (
                "⏳ The move is **underway** but history suggests "
                "there may be more room to extend."
            )
        else:
            outlook = "🔄 Based on history, there's likely **more expansion ahead**."

        # Assemble & render
        summary_lines = [
            (
                f"**P1 ({today['p1_type']})** was set at **{p1_time_str}** "
                f"during **{p1_sess}** — {direction_word} day so far."
            ),
            (
                f"Current expansion: **{today['expansion']:.2f}%** "
                f"(avg: {stats.get('avg_expansion', 0):.2f}%, "
                f"median: {stats.get('med_expansion', 0):.2f}%) "
                f"— **{progress:.0f}%** of typical range."
            ),
        ]
        if macro_line:
            summary_lines.append(macro_line)
        if p2_line:
            summary_lines.append(p2_line)
        if duration_line:
            summary_lines.append(duration_line)
        if weekly_line:
            summary_lines.append(weekly_line)
        summary_lines.append(outlook)

        st.markdown("\n\n".join(summary_lines))
        st.divider()

        # Gauge ┃ Candlestick
        g_col, c_col = st.columns([1, 2])

        with g_col:
            ceil = max(stats.get("p90_expansion", 5), today["expansion"]) * 1.3
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=today["expansion"],
                    number=dict(suffix="%", font=dict(size=38)),
                    gauge=dict(
                        axis=dict(range=[0, ceil], ticksuffix="%"),
                        bar=dict(
                            color=GREEN if today["p1_type"] == "Low" else RED
                        ),
                        bgcolor="#161b22",
                        borderwidth=0,
                        steps=[
                            dict(
                                range=[0, stats.get("p25_expansion", 1)],
                                color="#0d1117",
                            ),
                            dict(
                                range=[
                                    stats.get("p25_expansion", 1),
                                    stats.get("med_expansion", 2),
                                ],
                                color="#161b22",
                            ),
                            dict(
                                range=[
                                    stats.get("med_expansion", 2),
                                    stats.get("p75_expansion", 3),
                                ],
                                color="#1c2333",
                            ),
                            dict(
                                range=[
                                    stats.get("p75_expansion", 3),
                                    ceil,
                                ],
                                color="#21262d",
                            ),
                        ],
                        threshold=dict(
                            line=dict(color=GOLD, width=3),
                            thickness=0.8,
                            value=stats.get("avg_expansion", 2),
                        ),
                    ),
                    title=dict(text="Expansion Gauge", font=dict(size=14)),
                )
            )
            fig_g.update_layout(**_PLOTLY, height=340)
            st.plotly_chart(fig_g, use_container_width=True)

        with c_col:
            if not today_candles.empty:
                fig_c = go.Figure(
                    go.Candlestick(
                        x=today_candles.index,
                        open=today_candles["open"],
                        high=today_candles["high"],
                        low=today_candles["low"],
                        close=today_candles["close"],
                        increasing_line_color=GREEN,
                        decreasing_line_color=RED,
                        name="BTC",
                    )
                )
                # P1 marker
                fig_c.add_annotation(
                    x=today["p1_time"],
                    y=today["p1_price"],
                    text=f"P1 ({today['p1_type']})",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=GOLD,
                    font=dict(color=GOLD, size=11),
                    bgcolor="#161b22",
                    bordercolor=GOLD,
                )
                # Avg-P2 target line
                if today["p1_type"] == "Low":
                    tgt = today["p1_price"] * (1 + stats["avg_expansion"] / 100)
                else:
                    tgt = today["p1_price"] * (1 - stats["avg_expansion"] / 100)
                fig_c.add_hline(
                    y=tgt,
                    line_dash="dash",
                    line_color=GOLD,
                    annotation_text=f"Avg P2: ${tgt:,.0f}",
                    annotation_font_color=GOLD,
                )

                # ── Pivot Zone lines (daily only – weekly too far from intraday range) ──
                _pz_cfg = [
                    ("prev_d_p1", "Prev D·P1", "dash", DIM),
                    ("prev_d_p2", "Prev D·P2", "dash", DIM),
                ]
                for key, label, dash, color in _pz_cfg:
                    pz = pivots.get(key)
                    if pz is not None:
                        fig_c.add_hline(
                            y=pz["price"],
                            line_dash=dash,
                            line_color=color,
                            line_width=1,
                            annotation_text=f"{label} ${pz['price']:,.0f}",
                            annotation_font_color=color,
                            annotation_font_size=9,
                            annotation_position="top left" if "p1" in key else "bottom left",
                        )

                fig_c.update_layout(
                    **_PLOTLY,
                    height=340,
                    title=dict(text="Today's Session", font=dict(size=14)),
                    xaxis_rangeslider_visible=False,
                    showlegend=False,
                )
                st.plotly_chart(fig_c, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2 – Macro Map
# ─────────────────────────────────────────────

with tab_macro:
    if macro.empty:
        st.info("Not enough data to generate the Macro Map.")
    else:
        st.subheader("Macro Map  –  P1 Time → P2 Expansion Probability")
        st.caption(
            "Read: \"If P1 is set in [Time Window], what is the probability "
            "P2 expands ≥ X%?\""
        )

        # Heatmap
        thresh_cols = [c for c in macro.columns if c.startswith(">")]
        z = macro[thresh_cols].values

        fig_m = go.Figure(
            go.Heatmap(
                z=z,
                x=thresh_cols,
                y=macro["Time Window"],
                text=z,
                texttemplate="%{text:.0f}",
                textfont=dict(size=11),
                colorscale=[
                    [0.0,  "#0d1117"],
                    [0.20, "#1a3a5c"],
                    [0.40, "#1f6f8b"],
                    [0.60, "#e2d810"],
                    [0.80, "#ff9f1c"],
                    [1.0,  "#00d26a"],
                ],
                colorbar=dict(title="Prob %", ticksuffix="%"),
                hovertemplate=(
                    "P1 Window: %{y}<br>Threshold: %{x}<br>"
                    "Probability: %{z:.1f}%<extra></extra>"
                ),
            )
        )
        fig_m.update_layout(
            **_PLOTLY,
            height=max(420, len(macro) * 38),
            title=dict(
                text="Expansion Probability by P1 Time Window",
                font=dict(size=14),
            ),
            xaxis=dict(title="Expansion Threshold", side="top"),
            yaxis=dict(title="P1 Time Window (UTC)", autorange="reversed"),
        )
        st.plotly_chart(fig_m, use_container_width=True)

        # Summary table + bar chart
        st.divider()
        tl, tr = st.columns(2)
        with tl:
            st.markdown("**Sample Count & Average Expansion**")
            st.dataframe(
                macro[["Time Window", "N", "Avg %", "Med %"]],
                use_container_width=True,
                hide_index=True,
            )
        with tr:
            fig_n = go.Figure(
                go.Bar(
                    x=macro["Time Window"],
                    y=macro["N"],
                    marker_color=BLUE,
                    text=macro["N"],
                    textposition="auto",
                )
            )
            fig_n.update_layout(
                **_PLOTLY,
                height=320,
                title=dict(
                    text="Days per P1 Time Window", font=dict(size=13)
                ),
                xaxis=dict(tickangle=-45),
                yaxis=dict(title="Count"),
            )
            st.plotly_chart(fig_n, use_container_width=True)

        # Macro map as downloadable CSV
        csv_macro = macro.to_csv(index=False)
        st.download_button(
            "📥  Download Macro Map CSV",
            csv_macro,
            "macro_map.csv",
            "text/csv",
        )


# ─────────────────────────────────────────────
# TAB 3 – Session Breakdown  (always Full Day)
# ─────────────────────────────────────────────

SESSION_COLORS = {
    "Asian (00-08)":  "#58a6ff",
    "London (08-13)": "#bc8cff",
    "NY AM (13-17)":  "#00d26a",
    "NY PM (17-22)":  "#ff9f1c",
    "Late (22-00)":   "#484f58",
}

with tab_sess:
    if sess_brk is None:
        st.info("Session breakdown requires Full Day data.")
    else:
        st.subheader("Session Breakdown  –  Which Session Produces P1 & P2?")
        st.caption(
            "Based on **Full Day** analysis (independent of Session filter). "
            "Shows which trading session the daily high/low extreme typically falls in."
        )

        # ── Donut charts: P1 session / P2 session ──
        s1, s2 = st.columns(2)

        with s1:
            labels_p1 = sess_brk["p1_pct"].index.tolist()
            vals_p1   = sess_brk["p1_pct"].values.tolist()
            cols_p1   = [SESSION_COLORS.get(l, DIM) for l in labels_p1]
            fig_p1 = go.Figure(go.Pie(
                labels=labels_p1,
                values=vals_p1,
                hole=0.50,
                marker=dict(colors=cols_p1),
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_p1.update_layout(
                **_PLOTLY, height=340,
                title=dict(text="Which Session Produces P1?", font=dict(size=14)),
                showlegend=False,
            )
            st.plotly_chart(fig_p1, use_container_width=True)

        with s2:
            labels_p2 = sess_brk["p2_pct"].index.tolist()
            vals_p2   = sess_brk["p2_pct"].values.tolist()
            cols_p2   = [SESSION_COLORS.get(l, DIM) for l in labels_p2]
            fig_p2 = go.Figure(go.Pie(
                labels=labels_p2,
                values=vals_p2,
                hole=0.50,
                marker=dict(colors=cols_p2),
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_p2.update_layout(
                **_PLOTLY, height=340,
                title=dict(text="Which Session Produces P2?", font=dict(size=14)),
                showlegend=False,
            )
            st.plotly_chart(fig_p2, use_container_width=True)

        # ── Expansion by P1 session ──
        st.divider()
        st.subheader("Expansion by P1 Session")
        st.caption(
            "When P1 is set during a given session, how much does the market "
            "typically expand to P2?"
        )

        ex1, ex2 = st.columns([1, 1])

        with ex1:
            exp_tbl = sess_brk["exp_by_p1"].reset_index()
            exp_tbl.columns = ["Session", "Days", "Avg Expansion %", "Median Expansion %"]
            st.dataframe(exp_tbl, use_container_width=True, hide_index=True)

        with ex2:
            exp_data = sess_brk["exp_by_p1"].dropna()
            fig_ebar = go.Figure()
            fig_ebar.add_trace(go.Bar(
                x=exp_data.index,
                y=exp_data["Avg Exp %"],
                name="Average",
                marker_color=BLUE,
                text=exp_data["Avg Exp %"].apply(lambda v: f"{v:.2f}%"),
                textposition="outside",
            ))
            fig_ebar.add_trace(go.Bar(
                x=exp_data.index,
                y=exp_data["Med Exp %"],
                name="Median",
                marker_color=CYAN,
                text=exp_data["Med Exp %"].apply(lambda v: f"{v:.2f}%"),
                textposition="outside",
            ))
            fig_ebar.update_layout(
                **_PLOTLY, height=340, barmode="group",
                title=dict(text="Avg vs Median Expansion by P1 Session", font=dict(size=13)),
                yaxis=dict(title="Expansion %"),
                legend=dict(x=0.7, y=0.95),
            )
            st.plotly_chart(fig_ebar, use_container_width=True)

        # ── Cross-tabulation heatmap ──
        st.divider()
        st.subheader("P1 Session → P2 Session Flow")
        st.caption(
            "If P1 is set in [row session], which session does P2 land in? "
            "Read across a row to see the distribution."
        )

        cross = sess_brk["cross"]
        # Remove the 'All' margins for the heatmap
        cross_clean = cross.drop("All", axis=0, errors="ignore").drop("All", axis=1, errors="ignore")

        fig_cross = go.Figure(go.Heatmap(
            z=cross_clean.values,
            x=cross_clean.columns.tolist(),
            y=cross_clean.index.tolist(),
            text=cross_clean.values,
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale=[
                [0.0,  "#0d1117"],
                [0.25, "#1a3a5c"],
                [0.50, "#1f6f8b"],
                [0.75, "#e2d810"],
                [1.0,  "#00d26a"],
            ],
            hovertemplate="P1: %{y}<br>P2: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cross.update_layout(
            **_PLOTLY, height=350,
            title=dict(text="P1 Session × P2 Session (count)", font=dict(size=13)),
            xaxis=dict(title="P2 Session", side="top"),
            yaxis=dict(title="P1 Session", autorange="reversed"),
        )
        st.plotly_chart(fig_cross, use_container_width=True)

        # ── Weekly Sequence ──
        if not weekly_p1p2.empty:
            st.divider()
            st.subheader("Weekly Sequence  –  Which Day Produces Weekly P1 & P2?")
            st.caption(
                "Same P1/P2 logic applied to the full trading week. "
                "Shows which day of the week the weekly high/low typically forms."
            )

            wk1, wk2 = st.columns(2)

            _DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            with wk1:
                p1_day_counts = (
                    weekly_p1p2["p1_day"]
                    .value_counts()
                    .reindex(_DOW, fill_value=0)
                )
                fig_wp1 = go.Figure(go.Bar(
                    x=_DOW,
                    y=p1_day_counts.values,
                    marker_color=BLUE,
                    text=p1_day_counts.values,
                    textposition="outside",
                ))
                fig_wp1.update_layout(
                    **_PLOTLY, height=300,
                    title=dict(text="Weekly P1 by Day", font=dict(size=13)),
                    yaxis=dict(title="Count"),
                )
                st.plotly_chart(fig_wp1, use_container_width=True)

            with wk2:
                p2_day_counts = (
                    weekly_p1p2["p2_day"]
                    .value_counts()
                    .reindex(_DOW, fill_value=0)
                )
                fig_wp2 = go.Figure(go.Bar(
                    x=_DOW,
                    y=p2_day_counts.values,
                    marker_color=PURPLE,
                    text=p2_day_counts.values,
                    textposition="outside",
                ))
                fig_wp2.update_layout(
                    **_PLOTLY, height=300,
                    title=dict(text="Weekly P2 by Day", font=dict(size=13)),
                    yaxis=dict(title="Count"),
                )
                st.plotly_chart(fig_wp2, use_container_width=True)

            # Compact weekly stats row
            n_weeks = len(weekly_p1p2)
            w_bull = (weekly_p1p2["p1_type"] == "Low").sum()
            w_avg = weekly_p1p2["expansion_pct"].mean()
            w_med = weekly_p1p2["expansion_pct"].median()
            wm1, wm2, wm3, wm4 = st.columns(4)
            wm1.metric("Weeks Analysed", n_weeks)
            wm2.metric("Bullish Weeks", f"{w_bull}/{n_weeks} ({w_bull/max(n_weeks,1)*100:.0f}%)")
            wm3.metric("Avg Weekly Exp", f"{w_avg:.2f}%")
            wm4.metric("Med Weekly Exp", f"{w_med:.2f}%")


# ─────────────────────────────────────────────
# TAB 4 – Time Analysis
# ─────────────────────────────────────────────

with tab_time:
    hour_counts, dow_pivot = build_heatmap_matrix(p1p2, direction)

    cl, cr = st.columns(2)

    with cl:
        st.subheader("P1 Hour Distribution")
        fig_h = go.Figure(
            go.Bar(
                x=[f"{h:02d}:00" for h in range(24)],
                y=hour_counts.values,
                marker_color=BLUE,
                text=hour_counts.values,
                textposition="outside",
            )
        )
        fig_h.update_layout(
            **_PLOTLY,
            height=360,
            title=dict(
                text="When Does P1 Occur Most Often?", font=dict(size=13)
            ),
            xaxis=dict(title="Hour (UTC)"),
            yaxis=dict(title="Frequency"),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with cr:
        st.subheader("Day × Hour Heatmap")
        fig_dh = go.Figure(
            go.Heatmap(
                z=dow_pivot.values,
                x=dow_pivot.columns,
                y=dow_pivot.index,
                text=dow_pivot.values,
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorscale="Viridis",
                hovertemplate="%{y} %{x}:00 UTC  ·  Count: %{z}<extra></extra>",
            )
        )
        fig_dh.update_layout(
            **_PLOTLY,
            height=360,
            title=dict(
                text="P1 Frequency: Day-of-Week × Hour",
                font=dict(size=13),
            ),
            xaxis=dict(title="Hour (UTC)"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_dh, use_container_width=True)

    # Duration analysis
    st.divider()
    st.subheader("P1 → P2 Duration")
    d1, d2 = st.columns(2)

    with d1:
        fig_dur = go.Figure(
            go.Histogram(
                x=p1p2["p1_to_p2_minutes"] / 60,
                nbinsx=30,
                marker_color=PURPLE,
                opacity=0.85,
            )
        )
        fig_dur.add_vline(
            x=stats.get("avg_p1p2_hours", 0),
            line_dash="dash",
            line_color=GOLD,
            annotation_text=f"Avg: {stats.get('avg_p1p2_hours', 0):.1f} h",
        )
        fig_dur.update_layout(
            **_PLOTLY,
            height=310,
            title=dict(text="P1→P2 Duration (hours)", font=dict(size=13)),
            xaxis=dict(title="Hours"),
            yaxis=dict(title="Count"),
        )
        st.plotly_chart(fig_dur, use_container_width=True)

    with d2:
        fig_sc = go.Figure(
            go.Scatter(
                x=p1p2["p1_hour"],
                y=p1p2["p2_hour"],
                mode="markers",
                marker=dict(
                    color=p1p2["expansion_pct"],
                    colorscale="Turbo",
                    size=6,
                    opacity=0.65,
                    colorbar=dict(title="Exp %"),
                ),
                hovertemplate=(
                    "P1 %{x}:00 → P2 %{y}:00<br>"
                    "Expansion: %{marker.color:.2f}%<extra></extra>"
                ),
            )
        )
        fig_sc.update_layout(
            **_PLOTLY,
            height=310,
            title=dict(text="P1 Hour vs P2 Hour", font=dict(size=13)),
            xaxis=dict(title="P1 Hour (UTC)", dtick=2),
            yaxis=dict(title="P2 Hour (UTC)", dtick=2),
        )
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4 – Expansion Analysis
# ─────────────────────────────────────────────

with tab_exp:
    bull_exp = p1p2.loc[p1p2["p1_type"] == "Low", "expansion_pct"]
    bear_exp = p1p2.loc[p1p2["p1_type"] == "High", "expansion_pct"]

    e1, e2 = st.columns(2)

    with e1:
        st.subheader("Expansion Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=bull_exp,
                name="Bullish (P1=Low)",
                marker_color=GREEN,
                opacity=0.70,
                nbinsx=40,
            )
        )
        fig_dist.add_trace(
            go.Histogram(
                x=bear_exp,
                name="Bearish (P1=High)",
                marker_color=RED,
                opacity=0.70,
                nbinsx=40,
            )
        )
        fig_dist.add_vline(
            x=stats.get("avg_expansion", 0),
            line_dash="dash",
            line_color=GOLD,
            annotation_text=f"Avg: {stats.get('avg_expansion', 0):.2f}%",
        )
        fig_dist.update_layout(
            **_PLOTLY,
            height=360,
            barmode="overlay",
            title=dict(text="Expansion % Distribution", font=dict(size=13)),
            xaxis=dict(title="Expansion %"),
            yaxis=dict(title="Count"),
            legend=dict(x=0.65, y=0.95),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with e2:
        st.subheader("Expansion Over Time")
        dates = pd.to_datetime(p1p2["date"])
        rolling = p1p2["expansion_pct"].rolling(14, min_periods=1).mean()
        colours = [
            GREEN if t == "Low" else RED for t in p1p2["p1_type"]
        ]
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=dates,
                y=p1p2["expansion_pct"],
                mode="markers",
                marker=dict(size=4, color=colours, opacity=0.6),
                name="Daily",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=dates,
                y=rolling,
                mode="lines",
                line=dict(color=GOLD, width=2),
                name="14-day MA",
            )
        )
        fig_ts.update_layout(
            **_PLOTLY,
            height=360,
            title=dict(
                text="Daily Expansion with 14-day MA", font=dict(size=13)
            ),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Expansion %"),
            showlegend=False,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Breakdown cards
    st.divider()
    st.subheader("Bullish vs Bearish Breakdown")
    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown(
            f"**Bullish (P1=Low): {stats.get('bullish_days', 0)} days "
            f"({stats.get('bullish_pct', 0):.1f}%)**"
        )
        if not bull_exp.empty:
            st.markdown(
                f"Avg: **{bull_exp.mean():.2f}%** · "
                f"Med: **{bull_exp.median():.2f}%** · "
                f"Std: **{bull_exp.std():.2f}%**"
            )
    with b2:
        st.markdown(
            f"**Bearish (P1=High): {stats.get('bearish_days', 0)} days "
            f"({stats.get('bearish_pct', 0):.1f}%)**"
        )
        if not bear_exp.empty:
            st.markdown(
                f"Avg: **{bear_exp.mean():.2f}%** · "
                f"Med: **{bear_exp.median():.2f}%** · "
                f"Std: **{bear_exp.std():.2f}%**"
            )
    with b3:
        st.markdown(f"**Overall: {stats.get('total_days', 0)} days**")
        st.markdown(
            f"Avg: **{stats.get('avg_expansion', 0):.2f}%** · "
            f"P90: **{stats.get('p90_expansion', 0):.2f}%** · "
            f"Max: **{stats.get('max_expansion', 0):.2f}%**"
        )


# ─────────────────────────────────────────────
# TAB 5 – Data Explorer
# ─────────────────────────────────────────────

with tab_data:
    st.subheader("P1/P2 Sequence Data")

    show_df = p1p2[
        [
            "date",
            "p1_type",
            "p1_hour",
            "p1_price",
            "p2_type",
            "p2_hour",
            "p2_price",
            "expansion_pct",
            "p1_to_p2_minutes",
            "daily_range_pct",
            "volatility",
        ]
    ].copy()

    show_df.columns = [
        "Date",
        "P1",
        "P1 Hour",
        "P1 Price",
        "P2",
        "P2 Hour",
        "P2 Price",
        "Expansion %",
        "P1→P2 min",
        "Range %",
        "Vol",
    ]

    st.dataframe(
        show_df.style.format(
            {
                "P1 Price": "${:,.0f}",
                "P2 Price": "${:,.0f}",
                "Expansion %": "{:.2f}%",
                "P1→P2 min": "{:.0f}",
                "Range %": "{:.2f}%",
            }
        ),
        use_container_width=True,
        height=520,
    )

    csv = show_df.to_csv(index=False)
    st.download_button(
        "📥  Download CSV", csv, "btc_p1p2_data.csv", "text/csv"
    )


# ═════════════════════════════════════════════════════════════
# Footer
# ═════════════════════════════════════════════════════════════

st.divider()
st.caption(
    f"Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ·  "
    f"Source: {exchange_id.title()}  ·  "
    f"{len(p1p2)} sessions analysed  ·  "
    f"Inspired by [BrighterData](https://brighterdata.com)"
)
