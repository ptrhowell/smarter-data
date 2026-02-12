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

from data_fetcher import fetch_historical_ohlcv, fetch_today_candles
from analysis_engine import (
    SESSIONS,
    identify_p1_p2,
    classify_volatility,
    build_macro_map,
    build_heatmap_matrix,
    get_summary_stats,
    analyze_today,
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
        "Exchange", ["bybit", "okx", "binance", "coinbase"], index=0
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

raw = fetch_historical_ohlcv(exchange_id, symbol, timeframe, days)

if raw.empty:
    st.error(
        "No data returned from the exchange. "
        "Check your symbol / exchange settings and internet connection."
    )
    st.stop()

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

# Today
today_candles = fetch_today_candles(exchange_id, symbol, timeframe)
today = analyze_today(today_candles, stats)


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
    tab_time,
    tab_exp,
    tab_data,
) = st.tabs(
    [
        "📡  Live Dashboard",
        "🗺️  Macro Map",
        "🕐  Time Analysis",
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
            delta=f"Avg target: {today['avg_exp']:.2f}%",
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
    macro = build_macro_map(p1p2, hour_bin, direction)

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
# TAB 3 – Time Analysis
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
