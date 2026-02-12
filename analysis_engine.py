"""
analysis_engine.py
==================
Statistical analysis engine for P1/P2 Sequence Analysis.

Core responsibilities
---------------------
* Identify P1 (first extreme) and P2 (second extreme) per session/day.
* Build the "Macro Map" probability table.
* Generate heatmap matrices (hour, day-of-week).
* ATR-based volatility classification.
* Summarise expansion statistics.
* Analyse today's live session against historical baselines.
"""

import pandas as pd
import numpy as np
from datetime import time as dt_time


# ──────────────────────────────────────────────────────────────
# Session definitions  (all times in UTC)
# ──────────────────────────────────────────────────────────────

SESSIONS: dict[str, dict] = {
    "Full Day (00:00-23:59)":       {"start": dt_time(0, 0),  "end": dt_time(23, 59)},
    "Asian (00:00-08:00)":          {"start": dt_time(0, 0),  "end": dt_time(8, 0)},
    "London (08:00-13:00)":         {"start": dt_time(8, 0),  "end": dt_time(13, 0)},
    "NY AM (13:00-17:00)":          {"start": dt_time(13, 0), "end": dt_time(17, 0)},
    "NY PM (17:00-22:00)":          {"start": dt_time(17, 0), "end": dt_time(22, 0)},
    "London Close (15:00-17:00)":   {"start": dt_time(15, 0), "end": dt_time(17, 0)},
    "Asia Kill Zone (01:00-05:00)": {"start": dt_time(1, 0),  "end": dt_time(5, 0)},
    "London KZ (07:00-10:00)":      {"start": dt_time(7, 0),  "end": dt_time(10, 0)},
    "NY Kill Zone (12:00-15:00)":   {"start": dt_time(12, 0), "end": dt_time(15, 0)},
}


# ──────────────────────────────────────────────────────────────
# P1 / P2 identification
# ──────────────────────────────────────────────────────────────

def identify_p1_p2(
    intraday_df: pd.DataFrame,
    session_name: str = "Full Day (00:00-23:59)",
) -> pd.DataFrame:
    """
    Walk through each calendar day, filter to the chosen session
    window, and determine which extreme (High or Low) was printed
    first.

    Returns one row per day with: date, p1/p2 type, time, price,
    expansion %, duration, etc.
    """
    session = SESSIONS.get(session_name, SESSIONS["Full Day (00:00-23:59)"])
    df = intraday_df.copy()
    df["date"] = df.index.date
    rows: list[dict] = []

    for date_val, day_data in df.groupby("date"):
        # Filter to session window (skip filter for full-day)
        if "Full Day" in session_name:
            sdata = day_data
        else:
            sdata = day_data.between_time(
                session["start"], session["end"], inclusive="left"
            )

        if len(sdata) < 2:
            continue

        high_idx = sdata["high"].idxmax()
        low_idx  = sdata["low"].idxmin()
        high_px  = sdata.loc[high_idx, "high"]
        low_px   = sdata.loc[low_idx, "low"]

        if high_idx == low_idx:
            continue  # both extremes on same candle – indeterminate

        if low_idx < high_idx:
            p1_type, p2_type = "Low", "High"
            p1_t, p2_t = low_idx, high_idx
            p1_px, p2_px = low_px, high_px
        else:
            p1_type, p2_type = "High", "Low"
            p1_t, p2_t = high_idx, low_idx
            p1_px, p2_px = high_px, low_px

        expansion = abs(p2_px - p1_px) / p1_px * 100
        duration  = (p2_t - p1_t).total_seconds() / 60  # minutes

        rows.append({
            "date":              date_val,
            "p1_type":           p1_type,
            "p1_hour":           p1_t.hour,
            "p1_minute":         p1_t.minute,
            "p1_time":           p1_t,
            "p1_price":          p1_px,
            "p2_type":           p2_type,
            "p2_hour":           p2_t.hour,
            "p2_minute":         p2_t.minute,
            "p2_time":           p2_t,
            "p2_price":          p2_px,
            "expansion_pct":     expansion,
            "p1_to_p2_minutes":  duration,
            "session_open":      sdata.iloc[0]["open"],
            "session_close":     sdata.iloc[-1]["close"],
            "session_high":      high_px,
            "session_low":       low_px,
            "daily_range_pct":   (high_px - low_px) / low_px * 100,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Volatility classification (ATR-like)
# ──────────────────────────────────────────────────────────────

def classify_volatility(
    p1p2_df: pd.DataFrame,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Add rolling ATR proxy and High/Normal label."""
    df = p1p2_df.copy()
    df["atr"] = df["daily_range_pct"].rolling(
        window=atr_period, min_periods=1
    ).mean()
    median_atr = df["atr"].median()
    df["volatility"] = np.where(
        df["daily_range_pct"] > median_atr, "High", "Normal"
    )
    return df


# ──────────────────────────────────────────────────────────────
# Macro Map  (probability table)
# ──────────────────────────────────────────────────────────────

def build_macro_map(
    p1p2_df: pd.DataFrame,
    hour_bin_size: int = 2,
    direction_filter: str = "Both",
) -> pd.DataFrame:
    """
    Rows  : P1 time-of-day windows (hour bins).
    Cols  : Expansion thresholds  (>= X %).
    Values: Probability (0-100) that P2 reaches at least that level.
    """
    df = _apply_direction_filter(p1p2_df, direction_filter)
    if df.empty:
        return pd.DataFrame()

    bins = list(range(0, 25, hour_bin_size))
    if bins[-1] < 24:
        bins.append(24)
    labels = [
        f"{bins[i]:02d}:00-{bins[i+1]:02d}:00" for i in range(len(bins) - 1)
    ]
    df["tw"] = pd.cut(df["p1_hour"], bins=bins, labels=labels, right=False)

    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    result_rows: list[dict] = []

    for label in labels:
        subset = df[df["tw"] == label]
        n = len(subset)
        row: dict = {"Time Window": label, "N": n}

        if n == 0:
            for t in thresholds:
                row[f">{t}%"] = 0.0
            row["Avg %"] = 0.0
            row["Med %"] = 0.0
        else:
            for t in thresholds:
                row[f">{t}%"] = round(
                    (subset["expansion_pct"] >= t).mean() * 100, 1
                )
            row["Avg %"] = round(subset["expansion_pct"].mean(), 2)
            row["Med %"] = round(subset["expansion_pct"].median(), 2)

        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ──────────────────────────────────────────────────────────────
# Time-heatmap data
# ──────────────────────────────────────────────────────────────

def build_heatmap_matrix(
    p1p2_df: pd.DataFrame,
    direction_filter: str = "Both",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns
    -------
    hour_counts : pd.Series  – P1 frequency per hour (0-23).
    pivot       : pd.DataFrame – rows=day-of-week, cols=hour,
                                 values=count.
    """
    df = _apply_direction_filter(p1p2_df, direction_filter)

    hour_counts = (
        df["p1_hour"]
        .value_counts()
        .reindex(range(24), fill_value=0)
        .sort_index()
    )

    df = df.copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    pivot = (
        df.groupby(["dow", "p1_hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [f"{h:02d}" for h in range(24)]
    return hour_counts, pivot


# ──────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────

def get_summary_stats(p1p2_df: pd.DataFrame) -> dict:
    """Compute descriptive statistics for expansion & direction."""
    if p1p2_df.empty:
        return {}
    exp = p1p2_df["expansion_pct"]
    return {
        "total_days":        len(p1p2_df),
        "bullish_days":      int((p1p2_df["p1_type"] == "Low").sum()),
        "bearish_days":      int((p1p2_df["p1_type"] == "High").sum()),
        "bullish_pct":       round((p1p2_df["p1_type"] == "Low").mean() * 100, 1),
        "bearish_pct":       round((p1p2_df["p1_type"] == "High").mean() * 100, 1),
        "avg_expansion":     round(exp.mean(), 2),
        "med_expansion":     round(exp.median(), 2),
        "std_expansion":     round(exp.std(), 2),
        "min_expansion":     round(exp.min(), 2),
        "max_expansion":     round(exp.max(), 2),
        "p25_expansion":     round(exp.quantile(0.25), 2),
        "p75_expansion":     round(exp.quantile(0.75), 2),
        "p90_expansion":     round(exp.quantile(0.90), 2),
        "avg_p1p2_hours":    round(p1p2_df["p1_to_p2_minutes"].mean() / 60, 1),
        "med_p1p2_hours":    round(p1p2_df["p1_to_p2_minutes"].median() / 60, 1),
    }


# ──────────────────────────────────────────────────────────────
# Live / today analysis
# ──────────────────────────────────────────────────────────────

def analyze_today(today_df: pd.DataFrame, stats: dict) -> dict:
    """Compare today's running P1/P2 against historical baselines."""
    if today_df.empty or not stats:
        return {}

    high_idx = today_df["high"].idxmax()
    low_idx  = today_df["low"].idxmin()
    hi = today_df.loc[high_idx, "high"]
    lo = today_df.loc[low_idx,  "low"]
    last = today_df.iloc[-1]["close"]

    if low_idx < high_idx:
        p1_type, p1_px, p1_t = "Low", lo, low_idx
    elif high_idx < low_idx:
        p1_type, p1_px, p1_t = "High", hi, high_idx
    else:
        p1_type, p1_px, p1_t = "Undetermined", lo, low_idx

    expansion = (hi - lo) / lo * 100
    avg_exp = stats.get("avg_expansion", 1)
    med_exp = stats.get("med_expansion", 1)

    return {
        "price":       last,
        "high":        hi,
        "low":         lo,
        "p1_type":     p1_type,
        "p1_price":    p1_px,
        "p1_time":     p1_t,
        "p1_hour":     p1_t.hour,
        "expansion":   expansion,
        "avg_exp":     avg_exp,
        "med_exp":     med_exp,
        "progress":    min(expansion / avg_exp * 100, 200) if avg_exp else 0,
        "remaining":   max(avg_exp - expansion, 0),
    }


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _apply_direction_filter(
    df: pd.DataFrame, direction: str
) -> pd.DataFrame:
    if direction == "Bullish (P1=Low)":
        return df[df["p1_type"] == "Low"].copy()
    if direction == "Bearish (P1=High)":
        return df[df["p1_type"] == "High"].copy()
    return df.copy()
