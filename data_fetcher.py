"""
data_fetcher.py
===============
Data-fetching module for BTC Sequence Analysis Dashboard.

Handles exchange connectivity, historical OHLCV retrieval (with
pagination and Streamlit caching), and intraday data for the current
session.

NOTE: Binance blocks requests from most cloud server IPs.
      Bybit / OKX are used as cloud-friendly defaults.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time as _time
import streamlit as st


# ──────────────────────────────────────────────────────────────
# Exchange factory
# ──────────────────────────────────────────────────────────────

def create_exchange(exchange_id: str = "bybit") -> ccxt.Exchange:
    """Return a ccxt exchange instance configured for public access."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({
        "enableRateLimit": True,
        "timeout": 15000,           # 15 s per request
        "options": {"defaultType": "spot"},
    })


# ──────────────────────────────────────────────────────────────
# Historical data  (cached for 1 hour)
# ──────────────────────────────────────────────────────────────

MAX_RETRIES = 3          # per-request retry cap
MAX_CONSECUTIVE_FAILS = 5 # give up after this many back-to-back errors

@st.cache_data(
    ttl=3600,
    show_spinner="Downloading historical candles (first run may take 1-2 min)...",
)
def fetch_historical_ohlcv(
    exchange_id: str = "bybit",
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    days: int = 365,
) -> pd.DataFrame:
    """
    Paginate through the exchange REST API and return a clean
    DataFrame indexed by UTC datetime.

    Columns: timestamp, open, high, low, close, volume
    """
    exchange = create_exchange(exchange_id)
    since = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_candles: list[list] = []
    limit = 1000
    consecutive_fails = 0

    while since < now_ms:
        retries = 0
        batch = None

        while retries < MAX_RETRIES:
            try:
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
                consecutive_fails = 0   # reset on success
                break                    # got data
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                retries += 1
                _time.sleep(2 * retries)  # back off
            except ccxt.BaseError:
                retries += 1
                _time.sleep(2)
            except Exception:
                retries += 1
                _time.sleep(2)

        if batch is None or retries >= MAX_RETRIES:
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break          # stop — exchange likely unreachable
            since += limit * 900_000   # skip ahead ~1000 candles worth
            continue

        if not batch:
            break

        all_candles.extend(batch)
        since = batch[-1][0] + 1    # move past the last candle
        _time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ──────────────────────────────────────────────────────────────
# Today's candles  (cached for 5 min)
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_today_candles(
    exchange_id: str = "bybit",
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
) -> pd.DataFrame:
    """Fetch candles from today's midnight UTC to now."""
    exchange = create_exchange(exchange_id)
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    since = int(today_start.timestamp() * 1000)

    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    except Exception:
        return pd.DataFrame()

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    return df
