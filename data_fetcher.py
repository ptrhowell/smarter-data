"""
data_fetcher.py
===============
Data-fetching module for BTC Sequence Analysis Dashboard.

Includes auto-fallback across multiple exchanges and visible
diagnostic logging for cloud deployment debugging.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
import time as _time
import streamlit as st

# Exchanges ordered by cloud-friendliness
EXCHANGE_FALLBACK_ORDER = ["okx", "kraken", "bybit", "coinbase", "binance"]


# ──────────────────────────────────────────────────────────────
# Exchange factory
# ──────────────────────────────────────────────────────────────

def create_exchange(exchange_id: str) -> ccxt.Exchange:
    """Return a ccxt exchange instance configured for public access."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({
        "enableRateLimit": True,
        "timeout": 10000,           # 10 s per request
        "options": {"defaultType": "spot"},
    })


# ──────────────────────────────────────────────────────────────
# Connection test
# ──────────────────────────────────────────────────────────────

def test_connection(exchange_id: str, symbol: str = "BTC/USDT") -> bool:
    """Try fetching a single batch of candles. Returns True on success."""
    try:
        ex = create_exchange(exchange_id)
        since = int(
            (datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000
        )
        data = ex.fetch_ohlcv(symbol, "1h", since=since, limit=10)
        return len(data) > 0
    except Exception as exc:
        print(f"[test_connection] {exchange_id} failed: {exc}")
        return False


def find_working_exchange(
    preferred: str, symbol: str = "BTC/USDT"
) -> str | None:
    """
    Try the preferred exchange first, then fall through the
    EXCHANGE_FALLBACK_ORDER list.  Returns the first exchange
    that responds, or None.
    """
    # Try preferred first
    candidates = [preferred] + [
        e for e in EXCHANGE_FALLBACK_ORDER if e != preferred
    ]
    for eid in candidates:
        print(f"[find_working_exchange] Testing {eid}...")
        if test_connection(eid, symbol):
            print(f"[find_working_exchange] ✓ {eid} works!")
            return eid
        print(f"[find_working_exchange] ✗ {eid} unreachable")
    return None


# ──────────────────────────────────────────────────────────────
# Historical data  (cached for 1 hour)
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_ohlcv(
    exchange_id: str = "okx",
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    days: int = 365,
) -> pd.DataFrame:
    """
    Paginate through the exchange REST API and return a clean
    DataFrame indexed by UTC datetime.
    """
    print(f"[fetch] Starting: {exchange_id} {symbol} {timeframe} {days}d")
    exchange = create_exchange(exchange_id)

    since = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_candles: list[list] = []
    limit = 200          # smaller batches = more reliable on cloud
    consecutive_fails = 0
    request_num = 0

    while since < now_ms:
        request_num += 1
        batch = None
        retries = 0

        while retries < 3:
            try:
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
                consecutive_fails = 0
                break
            except Exception as exc:
                retries += 1
                print(f"[fetch] Req #{request_num} retry {retries}: {exc}")
                _time.sleep(2 * retries)

        if batch is None or retries >= 3:
            consecutive_fails += 1
            print(f"[fetch] Req #{request_num} FAILED (streak: {consecutive_fails})")
            if consecutive_fails >= 3:
                print("[fetch] Too many failures – stopping early")
                break
            # Skip ahead
            since += limit * 60_000 * _tf_to_minutes(timeframe)
            continue

        if not batch:
            break

        all_candles.extend(batch)
        since = batch[-1][0] + 1

        if request_num % 20 == 0:
            print(f"[fetch] {len(all_candles):,} candles so far...")

        _time.sleep(exchange.rateLimit / 1000)

    print(f"[fetch] Done – {len(all_candles):,} candles total")

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
    exchange_id: str = "okx",
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


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    unit = tf[-1]
    val = int(tf[:-1])
    return val * {"m": 1, "h": 60, "d": 1440}.get(unit, 1)
