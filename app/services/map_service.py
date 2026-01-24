"""
Map service - Batch market cap and price change for custom treemap (Finviz-style)
"""
import hashlib
import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from app.utils.constants import RATE_LIMIT_DELAY
from app.utils.logger import logger

try:
    from app import cache
    CACHE_AVAILABLE = cache is not None
except (ImportError, RuntimeError, AttributeError):
    CACHE_AVAILABLE = False
    cache = None


def _period_to_download(period: str) -> str:
    """Map API period to yfinance download period."""
    if period == '1d':
        return '5d'   # need at least 2 days for 1d change
    if period == '5d':
        return '5d'
    if period == '1m':
        return '1mo'
    return '5d'


def _compute_change_pct(close_series: pd.Series, period: str) -> Optional[float]:
    """Compute % change from Close series. period: 1d, 5d, 1m."""
    if close_series is None or len(close_series) < 2:
        return None
    close = close_series.dropna()
    if len(close) < 2:
        return None
    try:
        if period == '1d':
            return ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
        # 5d and 1m: first to last
        return ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    except (IndexError, ZeroDivisionError, TypeError):
        return None


def get_map_data(tickers: List[str], period: str = '1d') -> List[Dict]:
    """
    Fetch market_cap, name, and change_pct for each ticker.
    Uses yf.download for prices and yf.Ticker().info for market cap / name.

    Args:
        tickers: List of ticker symbols (max 80 enforced by route).
        period: '1d' | '5d' | '1m' for the change window.

    Returns:
        List of { ticker, name, market_cap, change_pct } sorted by market_cap DESC.
        Missing data as None.
    """
    if not tickers:
        return []

    # Cache key
    if CACHE_AVAILABLE and cache:
        key = f"map_data_{hashlib.md5((','.join(sorted(tickers)) + period).encode()).hexdigest()}"
        try:
            cached = cache.get(key)
            if cached is not None:
                logger.debug("Map data cache hit")
                return cached
        except Exception as e:
            logger.warning(f"Map cache get error: {e}")

    download_period = _period_to_download(period)
    change_by_ticker: Dict[str, Optional[float]] = {}

    # 1) Price history and % change
    try:
        # group_by='ticker' gives MultiIndex columns for multiple tickers
        df = yf.download(
            tickers,
            period=download_period,
            auto_adjust=True,
            group_by='ticker' if len(tickers) > 1 else None,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            pass
        elif len(tickers) == 1:
            # Single ticker: columns are Open, High, Low, Close, Volume
            if 'Close' in df.columns:
                change_by_ticker[tickers[0].upper()] = _compute_change_pct(df['Close'], period)
        elif isinstance(df.columns, pd.MultiIndex):
            # Multiple: (TICKER, OHLC)
            for t in tickers:
                tu = t.upper()
                if (tu, 'Close') in df.columns:
                    change_by_ticker[tu] = _compute_change_pct(df[(tu, 'Close')], period)
                elif 'Close' in df.columns:
                    # fallback if structure differs
                    change_by_ticker[tu] = _compute_change_pct(df['Close'], period)
        else:
            # sometimes columns are 'TICKER Close' or similar
            for t in tickers:
                tu = t.upper()
                for c in df.columns:
                    if c and (tu in str(c) or str(c).startswith(tu)) and 'Close' in str(c):
                        change_by_ticker[tu] = _compute_change_pct(df[c], period)
                        break
    except Exception as e:
        logger.warning(f"yf.download in map_service failed: {e}")

    # 2) Market cap and name per ticker
    items: List[Dict] = []
    for t in tickers:
        tu = t.upper()
        rec = {
            'ticker': tu,
            'name': tu,
            'market_cap': None,
            'change_pct': change_by_ticker.get(tu),
        }
        try:
            time.sleep(RATE_LIMIT_DELAY)
            info = yf.Ticker(tu).info
            if info:
                rec['market_cap'] = info.get('marketCap')
                rec['name'] = (info.get('shortName') or info.get('longName') or tu)
        except Exception as e:
            logger.debug(f"map_service: Ticker {tu} info error: {e}")
        items.append(rec)

    # 3) Sort by market_cap DESC (nulls last)
    items.sort(key=lambda x: (x['market_cap'] is None, -(x['market_cap'] or 0)))

    # 4) Cache
    if CACHE_AVAILABLE and cache:
        try:
            cache.set(key, items, timeout=120)
        except Exception as e:
            logger.warning(f"Map cache set error: {e}")

    return items
