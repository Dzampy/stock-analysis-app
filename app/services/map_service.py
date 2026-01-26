"""
Map service - Batch market cap and price change for custom treemap (Finviz-style)
"""
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    period_map = {
        '1d': '5d',      # need at least 2 days for 1d change
        '5d': '5d',
        '1w': '1mo',     # 1 week - use 1mo to ensure enough data
        '1m': '1mo',
        '3m': '3mo',
        '6m': '6mo',
        '1y': '1y',
        'ytd': '1y',     # YTD: use 1y and filter in _compute_change_pct
    }
    return period_map.get(period.lower(), '5d')


def _compute_change_pct(close_series: pd.Series, period: str) -> Optional[float]:
    """Compute % change from Close series. period: 1d, 5d, 1w, 1m, 3m, 6m, 1y, ytd."""
    if close_series is None or len(close_series) < 2:
        return None
    close = close_series.dropna()
    if len(close) < 2:
        return None
    try:
        period_lower = period.lower()
        if period_lower == '1d':
            # 1 day: yesterday to today
            return ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
        elif period_lower == 'ytd':
            # YTD: first trading day of year to today
            from datetime import datetime
            year_start = datetime(datetime.now().year, 1, 1)
            # Find first date >= year_start
            year_start_idx = close.index >= year_start
            if year_start_idx.any():
                first_idx = close.index[year_start_idx][0]
                first_value = close.loc[first_idx]
                return ((close.iloc[-1] - first_value) / first_value) * 100
            # Fallback to first to last
            return ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
        else:
            # All other periods: first to last
            return ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    except (IndexError, ZeroDivisionError, TypeError):
        return None


def _fetch_ticker_info(ticker: str) -> Dict:
    """Fetch info for single ticker (used in parallel execution)."""
    try:
        time.sleep(RATE_LIMIT_DELAY)  # Still rate limit per ticker
        info = yf.Ticker(ticker).info
        return {
            'ticker': ticker,
            'market_cap': info.get('marketCap'),
            'name': (info.get('shortName') or info.get('longName') or ticker),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'volume': info.get('volume'),
            'pe_ratio': info.get('trailingPE'),
            'error': None
        }
    except Exception as e:
        logger.debug(f"Ticker {ticker} info error: {e}")
        return {
            'ticker': ticker,
            'market_cap': None,
            'name': ticker,
            'sector': None,
            'industry': None,
            'volume': None,
            'pe_ratio': None,
            'error': str(e)
        }


def get_map_data(tickers: List[str], period: str = '1d') -> Dict:
    """
    Fetch market_cap, name, change_pct, and additional data for each ticker.
    Uses yf.download for prices and parallel yf.Ticker().info for market cap / name.

    Args:
        tickers: List of ticker symbols (max 80 enforced by route).
        period: '1d' | '5d' | '1w' | '1m' | '3m' | '6m' | '1y' | 'ytd' for the change window.

    Returns:
        Dict with:
        - 'items': List of { ticker, name, market_cap, change_pct, sector, industry, volume, pe_ratio } sorted by market_cap DESC.
        - 'failed_tickers': List of tickers that failed to load
        Missing data as None.
    """
    if not tickers:
        return {'items': [], 'failed_tickers': []}

    # Cache key
    if CACHE_AVAILABLE and cache:
        key = f"map_data_{hashlib.md5((','.join(sorted(tickers)) + period).encode()).hexdigest()}"
        try:
            cached = cache.get(key)
            if cached is not None:
                logger.debug("Map data cache hit")
                # Ensure cached data has the new structure
                if isinstance(cached, dict) and 'items' in cached:
                    return cached
                # Legacy format: convert to new format
                return {'items': cached if isinstance(cached, list) else [], 'failed_tickers': []}
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

    # 2) Market cap and name per ticker - PARALLEL EXECUTION
    ticker_info_by_ticker: Dict[str, Dict] = {}
    failed_tickers: List[str] = []
    
    # Use ThreadPoolExecutor for parallel fetching (max 5 workers to respect rate limits)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_ticker_info, t.upper()): t.upper() for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            ticker = result['ticker']
            ticker_info_by_ticker[ticker] = result
            if result.get('error'):
                failed_tickers.append(ticker)

    # 3) Merge price change data with ticker info
    items: List[Dict] = []
    for t in tickers:
        tu = t.upper()
        info = ticker_info_by_ticker.get(tu, {})
        rec = {
            'ticker': tu,
            'name': info.get('name', tu),
            'market_cap': info.get('market_cap'),
            'change_pct': change_by_ticker.get(tu),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'volume': info.get('volume'),
            'pe_ratio': info.get('pe_ratio'),
        }
        items.append(rec)

    # 4) Sort by market_cap DESC (nulls last)
    items.sort(key=lambda x: (x['market_cap'] is None, -(x['market_cap'] or 0)))

    # 5) Cache
    result = {'items': items, 'failed_tickers': failed_tickers}
    if CACHE_AVAILABLE and cache:
        try:
            cache.set(key, result, timeout=120)
        except Exception as e:
            logger.warning(f"Map cache set error: {e}")

    return result
