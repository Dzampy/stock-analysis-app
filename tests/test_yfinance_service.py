"""Unit tests for yfinance_service (mocked)."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


@patch('app.services.yfinance_service.time.sleep')
@patch('app.services.yfinance_service.yf')
@patch('app.services.yfinance_service.CACHE_AVAILABLE', False)
def test_get_stock_data_returns_dict_with_history_and_info(mock_yf, _mock_sleep):
    from app.services.yfinance_service import get_stock_data

    mock_yf.download.return_value = pd.DataFrame()  # force Ticker path
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({
        'Open': [100], 'High': [101], 'Low': [99], 'Close': [100.5], 'Volume': [1_000_000]
    }, index=pd.DatetimeIndex(['2024-01-15']))
    mock_ticker.info = {
        'symbol': 'AAPL', 'longName': 'Apple Inc.',
        'sector': 'Technology', 'industry': 'Consumer Electronics',
    }
    mock_yf.Ticker.return_value = mock_ticker

    out = get_stock_data('AAPL', '1y')

    assert out is not None
    assert 'history' in out
    assert 'info' in out
    assert not out['history'].empty
    assert out['info'].get('symbol') == 'AAPL'


@patch('app.services.yfinance_service.time.sleep')
@patch('app.services.yfinance_service.yf')
@patch('app.services.yfinance_service.CACHE_AVAILABLE', False)
def test_get_stock_data_empty_history_returns_none(mock_yf, _mock_sleep):
    from app.services.yfinance_service import get_stock_data

    mock_yf.download.return_value = pd.DataFrame()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker.info = {}
    mock_yf.Ticker.return_value = mock_ticker

    out = get_stock_data('INVALID', '1y')

    assert out is None
