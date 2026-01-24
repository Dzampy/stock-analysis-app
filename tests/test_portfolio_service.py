"""Unit tests for portfolio_service."""
import pytest
from unittest.mock import patch


@patch('app.services.portfolio_service.GEMINI_AVAILABLE', False)
def test_analyze_watchlist_news_with_ai_returns_error_when_gemini_unavailable():
    from app.services.portfolio_service import analyze_watchlist_news_with_ai

    out = analyze_watchlist_news_with_ai([])
    assert out.get('success') is False
    assert 'error' in out
