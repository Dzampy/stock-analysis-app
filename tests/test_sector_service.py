"""Unit tests for sector_service (mocked)."""
import pytest


def test_get_sector_averages_none_for_empty_sector():
    from app.services.sector_service import get_sector_averages

    assert get_sector_averages('') is None
    assert get_sector_averages('N/A') is None


def test_get_sector_historical_data_none_for_empty_sector():
    from app.services.sector_service import get_sector_historical_data

    assert get_sector_historical_data('') is None
    assert get_sector_historical_data('N/A') is None


def test_get_sector_historical_data_none_for_unknown_metric():
    from app.services.sector_service import get_sector_historical_data

    # 'unknown_metric' is not in metric_map
    assert get_sector_historical_data('Technology', metric='unknown_metric') is None
