"""
Integration tests for API routes
"""
import unittest
from unittest.mock import patch
from app import app
from app.utils.error_handler import ValidationError, NotFoundError


class TestRoutes(unittest.TestCase):
    """Test API routes"""

    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_index_route(self):
        """Test index route"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'html', response.data.lower())

    def test_stock_route_invalid_ticker(self):
        """Test stock route with invalid ticker"""
        response = self.client.get('/api/stock/INVALID123')
        self.assertIn(response.status_code, [404, 500])

    @patch('app.routes.stock.get_volume_analysis', return_value=None)
    @patch('app.routes.stock.get_short_interest_history', return_value=None)
    @patch('app.routes.stock.get_short_interest_from_finviz', return_value=None)
    @patch('app.routes.stock.generate_news_summary', return_value=None)
    @patch('app.routes.stock.get_stock_news', return_value=[])
    @patch('app.routes.stock.get_earnings_qoq', return_value=None)
    @patch('app.routes.stock.get_stock_data')
    def test_stock_route_returns_json_when_data_ok(self, mock_get, *_):
        """Test /api/stock/<ticker> returns JSON when service returns data."""
        import pandas as pd
        idx = pd.DatetimeIndex(['2024-01-14', '2024-01-15'])
        mock_get.return_value = {
            'history': pd.DataFrame({
                'Open': [99, 100], 'High': [101, 101], 'Low': [98, 99],
                'Close': [100, 100.5], 'Volume': [1e6, 1.1e6],
            }, index=idx),
            'info': {'symbol': 'AAPL', 'longName': 'Apple Inc.'},
        }
        response = self.client.get('/api/stock/AAPL?period=1y')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertIn('chart_data', data)

    def test_search_route(self):
        """Test search route"""
        response = self.client.get('/api/search/apple')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('success', data or {})

    def test_financials_route_invalid_ticker(self):
        """Test financials route with invalid ticker"""
        response = self.client.get('/api/financials/INVALID123')
        self.assertIn(response.status_code, [404, 500])

    @patch('app.routes.financials.get_financials_data')
    def test_financials_route_returns_json_when_ok(self, mock_get):
        """Test /api/financials/<ticker> returns JSON when service returns data."""
        mock_get.return_value = {
            'company_name': 'Apple Inc.', 'sector': 'Technology',
            'executive_snapshot': {}, 'income_statement': {}, 'balance_sheet': {},
            'margins': {}, 'info': {'symbol': 'AAPL'},
        }
        response = self.client.get('/api/financials/AAPL')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)

    def test_portfolio_data_requires_post_and_json(self):
        """Test /api/portfolio-data requires POST with JSON positions."""
        response = self.client.get('/api/portfolio-data')
        self.assertIn(response.status_code, [404, 405])
        response = self.client.post('/api/portfolio-data', json={})
        self.assertEqual(response.status_code, 400)

    def test_portfolio_data_post_empty_positions_returns_400(self):
        """Test /api/portfolio-data POST with empty positions returns 400."""
        response = self.client.post('/api/portfolio-data', json={'positions': []})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data or {})

    def test_error_handler_404(self):
        """Test 404 error handling"""
        response = self.client.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn('error', data)
        # App may return {'error': {'code': '...'}} or {'error': 'Not found'}
        if isinstance(data.get('error'), dict):
            self.assertIn('code', data['error'])


if __name__ == '__main__':
    unittest.main()




