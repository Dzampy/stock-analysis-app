"""
Integration tests for API routes
"""
import unittest
from flask import Flask
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
        # Should return 404 or 500 depending on implementation
        self.assertIn(response.status_code, [404, 500])
    
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
    
    def test_error_handler_404(self):
        """Test 404 error handling"""
        response = self.client.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('code', data['error'])


if __name__ == '__main__':
    unittest.main()




