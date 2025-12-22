"""
Unit tests for utility functions
"""
import unittest
from app.utils.validators import validate_ticker, validate_period, sanitize_input, validate_query
from app.utils.formatters import format_currency, format_percentage, format_number


class TestValidators(unittest.TestCase):
    """Test input validation functions"""
    
    def test_validate_ticker_valid(self):
        """Test valid ticker symbols"""
        self.assertTrue(validate_ticker('AAPL'))
        self.assertTrue(validate_ticker('MSFT'))
        self.assertTrue(validate_ticker('TSLA'))
        self.assertTrue(validate_ticker('aapl'))  # Should be case-insensitive
    
    def test_validate_ticker_invalid(self):
        """Test invalid ticker symbols"""
        self.assertFalse(validate_ticker(''))
        self.assertFalse(validate_ticker('AAPL123'))
        self.assertFalse(validate_ticker('TOOLONG'))
        self.assertFalse(validate_ticker(None))
        self.assertFalse(validate_ticker(123))
    
    def test_validate_period_valid(self):
        """Test valid periods"""
        self.assertTrue(validate_period('1d'))
        self.assertTrue(validate_period('1mo'))
        self.assertTrue(validate_period('1y'))
        self.assertTrue(validate_period('max'))
    
    def test_validate_period_invalid(self):
        """Test invalid periods"""
        self.assertFalse(validate_period(''))
        self.assertFalse(validate_period('1x'))
        self.assertFalse(validate_period(None))
        self.assertFalse(validate_period(123))
    
    def test_sanitize_input(self):
        """Test input sanitization"""
        self.assertEqual(sanitize_input('  test  '), 'test')
        self.assertEqual(sanitize_input('test<script>'), 'testscript')
        self.assertEqual(sanitize_input('a' * 200, max_length=50), 'a' * 50)
        self.assertEqual(sanitize_input(''), '')
    
    def test_validate_query(self):
        """Test query validation"""
        self.assertTrue(validate_query('apple'))
        self.assertTrue(validate_query('Apple Inc.'))
        self.assertFalse(validate_query(''))
        self.assertFalse(validate_query('a' * 101))  # Too long
        self.assertFalse(validate_query('test<script>'))  # Invalid chars


class TestFormatters(unittest.TestCase):
    """Test formatting functions"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        self.assertEqual(format_currency(1000), '$1,000.00')
        self.assertEqual(format_currency(1000000), '$1,000,000.00')
        self.assertEqual(format_currency(0), '$0.00')
        self.assertEqual(format_currency(None), 'N/A')
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        self.assertEqual(format_percentage(0.15), '15.00%')
        self.assertEqual(format_percentage(0.5), '50.00%')
        self.assertEqual(format_percentage(None), 'N/A')
    
    def test_format_number(self):
        """Test number formatting"""
        self.assertEqual(format_number(1000), '1,000')
        self.assertEqual(format_number(1000000), '1,000,000')
        self.assertEqual(format_number(None), 'N/A')


if __name__ == '__main__':
    unittest.main()

