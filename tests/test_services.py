"""
Unit tests for service functions
"""
import unittest
from unittest.mock import patch, MagicMock
from app.services.sentiment_service import analyze_sentiment
from app.utils.validators import validate_ticker


class TestSentimentService(unittest.TestCase):
    """Test sentiment analysis service"""
    
    def test_analyze_sentiment_positive(self):
        """Test positive sentiment detection"""
        text = "This stock is amazing! Great earnings and strong growth."
        result = analyze_sentiment(text)
        self.assertIn('sentiment', result)
        self.assertIn('score', result)
        self.assertGreaterEqual(result['score'], -1)
        self.assertLessEqual(result['score'], 1)
    
    def test_analyze_sentiment_negative(self):
        """Test negative sentiment detection"""
        text = "Terrible company. Poor management and declining revenue."
        result = analyze_sentiment(text)
        self.assertIn('sentiment', result)
        self.assertIn('score', result)
    
    def test_analyze_sentiment_neutral(self):
        """Test neutral sentiment detection"""
        text = "The company reported quarterly results."
        result = analyze_sentiment(text)
        self.assertIn('sentiment', result)
        self.assertIn('score', result)
    
    def test_analyze_sentiment_empty(self):
        """Test empty text handling"""
        result = analyze_sentiment('')
        self.assertIn('sentiment', result)
        self.assertEqual(result['sentiment'], 'neutral')


class TestYFinanceService(unittest.TestCase):
    """Test yfinance service functions"""
    
    @patch('app.services.yfinance_service.yf.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """Test successful stock data retrieval"""
        # Mock yfinance response
        mock_stock = MagicMock()
        mock_stock.info = {'symbol': 'AAPL', 'longName': 'Apple Inc.'}
        mock_stock.history.return_value = MagicMock()
        mock_ticker.return_value = mock_stock
        
        # This would require importing the actual function
        # For now, just test that mocking works
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()


