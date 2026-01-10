"""
Unit tests for error handling
"""
import unittest
from app.utils.error_handler import (
    AppError, ValidationError, NotFoundError, ExternalAPIError,
    create_error_response, handle_app_error
)


class TestErrorHandler(unittest.TestCase):
    """Test error handling functions"""
    
    def test_app_error_creation(self):
        """Test AppError creation"""
        error = AppError("Test error", 400, "TEST_ERROR")
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.error_code, "TEST_ERROR")
    
    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Invalid input", {'field': 'ticker'})
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.error_code, "VALIDATION_ERROR")
        self.assertIn('field', error.details)
    
    def test_not_found_error(self):
        """Test NotFoundError"""
        error = NotFoundError("Resource not found")
        self.assertEqual(error.status_code, 404)
        self.assertEqual(error.error_code, "NOT_FOUND")
    
    def test_external_api_error(self):
        """Test ExternalAPIError"""
        error = ExternalAPIError("API unavailable", service="yfinance")
        self.assertEqual(error.status_code, 502)
        self.assertEqual(error.error_code, "EXTERNAL_API_ERROR")
        self.assertEqual(error.details['service'], "yfinance")
    
    def test_create_error_response(self):
        """Test error response creation"""
        response, status_code = create_error_response(
            "Test error",
            400,
            "TEST_ERROR",
            {'field': 'test'}
        )
        self.assertEqual(status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])
        self.assertIn('error', data)
        self.assertEqual(data['error']['code'], "TEST_ERROR")
    
    def test_handle_app_error(self):
        """Test AppError handling"""
        error = ValidationError("Invalid input")
        response, status_code = handle_app_error(error)
        self.assertEqual(status_code, 400)
        data = response.get_json()
        self.assertEqual(data['error']['code'], "VALIDATION_ERROR")


if __name__ == '__main__':
    unittest.main()




