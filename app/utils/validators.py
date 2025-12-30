"""
Input validation functions
"""
import re
from typing import Optional

def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker symbol format
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Ticker should be 1-5 uppercase letters
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, ticker.upper()))

def validate_period(period: str) -> bool:
    """
    Validate time period format
    
    Args:
        period: Period string (e.g., '1d', '1mo', '1y')
        
    Returns:
        True if valid, False otherwise
    """
    if not period or not isinstance(period, str):
        return False
    
    # Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    valid_periods = ['1d', '5d', '1wk', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    return period.lower() in valid_periods

def sanitize_input(input_str: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input string
    
    Args:
        input_str: Input string to sanitize
        max_length: Maximum length (optional)
        
    Returns:
        Sanitized string
    """
    if not input_str:
        return ''
    
    # Strip whitespace
    sanitized = input_str.strip()
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', sanitized)
    
    # Limit length if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def validate_query(query: str) -> bool:
    """
    Validate search query
    
    Args:
        query: Search query string
        
    Returns:
        True if valid, False otherwise
    """
    if not query or not isinstance(query, str):
        return False
    
    # Query should be 1-100 characters, alphanumeric + spaces
    if len(query.strip()) < 1 or len(query) > 100:
        return False
    
    # Allow alphanumeric, spaces, and common punctuation
    pattern = r'^[a-zA-Z0-9\s\-_.,]+$'
    return bool(re.match(pattern, query))




