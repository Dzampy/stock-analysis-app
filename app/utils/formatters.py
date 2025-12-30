"""
Data formatting functions
"""
from typing import Optional

def format_currency(value: Optional[float], decimals: int = 2, show_currency: bool = True) -> str:
    """
    Format number as currency
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        show_currency: Whether to show $ symbol
        
    Returns:
        Formatted currency string
    """
    if value is None:
        return 'N/A'
    
    try:
        if abs(value) >= 1_000_000_000_000:
            formatted = f"{value / 1_000_000_000_000:.{decimals}f}T"
        elif abs(value) >= 1_000_000_000:
            formatted = f"{value / 1_000_000_000:.{decimals}f}B"
        elif abs(value) >= 1_000_000:
            formatted = f"{value / 1_000_000:.{decimals}f}M"
        elif abs(value) >= 1_000:
            formatted = f"{value / 1_000:.{decimals}f}K"
        else:
            formatted = f"{value:.{decimals}f}"
        
        if show_currency:
            return f"${formatted}"
        return formatted
    except (ValueError, TypeError):
        return 'N/A'

def format_percentage(value: Optional[float], decimals: int = 2, show_sign: bool = True) -> str:
    """
    Format number as percentage
    
    Args:
        value: Numeric value to format (0.05 = 5%)
        decimals: Number of decimal places
        show_sign: Whether to show + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return 'N/A'
    
    try:
        percentage = value * 100
        sign = '+' if show_sign and percentage > 0 else ''
        return f"{sign}{percentage:.{decimals}f}%"
    except (ValueError, TypeError):
        return 'N/A'

def format_number(value: Optional[float], decimals: int = 2) -> str:
    """
    Format number with thousand separators
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if value is None:
        return 'N/A'
    
    try:
        return f"{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return 'N/A'




