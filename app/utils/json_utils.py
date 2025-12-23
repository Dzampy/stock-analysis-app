"""
JSON serialization utilities
"""
import pandas as pd
import numpy as np
import math
from typing import Any

def clean_for_json(data: Any) -> Any:
    """
    Replace NaN and inf values with None for JSON serialization
    
    Args:
        data: Data to clean (can be dict, list, Series, DataFrame, scalar)
        
    Returns:
        JSON-serializable data structure
    """
    # Handle Timestamp first (before Series, as Series might contain Timestamps)
    if isinstance(data, pd.Timestamp):
        try:
            return data.strftime('%Y-%m-%d')
        except:
            return str(data)
    
    if isinstance(data, pd.Series):
        data = data.tolist()
    
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to dict of lists
        return {str(col): clean_for_json(data[col].tolist()) for col in data.columns}
    
    if isinstance(data, (list, tuple)):
        return [clean_for_json(item) for item in data]
    
    elif isinstance(data, dict):
        return {key: clean_for_json(value) for key, value in data.items()}
    
    elif isinstance(data, (bool, np.bool_)):
        return bool(data)
    
    elif isinstance(data, (int, np.integer)):
        return int(data)
    
    elif isinstance(data, (float, np.floating, np.number)):
        # Check for all types of invalid values
        if pd.isna(data) or math.isnan(data) or math.isinf(data) or not np.isfinite(data):
            return None
        val = float(data)
        # Double check after conversion
        if not np.isfinite(val) or math.isinf(val) or math.isnan(val):
            return None
        return val
    
    elif pd.isna(data):
        return None
    
    # Handle any other pandas/numpy types that might contain Timestamps
    try:
        if hasattr(data, 'item'):
            return clean_for_json(data.item())
    except:
        pass
    
    # For any other type, try to convert to string as last resort
    try:
        return str(data)
    except:
        return None



