"""
Configuration module - API keys, feature flags, constants
"""
import os
from app.utils.logger import logger

# Cache: Redis if CACHE_REDIS_URL is set, otherwise in-memory (simple)
_CACHE_REDIS_URL = os.getenv('CACHE_REDIS_URL', '').strip()
if _CACHE_REDIS_URL:
    CACHE_CONFIG = {
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': _CACHE_REDIS_URL,
        'CACHE_DEFAULT_TIMEOUT': 300,
        'CACHE_KEY_PREFIX': 'stockapp_',
    }
    logger.info("Cache configured to use Redis")
else:
    CACHE_CONFIG = {
        'CACHE_TYPE': 'simple',
        'CACHE_DEFAULT_TIMEOUT': 300,
        'CACHE_THRESHOLD': 1000,
    }
    logger.info("Cache configured to use in-memory (set CACHE_REDIS_URL for Redis)")

# Cache timeouts for different data types (in seconds)
CACHE_TIMEOUTS = {
    'yfinance': 300,  # 5 minutes - prices change frequently
    'finviz': 900,  # 15 minutes - analyst ratings, insider trading
    'ml_predictions': 600,  # 10 minutes - computationally expensive
    'news': 1800,  # 30 minutes - less frequent changes
    'financials': 3600,  # 1 hour - quarterly data changes rarely
    'analyst': 900,  # 15 minutes
    'insider': 900,  # 15 minutes
    'institutional': 1800,  # 30 minutes
}

# SEC API configuration
SEC_API_KEY = os.getenv('SEC_API_KEY')

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_AVAILABLE = GEMINI_API_KEY is not None

if not GEMINI_AVAILABLE:
    logger.warning("Google Gemini API key not found. Earnings call analysis will not be available.")
    logger.info("Get your free API key at: https://makersuite.google.com/app/apikey")

# Reddit API configuration (optional)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockAnalysisTool/1.0')
REDDIT_AVAILABLE = REDDIT_CLIENT_ID is not None and REDDIT_CLIENT_SECRET is not None

if not REDDIT_AVAILABLE:
    logger.warning("Reddit API credentials not found. Reddit sentiment will use web scraping fallback.")

# ML availability
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn not available. Using fallback models.")

# XGBoost and LightGBM are optional
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False


