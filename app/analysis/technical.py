"""
Technical analysis - Indicators, volume analysis, support/resistance
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from typing import Dict, Optional
import yfinance as yf
import time
from app.utils.constants import RATE_LIMIT_DELAY, VOLUME_UNUSUAL_THRESHOLD
from app.utils.logger import logger


def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
    """
    Calculate technical indicators
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dict with indicator arrays
    """
    indicators = {}
    
    # Moving Averages
    indicators['sma_20'] = SMAIndicator(df['Close'], window=20).sma_indicator().tolist()
    indicators['sma_50'] = SMAIndicator(df['Close'], window=50).sma_indicator().tolist()
    indicators['ema_12'] = EMAIndicator(df['Close'], window=12).ema_indicator().tolist()
    indicators['ema_26'] = EMAIndicator(df['Close'], window=26).ema_indicator().tolist()
    
    # RSI
    rsi = RSIIndicator(df['Close'], window=14)
    indicators['rsi'] = rsi.rsi().tolist()
    
    # MACD
    macd = MACD(df['Close'])
    indicators['macd'] = macd.macd().tolist()
    indicators['macd_signal'] = macd.macd_signal().tolist()
    indicators['macd_diff'] = macd.macd_diff().tolist()
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    indicators['bb_high'] = bb.bollinger_hband().tolist()
    indicators['bb_low'] = bb.bollinger_lband().tolist()
    indicators['bb_mid'] = bb.bollinger_mavg().tolist()
    
    # ADX
    try:
        if 'High' in df.columns and 'Low' in df.columns and len(df) > 14:
            adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            adx_values = adx.adx().tolist()
            indicators['adx'] = adx_values
        else:
            indicators['adx'] = []
    except Exception as e:
        logger.warning(f"ADX calculation error: {e}")
        indicators['adx'] = []
    
    # Stochastic Oscillator
    try:
        if 'High' in df.columns and 'Low' in df.columns and len(df) > 14:
            stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            indicators['stoch_k'] = stoch.stoch().tolist()
            indicators['stoch_d'] = stoch.stoch_signal().tolist()
        else:
            indicators['stoch_k'] = []
            indicators['stoch_d'] = []
    except Exception as e:
        logger.warning(f"Stochastic calculation error: {e}")
        indicators['stoch_k'] = []
        indicators['stoch_d'] = []
    
    # ATR
    try:
        if 'High' in df.columns and 'Low' in df.columns and len(df) > 14:
            atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            indicators['atr'] = atr.average_true_range().tolist()
        else:
            indicators['atr'] = []
    except Exception as e:
        logger.warning(f"ATR calculation error: {e}")
        indicators['atr'] = []
    
    return indicators


def get_volume_analysis(ticker: str) -> Optional[Dict]:
    """
    Calculate average daily volume vs recent volume for unusual activity detection
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with volume analysis data or None
    """
    try:
        stock = yf.Ticker(ticker)
        time.sleep(RATE_LIMIT_DELAY)
        
        # Get historical data for volume analysis
        hist = stock.history(period='6mo')  # 6 months for better averages
        
        if hist.empty or 'Volume' not in hist.columns:
            return None
        
        # Calculate moving averages
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_MA50'] = hist['Volume'].rolling(window=50).mean()
        hist['Volume_MA90'] = hist['Volume'].rolling(window=90).mean()
        
        # Get recent volumes (last 5 days)
        recent_volumes = hist['Volume'].tail(5).tolist()
        recent_dates = [d.strftime('%Y-%m-%d') for d in hist.tail(5).index]
        
        # Current volume
        current_volume = int(hist['Volume'].iloc[-1])
        
        # Average volumes
        avg_volume_20d = int(hist['Volume_MA20'].iloc[-1]) if not pd.isna(hist['Volume_MA20'].iloc[-1]) else None
        avg_volume_50d = int(hist['Volume_MA50'].iloc[-1]) if not pd.isna(hist['Volume_MA50'].iloc[-1]) else None
        avg_volume_90d = int(hist['Volume_MA90'].iloc[-1]) if not pd.isna(hist['Volume_MA90'].iloc[-1]) else None
        
        # Volume ratios
        volume_ratio_20d = (current_volume / avg_volume_20d) if avg_volume_20d and avg_volume_20d > 0 else None
        volume_ratio_50d = (current_volume / avg_volume_50d) if avg_volume_50d and avg_volume_50d > 0 else None
        volume_ratio_90d = (current_volume / avg_volume_90d) if avg_volume_90d and avg_volume_90d > 0 else None
        
        # Detect unusual activity (volume > 2x average)
        unusual_activity = False
        if volume_ratio_20d and volume_ratio_20d > VOLUME_UNUSUAL_THRESHOLD:
            unusual_activity = True
        
        # Prepare historical data for chart
        volume_data = []
        for i, (date, row) in enumerate(hist.iterrows()):
            if i >= 20:  # Only include data where we have MA20
                volume_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'volume': int(row['Volume']),
                    'ma20': int(row['Volume_MA20']) if not pd.isna(row['Volume_MA20']) else None,
                    'ma50': int(row['Volume_MA50']) if not pd.isna(row['Volume_MA50']) else None,
                    'ma90': int(row['Volume_MA90']) if not pd.isna(row['Volume_MA90']) else None
                })
        
        return {
            'current_volume': current_volume,
            'avg_volume_20d': avg_volume_20d,
            'avg_volume_50d': avg_volume_50d,
            'avg_volume_90d': avg_volume_90d,
            'volume_ratio_20d': round(volume_ratio_20d, 2) if volume_ratio_20d else None,
            'volume_ratio_50d': round(volume_ratio_50d, 2) if volume_ratio_50d else None,
            'volume_ratio_90d': round(volume_ratio_90d, 2) if volume_ratio_90d else None,
            'unusual_activity': unusual_activity,
            'recent_volumes': recent_volumes,
            'recent_dates': recent_dates,
            'volume_history': volume_data[-60:]  # Last 60 days for chart
        }
    
    except Exception as e:
        logger.exception(f"Error fetching volume analysis for {ticker}")
        import traceback
        traceback.print_exc()
        return None


# Functions will be moved here from app.py:
# - get_retail_activity_indicators()
# - Support/resistance calculation

