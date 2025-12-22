"""
ML service - Price predictions, backtesting, model management
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json
from app.config import ML_AVAILABLE
from app.utils.constants import MODEL_CACHE_VERSION
from app.utils.logger import logger
import time

# Model cache (module-level)
_model_cache = {}
_scaler_cache = {}

# Prediction history directory
_PREDICTION_HISTORY_DIR = Path('.ml_predictions')
_PREDICTION_HISTORY_DIR.mkdir(exist_ok=True)


def _clear_model_cache_if_needed():
    """Clear model cache if version changed"""
    if _model_cache and not any(k.startswith(f"rf_v{MODEL_CACHE_VERSION}_") for k in _model_cache.keys()):
        logger.info(f"Clearing model cache due to version change (new version: {MODEL_CACHE_VERSION})")
        _model_cache.clear()
        _scaler_cache.clear()


# Clear cache on module load
_clear_model_cache_if_needed()


def clear_ml_cache():
    """Clear ML model cache - useful when model structure changes"""
    global _model_cache, _scaler_cache
    cache_size_before = len(_model_cache)
    _model_cache.clear()
    _scaler_cache.clear()
    return {
        'success': True,
        'message': f'ML cache cleared (removed {cache_size_before} cached models)',
        'cache_version': MODEL_CACHE_VERSION
    }


def extract_ml_features(ticker: str, df: pd.DataFrame, info: Dict, indicators: Dict, metrics: Dict, news_list: List[Dict]) -> Dict:
    """
    Extract comprehensive features for ML models
    
    Args:
        ticker: Stock ticker symbol
        df: DataFrame with price data
        info: Stock info dict from yfinance
        indicators: Technical indicators dict
        metrics: Calculated metrics dict
        news_list: List of news articles with sentiment
        
    Returns:
        Dict with extracted features
    """
    features = {}
    
    current_price = df['Close'].iloc[-1]
    
    # Technical Features (20+)
    rsi_values = indicators.get('rsi', [])
    macd_values = indicators.get('macd', [])
    macd_signal = indicators.get('macd_signal', [])
    sma_20 = indicators.get('sma_20', [])
    sma_50 = indicators.get('sma_50', [])
    bb_high = indicators.get('bb_high', [])
    bb_low = indicators.get('bb_low', [])
    bb_mid = indicators.get('bb_mid', [])
    adx_values = indicators.get('adx', [])
    stoch_k_values = indicators.get('stoch_k', [])
    stoch_d_values = indicators.get('stoch_d', [])
    atr_values = indicators.get('atr', [])
    
    # RSI features
    if rsi_values and len(rsi_values) > 0:
        features['rsi'] = float(rsi_values[-1]) if not pd.isna(rsi_values[-1]) else 50.0
        features['rsi_7d_avg'] = float(np.mean(rsi_values[-7:])) if len(rsi_values) >= 7 else 50.0
    else:
        features['rsi'] = 50.0
        features['rsi_7d_avg'] = 50.0
    
    # MACD features
    if macd_values and macd_signal and len(macd_values) > 0:
        features['macd'] = float(macd_values[-1]) if not pd.isna(macd_values[-1]) else 0.0
        features['macd_signal'] = float(macd_signal[-1]) if not pd.isna(macd_signal[-1]) else 0.0
        features['macd_diff'] = features['macd'] - features['macd_signal']
        features['macd_bullish'] = 1.0 if features['macd_diff'] > 0 else 0.0
    else:
        features['macd'] = 0.0
        features['macd_signal'] = 0.0
        features['macd_diff'] = 0.0
        features['macd_bullish'] = 0.0
    
    # Moving Average features
    if sma_20 and len(sma_20) > 0:
        features['sma_20'] = float(sma_20[-1]) if not pd.isna(sma_20[-1]) else current_price
        features['price_vs_sma20'] = ((current_price - features['sma_20']) / features['sma_20']) * 100 if features['sma_20'] > 0 else 0.0
    else:
        features['sma_20'] = current_price
        features['price_vs_sma20'] = 0.0
    
    if sma_50 and len(sma_50) > 0:
        features['sma_50'] = float(sma_50[-1]) if not pd.isna(sma_50[-1]) else current_price
        features['price_vs_sma50'] = ((current_price - features['sma_50']) / features['sma_50']) * 100 if features['sma_50'] > 0 else 0.0
    else:
        features['sma_50'] = current_price
        features['price_vs_sma50'] = 0.0
    
    # Bollinger Bands features
    if bb_high and bb_low and bb_mid and len(bb_high) > 0:
        features['bb_high'] = float(bb_high[-1]) if not pd.isna(bb_high[-1]) else current_price * 1.1
        features['bb_low'] = float(bb_low[-1]) if not pd.isna(bb_low[-1]) else current_price * 0.9
        features['bb_mid'] = float(bb_mid[-1]) if not pd.isna(bb_mid[-1]) else current_price
        features['bb_width'] = ((features['bb_high'] - features['bb_low']) / features['bb_mid']) * 100 if features['bb_mid'] > 0 else 0.0
        features['bb_position'] = ((current_price - features['bb_low']) / (features['bb_high'] - features['bb_low'])) * 100 if (features['bb_high'] - features['bb_low']) > 0 else 50.0
    else:
        features['bb_high'] = current_price * 1.1
        features['bb_low'] = current_price * 0.9
        features['bb_mid'] = current_price
        features['bb_width'] = 20.0
        features['bb_position'] = 50.0
    
    # ADX features
    if adx_values and len(adx_values) > 0:
        features['adx'] = float(adx_values[-1]) if not pd.isna(adx_values[-1]) else 25.0
    else:
        features['adx'] = 25.0
    
    # Stochastic features
    if stoch_k_values and stoch_d_values and len(stoch_k_values) > 0:
        features['stoch_k'] = float(stoch_k_values[-1]) if not pd.isna(stoch_k_values[-1]) else 50.0
        features['stoch_d'] = float(stoch_d_values[-1]) if not pd.isna(stoch_d_values[-1]) else 50.0
        features['stoch_oversold'] = 1.0 if features['stoch_k'] < 20 else 0.0
        features['stoch_overbought'] = 1.0 if features['stoch_k'] > 80 else 0.0
    else:
        features['stoch_k'] = 50.0
        features['stoch_d'] = 50.0
        features['stoch_oversold'] = 0.0
        features['stoch_overbought'] = 0.0
    
    # ATR features
    if atr_values and len(atr_values) > 0:
        features['atr'] = float(atr_values[-1]) if not pd.isna(atr_values[-1]) else current_price * 0.02
        features['atr_pct'] = (features['atr'] / current_price) * 100 if current_price > 0 else 2.0
    else:
        features['atr'] = current_price * 0.02
        features['atr_pct'] = 2.0
    
    # Price momentum features
    if len(df) >= 5:
        features['price_change_5d'] = ((current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
    else:
        features['price_change_5d'] = 0.0
    
    if len(df) >= 10:
        features['price_change_10d'] = ((current_price - df['Close'].iloc[-10]) / df['Close'].iloc[-10]) * 100
    else:
        features['price_change_10d'] = 0.0
    
    if len(df) >= 30:
        features['price_change_30d'] = ((current_price - df['Close'].iloc[-30]) / df['Close'].iloc[-30]) * 100
    else:
        features['price_change_30d'] = 0.0
    
    # Volume features
    if 'Volume' in df.columns:
        avg_volume = df['Volume'].tail(20).mean() if len(df) >= 20 else df['Volume'].mean()
        current_volume = df['Volume'].iloc[-1]
        features['volume_ratio'] = (current_volume / avg_volume) if avg_volume > 0 else 1.0
    else:
        features['volume_ratio'] = 1.0
    
    # Fundamental features
    market_cap = info.get('marketCap')
    pe_ratio = info.get('trailingPE')
    pb_ratio = info.get('priceToBook')
    ps_ratio = info.get('priceToSalesTrailing12Months')
    revenue_growth = info.get('revenueGrowth')
    earnings_growth = info.get('earningsGrowth')
    roe = info.get('returnOnEquity')
    debt_to_equity = info.get('debtToEquity')
    beta = info.get('beta')
    
    features['market_cap'] = float(market_cap) if market_cap else 0.0
    features['pe_ratio'] = float(pe_ratio) if pe_ratio and not pd.isna(pe_ratio) else 0.0
    features['pb_ratio'] = float(pb_ratio) if pb_ratio and not pd.isna(pb_ratio) else 0.0
    features['ps_ratio'] = float(ps_ratio) if ps_ratio and not pd.isna(ps_ratio) else 0.0
    features['revenue_growth'] = float(revenue_growth) * 100 if revenue_growth and not pd.isna(revenue_growth) else 0.0
    features['earnings_growth'] = float(earnings_growth) * 100 if earnings_growth and not pd.isna(earnings_growth) else 0.0
    features['roe'] = float(roe) * 100 if roe and not pd.isna(roe) else 0.0
    features['debt_to_equity'] = float(debt_to_equity) if debt_to_equity and not pd.isna(debt_to_equity) else 0.0
    features['beta'] = float(beta) if beta and not pd.isna(beta) else 1.0
    
    # News sentiment features
    if news_list:
        sentiments = [article.get('sentiment_score', 0.0) for article in news_list[:10]]
        features['news_sentiment_avg'] = float(np.mean(sentiments)) if sentiments else 0.0
        features['news_count'] = len(news_list[:10])
    else:
        features['news_sentiment_avg'] = 0.0
        features['news_count'] = 0
    
    # Volatility features
    if len(df) >= 20:
        returns = df['Close'].pct_change().dropna()
        features['volatility'] = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 0 else 0.0
    else:
        features['volatility'] = 0.0
    
    return features


def _extract_historical_features(df, idx):
    """Extract features for a specific historical index"""
    if idx < 0 or idx >= len(df):
        return None
    
    # Get data up to this index
    df_slice = df.iloc[:idx+1]
    
    # Calculate indicators for this slice
    from app.analysis.technical import calculate_technical_indicators
    indicators = calculate_technical_indicators(df_slice)
    
    # Get info (would need to be passed in, but for now use empty dict)
    info = {}
    metrics = {}
    news_list = []
    
    # Extract features
    features = extract_ml_features('', df_slice, info, indicators, metrics, news_list)
    
    return features


def _train_random_forest_model(features_dict, current_price, df=None):
    """Train Random Forest model for price prediction"""
    if not ML_AVAILABLE:
        return None, None
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features
        feature_names = sorted([k for k in features_dict.keys() if k != 'ticker'])
        X = np.array([[features_dict.get(name, 0.0) for name in feature_names]])
        
        # For training, we'd need historical data
        # This is a simplified version
        if df is not None and len(df) > 30:
            # Use historical prices as targets
            prices = df['Close'].values[-30:]
            X_hist = []
            y_hist = []
            
            for i in range(30, len(df)):
                hist_features = _extract_historical_features(df, i-1)
                if hist_features:
                    X_hist.append([hist_features.get(name, 0.0) for name in feature_names])
                    y_hist.append(df['Close'].iloc[i])
            
            if len(X_hist) > 10:
                X_train = np.array(X_hist)
                y_train = np.array(y_hist)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                
                return model, scaler
        
        return None, None
    except Exception as e:
        logger.exception(f"Error training model: {e}")
        return None, None


def predict_price(features, current_price, df=None):
    """
    Predict future stock price using ML models
    
    Args:
        features: Dict with ML features
        current_price: Current stock price
        df: Historical price data (optional, for model training)
        
    Returns:
        Dict with price predictions and confidence intervals
    """
    if not ML_AVAILABLE:
        # Fallback predictions without ML
        return {
            'current_price': current_price,
            'predictions': {
                '1m': {'price': current_price * 1.02, 'confidence': 0.5},
                '3m': {'price': current_price * 1.05, 'confidence': 0.4},
                '6m': {'price': current_price * 1.10, 'confidence': 0.3},
                '12m': {'price': current_price * 1.15, 'confidence': 0.2}
            },
            'expected_returns': {
                '1m': 2.0,
                '3m': 5.0,
                '6m': 10.0,
                '12m': 15.0
            },
            'confidence_intervals': {
                '6m': {'lower': current_price * 0.85, 'upper': current_price * 1.35},
                '12m': {'lower': current_price * 0.75, 'upper': current_price * 1.55}
            },
            'model_used': 'fallback'
        }
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        ticker = features.get('ticker', '')
        cache_key = f"rf_v{MODEL_CACHE_VERSION}_{ticker}"
        
        # Check cache
        if cache_key in _model_cache:
            model = _model_cache[cache_key]
            scaler = _scaler_cache.get(cache_key)
        else:
            # Train new model
            model, scaler = _train_random_forest_model(features, current_price, df)
            if model:
                _model_cache[cache_key] = model
                if scaler:
                    _scaler_cache[cache_key] = scaler
        
        if not model:
            # Fallback if model training failed
            return {
                'current_price': current_price,
                'predictions': {
                    '1m': {'price': current_price * 1.02, 'confidence': 0.5},
                    '3m': {'price': current_price * 1.05, 'confidence': 0.4},
                    '6m': {'price': current_price * 1.10, 'confidence': 0.3},
                    '12m': {'price': current_price * 1.15, 'confidence': 0.2}
                },
                'expected_returns': {
                    '1m': 2.0,
                    '3m': 5.0,
                    '6m': 10.0,
                    '12m': 15.0
                },
                'confidence_intervals': {
                    '6m': {'lower': current_price * 0.85, 'upper': current_price * 1.35},
                    '12m': {'lower': current_price * 0.75, 'upper': current_price * 1.55}
                },
                'model_used': 'fallback'
            }
        
        # Prepare features for prediction
        feature_names = sorted([k for k in features.keys() if k != 'ticker'])
        X = np.array([[features.get(name, 0.0) for name in feature_names]])
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        
        # Calculate confidence intervals (simplified)
        # In production, use prediction intervals from ensemble
        std_dev = current_price * 0.15  # Simplified
        
        predictions = {
            '1m': {'price': current_price * (1 + (prediction - current_price) / current_price * 0.25), 'confidence': 0.6},
            '3m': {'price': current_price * (1 + (prediction - current_price) / current_price * 0.5), 'confidence': 0.5},
            '6m': {'price': prediction, 'confidence': 0.4},
            '12m': {'price': current_price * (1 + (prediction - current_price) / current_price * 1.5), 'confidence': 0.3}
        }
        
        expected_returns = {
            '1m': ((predictions['1m']['price'] - current_price) / current_price) * 100,
            '3m': ((predictions['3m']['price'] - current_price) / current_price) * 100,
            '6m': ((predictions['6m']['price'] - current_price) / current_price) * 100,
            '12m': ((predictions['12m']['price'] - current_price) / current_price) * 100
        }
        
        confidence_intervals = {
            '6m': {'lower': prediction - 2*std_dev, 'upper': prediction + 2*std_dev},
            '12m': {'lower': prediction - 3*std_dev, 'upper': prediction + 3*std_dev}
        }
        
        return {
            'current_price': current_price,
            'predictions': predictions,
            'expected_returns': expected_returns,
            'confidence_intervals': confidence_intervals,
            'model_used': 'random_forest'
        }
        
    except Exception as e:
        logger.exception(f"Error in predict_price: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        return {
            'current_price': current_price,
            'predictions': {
                '1m': {'price': current_price * 1.02, 'confidence': 0.5},
                '3m': {'price': current_price * 1.05, 'confidence': 0.4},
                '6m': {'price': current_price * 1.10, 'confidence': 0.3},
                '12m': {'price': current_price * 1.15, 'confidence': 0.2}
            },
            'expected_returns': {
                '1m': 2.0,
                '3m': 5.0,
                '6m': 10.0,
                '12m': 15.0
            },
            'confidence_intervals': {
                '6m': {'lower': current_price * 0.85, 'upper': current_price * 1.35},
                '12m': {'lower': current_price * 0.75, 'upper': current_price * 1.55}
            },
            'model_used': 'fallback',
            'error': str(e)
        }


def _save_prediction_history(ticker: str, current_price: float, prediction_result: Dict, score: Optional[float] = None):
    """Save prediction to history file"""
    try:
        history_file = _PREDICTION_HISTORY_DIR / f"{ticker.upper()}.json"
        
        # Load existing history
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        
        # Add new entry
        entry = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': current_price,
            'prediction_6m': prediction_result.get('predictions', {}).get('6m', {}).get('price'),
            'prediction_12m': prediction_result.get('predictions', {}).get('12m', {}).get('price'),
            'expected_return_6m': prediction_result.get('expected_returns', {}).get('6m'),
            'expected_return_12m': prediction_result.get('expected_returns', {}).get('12m'),
            'score': score
        }
        
        history.append(entry)
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logger.exception(f"Error saving prediction history: {e}")


def get_prediction_history(ticker: str, days: int = 30) -> List[Dict]:
    """Get prediction history for a ticker"""
    try:
        history_file = _PREDICTION_HISTORY_DIR / f"{ticker.upper()}.json"
        
        if not history_file.exists():
            return []
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Sort by date (newest first) and limit to requested days
        history = sorted(history, key=lambda x: x.get('date', ''), reverse=True)
        
        if days > 0:
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_history = []
            for entry in history:
                try:
                    entry_date = datetime.strptime(entry.get('date', ''), '%Y-%m-%d %H:%M:%S')
                    if entry_date >= cutoff_date:
                        filtered_history.append(entry)
                except:
                    continue
            history = filtered_history
        
        return history
    except Exception as e:
        logger.exception(f"Error loading prediction history: {e}")
        return []


def generate_ai_recommendations(ticker: str) -> Optional[Dict]:
    """
    Generate AI-powered stock recommendations
    
    NOTE: This is a stub implementation. Full implementation requires:
    - classify_trend() function
    - calculate_entry_tp_dca() function  
    - calculate_position_sizing() function
    
    These functions should be moved from app.py or reimplemented.
    """
    try:
        from app.services.yfinance_service import get_stock_data
        from app.services.news_service import get_stock_news
        from app.analysis.technical import calculate_technical_indicators
        from app.analysis.fundamental import calculate_metrics
        from app.analysis.risk import calculate_risk_score
        
        # Get stock data
        stock_data = get_stock_data(ticker, '1y')
        if not stock_data or stock_data['history'].empty:
            return None
        
        df = stock_data['history']
        info = stock_data.get('info', {})
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(df)
        
        # Get current metrics
        metrics = calculate_metrics(df, info)
        
        # Get news sentiment
        news_list = get_stock_news(ticker)
        
        # Extract ML features and run ML models
        ml_features = extract_ml_features(ticker, df, info, indicators, metrics, news_list)
        ml_features['ticker'] = ticker.upper()
        price_prediction = predict_price(ml_features, current_price, df)
        
        # Calculate risk score
        risk_analysis = calculate_risk_score(ml_features, metrics, info)
        
        # Stub implementations for missing functions
        trend_classification = {'trend_class': 'Neutral', 'confidence': 0.5}
        entry_tp_dca = {
            'entry_point': current_price,
            'take_profit': current_price * 1.15,
            'dca_levels': [current_price * 0.95, current_price * 0.90, current_price * 0.85]
        }
        # Calculate position sizing with proper structure
        risk_score = risk_analysis.get('risk_score', 50)
        ml_confidence = price_prediction.get('confidence', 50) if price_prediction else 50
        
        # Base position size calculation
        base_position = 5.0  # Default 5%
        
        # Adjust based on risk score (higher risk = smaller position)
        if risk_score >= 70:
            base_position = 1.0
        elif risk_score >= 50:
            base_position = 3.0
        elif risk_score >= 30:
            base_position = 5.0
        else:
            base_position = 7.0
        
        # Adjust based on ML confidence (higher confidence = larger position)
        confidence_multiplier = ml_confidence / 50.0  # 0.5x to 2.0x
        recommended_pct = base_position * confidence_multiplier
        
        # Cap at reasonable limits
        recommended_pct = max(0.5, min(15.0, recommended_pct))
        
        # Calculate range (conservative to aggressive)
        conservative_pct = recommended_pct * 0.6  # 60% of recommended
        aggressive_pct = recommended_pct * 1.5   # 150% of recommended
        conservative_pct = max(0.5, min(10.0, conservative_pct))
        aggressive_pct = max(1.0, min(20.0, aggressive_pct))
        
        # Determine size category and color
        if recommended_pct >= 10:
            size_category = 'Large'
            size_color = '#10b981'
        elif recommended_pct >= 5:
            size_category = 'Medium'
            size_color = '#fbbf24'
        else:
            size_category = 'Small'
            size_color = '#ef4444'
        
        # Calculate volatility (ATR-based)
        atr = ml_features.get('atr', current_price * 0.02)
        volatility_pct = (atr / current_price * 100) if current_price > 0 else 2.0
        
        position_sizing = {
            'recommended_pct': round(recommended_pct, 1),
            'size_category': size_category,
            'size_color': size_color,
            'range': {
                'conservative': round(conservative_pct, 1),
                'aggressive': round(aggressive_pct, 1)
            },
            'risk_score': round(risk_score, 1),
            'ml_confidence': round(ml_confidence, 1),
            'volatility_pct': round(volatility_pct, 2),
            'reasoning': f'Position size based on risk score ({risk_score:.1f}/100) and ML confidence ({ml_confidence:.1f}%). Lower risk and higher confidence allow for larger positions.',
            'confidence_factors': [
                f"Risk Score: {risk_score:.1f}/100",
                f"ML Confidence: {ml_confidence:.1f}%",
                f"Volatility: {volatility_pct:.2f}%"
            ],
            'adjustments': {
                'risk': round(risk_score / 50.0, 2),
                'confidence': round(ml_confidence / 50.0, 2),
                'volatility': round(volatility_pct / 2.0, 2),
                'risk_reward': 1.0
            }
        }
        
        # Calculate technical score
        technical_score = 50.0
        reasons = []
        warnings = []
        
        # Basic RSI analysis
        rsi_values = indicators.get('rsi', [])
        if rsi_values and len(rsi_values) > 0:
            current_rsi = rsi_values[-1]
            if current_rsi is not None and not pd.isna(current_rsi):
                if current_rsi < 30:
                    technical_score += 15
                    reasons.append("RSI indicates oversold conditions")
                elif current_rsi > 70:
                    technical_score -= 15
                    warnings.append("RSI indicates overbought conditions")
        
        # Prepare chart data
        dates = df.index.strftime('%Y-%m-%d').tolist()
        chart_data = {
            'dates': dates,
            'open': [float(x) for x in df['Open'].round(2).fillna(0).tolist()],
            'high': [float(x) for x in df['High'].round(2).fillna(0).tolist()],
            'low': [float(x) for x in df['Low'].round(2).fillna(0).tolist()],
            'close': [float(x) for x in df['Close'].round(2).fillna(0).tolist()],
            'volume': [int(x) for x in df['Volume'].fillna(0).astype(int).tolist()],
        }
        
        return {
            'ticker': ticker.upper(),
            'current_price': current_price,
            'technical_score': max(0, min(100, technical_score)),
            'reasons': reasons,
            'warnings': warnings,
            'price_prediction': price_prediction,
            'trend_classification': trend_classification,
            'risk_analysis': risk_analysis,
            'entry_tp_dca': entry_tp_dca,
            'position_sizing': position_sizing,
            'chart_data': chart_data
        }
        
    except Exception as e:
        logger.exception(f"Error in AI recommendations: {e}")
        import traceback
        traceback.print_exc()
        return None
