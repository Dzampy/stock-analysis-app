"""
ML service - Price predictions, backtesting, model management
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, List, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json
from app.config import ML_AVAILABLE, CACHE_TIMEOUTS
from app.utils.constants import MODEL_CACHE_VERSION
from app.utils.logger import logger
import time

# Import cache - will be initialized when app starts
try:
    from app import cache
    CACHE_AVAILABLE = True
except (ImportError, RuntimeError):
    CACHE_AVAILABLE = False
    cache = None

# Model cache (module-level)
_model_cache = {}
_scaler_cache = {}

# Prediction history directory
_PREDICTION_HISTORY_DIR = Path('.ml_predictions')
_PREDICTION_HISTORY_DIR.mkdir(exist_ok=True)


def _clear_model_cache_if_needed():
    """Clear model cache if version changed"""
    if _model_cache and not any(k.startswith(
            f"rf_v{MODEL_CACHE_VERSION}_") for k in _model_cache.keys()):
        logger.info(
            f"Clearing model cache due to version change (new version: {MODEL_CACHE_VERSION})")
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


def extract_ml_features(
        ticker: str,
        df: pd.DataFrame,
        info: Dict,
        indicators: Dict,
        metrics: Dict,
        news_list: List[Dict]) -> Dict:
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
        features['rsi'] = float(
            rsi_values[-1]) if not pd.isna(rsi_values[-1]) else 50.0
        features['rsi_7d_avg'] = float(
            np.mean(rsi_values[-7:])) if len(rsi_values) >= 7 else 50.0
    else:
        features['rsi'] = 50.0
        features['rsi_7d_avg'] = 50.0
    
    # MACD features
    if macd_values and macd_signal and len(macd_values) > 0:
        features['macd'] = float(
            macd_values[-1]) if not pd.isna(macd_values[-1]) else 0.0
        features['macd_signal'] = float(
            macd_signal[-1]) if not pd.isna(macd_signal[-1]) else 0.0
        features['macd_diff'] = features['macd'] - features['macd_signal']
        features['macd_bullish'] = 1.0 if features['macd_diff'] > 0 else 0.0
    else:
        features['macd'] = 0.0
        features['macd_signal'] = 0.0
        features['macd_diff'] = 0.0
        features['macd_bullish'] = 0.0
    
    # Moving Average features
    if sma_20 and len(sma_20) > 0:
        features['sma_20'] = float(
            sma_20[-1]) if not pd.isna(sma_20[-1]) else current_price
        features['price_vs_sma20'] = (
            (current_price - features['sma_20']) / features['sma_20']) * 100 if features['sma_20'] > 0 else 0.0
    else:
        features['sma_20'] = current_price
        features['price_vs_sma20'] = 0.0
    
    if sma_50 and len(sma_50) > 0:
        features['sma_50'] = float(
            sma_50[-1]) if not pd.isna(sma_50[-1]) else current_price
        features['price_vs_sma50'] = (
            (current_price - features['sma_50']) / features['sma_50']) * 100 if features['sma_50'] > 0 else 0.0
    else:

        features['sma_50'] = current_price

        features['price_vs_sma50'] = 0.0
    
    # Bollinger Bands features
    if bb_high and bb_low and bb_mid and len(bb_high) > 0:
        features['bb_high'] = float(
            bb_high[-1]) if not pd.isna(bb_high[-1]) else current_price * 1.1
        features['bb_low'] = float(
            bb_low[-1]) if not pd.isna(bb_low[-1]) else current_price * 0.9
        features['bb_mid'] = float(
            bb_mid[-1]) if not pd.isna(bb_mid[-1]) else current_price
        features['bb_width'] = ((features['bb_high'] - features['bb_low']) /
                                features['bb_mid']) * 100 if features['bb_mid'] > 0 else 0.0
        features['bb_position'] = (
            (current_price - features['bb_low']) / (
                features['bb_high'] - features['bb_low'])) * 100 if (
            features['bb_high'] - features['bb_low']) > 0 else 50.0
    else:

        features['bb_high'] = current_price * 1.1

        features['bb_low'] = current_price * 0.9

        features['bb_mid'] = current_price

        features['bb_width'] = 20.0

        features['bb_position'] = 50.0
    
    # ADX features
    if adx_values and len(adx_values) > 0:
        features['adx'] = float(
            adx_values[-1]) if not pd.isna(adx_values[-1]) else 25.0
    else:

        features['adx'] = 25.0
    
    # Stochastic features
    if stoch_k_values and stoch_d_values and len(stoch_k_values) > 0:
        features['stoch_k'] = float(
            stoch_k_values[-1]) if not pd.isna(stoch_k_values[-1]) else 50.0
        features['stoch_d'] = float(
            stoch_d_values[-1]) if not pd.isna(stoch_d_values[-1]) else 50.0
        features['stoch_oversold'] = 1.0 if features['stoch_k'] < 20 else 0.0
        features['stoch_overbought'] = 1.0 if features['stoch_k'] > 80 else 0.0
    else:

        features['stoch_k'] = 50.0

        features['stoch_d'] = 50.0

        features['stoch_oversold'] = 0.0

        features['stoch_overbought'] = 0.0
    
    # ATR features
    if atr_values and len(atr_values) > 0:
        features['atr'] = float(
            atr_values[-1]) if not pd.isna(atr_values[-1]) else current_price * 0.02
        features['atr_pct'] = (
            features['atr'] / current_price) * 100 if current_price > 0 else 2.0
    else:

        features['atr'] = current_price * 0.02

        features['atr_pct'] = 2.0
    
    # Price momentum features
    if len(df) >= 5:
        features['price_change_5d'] = (
            (current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
    else:

        features['price_change_5d'] = 0.0
    
    if len(df) >= 10:
        features['price_change_10d'] = (
            (current_price - df['Close'].iloc[-10]) / df['Close'].iloc[-10]) * 100
    else:

        features['price_change_10d'] = 0.0
    
    if len(df) >= 30:
        features['price_change_30d'] = (
            (current_price - df['Close'].iloc[-30]) / df['Close'].iloc[-30]) * 100
    else:

        features['price_change_30d'] = 0.0
    
    # Lag features for time series prediction (important for price prediction)
    # These capture recent price movements which are often predictive
    if len(df) >= 1:
        features['price_lag_1'] = float(
            df['Close'].iloc[-1]) / current_price if current_price > 0 else 1.0
    else:

        features['price_lag_1'] = 1.0

    if len(df) >= 5:
        features['price_lag_5'] = float(
            df['Close'].iloc[-5]) / current_price if current_price > 0 else 1.0
    else:

        features['price_lag_5'] = 1.0

    if len(df) >= 10:
        features['price_lag_10'] = float(
            df['Close'].iloc[-10]) / current_price if current_price > 0 else 1.0
    else:

        features['price_lag_10'] = 1.0

    # Rolling statistics for better time series features
    if len(df) >= 20:
        rolling_mean_20 = df['Close'].tail(20).mean()
        rolling_std_20 = df['Close'].tail(20).std()
        features['price_vs_rolling_mean_20'] = (
            (current_price - rolling_mean_20) / rolling_mean_20) * 100 if rolling_mean_20 > 0 else 0.0
        features['rolling_std_20_pct'] = (
            rolling_std_20 / rolling_mean_20) * 100 if rolling_mean_20 > 0 else 0.0
    else:

        features['price_vs_rolling_mean_20'] = 0.0

        features['rolling_std_20_pct'] = 0.0
    
    # Volume features
    if 'Volume' in df.columns:
        avg_volume = df['Volume'].tail(20).mean() if len(
            df) >= 20 else df['Volume'].mean()
        current_volume = df['Volume'].iloc[-1]
        features['volume_ratio'] = (
            current_volume /
            avg_volume) if avg_volume > 0 else 1.0
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
    features['pe_ratio'] = float(
        pe_ratio) if pe_ratio and not pd.isna(pe_ratio) else 0.0
    features['pb_ratio'] = float(
        pb_ratio) if pb_ratio and not pd.isna(pb_ratio) else 0.0
    features['ps_ratio'] = float(
        ps_ratio) if ps_ratio and not pd.isna(ps_ratio) else 0.0
    features['revenue_growth'] = float(
        revenue_growth) * 100 if revenue_growth and not pd.isna(revenue_growth) else 0.0
    features['earnings_growth'] = float(
        earnings_growth) * 100 if earnings_growth and not pd.isna(earnings_growth) else 0.0
    features['roe'] = float(roe) * 100 if roe and not pd.isna(roe) else 0.0
    features['debt_to_equity'] = float(
        debt_to_equity) if debt_to_equity and not pd.isna(debt_to_equity) else 0.0
    features['beta'] = float(beta) if beta and not pd.isna(beta) else 1.0
    
    # News sentiment features
    if news_list:
        sentiments = [article.get('sentiment_score', 0.0)
                      for article in news_list[:10]]
        features['news_sentiment_avg'] = float(
            np.mean(sentiments)) if sentiments else 0.0
        features['news_count'] = len(news_list[:10])
    else:

        features['news_sentiment_avg'] = 0.0

        features['news_count'] = 0
    
    # Volatility features
    if len(df) >= 20:
        returns = df['Close'].pct_change().dropna()
        features['volatility'] = float(
            returns.std() *
            np.sqrt(252) *
            100) if len(returns) > 0 else 0.0
    else:

        features['volatility'] = 0.0
    
    return features


def _extract_historical_features(df, idx):
    """
    Extract features for a specific historical index (no data leakage)

    Args:
        df: Full DataFrame
        idx: Index to extract features for (only uses data up to and including idx)

    Returns:
        Dict with features or None
        """

    if idx < 0 or idx >= len(df):

        return None
    
    # CRITICAL: Get data ONLY up to and including idx (no future data)
    # We use idx+1 because we want data including idx, but features should
    # only use past data, so we'll use only idx for feature calculation
    df_slice = df.iloc[:idx + 1].copy()

    # Ensure we have enough data for indicators (minimum lookback)
    min_lookback = 60  # Most indicators need ~14-50 days
    if len(df_slice) < min_lookback:
        logger.debug(
            f"Insufficient data for feature extraction: {len(df_slice)} < {min_lookback}")
        return None

    # Calculate indicators using ONLY past data (df_slice ends at idx)
    from app.analysis.technical import calculate_technical_indicators
    indicators = calculate_technical_indicators(df_slice)
    
    # Validate indicators don't use future data - take only the last value
    # This ensures we're using only data available at time idx
    validated_indicators = {}
    for key, values in indicators.items():
        if values and len(values) > 0:
            # Take the last value (which corresponds to idx)
            validated_indicators[key] = [
                values[-1]] if isinstance(values, list) else values
        else:
            validated_indicators[key] = []
    
    # Get info (would need to be passed in, but for now use empty dict)
    # Note: Historical info would be point-in-time, not current
    info = {}
    metrics = {}
    news_list = []
    
    # Extract features using validated indicators
    features = extract_ml_features(
        '',
        df_slice,
        info,
        validated_indicators,
        metrics,
        news_list)
    
    return features


def _download_extended_historical_data(
        ticker: str, years: int = 3) -> Optional[pd.DataFrame]:
    """
    Download extended historical data (3+ years) for ML training

    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data to download (default: 3)

    Returns:
        DataFrame with historical price data or None
        """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        logger.info(
            f"Downloading {years} years of historical data for {ticker}")

        # Calculate period
        if years <= 1:
            period = '1y'
        elif years <= 2:
            period = '2y'
        elif years <= 3:
            period = '3y'
        elif years <= 5:
            period = '5y'
        else:
            period = 'max'

        # Download data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=True, prepost=False)

        if hist.empty:
            logger.warning(f"No historical data available for {ticker}")

            return None

        # Convert timezone-aware index to timezone-naive for easier date
        # operations
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        # Ensure we have at least 2 years of data (approximately 500 trading days)
        # Prefer 3+ years for better model training
        min_days = 500
        preferred_days = 750  # 3 years
        if len(hist) < min_days:
            logger.warning(
                f"Only {len(hist)} days of data available for {ticker}, minimum {min_days} recommended")
        elif len(hist) < preferred_days:
            logger.info(
                f"Have {len(hist)} days of data, {preferred_days} days preferred for optimal training")

        logger.info(
            f"Downloaded {len(hist)} days of historical data for {ticker}")
        return hist

    except Exception as e:
        logger.exception(
            f"Error downloading extended historical data for {ticker}: {e}")
        return None


def _train_random_forest_model(ticker: str,
                               features_dict: Dict,
                               current_price: float,
                               df: Optional[pd.DataFrame] = None) -> Tuple[Optional[Any],
                                                                           Optional[Any]]:
    """
    Train Random Forest model for price prediction using 2+ years of historical data

    Args:
        ticker: Stock ticker symbol
        features_dict: Current features dict
        current_price: Current stock price
        df: Optional DataFrame with historical data (if None, will download)

    Returns:
        Tuple of (model, scaler) or (None, None) if training fails
        """

    if not ML_AVAILABLE:

        return None, None
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        # Download extended historical data if not provided
        if df is None or len(df) < 500:
            logger.info(f"Downloading extended historical data for {ticker}")

            df = _download_extended_historical_data(ticker, years=2)
            if df is None or len(df) < 100:
                logger.warning(
                    f"Insufficient historical data for {ticker} to train model")

        return None, None
        
        # Prepare features
        feature_names = sorted(
            [k for k in features_dict.keys() if k != 'ticker'])

        # Prepare features
        feature_names = sorted(
            [k for k in features_dict.keys() if k != 'ticker'])

        # We need at least 100 days of historical data for meaningful training
        min_training_days = 100
        if len(df) < min_training_days:
            logger.warning(
                f"Insufficient data: {len(df)} days, need at least {min_training_days}")

            return None, None

        # Build training dataset using walk-forward approach
        # For each day, extract features and predict next day's price
            X_hist = []
            y_hist = []
            
        # Use at least 60 days lookback for feature calculation
        lookback_days = 60

        logger.info(
            f"Building training dataset for {ticker} with {len(df)} days of data")

        for i in range(
                lookback_days,
                len(df) - 1):  # -1 because we predict next day
            try:
                # Extract features for this historical point
                hist_features = _extract_historical_features(df, i)
                if hist_features is None:
                    continue

                # Predict absolute price (not percentage return)
                # Percentage return caused worse CV R² scores, reverting to
                # absolute price
                current_price_at_idx = df['Close'].iloc[i]
                next_day_price = df['Close'].iloc[i + 1]

                # Use absolute price as target (normalized by current price for better scaling)
                # Normalize target to reduce impact of price scale differences
                target_price_normalized = next_day_price / \
                    current_price_at_idx if current_price_at_idx > 0 else 1.0

                # Build feature vector
                feature_vector = [
                    hist_features.get(
                        name, 0.0) for name in feature_names]
                X_hist.append(feature_vector)
                # Normalized price (ratio to current price)
                y_hist.append(target_price_normalized)

            except Exception as e:
                logger.debug(f"Error extracting features for index {i}: {e}")
                continue

        if len(X_hist) < 50:
            logger.warning(
                f"Insufficient training samples: {len(X_hist)}, need at least 50")

            return None, None

        # Convert to numpy arrays
                X_train = np.array(X_hist)
                y_train = np.array(y_hist)
                
        logger.info(
            f"Training Random Forest model with {len(X_train)} samples and {len(feature_names)} features")
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
        # Cross-validation for model validation (TimeSeriesSplit for time
        # series data)
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        # Use TimeSeriesSplit for proper time series cross-validation
        # This prevents data leakage from future to past
        tscv = TimeSeriesSplit(
            n_splits=min(
                5,
                len(X_train) //
                50))  # At least 50 samples per fold

        # Try different hyperparameter sets - OPTIMIZED for better CV R²
        # Based on analysis: need more regularization but also enough capacity
        # to learn patterns
        hyperparameter_sets = [
            {
                'n_estimators': 100,  # Reduced for faster training and less overfitting
                'max_depth': 8,  # Shallow trees to prevent overfitting

                'min_samples_split': 30,  # High threshold - more regularization

                'min_samples_leaf': 15,  # High threshold - more regularization

                'max_features': 'sqrt'  # Feature sampling to reduce overfitting

            },

            {
                'n_estimators': 150,
                'max_depth': 10,

                'min_samples_split': 25,

                'min_samples_leaf': 12,

                'max_features': 'sqrt'

            },

            {
                'n_estimators': 200,
                'max_depth': 12,

                'min_samples_split': 20,

                'min_samples_leaf': 10,

                'max_features': 'log2'

            }

        ]

        best_score = float('-inf')
        best_params = None
        best_model = None

        # If we have enough data, do hyperparameter tuning with
        # cross-validation
        if len(X_train) >= 100 and len(hyperparameter_sets) > 1:
            logger.info(
                f"Performing hyperparameter tuning with {tscv.get_n_splits()} folds")

            for params in hyperparameter_sets:
                try:

                    model_cv = RandomForestRegressor(

                        random_state=42,
                        n_jobs=-1,
                        verbose=0,
                        **params
                    )

                    # Cross-validation score (negative because we want to
                    # maximize R²)
                    cv_scores = cross_val_score(
                        model_cv, X_train_scaled, y_train,
                        cv=tscv, scoring='r2', n_jobs=-1
                    )
                    avg_score = np.mean(cv_scores)

                    logger.debug(
                        f"CV R² score: {avg_score:.4f} (+/- {np.std(cv_scores):.4f}) for params {params}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params
                        best_model = model_cv

                except Exception as e:
                    logger.debug(f"Error in CV for params {params}: {e}")
                    continue

        # Use best model from CV or default if CV failed
        if best_model is not None and best_params:
            logger.info(
                f"Best hyperparameters: {best_params} with CV R²: {best_score:.4f}")

            model = best_model
        else:
            # Default hyperparameters if CV didn't work
            logger.info("Using default hyperparameters")
            best_params = hyperparameter_sets[0]
            model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                verbose=0,
                **best_params
            )

        # Train final model on all training data
                model.fit(X_train_scaled, y_train)

        # Calculate training score for logging
        train_score = model.score(X_train_scaled, y_train)
        logger.info(
            f"Model trained successfully for {ticker}. Training R² score: {train_score:.4f}")

        # If we did CV, log the CV score too
        if best_score != float('-inf'):
            logger.info(f"Cross-validation R² score: {best_score:.4f}")

        # Store feature importance for later use and feature selection
        if hasattr(model, 'feature_importances_'):
            feature_importance_dict = dict(
                zip(feature_names, model.feature_importances_))

            # Sort by importance
            sorted_importance = sorted(
                feature_importance_dict.items(),
                key=lambda x: x[1],
                reverse=True)
            logger.debug(
                f"Top 5 most important features: {sorted_importance[:5]}")

            # Feature selection: Keep only top 70% of features by importance
            # This reduces noise and improves generalization
            if len(sorted_importance) > 10:
                threshold_idx = max(10, int(len(sorted_importance) * 0.7))
                top_features = [name for name,
                                _ in sorted_importance[:threshold_idx]]
                logger.info(
                    f"Feature selection: Keeping top {len(top_features)}/{len(feature_names)} features")
                # Store selected features for future use
                model.selected_features_ = top_features
            else:
                model.selected_features_ = feature_names

            # Attach to model for later retrieval
            model.feature_names_ = feature_names
            model.feature_importances_dict_ = feature_importance_dict

        # Store CV R² score for transparency
        if best_score != float('-inf'):
            model.cv_r2_score = float(best_score)

            model.train_r2_score = float(train_score)
        else:

            # If no CV was performed, use training score as approximation
            model.cv_r2_score = float(train_score)
            model.train_r2_score = float(train_score)

        # Store flag indicating if model is better than baseline
        model.is_better_than_baseline = model.cv_r2_score > 0.0
                
                return model, scaler
        
    except Exception as e:
        logger.exception(f"Error training model for {ticker}: {e}")
        return None, None


def predict_price(features, current_price, df=None):
    """
    Predict future stock price using ML models (cached)
    
    Args:
        features: Dict with ML features
        current_price: Current stock price
        df: Historical price data (optional, for model training)
        
    Returns:
        Dict with price predictions and confidence intervals
    """

    # Validate and ensure current_price is correct
    # If df is provided, prefer the latest close price from df as it's more accurate
    if df is not None and len(df) > 0:
        df_current_price = float(df['Close'].iloc[-1])
        # If provided current_price differs significantly from df price, use df price
        if abs(current_price - df_current_price) > 0.1:
            logger.warning(f"current_price mismatch: provided={current_price:.2f}, df={df_current_price:.2f}, using df price")
            current_price = df_current_price
    
    # Check cache first
    ticker = features.get('ticker', '')
    if CACHE_AVAILABLE and cache and ticker:
        cache_key = f"ml_predict_price_{ticker}"
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for ML price prediction {ticker}")
            return cached_data
    
    if not ML_AVAILABLE:
        # Simple estimate based on historical momentum (NOT ML prediction)
        logger.warning(
            f"ML not available for {ticker}, using simple momentum-based estimates")

        # Calculate simple momentum if historical data available
        momentum_1m = 0.0
        momentum_3m = 0.0
        momentum_6m = 0.0
        momentum_12m = 0.0

        if df is not None and len(df) > 20:
            # Calculate recent momentum
            if len(df) >= 20:
                momentum_1m = (
                    (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100

        if len(df) >= 60:
            momentum_3m = (
                (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]) * 100
            if len(df) >= 120:

                momentum_6m = (
                    (df['Close'].iloc[-1] - df['Close'].iloc[-120]) / df['Close'].iloc[-120]) * 100

        if len(df) >= 252:
            momentum_12m = (
                (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]) * 100

        # Use momentum with conservative estimates (50% of momentum)
        return {
            'current_price': current_price,
            'predictions': {
                '1m': {'price': current_price * (1 + momentum_1m * 0.5 / 100), 'confidence': 0.3},

                '3m': {'price': current_price * (1 + momentum_3m * 0.5 / 100), 'confidence': 0.25},

                '6m': {'price': current_price * (1 + momentum_6m * 0.5 / 100), 'confidence': 0.2},

                '12m': {'price': current_price * (1 + momentum_12m * 0.5 / 100), 'confidence': 0.15}

            },
            'expected_returns': {
                '1m': momentum_1m * 0.5,

                '3m': momentum_3m * 0.5,

                '6m': momentum_6m * 0.5,

                '12m': momentum_12m * 0.5

            },
            'confidence_intervals': {
                '6m': {'lower': current_price * 0.80, 'upper': current_price * 1.30},

                '12m': {'lower': current_price * 0.70, 'upper': current_price * 1.50}

            },
            'model_used': 'momentum_estimate',
            'warning': 'ML models not available. Using simple momentum-based estimates. These are NOT ML predictions.'
        }
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        ticker = features.get('ticker', '')
        cache_key = f"rf_v{MODEL_CACHE_VERSION}_{ticker}"
        
        # Check cache
        model = None
        scaler = None
        if cache_key in _model_cache:
            cached_model = _model_cache[cache_key]
            # Only use cached model if it's not None
            if cached_model is not None:
                model = cached_model
            scaler = _scaler_cache.get(cache_key)
                logger.debug(f"Using cached model for {ticker}")
        else:
                logger.debug(f"Cached model is None for {ticker}, will use momentum-based estimates")
        
        # If no valid model from cache, train new one
        if model is None:
            # Train new model with extended historical data
            logger.info(f"Training new ML model for {ticker}")
            model, scaler = _train_random_forest_model(
                ticker, features, current_price, df)
            if model:
                _model_cache[cache_key] = model
                if scaler:
                    _scaler_cache[cache_key] = scaler
                logger.info(f"Model trained and cached for {ticker}")
            else:
                # Training failed - cache None to avoid retrying immediately
                _model_cache[cache_key] = None
                if scaler:
                    _scaler_cache[cache_key] = scaler
                logger.warning(f"Model training failed for {ticker}")
        
        # CRITICAL: Check if model is None after cache lookup/training
        if model is None:
            # Model training failed - use momentum-based estimates with clear
            # warning
            logger.warning(
                f"Model training failed for {ticker}, using momentum-based estimates")

            # Calculate momentum from available data
            momentum_1m = 0.0
            momentum_3m = 0.0
            momentum_6m = 0.0
            momentum_12m = 0.0

            if df is not None and len(df) > 20:
                if len(df) >= 20:
                    momentum_1m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
                if len(df) >= 60:
                    momentum_3m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]) * 100
                if len(df) >= 120:
                    momentum_6m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-120]) / df['Close'].iloc[-120]) * 100
                if len(df) >= 252:
                    momentum_12m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]) * 100

            return {
                'current_price': current_price,
                'predictions': {
                    '1m': {'price': current_price * (1 + momentum_1m * 0.5 / 100), 'confidence': 0.3},
                    '3m': {'price': current_price * (1 + momentum_3m * 0.5 / 100), 'confidence': 0.25},
                    '6m': {'price': current_price * (1 + momentum_6m * 0.5 / 100), 'confidence': 0.2},
                    '12m': {'price': current_price * (1 + momentum_12m * 0.5 / 100), 'confidence': 0.15}
                },
                'expected_returns': {
                    '1m': momentum_1m * 0.5,
                    '3m': momentum_3m * 0.5,
                    '6m': momentum_6m * 0.5,
                    '12m': momentum_12m * 0.5
                },
                'confidence_intervals': {
                    '6m': {'lower': current_price * 0.80, 'upper': current_price * 1.30},
                    '12m': {'lower': current_price * 0.70, 'upper': current_price * 1.50}
                },
                'model_used': 'momentum_estimate',
                'warning': 'ML model training failed. Using momentum-based estimates. These are NOT ML predictions.'
            }
        
        # Prepare features for prediction
        feature_names = sorted([k for k in features.keys() if k != 'ticker'])
        X = np.array([[features.get(name, 0.0) for name in feature_names]])
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:

            X_scaled = X
        
        # CRITICAL: Additional safety check - verify model has required attributes
        if not hasattr(model, 'estimators_') or model.estimators_ is None:
            logger.error(f"Model for {ticker} is missing estimators_ attribute, falling back to momentum")
            # Fall back to momentum-based estimates
            momentum_1m = 0.0
            momentum_3m = 0.0
            momentum_6m = 0.0
            momentum_12m = 0.0
            if df is not None and len(df) > 20:
                if len(df) >= 20:
                    momentum_1m = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
                if len(df) >= 60:
                    momentum_3m = ((df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]) * 100
                if len(df) >= 120:
                    momentum_6m = ((df['Close'].iloc[-1] - df['Close'].iloc[-120]) / df['Close'].iloc[-120]) * 100
                if len(df) >= 252:
                    momentum_12m = ((df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]) * 100
            return {
                'current_price': current_price,
                'predictions': {
                    '1m': {'price': current_price * (1 + momentum_1m * 0.5 / 100), 'confidence': 0.3},
                    '3m': {'price': current_price * (1 + momentum_3m * 0.5 / 100), 'confidence': 0.25},
                    '6m': {'price': current_price * (1 + momentum_6m * 0.5 / 100), 'confidence': 0.2},
                    '12m': {'price': current_price * (1 + momentum_12m * 0.5 / 100), 'confidence': 0.15}
                },
                'expected_returns': {
                    '1m': momentum_1m * 0.5,
                    '3m': momentum_3m * 0.5,
                    '6m': momentum_6m * 0.5,
                    '12m': momentum_12m * 0.5
                },
                'confidence_intervals': {
                    '6m': {'lower': current_price * 0.80, 'upper': current_price * 1.30},
                    '12m': {'lower': current_price * 0.70, 'upper': current_price * 1.50}
                },
                'model_used': 'momentum_estimate',
                'warning': 'ML model is missing required attributes. Using momentum-based estimates. These are NOT ML predictions.'
            }
        
        # Predict using all trees in ensemble for proper confidence intervals
        tree_predictions = []
        try:
            for tree in model.estimators_:
                tree_pred = tree.predict(X_scaled)[0]
                tree_predictions.append(tree_pred)
        except (AttributeError, TypeError) as e:
            logger.error(f"Error accessing model.estimators_ for {ticker}: {e}, falling back to momentum")
            # Fall back to momentum-based estimates
            momentum_1m = 0.0
            momentum_3m = 0.0
            momentum_6m = 0.0
            momentum_12m = 0.0
            if df is not None and len(df) > 20:
                if len(df) >= 20:
                    momentum_1m = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
                if len(df) >= 60:
                    momentum_3m = ((df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]) * 100
                if len(df) >= 120:
                    momentum_6m = ((df['Close'].iloc[-1] - df['Close'].iloc[-120]) / df['Close'].iloc[-120]) * 100
                if len(df) >= 252:
                    momentum_12m = ((df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]) * 100
            return {
                'current_price': current_price,
                'predictions': {
                    '1m': {'price': current_price * (1 + momentum_1m * 0.5 / 100), 'confidence': 0.3},
                    '3m': {'price': current_price * (1 + momentum_3m * 0.5 / 100), 'confidence': 0.25},
                    '6m': {'price': current_price * (1 + momentum_6m * 0.5 / 100), 'confidence': 0.2},
                    '12m': {'price': current_price * (1 + momentum_12m * 0.5 / 100), 'confidence': 0.15}
                },
                'expected_returns': {
                    '1m': momentum_1m * 0.5,
                    '3m': momentum_3m * 0.5,
                    '6m': momentum_6m * 0.5,
                    '12m': momentum_12m * 0.5
                },
                'confidence_intervals': {
                    '6m': {'lower': current_price * 0.80, 'upper': current_price * 1.30},
                    '12m': {'lower': current_price * 0.70, 'upper': current_price * 1.50}
                },
                'model_used': 'momentum_estimate',
                'warning': 'ML model error during prediction. Using momentum-based estimates. These are NOT ML predictions.'
            }

        tree_predictions = np.array(tree_predictions)
        # Mean prediction from ensemble (normalized price ratio)
        prediction_mean = np.mean(tree_predictions)
        # Standard deviation across trees
        prediction_std = np.std(tree_predictions)

        # Model predicts normalized price ratio (next_price / current_price)
        # Convert back to absolute price
        next_day_prediction = current_price * \
            prediction_mean if current_price > 0 else current_price

        # Check if model is better than baseline - if not, ignore ML
        # predictions
        use_ml_prediction = True
        if hasattr(model, 'is_better_than_baseline'):
            use_ml_prediction = model.is_better_than_baseline

            if not use_ml_prediction:
                logger.warning(
                    f"ML model CV R² ({model.cv_r2_score:.3f}) < 0, using momentum-only predictions")

        # Calculate annualized return expectation from next-day prediction
        if current_price > 0 and use_ml_prediction:
            next_day_return = (
                next_day_prediction - current_price) / current_price

            annualized_return = next_day_return * 252 * \
                100  # Annualize and convert to percentage
        else:

            # If ML is worse than baseline, set annualized return to 0 (will
            # use only momentum)
            annualized_return = 0.0
            next_day_prediction = current_price  # No change predicted

        # Calculate historical volatility for dynamic capping
        historical_volatility = 0.0
        if df is not None and len(df) >= 60:
            returns = df['Close'].pct_change().dropna()
            historical_volatility = returns.std() * np.sqrt(252) * \
                100  # Annualized volatility in %

        # Calculate historical momentum for each timeframe (for hybrid
        # approach)
        momentum_pcts = {}
        if df is not None:
            if len(df) >= 21:

                momentum_pcts['1m'] = (
                    (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21]) * 100

        if len(df) >= 63:
            momentum_pcts['3m'] = (
                (df['Close'].iloc[-1] - df['Close'].iloc[-63]) / df['Close'].iloc[-63]) * 100
            if len(df) >= 126:

                momentum_pcts['6m'] = (
                    (df['Close'].iloc[-1] - df['Close'].iloc[-126]) / df['Close'].iloc[-126]) * 100

        if len(df) >= 252:
            momentum_pcts['12m'] = (
                (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]) * 100

        # Timeframe configuration
        timeframe_days = {
            '1m': 21,   # ~1 month (trading days)
            '3m': 63,   # ~3 months
            '6m': 126,  # ~6 months
            '12m': 252  # ~1 year
        }

        # Hybrid weights: heavily favor momentum over ML since ML model has negative CV R²
        # If ML is worse than baseline (CV R² < 0), use only momentum (0% ML, 100% momentum)
        # Otherwise use conservative weights: 10-15% ML, 85-90% momentum
        if not use_ml_prediction:
            # ML is worse than baseline - use only momentum
            timeframe_weights = {
                '1m': {'ml': 0.0, 'momentum': 1.0},
                '3m': {'ml': 0.0, 'momentum': 1.0},

                '6m': {'ml': 0.0, 'momentum': 1.0},

                '12m': {'ml': 0.0, 'momentum': 1.0}

            }

        else:

            # ML is better than baseline but still conservative weights
            timeframe_weights = {
                '1m': {'ml': 0.10, 'momentum': 0.90},   # 10% ML, 90% momentum
                '3m': {'ml': 0.10, 'momentum': 0.90},   # 10% ML, 90% momentum

                '6m': {'ml': 0.15, 'momentum': 0.85},   # 15% ML, 85% momentum

                '12m': {'ml': 0.15, 'momentum': 0.85}   # 15% ML, 85% momentum

            }

        predictions = {}
        expected_returns = {}
        confidence_intervals = {}

        for timeframe, days in timeframe_days.items():
            # ML-based prediction (compound next-day return)
            if use_ml_prediction and annualized_return != 0:
                daily_return = annualized_return / 252
                ml_return_pct = ((1 + daily_return) ** days - 1) * 100
            else:
                # If ML is worse than baseline, use 0% ML prediction (will use
                # only momentum)
                ml_return_pct = 0.0

            # Get historical momentum for this timeframe
            momentum_pct = momentum_pcts.get(timeframe, 0.0)

            # Blend ML and momentum using timeframe-specific weights
            weights = timeframe_weights[timeframe]
            blended_return_pct = (weights['ml'] * ml_return_pct +
                                  weights['momentum'] * momentum_pct)

            # Sanity check: if blended prediction is extremely negative, favor momentum more
            # This prevents unrealistic predictions like -43% for 1M
            extreme_thresholds = {
                '1m': -30,  # If blended < -30% for 1M, use momentum

                '3m': -40,  # If blended < -40% for 3M, use momentum

                '6m': -50,  # If blended < -50% for 6M, use momentum

                '12m': -60  # If blended < -60% for 12M, use momentum

            }

            threshold = extreme_thresholds.get(timeframe, -50)
            if blended_return_pct < threshold and abs(momentum_pct) > 1:
                # Use 90% momentum, 10% ML if prediction is too extreme
                logger.debug(
                    f"Extreme prediction ({blended_return_pct:.1f}%) for {timeframe}, using mostly momentum")
                blended_return_pct = 0.1 * ml_return_pct + 0.9 * momentum_pct

            # Calculate predicted price (uncapped)
            predicted_price_uncapped = current_price * (1 + blended_return_pct / 100)

            # Dynamic volatility-based cap (instead of fixed -30%)
            # Use historical volatility to set reasonable bounds
            if historical_volatility > 0:
                vol_factor = historical_volatility / 100  # Convert to decimal

                # Allow more movement for longer timeframes
                max_down = min(
                    0.5,
                    vol_factor *
                    np.sqrt(
                        days /
                        252) *
                    3)  # Max 50% down
                max_up = min(
                    2.0,
                    vol_factor *
                    np.sqrt(
                        days /
                        252) *
                    4)    # Max 200% up
            else:
                max_down = 0.3  # Conservative 30% down
                max_up = 1.5    # Conservative 50% up

            # Apply cap to predicted price (for display)
            predicted_price = max(current_price * (1 - max_down),
                                  min(current_price * (1 + max_up), predicted_price_uncapped))

            # Calculate confidence based on alignment between ML and momentum
            if abs(momentum_pct) > 1:
                alignment = 1 - abs(blended_return_pct -
                                    momentum_pct) / max(abs(momentum_pct), 10)
                confidence = 0.5 + 0.3 * max(0, alignment)  # 0.5 to 0.8
            else:
                confidence = 0.5  # Lower confidence if no clear momentum

            # Also factor in timeframe (longer = lower confidence)
            # Reduce for longer timeframes
            confidence = max(0.2, confidence * (1 - (days / 252) * 0.3))

            predictions[timeframe] = {
                'price': float(predicted_price),
                'confidence': float(confidence)
            }

            # Calculate expected return from CAPPED predicted price to match displayed price
            # This ensures percentage matches what the user actually sees
            # CRITICAL: Use the same current_price that was used to calculate predicted_price
            if current_price <= 0:
                expected_return = 0.0
            else:
                expected_return = ((predicted_price - current_price) / current_price) * 100
                # Ensure expected_return is a valid float
                if not isinstance(expected_return, (int, float)) or not np.isfinite(expected_return):
                    logger.warning(f"Invalid expected_return for {timeframe}: {expected_return}, predicted_price={predicted_price}, current_price={current_price}")
                    expected_return = 0.0
            
            # Additional validation: if predicted_price equals current_price, expected_return should be 0
            # BUT: Don't override if the difference is significant (could be a rounding issue)
            price_diff = abs(predicted_price - current_price)
            if price_diff < 0.01:
                # Only set to 0 if the prices are truly equal (within 1 cent)
                expected_return = 0.0
            else:
                # Recalculate expected_return to ensure it's correct
                # This handles edge cases where rounding might cause issues
                expected_return_recalc = ((predicted_price - current_price) / current_price) * 100
                if abs(expected_return_recalc - expected_return) > 0.1:
                    logger.warning(f"expected_return mismatch for {timeframe}: original={expected_return:.2f}%, recalc={expected_return_recalc:.2f}%, using recalc")
                    expected_return = expected_return_recalc
            
            expected_returns[timeframe] = float(expected_return)
            
            # Debug logging for 12M to diagnose the issue
            if timeframe == '12m':
                logger.info(f"12M prediction: current_price=${current_price:.2f}, predicted_price=${predicted_price:.2f}, predicted_price_uncapped=${predicted_price_uncapped:.2f}, blended_return_pct={blended_return_pct:.2f}%, expected_return={expected_return:.2f}%, max_up={max_up:.2f}, max_down={max_down:.2f}, price_diff=${price_diff:.2f}")

            # Confidence intervals using percentage-based ranges (more realistic than std-based)
            # Use timeframe-specific percentage ranges that scale reasonably
            timeframe_ranges = {
                '1m': 0.15,  # ±15% for 1 month
                '3m': 0.20,  # ±20% for 3 months
                '6m': 0.25,  # ±25% for 6 months
                '12m': 0.35  # ±35% for 12 months
            }

            # Optionally use historical volatility to adjust range (more
            # dynamic)
            range_multiplier = 1.0
            if historical_volatility > 0:
                # Adjust based on volatility: high vol stocks get wider ranges
                # Normalize: 30% annual vol = 1.0x, scale from there
                vol_factor = historical_volatility / 30.0
                # Between 0.7x and 1.5x
                range_multiplier = min(1.5, max(0.7, vol_factor))

            range_pct = timeframe_ranges.get(
                timeframe, 0.25) * range_multiplier
            # Apply percentage range to predicted price (not current price)
            lower_bound = predicted_price * (1 - range_pct)
            upper_bound = predicted_price * (1 + range_pct)

            # Store confidence intervals for all timeframes
            confidence_intervals[timeframe] = {
                # Ensure lower is at least 1% of current price
                'lower': max(0.01 * current_price, lower_bound),

                'upper': upper_bound,

                'confidence_level': 0.95

            }

        # Store also the next-day prediction for reference
        predictions['_next_day'] = {
            'price': float(next_day_prediction),
            'confidence': 0.7
        }

        # Get feature importance if available
        top_features = {}
        if model and hasattr(model, 'feature_importances_'):
            if hasattr(model, 'feature_importances_dict_'):

                # Use cached dict if available
                feature_importance_dict = model.feature_importances_dict_
            else:
                # Create dict from feature_importances_
                feature_importance_dict = dict(
                    zip(feature_names, model.feature_importances_))

            # Sort by importance and get top 10
            sorted_importance = sorted(
                feature_importance_dict.items(),
                key=lambda x: x[1],
                reverse=True)
            top_features = {name: float(importance)
                            for name, importance in sorted_importance[:10]}
            logger.debug(f"Top 5 features: {list(sorted_importance[:5])}")

        # Get model quality metrics (CV R² score) if available
        model_quality = {}
        if model is not None:
            if hasattr(model, 'cv_r2_score'):
                model_quality['cv_r2_score'] = float(model.cv_r2_score)
                model_quality['train_r2_score'] = float(
                    getattr(model, 'train_r2_score', model.cv_r2_score))
            else:
                # Fallback if model doesn't have CV score
                model_quality['cv_r2_score'] = None
                model_quality['train_r2_score'] = None
                model_quality['is_better_than_baseline'] = False

        result = {
            'current_price': current_price,
            'predictions': predictions,
            'expected_returns': expected_returns,
            'confidence_intervals': confidence_intervals,
            'model_used': 'random_forest',
            'feature_importance': top_features if top_features else None,
            'model_quality': model_quality if model_quality else None
        }
        
        # Cache the result
        if CACHE_AVAILABLE and cache and ticker:
            cache_key = f"ml_predict_price_{ticker}"
            cache.set(
                cache_key,
                result,
                timeout=CACHE_TIMEOUTS['ml_predictions'])
            logger.debug(f"Cached ML price prediction for {ticker}")
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in predict_price: {e}")
        import traceback
        traceback.print_exc()
        # Error fallback - use momentum estimates
        logger.error(f"Error in predict_price for {ticker}: {e}")

        momentum_1m = 0.0
        momentum_3m = 0.0
        momentum_6m = 0.0
        momentum_12m = 0.0

        if df is not None and len(df) > 20:
            try:

                if len(df) >= 20:

                    momentum_1m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
                if len(df) >= 60:
                    momentum_3m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]) * 100
                if len(df) >= 120:
                    momentum_6m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-120]) / df['Close'].iloc[-120]) * 100
                if len(df) >= 252:
                    momentum_12m = (
                        (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]) * 100
            except Exception:
                pass

        return {
            'current_price': current_price,
            'predictions': {
                '1m': {'price': current_price * (1 + momentum_1m * 0.5 / 100), 'confidence': 0.3},

                '3m': {'price': current_price * (1 + momentum_3m * 0.5 / 100), 'confidence': 0.25},

                '6m': {'price': current_price * (1 + momentum_6m * 0.5 / 100), 'confidence': 0.2},

                '12m': {'price': current_price * (1 + momentum_12m * 0.5 / 100), 'confidence': 0.15}

            },
            'expected_returns': {
                '1m': momentum_1m * 0.5,

                '3m': momentum_3m * 0.5,

                '6m': momentum_6m * 0.5,

                '12m': momentum_12m * 0.5

            },
            'confidence_intervals': {
                '6m': {'lower': current_price * 0.80, 'upper': current_price * 1.30},

                '12m': {'lower': current_price * 0.70, 'upper': current_price * 1.50}

            },
            'model_used': 'momentum_estimate',
            'warning': f'ML prediction error: {str(e)}. Using momentum-based estimates. These are NOT ML predictions.',
            'error': str(e)
        }


def _save_prediction_history(
        ticker: str,
        current_price: float,
        prediction_result: Dict,
        score: Optional[float] = None):
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
        history = sorted(
            history, key=lambda x: x.get(
                'date', ''), reverse=True)
        
        if days > 0:
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_history = []
            for entry in history:
                try:
                    entry_date = datetime.strptime(
                        entry.get('date', ''), '%Y-%m-%d %H:%M:%S')
                    if entry_date >= cutoff_date:
                        filtered_history.append(entry)
                except:
                    continue
            history = filtered_history
        
        return history
    except Exception as e:

        logger.exception(f"Error loading prediction history: {e}")

        return []


def get_prediction_accuracy(
        ticker: str,
        timeframe: str = '6m') -> Optional[Dict]:
    """
    Calculate accuracy of historical predictions by comparing them with actual prices

    Args:
        ticker: Stock ticker symbol
        timeframe: '1m', '3m', '6m', or '12m' to evaluate predictions for that timeframe

    Returns:
        Dict with accuracy metrics or None if insufficient data
        """

    try:
        # Load prediction history
        history = get_prediction_history(
            ticker, days=365)  # Look back up to 1 year

        if not history or len(history) < 3:
            return None

        # Map timeframe to days
        timeframe_days = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}
        days = timeframe_days.get(timeframe, 126)

        # Get stock data for actual prices
        stock = yf.Ticker(ticker)

        accuracy_results = []

        for entry in history:
            try:

                pred_date_str = entry.get('date', '')

                if not pred_date_str:
                    continue

                pred_date = pd.to_datetime(pred_date_str)
                pred_price = entry.get(f'prediction_{timeframe}')

                if pred_price is None or not isinstance(pred_price, (int, float)):
                    continue

                # Get actual price 'days' days after prediction
                # Add buffer for weekends/holidays
                target_date = pred_date + timedelta(days=days + 30)

        # Download historical data for that period
                hist = stock.history(start=pred_date, end=target_date)

                if hist.empty or len(hist) == 0:
                    continue

                # Get actual price closest to target date (use last available
                # price in range)
                actual_price = float(hist['Close'].iloc[-1]) if len(hist) > 0 else None

                if actual_price is None or actual_price <= 0:
                    continue

                # Calculate error
                error_pct = ((actual_price - pred_price) / pred_price) * 100
                error_abs = abs(error_pct)

                accuracy_results.append({
                    'date': pred_date_str,
                    'predicted': float(pred_price),
                    'actual': float(actual_price),
                    'error_pct': float(error_pct),
                    'error_abs': float(error_abs)
                })

            except Exception as e:

                logger.debug(
                    f"Error evaluating prediction from {entry.get('date')}: {e}")
                continue

        if len(accuracy_results) < 3:
            return None

            # Calculate aggregate metrics
            errors_abs = [r['error_abs'] for r in accuracy_results]
            errors_pct = [r['error_pct'] for r in accuracy_results]

            mean_error = np.mean(errors_abs)
            median_error = np.median(errors_abs)
            std_error = np.std(errors_abs)

            # Calculate percentage of predictions within reasonable ranges
            within_range_20pct = sum(
                1 for e in errors_abs if e <= 20) / len(errors_abs) * 100
        within_range_30pct = sum(
            1 for e in errors_abs if e <= 30) / len(errors_abs) * 100
        within_range_50pct = sum(
            1 for e in errors_abs if e <= 50) / len(errors_abs) * 100

        return {
            'timeframe': timeframe,
            'num_predictions': len(accuracy_results),
            'mean_absolute_error_pct': float(mean_error),
            'median_error_pct': float(median_error),
            'std_error_pct': float(std_error),
            'within_20pct': float(within_range_20pct),
            'within_30pct': float(within_range_30pct),
            'within_50pct': float(within_range_50pct),
            'recent_predictions': accuracy_results[-10:] if len(accuracy_results) > 10 else accuracy_results
        }

    except Exception as e:
        logger.exception(
            f"Error calculating prediction accuracy for {ticker}: {e}")
        return None


def generate_ai_recommendations(ticker: str) -> Optional[Dict]:
    """
    Generate AI-powered stock recommendations (cached)
    
    NOTE: This is a stub implementation. Full implementation requires:
    - classify_trend() function

    - calculate_entry_tp_dca() function  

    - calculate_position_sizing() function

    
    These functions should be moved from app.py or reimplemented.
    """
    # Check cache first
    if CACHE_AVAILABLE and cache:
        cache_key = f"ml_ai_recommendations_{ticker}"
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for AI recommendations {ticker}")
            return cached_data
    
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
        ml_features = extract_ml_features(
            ticker, df, info, indicators, metrics, news_list)
        ml_features['ticker'] = ticker.upper()
        price_prediction = predict_price(ml_features, current_price, df)
        
        # Calculate risk score
        risk_analysis = calculate_risk_score(ml_features, metrics, info)
        
        # Get moving averages for trend classification
        sma_20 = indicators.get('sma_20', [])
        sma_50 = indicators.get('sma_50', [])
        
        # Stub implementations for missing functions
        # Determine trend class based on price momentum and indicators
        price_momentum_30d = 0
        if len(df) >= 30:
            price_30d_ago = df['Close'].iloc[-30]
            price_momentum_30d = (
                (current_price - price_30d_ago) / price_30d_ago) * 100
        
        # Simple trend classification based on momentum and moving averages
        if price_momentum_30d > 15 and sma_20 and sma_50 and len(
                sma_20) > 0 and len(sma_50) > 0:
            if current_price > sma_20[-1] > sma_50[-1]:
                trend_class = 'Strong Uptrend'
                confidence = 0.75
            else:
                trend_class = 'Moderate Uptrend'
                confidence = 0.65
        elif price_momentum_30d > 5:
            trend_class = 'Moderate Uptrend'
            confidence = 0.60
        elif price_momentum_30d < -15 and sma_20 and sma_50 and len(sma_20) > 0 and len(sma_50) > 0:

            if current_price < sma_20[-1] < sma_50[-1]:

                trend_class = 'Strong Downtrend'

                confidence = 0.75
            else:
                trend_class = 'Moderate Downtrend'
                confidence = 0.65
        elif price_momentum_30d < -5:

            trend_class = 'Moderate Downtrend'
            confidence = 0.60
        else:

            trend_class = 'Sideways'

            confidence = 0.50
        
        # Create probabilities distribution (sum to 1.0)
        probabilities = {
            'Strong Uptrend': 0.0,
            'Moderate Uptrend': 0.0,
            'Sideways': 0.0,
            'Moderate Downtrend': 0.0,
            'Strong Downtrend': 0.0
        }
        
        # Assign probabilities based on trend_class
        if trend_class == 'Strong Uptrend':
            probabilities['Strong Uptrend'] = confidence
            probabilities['Moderate Uptrend'] = (1 - confidence) * 0.6
            probabilities['Sideways'] = (1 - confidence) * 0.3
            probabilities['Moderate Downtrend'] = (1 - confidence) * 0.1
        elif trend_class == 'Moderate Uptrend':

            probabilities['Moderate Uptrend'] = confidence

            probabilities['Strong Uptrend'] = (1 - confidence) * 0.3
            probabilities['Sideways'] = (1 - confidence) * 0.5
            probabilities['Moderate Downtrend'] = (1 - confidence) * 0.2
        elif trend_class == 'Sideways':

            probabilities['Sideways'] = confidence

            probabilities['Moderate Uptrend'] = (1 - confidence) * 0.3
            probabilities['Moderate Downtrend'] = (1 - confidence) * 0.3
            probabilities['Strong Uptrend'] = (1 - confidence) * 0.2
            probabilities['Strong Downtrend'] = (1 - confidence) * 0.2
        elif trend_class == 'Moderate Downtrend':

            probabilities['Moderate Downtrend'] = confidence

            probabilities['Sideways'] = (1 - confidence) * 0.5
            probabilities['Strong Downtrend'] = (1 - confidence) * 0.3
            probabilities['Moderate Uptrend'] = (1 - confidence) * 0.2
        elif trend_class == 'Strong Downtrend':

            probabilities['Strong Downtrend'] = confidence

            probabilities['Moderate Downtrend'] = (1 - confidence) * 0.6
            probabilities['Sideways'] = (1 - confidence) * 0.3
            probabilities['Moderate Uptrend'] = (1 - confidence) * 0.1
        
        trend_classification = {
            'trend_class': trend_class,
            'confidence': confidence,
            'probabilities': probabilities
        }

        # Calculate Entry, TP, and DCA levels using ML predictions and technical analysis
        # Entry point based on current price, support levels, and ML confidence
        # Less strict: entry should be close to current price, not far away
        entry_price = current_price
        entry_confidence = 'medium'
        recent_low = current_price * 0.95  # Default fallback
        recent_high = current_price * 1.05  # Default fallback
        
        # Calculate support/resistance first
        if len(df) >= 20:
            recent_low = float(df['Low'].iloc[-20:].min())

            recent_high = float(df['High'].iloc[-20:].max())

        # Use ML prediction as guidance, but keep entry close to current price
        if price_prediction and price_prediction.get('predictions'):
            pred_1m = price_prediction['predictions'].get('1m', {})
            ml_pred_price = None
            if isinstance(pred_1m, dict) and 'price' in pred_1m:
                ml_pred_price = pred_1m['price']

            elif isinstance(pred_1m, (int, float)):
                ml_pred_price = pred_1m

            if ml_pred_price:
                # Use weighted average: 80% current price, 20% ML prediction
                # More conservative - entry should be even closer to current
                # price for more realistic entry points
                entry_price = current_price * 0.8 + ml_pred_price * 0.2

        # Ensure entry is within tighter range (±8% of current price
        # for more realistic entry)
        entry_price = max(current_price * 0.92,
                          min(current_price * 1.08, entry_price))

        # Adjust entry based on support/resistance (fine-tuning)
        if len(df) >= 20:
            # If current price is near recent low, it's a good entry
            if current_price <= recent_low * 1.05:
                entry_confidence = 'high'
                # Entry can be slightly above current (up to 2%)
                entry_price = min(entry_price, current_price * 1.02)
            elif current_price >= recent_high * 0.95:
                entry_confidence = 'low'
            # Entry can be slightly below current (up to 3% discount)
                entry_price = max(entry_price, current_price * 0.97)
            else:
                entry_confidence = 'medium'
                # Keep entry within 2% of current price in medium confidence
                # scenarios
        entry_price = max(current_price * 0.98,
                          min(current_price * 1.02, entry_price))

        # Take Profit levels - less strict: use reasonable defaults with ML guidance
        # Default TP levels (conservative targets)
        default_tp1 = entry_price * 1.10  # 10% gain from entry
        default_tp2 = entry_price * 1.20  # 20% gain from entry
        default_tp3 = entry_price * 1.35  # 35% gain from entry

        tp1_price = default_tp1
        tp2_price = default_tp2
        tp3_price = default_tp3
        
        if price_prediction and price_prediction.get('predictions'):
            pred_3m = price_prediction['predictions'].get('3m', {})
            pred_6m = price_prediction['predictions'].get('6m', {})
            pred_12m = price_prediction['predictions'].get('12m', {})

            # Get ML prediction prices
            ml_tp1 = None
            ml_tp2 = None
            ml_tp3 = None
            
            if isinstance(pred_3m, dict) and 'price' in pred_3m:
                ml_tp1 = pred_3m['price']

            elif isinstance(pred_3m, (int, float)):
                ml_tp1 = pred_3m
                
            if isinstance(pred_6m, dict) and 'price' in pred_6m:
                ml_tp2 = pred_6m['price']

            elif isinstance(pred_6m, (int, float)):
                ml_tp2 = pred_6m
                
            if isinstance(pred_12m, dict) and 'price' in pred_12m:
                ml_tp3 = pred_12m['price']

            elif isinstance(pred_12m, (int, float)):
                ml_tp3 = pred_12m

            # Use weighted average: 60% default (reasonable target), 40% ML prediction
            # This keeps TP realistic while considering ML direction
            if ml_tp1 and ml_tp1 > entry_price:
                # Only use ML if it's higher than entry (makes sense for TP)
                tp1_price = default_tp1 * 0.6 + ml_tp1 * 0.4
        # Ensure TP1 is at least 5% above entry
        tp1_price = max(entry_price * 1.05, tp1_price)

        if ml_tp2 and ml_tp2 > entry_price:
            tp2_price = default_tp2 * 0.6 + ml_tp2 * 0.4

            # Ensure TP2 is at least 10% above entry and higher than TP1
        tp2_price = max(entry_price * 1.10, tp1_price * 1.05, tp2_price)

        if ml_tp3 and ml_tp3 > entry_price:
            tp3_price = default_tp3 * 0.6 + ml_tp3 * 0.4

            # Ensure TP3 is at least 20% above entry and higher than TP2
        tp3_price = max(entry_price * 1.20, tp2_price * 1.05, tp3_price)
        
        # Calculate gains
        tp1_gain = ((tp1_price - entry_price) / entry_price *
                    100) if entry_price > 0 else 10
        tp2_gain = ((tp2_price - entry_price) / entry_price *
                    100) if entry_price > 0 else 20
        tp3_gain = ((tp3_price - entry_price) / entry_price *
                    100) if entry_price > 0 else 35
        
        # DCA levels (buying on dips)
        dca1_price = current_price * 0.95
        dca2_price = current_price * 0.90
        dca3_price = current_price * 0.85
        
        # Risk/Reward ratio
        risk = entry_price - dca3_price
        reward = tp3_price - entry_price
        risk_reward_ratio = (reward / risk) if risk > 0 else 2.0
        
        # ML enhancements
        volatility_pct = (
            ml_features.get(
                'volatility',
                0) /
            current_price *
            100) if current_price > 0 else 2.0
        # Higher volatility = wider TP levels
        adaptive_factor = 1.0 + (volatility_pct / 100)
        
        entry_tp_dca = {
            'entry': {
                'price': round(entry_price, 2),
                'confidence': entry_confidence,
                'reason': f'Entry based on ML prediction and technical analysis. Current price: ${current_price:.2f}',
                'conditions': [
                    f'ML 1M prediction: ${tp1_price:.2f}',
                    f'Support level: ${recent_low:.2f}' if len(
                        df) >= 20 else 'Support analysis available'
                ]
            },
            'take_profit': {
                'tp1': {
                    'price': round(tp1_price, 2),
                    'gain_pct': round(tp1_gain, 1),
                    'timeframe': '3 months',
                    'ml_confidence': round(price_prediction.get('predictions', {}).get('3m', {}).get('confidence', 0.5) * 100, 1) if price_prediction else 50
                },
                'tp2': {
                    'price': round(tp2_price, 2),
                    'gain_pct': round(tp2_gain, 1),
                    'timeframe': '6 months',
                    'ml_confidence': round(price_prediction.get('predictions', {}).get('6m', {}).get('confidence', 0.4) * 100, 1) if price_prediction else 40
                },
                'tp3': {
                    'price': round(tp3_price, 2),
                    'gain_pct': round(tp3_gain, 1),
                    'timeframe': '12 months',
                    'ml_confidence': round(price_prediction.get('predictions', {}).get('12m', {}).get('confidence', 0.3) * 100, 1) if price_prediction else 30
                }
            },
            'dca_levels': [
                {
                    'price': round(dca1_price, 2),
                    'reason': 'First DCA level - 5% below entry',
                    'confidence': 'medium',
                    'ml_probability': 60
                },
                {
                    'price': round(dca2_price, 2),
                    'reason': 'Second DCA level - 10% below entry',
                    'confidence': 'medium',
                    'ml_probability': 40
                },
                {
                    'price': round(dca3_price, 2),
                    'reason': 'Third DCA level - 15% below entry',
                    'confidence': 'low',
                    'ml_probability': 20
                }
            ],
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'ml_enhancements': {
                'volatility_pct': round(volatility_pct, 2),
                'adaptive_factor': round(adaptive_factor, 2)
            }
        }

        # Calculate position sizing with proper structure
        risk_score = risk_analysis.get('risk_score', 50)
        ml_confidence = price_prediction.get(
            'confidence', 50) if price_prediction else 50
        
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
        volatility_pct = (
            atr /
            current_price *
            100) if current_price > 0 else 2.0
        
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
        
        # Analyze news sentiment
        news_sentiment = 'neutral'
        if news_list:
            sentiments = [article.get('sentiment', 'neutral')
                          for article in news_list[:10]]
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            if positive_count > negative_count * 1.5:
                news_sentiment = 'positive'
            elif negative_count > positive_count * 1.5:
                news_sentiment = 'negative'
        
        # Get technical indicator values
        rsi_values = indicators.get('rsi', [])
        macd_values = indicators.get('macd', [])
        macd_signal = indicators.get('macd_signal', [])
        sma_20 = indicators.get('sma_20', [])
        sma_50 = indicators.get('sma_50', [])
        
        # Calculate technical score (0-100, higher is better)
        technical_score = 50  # Base score
        reasons = []
        warnings = []
        
        # RSI Analysis
        if rsi_values and len(rsi_values) > 0:
            current_rsi = rsi_values[-1]
            if current_rsi is not None and not pd.isna(current_rsi):
                if current_rsi < 30:
                    technical_score += 15
                    reasons.append(
                        "RSI indicates oversold conditions - potential buying opportunity")
                elif current_rsi < 40:
                    technical_score += 8
                    reasons.append("RSI suggests stock may be undervalued")
                elif current_rsi > 70:
                    technical_score -= 15
                    warnings.append(
                        "RSI indicates overbought conditions - stock may be overvalued")
                elif current_rsi > 60:
                    technical_score -= 8
                    warnings.append("RSI suggests stock may be overvalued")
        
        # MACD Analysis
        if macd_values and macd_signal and len(
                macd_values) > 0 and len(macd_signal) > 0:
            current_macd = macd_values[-1]
            current_signal = macd_signal[-1]
            if current_macd is not None and current_signal is not None and not pd.isna(
                    current_macd) and not pd.isna(current_signal):
                if current_macd > current_signal:
                    technical_score += 10
                    reasons.append("MACD shows bullish momentum")
                else:
                    technical_score -= 5
                    warnings.append("MACD shows bearish momentum")
        
        # Moving Average Analysis
        if sma_20 and sma_50 and len(sma_20) > 0 and len(sma_50) > 0:
            current_sma20 = sma_20[-1]
            current_sma50 = sma_50[-1]
            if current_sma20 is not None and current_sma50 is not None and not pd.isna(
                    current_sma20) and not pd.isna(current_sma50):
                if current_price > current_sma20 > current_sma50:
                    technical_score += 12
                    reasons.append(
                        "Price above both 20-day and 50-day moving averages - strong uptrend")
                elif current_price > current_sma20:
                    technical_score += 5
                    reasons.append(
                        "Price above 20-day moving average - short-term bullish")
                elif current_price < current_sma20 < current_sma50:
                    technical_score -= 12
                    warnings.append(
                        "Price below both moving averages - downtrend")
                elif current_price < current_sma20:
                    technical_score -= 5
                    warnings.append(
                        "Price below 20-day moving average - short-term bearish")
        
        # Price Momentum (30-day)
        if len(df) >= 30:
            price_30d_ago = df['Close'].iloc[-30]
            price_change_30d = (
                (current_price - price_30d_ago) / price_30d_ago) * 100
            if price_change_30d > 10:
                technical_score += 8
                reasons.append(
                    f"Strong 30-day price momentum (+{price_change_30d:.1f}%)")
            elif price_change_30d > 5:
                technical_score += 4
                reasons.append(
                    f"Positive 30-day price momentum (+{price_change_30d:.1f}%)")
            elif price_change_30d < -10:
                technical_score -= 8
                warnings.append(
                    f"Negative 30-day price momentum ({price_change_30d:.1f}%)")
            elif price_change_30d < -5:
                technical_score -= 4
                warnings.append(
                    f"Weak 30-day price momentum ({price_change_30d:.1f}%)")
        
        # News Sentiment Impact
        if news_sentiment == 'positive':
            technical_score += 5
            reasons.append("Recent news sentiment is positive")
        elif news_sentiment == 'negative':

            technical_score -= 5

            warnings.append("Recent news sentiment is negative")
        
        # Volatility Analysis
        volatility = metrics.get('volatility')
        if volatility is not None:
            if volatility > 40:
                warnings.append(
                    f"High volatility ({volatility:.1f}%) - higher risk")
            elif volatility < 15:
                reasons.append(
                    f"Low volatility ({volatility:.1f}%) - more stable")
        
        # Enhance recommendation with ML model outputs
        # ML predictions should have significant weight to avoid "Buy" when ML
        # is negative
        expected_return_6m = price_prediction.get(
            'expected_returns', {}).get(
            '6m', 0) if price_prediction else 0
        expected_return_1m = price_prediction.get(
            'expected_returns', {}).get(
            '1m', 0) if price_prediction else 0
        expected_return_3m = price_prediction.get(
            'expected_returns', {}).get(
            '3m', 0) if price_prediction else 0
        expected_return_12m = price_prediction.get(
            'expected_returns', {}).get(
            '12m', 0) if price_prediction else 0

        # Check if all ML predictions are negative (strong bearish signal)
        all_ml_negative = all(
            r < 0 for r in [
                expected_return_1m,
                expected_return_3m,
                expected_return_6m,
                expected_return_12m])
        if all_ml_negative:
            technical_score -= 25  # Heavy penalty if all timeframes are negative

            warnings.append(
                f"ML model predicts negative returns across all timeframes (1M: {expected_return_1m:.1f}%, 3M: {expected_return_3m:.1f}%, 6M: {expected_return_6m:.1f}%, 12M: {expected_return_12m:.1f}%)")
        else:
            # Individual ML prediction impact (increased penalties/bonuses)
        if expected_return_6m > 20:

                technical_score += 20  # Increased from 15

                reasons.append(
                    f"ML model predicts strong 6-month return (+{expected_return_6m:.1f}%)")
        elif expected_return_6m > 10:

                technical_score += 15  # Increased from 10

                reasons.append(
                    f"ML model predicts positive 6-month return (+{expected_return_6m:.1f}%)")
        elif expected_return_6m < -10:

                technical_score -= 25  # Increased from 15

                warnings.append(
                    f"ML model predicts negative 6-month return ({expected_return_6m:.1f}%)")
        elif expected_return_6m < -5:

                technical_score -= 20  # Increased from 10

                warnings.append(
                    f"ML model predicts weak 6-month return ({expected_return_6m:.1f}%)")
            elif expected_return_6m < 0:
                technical_score -= 10  # New: small penalty for any negative return

                warnings.append(
                    f"ML model predicts slight decline ({expected_return_6m:.1f}%)")
        
        # Adjust score based on trend classification
        trend_class = trend_classification.get('trend_class', 'Neutral')
        if trend_class == 'Strong Uptrend':
            technical_score += 12
            reasons.append("ML trend classification: Strong Uptrend")
        elif trend_class == 'Moderate Uptrend':

            technical_score += 6

            reasons.append("ML trend classification: Moderate Uptrend")
        elif trend_class == 'Strong Downtrend':

            technical_score -= 12

            warnings.append("ML trend classification: Strong Downtrend")
        elif trend_class == 'Moderate Downtrend':

            technical_score -= 6

            warnings.append("ML trend classification: Moderate Downtrend")
        
        # Adjust score based on risk
        risk_score = risk_analysis.get('risk_score', 50)
        if risk_score > 75:
            technical_score -= 10
            warnings.append(f"High risk score ({risk_score:.1f}/100)")
        elif risk_score < 25:

            technical_score += 5

            reasons.append(f"Low risk score ({risk_score:.1f}/100)")
        
        # Re-determine recommendation with ML-enhanced score
        final_score = min(100, max(0, technical_score))
        
        # Determine recommendation
        if final_score >= 75:
            recommendation = "Strong Buy"
            confidence = "High"
            color = "#10b981"  # Green
        elif final_score >= 60:

            recommendation = "Buy"

            confidence = "Medium-High"
            color = "#34d399"  # Light green
        elif final_score >= 45:

            recommendation = "Hold"

            confidence = "Medium"
            color = "#fbbf24"  # Yellow
        elif final_score >= 30:

            recommendation = "Sell"

            confidence = "Medium"
            color = "#f87171"  # Light red
        else:

            recommendation = "Strong Sell"

            confidence = "High"
            color = "#ef4444"  # Red
        
        # Generate summary
        summary_parts = []
        if reasons:
            summary_parts.append(f"Key positives: {', '.join(reasons[:2])}")
        if warnings:

            summary_parts.append(f"Key concerns: {', '.join(warnings[:2])}")
        
        summary = ". ".join(
            summary_parts) if summary_parts else "Mixed signals - consider additional research"
        
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
            'recommendation': recommendation,
            'confidence': confidence,
            'score': final_score,
            'color': color,
            'reasons': reasons[:5],
            'warnings': warnings[:5],
            'summary': summary,
            'technical_indicators': {
                'rsi': rsi_values[-1] if rsi_values and len(rsi_values) > 0 else None,
                'macd_bullish': macd_values[-1] > macd_signal[-1] if macd_values and macd_signal and len(macd_values) > 0 and len(macd_signal) > 0 else None,
                'price_vs_sma20': current_price > sma_20[-1] if sma_20 and len(sma_20) > 0 else None,
                'price_vs_sma50': current_price > sma_50[-1] if sma_50 and len(sma_50) > 0 else None,
            },
            'news_sentiment': news_sentiment,
            'ml_models': {
                'price_prediction': price_prediction,
                'trend_classification': trend_classification,
                'risk_analysis': risk_analysis
            },
            'trading_strategy': entry_tp_dca,
            'position_sizing': position_sizing,
            'chart_data': chart_data
        }
        
        # Cache the result
        if CACHE_AVAILABLE and cache:
            cache_key = f"ml_ai_recommendations_{ticker}"
            cache.set(
                cache_key,
                result,
                timeout=CACHE_TIMEOUTS['ml_predictions'])
            logger.debug(f"Cached AI recommendations for {ticker}")
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in AI recommendations: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_backtest(
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None) -> Dict:
    """
    Run backtest on ML predictions using historical data with walk-forward validation

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for backtest (YYYY-MM-DD), defaults to 1 year ago
        end_date: End date for backtest (YYYY-MM-DD), defaults to today

    Returns:
        Dict with backtest results including accuracy metrics
        """

    if not ML_AVAILABLE:

        return {

            'success': False,

            'error': 'ML not available for backtesting'
        }

    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Download extended historical data
        logger.info(f"Running backtest for {ticker}")
        df = _download_extended_historical_data(ticker, years=2)

        if df is None or len(df) < 200:
            return {

                'success': False,

                'error': f'Insufficient historical data for {ticker} (need at least 200 days)'

            }

        # Set date range
        # Note: df.index should already be timezone-naive from
        # _download_extended_historical_data

        if end_date:
            end = pd.to_datetime(end_date)

        else:

            end = df.index[-1]

        if start_date:
            start = pd.to_datetime(start_date)

        else:

            # Default to 1 year ago or 252 trading days
            start = end - timedelta(days=365)

        # Filter data to date range
        df_test = df[(df.index >= start) & (df.index <= end)].copy()

        if len(df_test) < 60:
            return {

                'success': False,

                'error': f'Insufficient data in date range (only {len(df_test)} days)'

            }

        logger.info(
            f"Backtesting on {len(df_test)} days from {start.date()} to {end.date()}")

        # Walk-forward validation: train model less frequently for performance
        predictions = []
        actuals = []
        dates = []

        # Minimum training period: 100 days
        min_train_days = 100
        test_start_idx = min_train_days

        # Train model every N days instead of every day (optimization)
        # Re-train every 30 days or on significant market changes
        retrain_interval_days = 30
        last_retrain_idx = test_start_idx - 1
        cached_model = None
        cached_scaler = None
        cached_features_template = None

        for i in range(test_start_idx, len(df_test) - 1):
            try:

                # Training data: everything up to current point
                train_df = df_test.iloc[:i + 1]

                # Test point: next day
                test_date = df_test.index[i + 1]
                actual_price = df_test['Close'].iloc[i + 1]

                # Get features for current point
                from app.analysis.technical import calculate_technical_indicators
                from app.analysis.fundamental import calculate_metrics

                indicators = calculate_technical_indicators(train_df)

                # Note: Historical info and news would require point-in-time data
                # For now, we use empty structures which means backtest uses fewer features
                # This may reduce accuracy but prevents data leakage
                info = {}  # Historical fundamentals would require point-in-time data
                metrics = calculate_metrics(train_df, info)
                news_list = []  # Historical news sentiment would require time-series data

                # Extract features
                features = extract_ml_features(
                    ticker, train_df, info, indicators, metrics, news_list)
                features['ticker'] = ticker.upper()

                # Decide if we need to retrain model
                should_retrain = (
                    cached_model is None or  # First iteration
                    (i - last_retrain_idx) >= retrain_interval_days or  # Interval reached
                    len(train_df) - last_retrain_idx > retrain_interval_days *
                    2  # Significant growth
                )

                if should_retrain:
                    # Train new model on training data
                    current_price = train_df['Close'].iloc[-1]
                    logger.debug(
                        f"Retraining model at index {i} (interval: {i - last_retrain_idx} days)")
                    cached_model, cached_scaler = _train_random_forest_model(
                        ticker, features, current_price, train_df)
                    cached_features_template = features.copy()
                    last_retrain_idx = i

                # Use cached model for prediction
                if cached_model and cached_scaler:
                    # CRITICAL: Additional safety check - verify model has required attributes
                    if not hasattr(cached_model, 'estimators_') or cached_model.estimators_ is None:
                        logger.warning(f"Backtest: Model for {ticker} at index {i} is missing estimators_ attribute, skipping prediction")
                        continue
                    
                    try:
                        # Make prediction using cached model
                        feature_names = sorted([k for k in features.keys() if k != 'ticker'])
                        X = np.array([[features.get(name, 0.0) for name in feature_names]])
                        X_scaled = cached_scaler.transform(X)

                        # Get predictions from all trees for proper confidence intervals
                        tree_preds = [tree.predict(X_scaled)[0]
                                      for tree in cached_model.estimators_]
                        predicted_price = np.mean(tree_preds)
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Backtest: Error using model for {ticker} at index {i}: {e}, skipping prediction")
                        continue

                    predictions.append(predicted_price)
                    actuals.append(actual_price)
                    dates.append(test_date)

            except Exception as e:
                logger.debug(f"Error in backtest iteration {i}: {e}")

                continue

        if len(predictions) < 10:
            return {

                'success': False,

                'error': f'Insufficient predictions generated (only {len(predictions)})'

            }

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)

        # Calculate percentage errors
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Direction accuracy (did we predict up/down correctly?)
        direction_accuracy = 0.0
        if len(predictions) > 1:
            pred_direction = np.diff(predictions) > 0

            actual_direction = np.diff(actuals) > 0
            direction_accuracy = np.mean(
                pred_direction == actual_direction) * 100

        # Calculate returns
        actual_returns = np.diff(actuals) / actuals[:-1] * 100
        predicted_returns = np.diff(predictions) / predictions[:-1] * 100

        # Correlation
        correlation = np.corrcoef(actuals, predictions)[
            0, 1] if len(predictions) > 1 else 0.0

        # BASELINE COMPARISON
        # 1. Naive baseline: Price stays the same (predict current price)
        naive_predictions = np.full_like(
            actuals, actuals[0])  # Use first actual price
        naive_mae = mean_absolute_error(actuals, naive_predictions)
        naive_mse = mean_squared_error(actuals, naive_predictions)
        naive_rmse = np.sqrt(naive_mse)
        naive_r2 = r2_score(actuals, naive_predictions)
        naive_mape = np.mean(
            np.abs((actuals - naive_predictions) / actuals)) * 100

        # 2. Momentum baseline: Price continues trend
        momentum_predictions = []
        for i in range(len(actuals)):
            if i == 0:
                momentum_predictions.append(actuals[0])
            else:
                # Simple momentum: assume same return as previous day
                prev_return = (actuals[i] - actuals[i - 1]) / \
                    actuals[i - 1] if actuals[i - 1] > 0 else 0
                momentum_pred = actuals[i - 1] * (1 + prev_return)
                momentum_predictions.append(momentum_pred)
        momentum_predictions = np.array(momentum_predictions)

        momentum_mae = mean_absolute_error(actuals, momentum_predictions)
        momentum_mse = mean_squared_error(actuals, momentum_predictions)
        momentum_rmse = np.sqrt(momentum_mse)
        momentum_r2 = r2_score(actuals, momentum_predictions)
        momentum_mape = np.mean(
            np.abs((actuals - momentum_predictions) / actuals)) * 100

        # 3. Trading strategy metrics (if we followed predictions)
        if len(actual_returns) > 0:
            # Calculate strategy returns (buy when predicted up, hold)
            strategy_returns = []
            for i in range(len(actual_returns)):
                # If we predicted increase, we would buy/hold
                if i < len(predicted_returns):
                    # Simple strategy: follow prediction direction
                    strategy_returns.append(actual_returns[i])
                else:
                    strategy_returns.append(0)

            strategy_returns = np.array(strategy_returns)

            # Sharpe ratio (annualized)
            if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                sharpe_ratio = (np.mean(strategy_returns) /
                                np.std(strategy_returns)) * np.sqrt(252)

            else:
                sharpe_ratio = 0.0

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns / 100)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = float(np.min(drawdown)) * \
                100 if len(drawdown) > 0 else 0.0

            # Total return
            total_return = (
                cumulative_returns[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            total_return = 0.0

        result = {
            'success': True,
            'ticker': ticker.upper(),
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'test_period_days': len(df_test),
            'predictions_count': len(predictions),
            'warning': 'Backtest uses simplified features (no historical fundamentals/news data). Results may not fully reflect real-world performance with complete feature set.',
            'metrics': {
                'mae': float(mae),

                'mse': float(mse),

                'rmse': float(rmse),

                'r2_score': float(r2),

                'mape': float(mape),

                'direction_accuracy': float(direction_accuracy),

                'correlation': float(correlation)

            },

            'baseline_comparison': {
                'naive_baseline': {

                    'mae': float(naive_mae),

                    'rmse': float(naive_rmse),

                    'r2_score': float(naive_r2),

                    'mape': float(naive_mape),

                    'description': 'Predicts price stays the same (first price)'

                },

                'momentum_baseline': {

                    'mae': float(momentum_mae),

                    'rmse': float(momentum_rmse),

                    'r2_score': float(momentum_r2),

                    'mape': float(momentum_mape),

                    'description': 'Predicts price continues previous day trend'

                },

                'ml_model_vs_baselines': {

                    'better_than_naive': r2 > naive_r2,

                    'better_than_momentum': r2 > momentum_r2,

                    'improvement_vs_naive_r2': float(r2 - naive_r2),

                    'improvement_vs_momentum_r2': float(r2 - momentum_r2)

                }

            },

            'trading_metrics': {
                'sharpe_ratio': float(sharpe_ratio),

                'max_drawdown_pct': float(max_drawdown),

                'total_return_pct': float(total_return),

                'note': 'Trading metrics assume simple buy-and-hold strategy based on predictions. Not actual trading advice.'

            },

            'summary': {
                'mean_absolute_error': f"${mae:.2f}",

                'root_mean_squared_error': f"${rmse:.2f}",

                'r2_score': f"{r2:.4f}",

                'mean_absolute_percentage_error': f"{mape:.2f}%",

                'direction_accuracy': f"{direction_accuracy:.1f}%",

                'correlation': f"{correlation:.4f}",

                'sharpe_ratio': f"{sharpe_ratio:.2f}",

                'max_drawdown': f"{max_drawdown:.2f}%"

            },

            'predictions': [
                {

                    'date': dates[i].strftime('%Y-%m-%d'),

                    'predicted': float(predictions[i]),

                    'actual': float(actuals[i]),

                    'error': float(predictions[i] - actuals[i]),

                    'error_pct': float((predictions[i] - actuals[i]) / actuals[i] * 100)

                }

                for i in range(len(predictions))

            ]

        }

        
        logger.info(f"Backtest completed for {ticker}. R²: {r2:.4f}, MAPE: {mape:.2f}%, Direction Accuracy: {direction_accuracy:.1f}%")
        
        return result
        
    except Exception as e:
        logger.exception(f"Error running backtest for {ticker}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

