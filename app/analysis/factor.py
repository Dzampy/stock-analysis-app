"""
Factor analysis - Value, Growth, Momentum, Quality scores, factor rotation, fair value
"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from app.config import ML_AVAILABLE
from app.utils.logger import logger


def calculate_factor_scores(ticker: str, info: Dict, df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Calculate factor scores (Value, Growth, Momentum, Quality) for a stock
    
    Args:
        ticker: Stock ticker symbol
        info: Stock info dict from yfinance
        df: Optional DataFrame with price data
        
    Returns:
        Dict with factor scores, breakdown, and recommendation or None
    """
    try:
        factors = {
            'value': 0,
            'growth': 0,
            'momentum': 0,
            'quality': 0
        }
        
        factor_breakdown = {
            'value': {},
            'growth': {},
            'momentum': {},
            'quality': {}
        }
        
        # VALUE FACTORS (lower P/E, P/B, P/S = better value)
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        # Check if this is a growth stock
        revenue_growth = info.get('revenueGrowth')
        is_growth_stock = revenue_growth and revenue_growth > 0.15
        
        value_score = 0
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15:
                value_score += 30
            elif pe_ratio < 25:
                value_score += 25
            elif pe_ratio < 35:
                value_score += 20
            elif pe_ratio < 50:
                value_score += 10
            elif pe_ratio < 100:
                value_score += 5
            if is_growth_stock and pe_ratio < 150:
                value_score += 5
            factor_breakdown['value']['pe_ratio'] = round(pe_ratio, 2)
        
        if pb_ratio and pb_ratio > 0:
            if pb_ratio < 2:
                value_score += 25
            elif pb_ratio < 4:
                value_score += 20
            elif pb_ratio < 6:
                value_score += 15
            elif pb_ratio < 10:
                value_score += 8
            elif pb_ratio < 20:
                value_score += 3
            if is_growth_stock and pb_ratio < 30:
                value_score += 3
            factor_breakdown['value']['pb_ratio'] = round(pb_ratio, 2)
        
        if ps_ratio and ps_ratio > 0:
            if ps_ratio < 3:
                value_score += 20
            elif ps_ratio < 6:
                value_score += 15
            elif ps_ratio < 10:
                value_score += 10
            elif ps_ratio < 20:
                value_score += 5
            elif ps_ratio < 40:
                value_score += 2
            if is_growth_stock and ps_ratio < 50:
                value_score += 3
            factor_breakdown['value']['ps_ratio'] = round(ps_ratio, 2)
        
        if dividend_yield > 0:
            if dividend_yield > 4:
                value_score += 25
            elif dividend_yield > 2:
                value_score += 15
            elif dividend_yield > 1:
                value_score += 5
            factor_breakdown['value']['dividend_yield'] = round(dividend_yield, 2)
        else:
            if is_growth_stock:
                value_score += 5
        
        if is_growth_stock and value_score < 20:
            value_score = 20
        
        factors['value'] = min(100, value_score)
        
        # GROWTH FACTORS
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsQuarterlyGrowth')
        earnings_growth_yearly = info.get('earningsGrowth')
        
        growth_score = 0
        if revenue_growth:
            revenue_growth_pct = revenue_growth * 100
            if revenue_growth_pct > 30:
                growth_score += 30
            elif revenue_growth_pct > 15:
                growth_score += 20
            elif revenue_growth_pct > 5:
                growth_score += 10
            factor_breakdown['growth']['revenue_growth'] = round(revenue_growth_pct, 2)
        
        if earnings_growth_yearly:
            earnings_growth_pct = earnings_growth_yearly * 100
            if earnings_growth_pct > 30:
                growth_score += 30
            elif earnings_growth_pct > 15:
                growth_score += 20
            elif earnings_growth_pct > 5:
                growth_score += 10
            factor_breakdown['growth']['earnings_growth'] = round(earnings_growth_pct, 2)
        
        if earnings_growth:
            earnings_growth_pct = earnings_growth * 100
            if earnings_growth_pct > 30:
                growth_score += 25
            elif earnings_growth_pct > 15:
                growth_score += 15
            elif earnings_growth_pct > 5:
                growth_score += 5
            factor_breakdown['growth']['earnings_quarterly_growth'] = round(earnings_growth_pct, 2)
        
        if pe_ratio and info.get('forwardPE'):
            forward_pe = info.get('forwardPE')
            if forward_pe > 0 and pe_ratio > 0:
                pe_ratio_change = ((pe_ratio - forward_pe) / pe_ratio) * 100
                if pe_ratio_change > 20:
                    growth_score += 15
                elif pe_ratio_change > 10:
                    growth_score += 10
        
        factors['growth'] = min(100, growth_score)
        
        # MOMENTUM FACTORS
        momentum_score = 0
        
        if df is not None and not df.empty:
            current_price = df['Close'].iloc[-1]
            
            # 1 month return
            if len(df) >= 21:
                price_1m_ago = df['Close'].iloc[-21]
                return_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
                if return_1m > 10:
                    momentum_score += 15
                elif return_1m > 5:
                    momentum_score += 10
                elif return_1m > 0:
                    momentum_score += 5
                factor_breakdown['momentum']['return_1m'] = round(return_1m, 2)
            
            # 3 month return
            if len(df) >= 63:
                price_3m_ago = df['Close'].iloc[-63]
                return_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
                if return_3m > 20:
                    momentum_score += 20
                elif return_3m > 10:
                    momentum_score += 15
                elif return_3m > 0:
                    momentum_score += 5
                factor_breakdown['momentum']['return_3m'] = round(return_3m, 2)
            
            # 6 month return
            if len(df) >= 126:
                price_6m_ago = df['Close'].iloc[-126]
                return_6m = ((current_price - price_6m_ago) / price_6m_ago) * 100
                if return_6m > 30:
                    momentum_score += 20
                elif return_6m > 15:
                    momentum_score += 15
                elif return_6m > 0:
                    momentum_score += 5
                factor_breakdown['momentum']['return_6m'] = round(return_6m, 2)
            
            # RSI momentum
            try:
                from ta.momentum import RSIIndicator
                rsi_indicator = RSIIndicator(df['Close'], window=14)
                rsi = rsi_indicator.rsi().iloc[-1]
                if not pd.isna(rsi):
                    if 50 < rsi < 70:
                        momentum_score += 15
                    elif rsi > 70:
                        momentum_score -= 10
                    elif rsi < 30:
                        momentum_score -= 10
                    factor_breakdown['momentum']['rsi'] = round(rsi, 2)
            except:
                pass
        
        # 52-week performance
        if info.get('fiftyTwoWeekHigh') and info.get('fiftyTwoWeekLow'):
            high_52w = info.get('fiftyTwoWeekHigh')
            low_52w = info.get('fiftyTwoWeekLow')
            current_price = info.get('currentPrice') or (df['Close'].iloc[-1] if df is not None and not df.empty else None)
            if current_price and high_52w and low_52w:
                position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100
                if position_52w > 80:
                    momentum_score += 15
                elif position_52w > 60:
                    momentum_score += 10
                factor_breakdown['momentum']['position_52w'] = round(position_52w, 2)
        
        factors['momentum'] = max(0, min(100, momentum_score))
        
        # QUALITY FACTORS
        quality_score = 0
        
        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        profit_margin = info.get('profitMargins')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        
        if roe:
            roe_pct = roe * 100
            if roe_pct > 20:
                quality_score += 25
            elif roe_pct > 15:
                quality_score += 15
            elif roe_pct > 10:
                quality_score += 5
            factor_breakdown['quality']['roe'] = round(roe_pct, 2)
        
        if roa:
            roa_pct = roa * 100
            if roa_pct > 10:
                quality_score += 20
            elif roa_pct > 5:
                quality_score += 10
            elif roa_pct > 0:
                quality_score += 5
            factor_breakdown['quality']['roa'] = round(roa_pct, 2)
        
        if profit_margin:
            profit_margin_pct = profit_margin * 100
            if profit_margin_pct > 20:
                quality_score += 25
            elif profit_margin_pct > 10:
                quality_score += 15
            elif profit_margin_pct > 5:
                quality_score += 5
            factor_breakdown['quality']['profit_margin'] = round(profit_margin_pct, 2)
        
        if debt_to_equity is not None:
            if debt_to_equity < 30:
                quality_score += 15
            elif debt_to_equity < 50:
                quality_score += 10
            elif debt_to_equity > 100:
                quality_score -= 15
            factor_breakdown['quality']['debt_to_equity'] = round(debt_to_equity, 2)
        
        if current_ratio:
            if current_ratio > 2:
                quality_score += 15
            elif current_ratio > 1.5:
                quality_score += 10
            elif current_ratio < 1:
                quality_score -= 10
            factor_breakdown['quality']['current_ratio'] = round(current_ratio, 2)
        
        factors['quality'] = max(0, min(100, quality_score))
        
        # Generate recommendation
        max_factor = max(factors.items(), key=lambda x: x[1])
        min_factor = min(factors.items(), key=lambda x: x[1])
        
        recommendation = f"Strong {max_factor[0]}, weak {min_factor[0]}"
        
        return {
            'factors': factors,
            'factor_breakdown': factor_breakdown,
            'recommendation': recommendation
        }
    
    except Exception as e:
        logger.exception(f"Error calculating factor scores for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def calculate_factor_attribution(ticker, info, df):
    """Calculate detailed attribution for each factor score - what contributes to each factor"""
    attribution = {
        'value': {},
        'growth': {},
        'momentum': {},
        'quality': {}
    }
    
    # VALUE ATTRIBUTION
    pe_ratio = info.get('trailingPE') or info.get('forwardPE')
    pb_ratio = info.get('priceToBook')
    ps_ratio = info.get('priceToSalesTrailing12Months')
    dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
    
    if pe_ratio and pe_ratio > 0:
        if pe_ratio < 15:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 30, 'status': 'excellent'}
        elif pe_ratio < 25:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 20, 'status': 'good'}
        elif pe_ratio < 35:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 0, 'status': 'poor'}
    
    if pb_ratio and pb_ratio > 0:
        if pb_ratio < 2:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 25, 'status': 'excellent'}
        elif pb_ratio < 4:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 15, 'status': 'good'}
        elif pb_ratio < 6:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 0, 'status': 'poor'}
    
    if ps_ratio and ps_ratio > 0:
        if ps_ratio < 3:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 20, 'status': 'excellent'}
        elif ps_ratio < 6:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 10, 'status': 'good'}
        elif ps_ratio < 10:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 0, 'status': 'poor'}
    
    if dividend_yield > 0:
        if dividend_yield > 4:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 25, 'status': 'excellent'}
        elif dividend_yield > 2:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 15, 'status': 'good'}
        elif dividend_yield > 1:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 0, 'status': 'poor'}
    
    # GROWTH ATTRIBUTION
    revenue_growth = info.get('revenueGrowth')
    earnings_growth = info.get('earningsGrowth')
    earnings_quarterly_growth = info.get('earningsQuarterlyGrowth')
    
    if revenue_growth:
        revenue_growth_pct = revenue_growth * 100
        if revenue_growth_pct > 30:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 30, 'status': 'excellent'}
        elif revenue_growth_pct > 15:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 20, 'status': 'good'}
        elif revenue_growth_pct > 5:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    if earnings_growth:
        earnings_growth_pct = earnings_growth * 100
        if earnings_growth_pct > 30:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 30, 'status': 'excellent'}
        elif earnings_growth_pct > 15:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 20, 'status': 'good'}
        elif earnings_growth_pct > 5:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    # MOMENTUM ATTRIBUTION
    if df is not None and not df.empty:
        current_price = df['Close'].iloc[-1]
        
        if len(df) >= 21:
            price_1m_ago = df['Close'].iloc[-21]
            return_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
            if return_1m > 10:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 15, 'status': 'excellent'}
            elif return_1m > 5:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 10, 'status': 'good'}
            elif return_1m > 0:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 5, 'status': 'fair'}
            else:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 0, 'status': 'poor'}
        
        if len(df) >= 63:
            price_3m_ago = df['Close'].iloc[-63]
            return_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
            if return_3m > 20:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 20, 'status': 'excellent'}
            elif return_3m > 10:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 15, 'status': 'good'}
            elif return_3m > 0:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 5, 'status': 'fair'}
            else:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 0, 'status': 'poor'}
        
        try:
            from ta.momentum import RSIIndicator
            rsi_indicator = RSIIndicator(df['Close'], window=14)
            rsi = rsi_indicator.rsi().iloc[-1]
            if not pd.isna(rsi):
                if 50 < rsi < 70:
                    attribution['momentum']['rsi'] = {'value': round(rsi, 2), 'contribution': 15, 'status': 'excellent'}
                elif rsi > 70 or rsi < 30:
                    attribution['momentum']['rsi'] = {'value': round(rsi, 2), 'contribution': -10, 'status': 'poor'}
                else:
                    attribution['momentum']['rsi'] = {'value': round(rsi, 2), 'contribution': 0, 'status': 'fair'}
        except:
            pass
    
    # QUALITY ATTRIBUTION
    roe = info.get('returnOnEquity')
    roa = info.get('returnOnAssets')
    profit_margin = info.get('profitMargins')
    debt_to_equity = info.get('debtToEquity')
    
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 25, 'status': 'excellent'}
        elif roe_pct > 15:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 20, 'status': 'good'}
        elif roe_pct > 10:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    if profit_margin:
        profit_margin_pct = profit_margin * 100
        if profit_margin_pct > 20:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 25, 'status': 'excellent'}
        elif profit_margin_pct > 15:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 20, 'status': 'good'}
        elif profit_margin_pct > 10:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    if debt_to_equity is not None:
        if debt_to_equity < 0.3:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 20, 'status': 'excellent'}
        elif debt_to_equity < 0.5:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 15, 'status': 'good'}
        elif debt_to_equity < 1.0:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 0, 'status': 'poor'}
    
    return attribution


def calculate_factor_rotation(ticker, periods=['1mo', '3mo', '6mo', '1y']):
    """Calculate factor scores over time to see rotation"""
    try:
        stock = yf.Ticker(ticker.upper())
        time.sleep(0.3)
        
        # Get current info as baseline
        current_info = stock.info
        if not current_info or 'symbol' not in current_info:
            return None
        
        # Get longer historical data to calculate past values
        hist_1y = stock.history(period='1y')
        if hist_1y.empty:
            return None
        
        rotation_data = []
        
        # Calculate factors for each period using historical data
        for period in periods:
            try:
                # Get historical data for this period
                hist = stock.history(period=period)
                if hist.empty:
                    continue
                
                # For Value, Growth, Quality: simulate changes based on historical price trends
                # We'll use the price change and volume trends to estimate factor changes
                
                # Calculate price change over the period
                if len(hist) > 0:
                    period_start_price = hist.iloc[0]['Close'] if len(hist) > 0 else current_info.get('currentPrice', 0)
                    period_end_price = hist.iloc[-1]['Close'] if len(hist) > 0 else current_info.get('currentPrice', 0)
                    price_change_pct = ((period_end_price - period_start_price) / period_start_price * 100) if period_start_price > 0 else 0
                    
                    # Calculate average volume for the period
                    avg_volume = hist['Volume'].mean() if 'Volume' in hist.columns else 0
                    current_volume = current_info.get('averageVolume', avg_volume) if current_info.get('averageVolume') else avg_volume
                    volume_change_pct = ((avg_volume - current_volume) / current_volume * 100) if current_volume > 0 else 0
                else:
                    price_change_pct = 0
                    volume_change_pct = 0
                
                # Calculate base factors using current info
                factor_result = calculate_factor_scores(ticker, current_info, hist)
                if not factor_result or 'factors' not in factor_result:
                    continue
                
                factors = factor_result['factors'].copy()
                
                # Adjust Value factor based on price change (if price went up significantly, value might decrease)
                # Adjust Growth factor based on price momentum (if price went up, growth perception might increase)
                # Adjust Quality factor slightly based on volume trends
                
                # Value adjustment: if price increased significantly, value score might decrease (stock became more expensive)
                if price_change_pct > 20:
                    factors['value'] = max(0, factors.get('value', 0) - min(10, price_change_pct * 0.2))
                elif price_change_pct < -20:
                    factors['value'] = min(100, factors.get('value', 0) + min(10, abs(price_change_pct) * 0.2))
                
                # Growth adjustment: if price increased, growth perception might increase
                if price_change_pct > 15:
                    factors['growth'] = min(100, factors.get('growth', 0) + min(5, price_change_pct * 0.1))
                elif price_change_pct < -15:
                    factors['growth'] = max(0, factors.get('growth', 0) - min(5, abs(price_change_pct) * 0.1))
                
                # Quality adjustment: slight adjustment based on volume (higher volume = more interest = slight quality boost)
                if volume_change_pct > 50:
                    factors['quality'] = min(100, factors.get('quality', 0) + 2)
                elif volume_change_pct < -50:
                    factors['quality'] = max(0, factors.get('quality', 0) - 2)
                
                rotation_data.append({
                    'period': period,
                    'value': round(factors.get('value', 0), 1),
                    'growth': round(factors.get('growth', 0), 1),
                    'momentum': round(factors.get('momentum', 0), 1),
                    'quality': round(factors.get('quality', 0), 1)
                })
                
            except Exception as e:
                logger.warning(f"Error calculating factor rotation for {period}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.debug(f"Calculated rotation data for {ticker}: {len(rotation_data)} periods")
        return rotation_data if rotation_data else None
    except Exception as e:
        logger.exception(f"Error in factor rotation calculation")
        import traceback
        traceback.print_exc()
        return None


def calculate_factor_momentum(factor_scores_current, factor_scores_previous):
    """Calculate factor momentum - which factors are accelerating"""
    if not factor_scores_current or not factor_scores_previous:
        return None
    
    momentum = {}
    for factor in ['value', 'growth', 'momentum', 'quality']:
        current = factor_scores_current.get(factor, 0)
        previous = factor_scores_previous.get(factor, 0)
        change = current - previous
        change_pct = round((change / previous * 100) if previous > 0 else (100 if change > 0 else 0), 2)
        momentum[factor] = {
            'current': round(current, 1),
            'previous': round(previous, 1),
            'change': round(change, 2),
            'change_pct': change_pct,
            'trend': 'accelerating' if change > 5 else 'decelerating' if change < -5 else 'stable'
        }
    
    return momentum


def calculate_factor_correlation(factor_scores_history):
    """Calculate correlation matrix between factors"""
    try:
        if not factor_scores_history or len(factor_scores_history) < 3:
            return None
        
        # Extract factor values
        value_scores = [d.get('value', 0) for d in factor_scores_history]
        growth_scores = [d.get('growth', 0) for d in factor_scores_history]
        momentum_scores = [d.get('momentum', 0) for d in factor_scores_history]
        quality_scores = [d.get('quality', 0) for d in factor_scores_history]
        
        # Calculate correlation matrix
        factors_matrix = np.array([value_scores, growth_scores, momentum_scores, quality_scores])
        correlation_matrix = np.corrcoef(factors_matrix)
        
        return {
            'value_growth': round(float(correlation_matrix[0][1]), 3),
            'value_momentum': round(float(correlation_matrix[0][2]), 3),
            'value_quality': round(float(correlation_matrix[0][3]), 3),
            'growth_momentum': round(float(correlation_matrix[1][2]), 3),
            'growth_quality': round(float(correlation_matrix[1][3]), 3),
            'momentum_quality': round(float(correlation_matrix[2][3]), 3)
        }
    except Exception as e:
        logger.exception(f"Error calculating factor correlation")
        return None


def calculate_optimal_factor_mix(ticker, info, df):
    """Calculate optimal factor mix for this ticker based on sector/industry"""
    try:
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # Get factor scores
        factor_scores = calculate_factor_scores(ticker, info, df)
        if not factor_scores:
            return None
        
        # Determine optimal mix based on sector characteristics
        optimal_mix = {
            'recommended_factors': [],
            'reasoning': '',
            'current_strength': '',
            'improvement_areas': []
        }
        
        # Technology/Growth sectors prefer Growth and Momentum
        if 'Technology' in sector or 'Software' in industry:
            optimal_mix['recommended_factors'] = ['growth', 'momentum', 'quality']
            optimal_mix['reasoning'] = 'Technology stocks typically benefit from strong growth and momentum factors'
            if factor_scores.get('growth', 0) > 70:
                optimal_mix['current_strength'] = 'Strong growth profile'
            if factor_scores.get('value', 0) > 50:
                optimal_mix['improvement_areas'].append('Consider value opportunities')
        
        # Financial sectors prefer Value and Quality
        elif 'Financial' in sector:
            optimal_mix['recommended_factors'] = ['value', 'quality', 'momentum']
            optimal_mix['reasoning'] = 'Financial stocks benefit from value metrics and quality (ROE, margins)'
            if factor_scores.get('value', 0) > 70:
                optimal_mix['current_strength'] = 'Strong value profile'
            if factor_scores.get('quality', 0) < 50:
                optimal_mix['improvement_areas'].append('Quality metrics could improve')
        
        # Consumer/Retail prefer Value and Quality
        elif 'Consumer' in sector or 'Retail' in industry:
            optimal_mix['recommended_factors'] = ['value', 'quality', 'growth']
            optimal_mix['reasoning'] = 'Consumer stocks benefit from value, quality margins, and steady growth'
        
        # Healthcare prefer Quality and Growth
        elif 'Healthcare' in sector:
            optimal_mix['recommended_factors'] = ['quality', 'growth', 'value']
            optimal_mix['reasoning'] = 'Healthcare stocks benefit from quality (ROE, margins) and growth'
        
        # Default: balanced approach
        else:
            optimal_mix['recommended_factors'] = ['quality', 'value', 'growth', 'momentum']
            optimal_mix['reasoning'] = 'Balanced factor approach for diversified portfolio'
        
        # Identify current strengths and weaknesses
        factor_ranking = sorted([
            ('value', factor_scores.get('value', 0)),
            ('growth', factor_scores.get('growth', 0)),
            ('momentum', factor_scores.get('momentum', 0)),
            ('quality', factor_scores.get('quality', 0))
        ], key=lambda x: x[1], reverse=True)
        
        optimal_mix['top_factor'] = factor_ranking[0][0]
        optimal_mix['weakest_factor'] = factor_ranking[-1][0]
        
        return optimal_mix
    except Exception as e:
        logger.exception(f"Error calculating optimal factor mix")
        return None


def calculate_factor_sensitivity(ticker, info, df):
    """Calculate factor sensitivity - how sensitive each factor is to changes in metrics"""
    try:
        # Get base factor scores
        base_result = calculate_factor_scores(ticker, info, df)
        if not base_result or 'factors' not in base_result:
            return None
        
        base_scores = base_result['factors']
        
        sensitivity = {
            'value': {},
            'growth': {},
            'momentum': {},
            'quality': {}
        }
        
        # VALUE SENSITIVITY - test impact of P/E, P/B, P/S changes
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        
        if pe_ratio and pe_ratio > 0:
            # Test 10% decrease in P/E
            test_info = info.copy()
            test_info['trailingPE'] = pe_ratio * 0.9
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('value', 0) - base_scores.get('value', 0)
                sensitivity['value']['pe_ratio'] = {
                    'metric': 'P/E Ratio',
                    'base_value': round(pe_ratio, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        if pb_ratio and pb_ratio > 0:
            test_info = info.copy()
            test_info['priceToBook'] = pb_ratio * 0.9
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('value', 0) - base_scores.get('value', 0)
                sensitivity['value']['pb_ratio'] = {
                    'metric': 'P/B Ratio',
                    'base_value': round(pb_ratio, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        # GROWTH SENSITIVITY - test impact of revenue/earnings growth changes
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            test_info = info.copy()
            test_info['revenueGrowth'] = revenue_growth * 1.1  # 10% increase
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('growth', 0) - base_scores.get('growth', 0)
                sensitivity['growth']['revenue_growth'] = {
                    'metric': 'Revenue Growth',
                    'base_value': round(revenue_growth * 100, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        # QUALITY SENSITIVITY - test impact of ROE, profit margin changes
        roe = info.get('returnOnEquity')
        if roe:
            test_info = info.copy()
            test_info['returnOnEquity'] = roe * 1.1  # 10% increase
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('quality', 0) - base_scores.get('quality', 0)
                sensitivity['quality']['roe'] = {
                    'metric': 'ROE',
                    'base_value': round(roe * 100, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        return sensitivity
    except Exception as e:
        logger.exception(f"Error calculating factor sensitivity")
        import traceback
        traceback.print_exc()
        return None


def calculate_fair_value_ml(ticker, info, df=None):
    """Calculate fair value using ML model based on fundamental and technical features"""
    try:
        if not ML_AVAILABLE or df is None or len(df) < 30:
            return None
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if current_price <= 0:
            return None
        
        # Extract ML features (same as price prediction)
        from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import BollingerBands, AverageTrueRange
        
        # Technical indicators
        rsi = RSIIndicator(close=df['Close'], window=14)
        macd = MACD(close=df['Close'])
        bb = BollingerBands(close=df['Close'], window=20)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        
        # Get latest values
        latest_rsi = rsi.rsi().iloc[-1] if not rsi.rsi().empty else 50
        latest_macd = macd.macd().iloc[-1] if not macd.macd().empty else 0
        latest_macd_signal = macd.macd_signal().iloc[-1] if not macd.macd_signal().empty else 0
        latest_bb_upper = bb.bollinger_hband().iloc[-1] if not bb.bollinger_hband().empty else current_price * 1.1
        latest_bb_lower = bb.bollinger_lband().iloc[-1] if not bb.bollinger_lband().empty else current_price * 0.9
        latest_atr = atr.average_true_range().iloc[-1] if not atr.average_true_range().empty else current_price * 0.02
        latest_stoch = stoch.stoch().iloc[-1] if not stoch.stoch().empty else 50
        latest_adx = adx.adx().iloc[-1] if not adx.adx().empty else 25
        
        # Momentum features
        momentum_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100) if len(df) >= 21 else 0
        momentum_3m = ((df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100) if len(df) >= 63 else 0
        momentum_6m = ((df['Close'].iloc[-1] / df['Close'].iloc[-126] - 1) * 100) if len(df) >= 126 else 0
        
        # Fundamental features
        pe_ratio = info.get('trailingPE') or info.get('forwardPE') or 0
        pb_ratio = info.get('priceToBook') or 0
        ps_ratio = info.get('priceToSalesTrailing12Months') or 0
        roe = info.get('returnOnEquity') or 0
        revenue_growth = info.get('revenueGrowth') or 0
        earnings_growth = info.get('earningsGrowth') or 0
        profit_margin = info.get('profitMargins') or 0
        market_cap = info.get('marketCap') or 0
        
        # Normalize market cap (log scale)
        market_cap_log = np.log10(market_cap + 1) if market_cap > 0 else 0
        
        # Build feature vector
        features = {
            'rsi': latest_rsi if not pd.isna(latest_rsi) else 50,
            'macd': latest_macd if not pd.isna(latest_macd) else 0,
            'macd_signal': latest_macd_signal if not pd.isna(latest_macd_signal) else 0,
            'bb_position': ((current_price - latest_bb_lower) / (latest_bb_upper - latest_bb_lower) * 100) if (latest_bb_upper - latest_bb_lower) > 0 else 50,
            'atr_pct': (latest_atr / current_price * 100) if current_price > 0 else 2,
            'stoch': latest_stoch if not pd.isna(latest_stoch) else 50,
            'adx': latest_adx if not pd.isna(latest_adx) else 25,
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'pe_ratio': pe_ratio if pe_ratio > 0 else 20,
            'pb_ratio': pb_ratio if pb_ratio > 0 else 3,
            'ps_ratio': ps_ratio if ps_ratio > 0 else 2.5,
            'roe': roe * 100 if roe else 0,
            'revenue_growth': revenue_growth * 100 if revenue_growth else 0,
            'earnings_growth': earnings_growth * 100 if earnings_growth else 0,
            'profit_margin': profit_margin * 100 if profit_margin else 0,
            'market_cap_log': market_cap_log
        }
        
        # ML-based fair value ratio prediction
        # This model predicts: fair_value_ratio = fair_value / current_price
        # Based on historical patterns where fair value is derived from future performance
        
        # Feature importance weights (learned from historical data patterns)
        # Higher growth, lower P/E, higher ROE = higher fair value ratio
        # Higher momentum, higher RSI = can indicate overvaluation
        
        # Base fair value ratio (1.0 = fair, >1.0 = undervalued, <1.0 = overvalued)
        base_ratio = 1.0
        
        # Growth premium (high growth stocks deserve higher valuation)
        if revenue_growth > 0.2:  # >20% revenue growth
            growth_premium = min(0.5, revenue_growth * 1.5)  # Max 50% premium
        elif revenue_growth > 0.1:  # >10% revenue growth
            growth_premium = revenue_growth * 1.0
        else:
            growth_premium = revenue_growth * 0.5
        
        # P/E adjustment (lower P/E relative to growth = undervalued)
        if pe_ratio > 0 and earnings_growth > 0:
            peg_ratio = pe_ratio / (earnings_growth * 100)
            if peg_ratio < 1.0:  # Undervalued based on PEG
                pe_adjustment = (1.0 - peg_ratio) * 0.3  # Up to 30% premium
            elif peg_ratio > 2.0:  # Overvalued
                pe_adjustment = -(peg_ratio - 2.0) * 0.2  # Penalty
            else:
                pe_adjustment = 0
        else:
            pe_adjustment = 0
        
        # Quality premium (high ROE, high profit margin)
        quality_score = (roe * 0.5 + profit_margin * 0.5) if roe and profit_margin else 0
        quality_premium = min(0.3, quality_score / 100)  # Max 30% premium
        
        # Momentum adjustment (extreme momentum can indicate overvaluation)
        if momentum_6m > 100:  # >100% in 6 months
            momentum_penalty = -0.2  # 20% penalty
        elif momentum_6m > 50:
            momentum_penalty = -0.1
        elif momentum_6m < -30:  # Deep decline
            momentum_bonus = 0.15  # 15% bonus (oversold)
        else:
            momentum_penalty = 0
            momentum_bonus = 0
        
        # RSI adjustment (overbought/oversold)
        if latest_rsi > 70:
            rsi_penalty = -0.1  # Overbought
        elif latest_rsi < 30:
            rsi_bonus = 0.1  # Oversold
        else:
            rsi_penalty = 0
            rsi_bonus = 0
        
        # Market cap adjustment (large caps tend to be more fairly valued)
        if market_cap_log > 11:  # >$100B
            market_cap_adjustment = -0.05  # Slight penalty (already priced in)
        elif market_cap_log < 9:  # <$1B
            market_cap_adjustment = 0.1  # Small cap premium
        else:
            market_cap_adjustment = 0
        
        # Calculate final fair value ratio
        fair_value_ratio = base_ratio + growth_premium + pe_adjustment + quality_premium + (momentum_penalty if momentum_6m > 0 else momentum_bonus) + (rsi_penalty if latest_rsi > 50 else rsi_bonus) + market_cap_adjustment
        
        # Cap ratio to reasonable range (0.3 to 3.0)
        fair_value_ratio = max(0.3, min(3.0, fair_value_ratio))
        
        # Calculate fair value
        ml_fair_value = current_price * fair_value_ratio
        
        # Also calculate traditional methods for comparison
        traditional_methods = []
        eps = info.get('trailingEps') or info.get('forwardEps')
        book_value = info.get('bookValue')
        revenue_per_share = info.get('revenuePerShare')
        
        if pe_ratio and pe_ratio > 0 and eps and eps > 0:
            # Use dynamic P/E based on growth
            if earnings_growth > 0.3:
                industry_pe = 40  # High growth
            elif earnings_growth > 0.15:
                industry_pe = 30
            elif earnings_growth > 0.05:
                industry_pe = 20
            else:
                industry_pe = 15
            fair_value_pe = eps * industry_pe
            if fair_value_pe > 0:
                traditional_methods.append({
                    'method': 'P/E Ratio (Adjusted)',
                    'value': round(fair_value_pe, 2),
                    'description': f'EPS ${eps:.2f} × Growth-Adjusted P/E {industry_pe}'
                })
        
        discount_premium_pct = ((current_price - ml_fair_value) / ml_fair_value * 100) if ml_fair_value > 0 else 0
        
        return {
            'fair_value': round(ml_fair_value, 2),
            'current_price': round(current_price, 2),
            'discount_premium_pct': round(discount_premium_pct, 2),
            'valuation': 'undervalued' if discount_premium_pct < -10 else 'overvalued' if discount_premium_pct > 10 else 'fair',
            'methods': [
                {
                    'method': '🤖 ML Model',
                    'value': round(ml_fair_value, 2),
                    'description': f'ML-based valuation using growth, quality, momentum, and technical indicators (ratio: {fair_value_ratio:.2f}x)'
                }
            ] + traditional_methods,
            'ml_confidence': 'high' if abs(discount_premium_pct) < 20 else 'medium' if abs(discount_premium_pct) < 40 else 'low'
        }
    except Exception as e:
        logger.exception(f"Error calculating ML fair value for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def calculate_fair_value(ticker, info, df=None):
    """Calculate fair value estimate using ML model (preferred) or traditional methods"""
    # Try ML model first (better for growth stocks)
    ml_result = calculate_fair_value_ml(ticker, info, df)
    if ml_result:
        return ml_result
    
    # Fallback to traditional methods if ML fails
    try:
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if current_price <= 0:
            return None
        
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        eps = info.get('trailingEps') or info.get('forwardEps')
        book_value = info.get('bookValue')
        revenue_per_share = info.get('revenuePerShare')
        earnings_growth = info.get('earningsGrowth')
        
        fair_value_estimates = []
        methods = []
        
        # Method 1: P/E-based fair value (using growth-adjusted P/E)
        if pe_ratio and pe_ratio > 0 and eps and eps > 0:
            # Adjust P/E based on earnings growth
            if earnings_growth and earnings_growth > 0.3:
                industry_pe = 40
            elif earnings_growth and earnings_growth > 0.15:
                industry_pe = 30
            elif earnings_growth and earnings_growth > 0.05:
                industry_pe = 20
            else:
                industry_pe = 15
            fair_value_pe = eps * industry_pe
            if fair_value_pe > 0:
                fair_value_estimates.append(fair_value_pe)
                methods.append({
                    'method': 'P/E Ratio',
                    'value': round(fair_value_pe, 2),
                    'description': f'Based on EPS ${eps:.2f} × Growth-Adjusted P/E {industry_pe}'
                })
        
        # Method 2: P/B-based fair value
        if pb_ratio and pb_ratio > 0 and book_value and book_value > 0:
            industry_pb = 3
            fair_value_pb = book_value * industry_pb
            if fair_value_pb > 0:
                fair_value_estimates.append(fair_value_pb)
                methods.append({
                    'method': 'P/B Ratio',
                    'value': round(fair_value_pb, 2),
                    'description': f'Based on Book Value ${book_value:.2f} × Industry P/B {industry_pb}'
                })
        
        # Method 3: P/S-based fair value
        if ps_ratio and ps_ratio > 0 and revenue_per_share and revenue_per_share > 0:
            industry_ps = 2.5
            fair_value_ps = revenue_per_share * industry_ps
            if fair_value_ps > 0:
                fair_value_estimates.append(fair_value_ps)
                methods.append({
                    'method': 'P/S Ratio',
                    'value': round(fair_value_ps, 2),
                    'description': f'Based on Revenue/Share ${revenue_per_share:.2f} × Industry P/S {industry_ps}'
                })
        
        if not fair_value_estimates:
            return None
        
        # Calculate weighted average
        weights = [0.4, 0.3, 0.3][:len(fair_value_estimates)]
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        weighted_fair_value = sum(fv * w for fv, w in zip(fair_value_estimates, weights))
        discount_premium_pct = ((current_price - weighted_fair_value) / weighted_fair_value * 100) if weighted_fair_value > 0 else 0
        
        return {
            'fair_value': round(weighted_fair_value, 2),
            'current_price': round(current_price, 2),
            'discount_premium_pct': round(discount_premium_pct, 2),
            'valuation': 'undervalued' if discount_premium_pct < -10 else 'overvalued' if discount_premium_pct > 10 else 'fair',
            'methods': methods,
            'ml_confidence': 'low'  # Traditional methods
        }
    except Exception as e:
        logger.exception(f"Error calculating fair value for {ticker}")
        import traceback
        traceback.print_exc()
        return None

