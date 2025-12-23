"""Analyst and insider routes"""
from flask import Blueprint, jsonify
from app.services.finviz_service import get_finviz_analyst_ratings, get_finviz_insider_trading
from app.services.analyst_service import get_marketbeat_insider_trading, get_tipranks_insider_trading, get_sec_api_insider_trading
from app.utils.json_utils import clean_for_json
from app.utils.logger import logger
from app.utils.error_handler import NotFoundError, ExternalAPIError
import yfinance as yf
import pandas as pd
import time

bp = Blueprint('analyst', __name__)


@bp.route('/api/analyst-data/<ticker>')
def get_analyst_data(ticker):
    """Get analyst ratings and price targets"""
    try:
        from app.services.analyst_service import get_benzinga_analyst_ratings, get_marketbeat_analyst_ratings
        
        ticker_upper = ticker.upper()
        stock = yf.Ticker(ticker_upper)
        time.sleep(0.3)
        info = stock.info
        
        if not info or 'symbol' not in info:
            return jsonify({'error': 'Stock not found'}), 404
        
        # Try to get recommendations from multiple sources
        recommendations = []
        try:
            # Try Finviz first
            finviz_recs = get_finviz_analyst_ratings(ticker_upper)
            if finviz_recs and len(finviz_recs) > 0:
                recommendations = finviz_recs
                logger.info(f"Found {len(recommendations)} recommendations from Finviz for {ticker}")
            else:
                # Try MarketBeat
                mb_recs = get_marketbeat_analyst_ratings(ticker_upper)
                if mb_recs and len(mb_recs) > 0:
                    recommendations = mb_recs
                    logger.info(f"Found {len(recommendations)} recommendations from MarketBeat for {ticker}")
                else:
                    # Try Benzinga
                    bz_recs = get_benzinga_analyst_ratings(ticker_upper)
                    if bz_recs and len(bz_recs) > 0:
                        recommendations = bz_recs
                        logger.info(f"Found {len(recommendations)} recommendations from Benzinga for {ticker}")
        except Exception as e:
            logger.warning(f"Error getting recommendations: {str(e)}")
            recommendations = []
        
        # Get recommendation summary
        recommendation_summary = None
        try:
            rec_summary = stock.recommendations_summary
            if rec_summary is not None and not rec_summary.empty:
                latest = rec_summary.iloc[-1] if len(rec_summary) > 0 else None
                if latest is not None:
                    recommendation_summary = {
                        'strong_buy': int(latest.get('strongBuy', 0)) if pd.notna(latest.get('strongBuy', 0)) else 0,
                        'buy': int(latest.get('buy', 0)) if pd.notna(latest.get('buy', 0)) else 0,
                        'hold': int(latest.get('hold', 0)) if pd.notna(latest.get('hold', 0)) else 0,
                        'sell': int(latest.get('sell', 0)) if pd.notna(latest.get('sell', 0)) else 0,
                        'strong_sell': int(latest.get('strongSell', 0)) if pd.notna(latest.get('strongSell', 0)) else 0
                    }
        except Exception:
            pass
        
        # Get price targets
        target_mean_price = info.get('targetMeanPrice')
        target_high_price = info.get('targetHighPrice')
        target_low_price = info.get('targetLowPrice')
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Calculate upside/downside
        upside_pct = None
        if target_mean_price and current_price:
            upside_pct = ((target_mean_price - current_price) / current_price) * 100
        
        # Also add individual price targets from recommendations if available
        individual_targets = []
        if recommendations:
            for rec in recommendations:
                if rec.get('target_price'):
                    individual_targets.append({
                        'firm': rec.get('firm', 'N/A'),
                        'target_price': rec.get('target_price'),
                        'rating': rec.get('to_grade', 'N/A'),
                        'date': rec.get('date', 'N/A')
                    })
        
        analyst_data = {
            'recommendations': recommendations[:20] if recommendations else [],
            'recommendation_summary': recommendation_summary,
            'target_mean_price': target_mean_price,
            'target_high_price': target_high_price,
            'target_low_price': target_low_price,
            'current_price': current_price,
            'upside_pct': upside_pct,
            'number_of_analysts': info.get('numberOfAnalystOpinions'),
            'individual_targets': individual_targets[:10] if individual_targets else []
        }
        
        return jsonify(clean_for_json(analyst_data))
        
    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Error fetching analyst data for {ticker}: {str(e)}")
        # Return empty data structure instead of raising error, so frontend can still display the page
        return jsonify(clean_for_json({
            'recommendations': [],
            'recommendation_summary': None,
            'target_mean_price': None,
            'target_high_price': None,
            'target_low_price': None,
            'current_price': None,
            'upside_pct': None,
            'number_of_analysts': None,
            'individual_targets': [],
            'error': f'Unable to fetch analyst data: {str(e)}'
        })), 200  # Return 200 so frontend doesn't show error banner


@bp.route('/api/insider-trading/<ticker>')
def get_insider_trading(ticker):
    """Get insider trading activity from SEC API (primary), Finviz/MarketBeat (fallback), yfinance (last resort)"""
    try:
        ticker_upper = ticker.upper()
        time.sleep(0.3)  # Rate limiting
        
        insider_transactions = []
        
        # Try SEC API first (most reliable and official source)
        sec_data = get_sec_api_insider_trading(ticker_upper)
        if sec_data and len(sec_data) > 0:
            logger.info(f"SEC API returned {len(sec_data)} transactions for {ticker_upper}")
            insider_transactions = sec_data
        else:
            # Fallback to Finviz
            logger.debug(f"SEC API returned no data for {ticker_upper}, trying Finviz")
            finviz_data = get_finviz_insider_trading(ticker_upper)
            if finviz_data and len(finviz_data) > 0:
                logger.info(f"Finviz returned {len(finviz_data)} transactions for {ticker_upper}")
                insider_transactions = finviz_data
            else:
                # Fallback to MarketBeat
                logger.debug(f"Finviz returned no data for {ticker_upper}, trying MarketBeat")
                marketbeat_data = get_marketbeat_insider_trading(ticker_upper)
                if marketbeat_data and len(marketbeat_data) > 0:
                    logger.info(f"MarketBeat returned {len(marketbeat_data)} transactions for {ticker_upper}")
                    insider_transactions = marketbeat_data
                else:
                    # Last resort: yfinance
                    logger.debug(f"MarketBeat returned no data for {ticker_upper}, trying yfinance fallback")
                    try:
                        stock = yf.Ticker(ticker_upper)
                        insider_df = stock.insider_transactions
                        if insider_df is not None and not insider_df.empty:
                            for idx, row in insider_df.tail(30).iterrows():
                                try:
                                    row_dict = row.to_dict()
                                    
                                    transaction_type = None
                                    text = str(row_dict.get('Text', '')).lower() if row_dict.get('Text') else ''
                                    
                                    if 'sale' in text or 'sell' in text:
                                        transaction_type = 'sell'
                                    elif ('purchase' in text or 'buy' in text or 'acquisition' in text or
                                          'option exercise' in text.lower() or 'exercise' in text.lower() or
                                          'grant' in text.lower() or 'award' in text.lower() or
                                          'conversion' in text.lower() or 'convert' in text.lower()):
                                        transaction_type = 'buy'
                                    
                                    value = None
                                    val = row_dict.get('Value')
                                    if val is not None and pd.notna(val):
                                        try:
                                            value = float(val)
                                        except:
                                            pass
                                    
                                    shares = None
                                    sh = row_dict.get('Shares')
                                    if sh is not None and pd.notna(sh):
                                        try:
                                            shares = int(sh)
                                        except:
                                            pass
                                    
                                    insider = 'N/A'
                                    ins = row_dict.get('Insider')
                                    if ins is not None and pd.notna(ins):
                                        insider = str(ins)
                                    
                                    date_str = 'N/A'
                                    date_val = row_dict.get('Start Date')
                                    if date_val is not None and pd.notna(date_val):
                                        if hasattr(date_val, 'strftime'):
                                            date_str = date_val.strftime('%Y-%m-%d')
                                        else:
                                            date_str = str(date_val)
                                    
                                    if transaction_type and value and value > 0:
                                        insider_transactions.append({
                                            'date': date_str,
                                            'transaction_type': transaction_type,
                                            'value': value,
                                            'shares': shares,
                                            'insider': insider,
                                            'position': 'N/A',
                                            'text': str(row_dict.get('Text', ''))
                                        })
                                except Exception as row_error:
                                    logger.warning(f"Error parsing yfinance row: {str(row_error)}")
                                    continue
                    except Exception as yf_error:
                        logger.warning(f"Error getting yfinance insider data: {str(yf_error)}")
        
        return jsonify(clean_for_json({
            'ticker': ticker_upper,
            'transactions': insider_transactions[:30] if insider_transactions else [],
            'total': len(insider_transactions) if insider_transactions else 0
        }))
        
    except Exception as e:
        logger.exception(f"Error in insider trading endpoint for {ticker}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch insider trading: {str(e)}'}), 500


@bp.route('/api/institutional-analysis/<ticker>')
def get_institutional_analysis(ticker):
    """Get comprehensive institutional analysis (ownership, flow, retail indicators, whales)"""
    try:
        from app.services.yfinance_service import get_institutional_ownership, get_retail_activity_indicators
        from app.services.sec_service import get_institutional_flow, get_whale_watching
        
        ticker = ticker.upper()
        
        ownership = get_institutional_ownership(ticker)
        flow = get_institutional_flow(ticker)
        retail_indicators = get_retail_activity_indicators(ticker)
        whales = get_whale_watching(ticker)
        
        return jsonify(clean_for_json({
            'ownership': ownership,
            'flow': flow,
            'retail_indicators': retail_indicators,
            'whales': whales
        }))
    
    except Exception as e:
        logger.exception(f"Error in institutional analysis endpoint for {ticker}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to get institutional analysis: {str(e)}'}), 500


@bp.route('/api/earnings-calendar')
def get_earnings_calendar():
    """Get earnings calendar for popular stocks"""
    try:
        import yfinance as yf
        import pandas as pd
        import time
        from datetime import datetime, timedelta
        
        # Reduced list of most popular stocks to speed up loading
        popular_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC',
            'ADBE', 'PYPL', 'CMCSA', 'NKE', 'XOM', 'VZ', 'CVX', 'MRK', 'PFE'
        ]
        
        earnings_calendar = []
        today = datetime.now().date()
        future_date = today + timedelta(days=90)  # Next 90 days
        
        # Process stocks with minimal delays
        for i, ticker in enumerate(popular_tickers):
            try:
                stock = yf.Ticker(ticker)
                
                # Get earnings dates first (faster)
                earnings_dates = None
                try:
                    earnings_dates = stock.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        # Get company info only if we have earnings dates
                        info = {}
                        try:
                            info = stock.info
                        except:
                            info = {'longName': ticker, 'sector': 'N/A'}
                        
                        # Process each earnings date
                        for date_idx, row in earnings_dates.iterrows():
                            try:
                                earnings_date = pd.Timestamp(date_idx)
                                if hasattr(earnings_date, 'date'):
                                    earnings_date = earnings_date.date()
                                else:
                                    earnings_date = earnings_date.to_pydatetime().date()
                                
                                # Include future earnings and also recent past (last 7 days) for context
                                days_diff = (earnings_date - today).days
                                if days_diff >= -7 and days_diff <= 90:
                                    eps_estimate = None
                                    eps_reported = None
                                    surprise_pct = None
                                    
                                    if 'EPS Estimate' in row.index:
                                        eps_estimate = float(row['EPS Estimate']) if pd.notna(row['EPS Estimate']) else None
                                    if 'Reported EPS' in row.index:
                                        eps_reported = float(row['Reported EPS']) if pd.notna(row['Reported EPS']) else None
                                    if 'Surprise(%)' in row.index:
                                        surprise_pct = float(row['Surprise(%)']) if pd.notna(row['Surprise(%)']) else None
                                    
                                    earnings_calendar.append({
                                        'ticker': ticker,
                                        'company_name': info.get('longName', ticker),
                                        'sector': info.get('sector', 'N/A'),
                                        'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                                        'earnings_date_display': earnings_date.strftime('%B %d, %Y'),
                                        'eps_estimate': eps_estimate,
                                        'eps_reported': eps_reported,
                                        'surprise_pct': surprise_pct,
                                        'is_past': earnings_date < today
                                    })
                            except Exception as e:
                                logger.warning(f"Error processing earnings date for {ticker}: {str(e)}")
                                continue
                except Exception as e:
                    logger.warning(f"Error fetching earnings_dates for {ticker}: {str(e)}")
                    continue
                
                # Small delay only every 5 stocks to avoid rate limiting
                if (i + 1) % 5 == 0:
                    time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Error fetching earnings for {ticker}: {str(e)}")
                continue
        
        # Sort by earnings date
        earnings_calendar.sort(key=lambda x: x['earnings_date'])
        
        return jsonify({
            'earnings': clean_for_json(earnings_calendar),
            'total': len(earnings_calendar)
        })
        
    except Exception as e:
        logger.exception(f"Error in earnings calendar")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch earnings calendar: {str(e)}', 'earnings': [], 'total': 0}), 500

