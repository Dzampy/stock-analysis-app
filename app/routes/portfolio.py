"""Portfolio routes"""
from flask import Blueprint, jsonify, request
import yfinance as yf
from app.utils.json_utils import clean_for_json

bp = Blueprint('portfolio', __name__)


@bp.route('/api/portfolio-data', methods=['POST'])
def get_portfolio_data():
    """Calculate portfolio performance for given positions"""
    try:
        positions = request.json.get('positions', [])
        if not positions:
            return jsonify({'error': 'No positions provided'}), 400
        
        results = []
        total_cost = 0
        total_value = 0
        
        for pos in positions:
            try:
                ticker = pos.get('ticker', '').upper().strip()
                if not ticker:
                    continue
                
                shares = float(pos.get('shares', 0))
                purchase_price = float(pos.get('purchase_price', 0))
                purchase_date = pos.get('purchase_date', '')
                
                if shares <= 0 or purchase_price <= 0:
                    continue
                
                cost_basis = shares * purchase_price
                
                # Get current price
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1d')
                
                if hist.empty:
                    # Try to get info for current price
                    info = stock.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or purchase_price
                else:
                    current_price = float(hist['Close'].iloc[-1])
                
                current_value = shares * current_price
                pnl = current_value - cost_basis
                pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                # Get sector info
                try:
                    info = stock.info
                    sector = info.get('sector', 'N/A')
                    company_name = info.get('longName', ticker)
                except:
                    sector = 'N/A'
                    company_name = ticker
                
                results.append({
                    'ticker': ticker,
                    'company_name': company_name,
                    'shares': shares,
                    'purchase_price': purchase_price,
                    'purchase_date': purchase_date,
                    'current_price': current_price,
                    'cost_basis': cost_basis,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'sector': sector
                })
                
                total_cost += cost_basis
                total_value += current_value
                
            except Exception as e:
                print(f"Error processing position {pos.get('ticker', 'unknown')}: {str(e)}")
                continue
        
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        return jsonify(clean_for_json({
            'positions': results,
            'summary': {
                'total_cost': total_cost,
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent,
                'position_count': len(results)
            }
        }))
        
    except Exception as e:
        print(f"Error calculating portfolio data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to calculate portfolio data: {str(e)}'}), 500


@bp.route('/api/analyze-watchlist-summary', methods=['POST'])
def analyze_watchlist_summary():
    """Analyze watchlist and provide summary"""
    try:
        from app.services.portfolio_service import analyze_watchlist_news_with_ai
        
        data = request.json
        all_news_data = data.get('news_data', [])
        
        if not all_news_data or len(all_news_data) == 0:
            return jsonify({'error': 'No news data provided'}), 400
        
        # Analyze with AI
        result = analyze_watchlist_news_with_ai(all_news_data)
        
        if not result.get('success'):
            return jsonify({'error': result.get('error', 'AI analysis failed')}), 500
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        print(f"Error analyzing watchlist: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to analyze watchlist: {str(e)}'}), 500


@bp.route('/api/alerts-dashboard', methods=['POST'])
def get_alerts_dashboard():
    """Get all alerts (earnings, news) for watchlist stocks"""
    try:
        import yfinance as yf
        import pandas as pd
        import time
        from datetime import datetime, timedelta
        from app.services.news_service import get_stock_news
        
        data = request.get_json()
        watchlist = data.get('watchlist', [])
        
        if not watchlist:
            return jsonify({
                'earnings_alerts': [],
                'news_alerts': [],
                'total': 0
            })
        
        today = datetime.now().date()
        earnings_alerts = []
        news_alerts = []
        
        # Process each ticker in watchlist
        for i, ticker in enumerate(watchlist[:20]):  # Limit to 20 to avoid timeout
            try:
                stock = yf.Ticker(ticker)
                
                # Check for upcoming earnings (next 90 days for better coverage)
                try:
                    earnings_dates = stock.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        info = {}
                        try:
                            info = stock.info
                        except:
                            info = {'longName': ticker}
                        
                        # Try to find next earnings date
                        found_earnings = False
                        for date_idx, row in earnings_dates.iterrows():
                            try:
                                earnings_date = pd.Timestamp(date_idx)
                                if hasattr(earnings_date, 'date'):
                                    earnings_date = earnings_date.date()
                                else:
                                    earnings_date = earnings_date.to_pydatetime().date()
                                
                                days_diff = (earnings_date - today).days
                                if -7 <= days_diff <= 90:
                                    earnings_alerts.append({
                                        'ticker': ticker,
                                        'company_name': info.get('longName', ticker),
                                        'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                                        'earnings_date_display': earnings_date.strftime('%B %d, %Y'),
                                        'days_until': days_diff,
                                        'priority': 'high' if days_diff <= 2 else ('medium' if days_diff <= 7 else 'low'),
                                        'type': 'earnings'
                                    })
                                    found_earnings = True
                                    break
                            except Exception as e:
                                print(f"Error processing earnings date for {ticker}: {str(e)}")
                                continue
                except Exception as e:
                    print(f"Error fetching earnings_dates for {ticker}: {str(e)}")
                    pass
                
                # Get recent news with high sentiment
                try:
                    news = get_stock_news(ticker, max_news=5)
                    for article in news:
                        sentiment_score = article.get('sentiment_score', 0)
                        if abs(sentiment_score) > 0.3:  # Strong sentiment
                            news_alerts.append({
                                'ticker': ticker,
                                'title': article.get('title', ''),
                                'summary': article.get('summary', '')[:200],
                                'link': article.get('link', ''),
                                'publisher': article.get('publisher', ''),
                                'published': article.get('published', ''),
                                'sentiment': article.get('sentiment', 'neutral'),
                                'sentiment_score': sentiment_score,
                                'priority': 'high' if abs(sentiment_score) > 0.5 else 'medium',
                                'type': 'news'
                            })
                except:
                    pass
                
                # Small delay to avoid rate limiting
                if (i + 1) % 3 == 0:
                    time.sleep(0.3)
                    
            except Exception as e:
                print(f"Error processing alerts for {ticker}: {str(e)}")
                continue
        
        # Sort earnings alerts by date
        earnings_alerts.sort(key=lambda x: x['earnings_date'])
        
        # Sort news alerts by sentiment score (absolute value)
        news_alerts.sort(key=lambda x: abs(x.get('sentiment_score', 0)), reverse=True)
        
        return jsonify({
            'earnings_alerts': clean_for_json(earnings_alerts),
            'news_alerts': clean_for_json(news_alerts[:20]),  # Limit to top 20 news
            'total': len(earnings_alerts) + len(news_alerts)
        })
        
    except Exception as e:
        print(f"Error in alerts dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to fetch alerts: {str(e)}',
            'earnings_alerts': [],
            'news_alerts': [],
            'total': 0
        }), 500


