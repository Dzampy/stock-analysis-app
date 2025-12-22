"""Stock analysis routes"""
from flask import Blueprint, jsonify, request, render_template
from app.services.yfinance_service import get_stock_data, get_earnings_qoq
from app.analysis.technical import calculate_technical_indicators, get_volume_analysis
from app.analysis.fundamental import calculate_metrics
from app.services.finviz_service import get_short_interest_from_finviz, get_short_interest_history
from app.services.news_service import get_stock_news
from app.services.ai_service import generate_news_summary
from app.utils.json_utils import clean_for_json

bp = Blueprint('stock', __name__)


@bp.route('/')
def index():
    """Main page - render index.html"""
    return render_template('index.html')


@bp.route('/api/stock/<ticker>')
def get_stock(ticker):
    """Get stock data with technical indicators and metrics"""
    period = request.args.get('period', '1y')
    
    try:
        data = get_stock_data(ticker.upper(), period)
        
        if data is None:
            # Check if it's an intraday timeframe that might not be available
            intraday_timeframes = ['1m', '5m', '15m', '1h', '4h']
            if period in intraday_timeframes:
                error_msg = f'Intraday data ({period}) may not be available for this stock. Try a different timeframe or stock like AAPL, MSFT, or TSLA.'
            else:
                error_msg = 'Stock not found or data unavailable'
            return jsonify({'error': error_msg}), 404
    except Exception as e:
        print(f"Error in get_stock endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        intraday_timeframes = ['1m', '5m', '15m', '1h', '4h']
        if period in intraday_timeframes:
            error_msg = f'Error fetching intraday data ({period}): {str(e)}. Intraday data may not be available for this stock.'
        else:
            error_msg = f'Error fetching stock data: {str(e)}'
        return jsonify({'error': error_msg}), 500
    
    df = data['history']
    info = data['info']
    
    # Prepare chart data
    # For intraday data, include time in date format
    is_intraday = period in ['1m', '5m', '15m', '1h', '4h']
    if is_intraday:
        # Include time for intraday data: YYYY-MM-DD HH:MM:SS
        dates = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    else:
        # Just date for daily/weekly/monthly data
        dates = df.index.strftime('%Y-%m-%d').tolist()
    
    chart_data = {
        'dates': dates,
        'open': df['Open'].round(2).fillna(0).tolist(),
        'high': df['High'].round(2).fillna(0).tolist(),
        'low': df['Low'].round(2).fillna(0).tolist(),
        'close': df['Close'].round(2).fillna(0).tolist(),
        'volume': df['Volume'].fillna(0).astype(int).tolist(),
    }
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(df)
    
    # Calculate metrics
    metrics = calculate_metrics(df, info)
    
    # Get earnings QoQ data
    # #region agent log
    print(f"[DEBUG] /api/stock/{ticker}: Calling get_earnings_qoq")
    # #endregion
    earnings_qoq = get_earnings_qoq(ticker.upper())
    # #region agent log
    print(f"[DEBUG] /api/stock/{ticker}: get_earnings_qoq returned: {earnings_qoq is not None}")
    if earnings_qoq:
        print(f"[DEBUG] /api/stock/{ticker}: earnings_qoq['eps'] = {earnings_qoq.get('eps', [])[:4]}")
    # #endregion
    
    # Get news with sentiment analysis
    news = get_stock_news(ticker.upper(), max_news=10)
    
    # Generate AI news summary
    news_summary = generate_news_summary(news, ticker.upper())
    
    # Get short interest data
    short_interest = get_short_interest_from_finviz(ticker.upper())
    
    # Get short interest history
    short_interest_history = get_short_interest_history(ticker.upper())
    if short_interest and short_interest_history:
        short_interest['history'] = short_interest_history
    
    # Get volume analysis
    volume_analysis = get_volume_analysis(ticker.upper())
    
    # Company info
    company_info = {
        'name': info.get('longName', ticker),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'description': info.get('longBusinessSummary', ''),
    }
    
    # Clean all data for JSON (replace NaN with None)
    # #region agent log
    if earnings_qoq:
        print(f"[DEBUG] /api/stock/{ticker}: Before clean_for_json - earnings_qoq['eps'] = {earnings_qoq.get('eps', [])[:4]}")
    # #endregion
    
    response_data = {
        'ticker': ticker.upper(),
        'chart_data': clean_for_json(chart_data),
        'indicators': clean_for_json(indicators),
        'metrics': clean_for_json(metrics),
        'company_info': clean_for_json(company_info),
        'earnings_qoq': clean_for_json(earnings_qoq) if earnings_qoq else None,
        'news': clean_for_json(news),
        'news_summary': clean_for_json(news_summary),
        'short_interest': clean_for_json(short_interest) if short_interest else None,
        'volume_analysis': clean_for_json(volume_analysis) if volume_analysis else None
    }
    
    # #region agent log
    if response_data.get('earnings_qoq'):
        print(f"[DEBUG] /api/stock/{ticker}: After clean_for_json - earnings_qoq['eps'] = {response_data['earnings_qoq'].get('eps', [])[:4]}")
        print(f"[DEBUG] /api/stock/{ticker}: Full earnings_qoq['eps'] = {response_data['earnings_qoq'].get('eps', [])}")
        print(f"[DEBUG] /api/stock/{ticker}: earnings_qoq['eps'] None count = {sum(1 for x in response_data['earnings_qoq'].get('eps', []) if x is None)}")
    else:
        print(f"[DEBUG] /api/stock/{ticker}: earnings_qoq is None or missing after clean_for_json")
    # #endregion
    
    return jsonify(response_data)

