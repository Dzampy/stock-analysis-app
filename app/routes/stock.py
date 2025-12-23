"""Stock analysis routes"""
from flask import Blueprint, jsonify, request, render_template
from app.services.yfinance_service import get_stock_data, get_earnings_qoq
from app.analysis.technical import calculate_technical_indicators, get_volume_analysis
from app.analysis.fundamental import calculate_metrics
from app.services.finviz_service import get_short_interest_from_finviz, get_short_interest_history
from app.services.news_service import get_stock_news
from app.services.ai_service import generate_news_summary
from app.utils.json_utils import clean_for_json
from app.utils.logger import logger
from app.utils.error_handler import NotFoundError, ExternalAPIError, create_error_response
from app.config import CACHE_TIMEOUTS

bp = Blueprint('stock', __name__)

# Import cache for route caching
try:
    from app import cache
    CACHE_AVAILABLE = cache is not None
except (ImportError, RuntimeError, AttributeError):
    CACHE_AVAILABLE = False
    cache = None

# Helper function to conditionally apply cache decorator
def cache_if_available(timeout):
    """Return cache decorator if cache is available, else return no-op decorator"""
    if CACHE_AVAILABLE and cache:
        return cache.cached(timeout=timeout)
    else:
        # Return no-op decorator that does nothing
        def noop_decorator(func):
            return func
        return noop_decorator


@bp.route('/')
def index():
    """Main page - render index.html"""
    from flask import current_app, make_response
    from pathlib import Path
    import os
    
    # Try to find template using multiple strategies
    template_path = None
    template_content = None
    
    # Strategy 1: Use Flask's template folder
    if current_app.template_folder:
        template_path = Path(current_app.template_folder) / 'index.html'
        if template_path.exists():
            logger.info(f"Found template at Flask template_folder: {template_path}")
            try:
                response = make_response(render_template('index.html'))
                # Don't cache HTML - always serve fresh
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                return response
            except Exception as e:
                logger.warning(f"render_template failed: {e}")
    
    # Strategy 2: Try absolute path from app.py location
    base_dir = Path(__file__).parent.parent.parent  # Go up from app/routes/stock.py to project root
    alt_template_path = base_dir / 'templates' / 'index.html'
    if alt_template_path.exists():
        logger.info(f"Found template at alternative path: {alt_template_path}")
        try:
            # Read template directly and use render_template_string
            with open(alt_template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            from flask import render_template_string, make_response
            response = make_response(render_template_string(template_content))
            # Don't cache HTML - always serve fresh
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        except Exception as e:
            logger.warning(f"Direct template read failed: {e}")
    
    # Strategy 3: Try common Render paths
    render_paths = [
        Path('/opt/render/project/src/templates/index.html'),
        Path('/opt/render/project/src') / 'templates' / 'index.html',
        Path.cwd() / 'templates' / 'index.html',
    ]
    for render_path in render_paths:
        if render_path.exists():
            logger.info(f"Found template at Render path: {render_path}")
            try:
                with open(render_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                from flask import render_template_string, make_response
                response = make_response(render_template_string(template_content))
                # Don't cache HTML - always serve fresh
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                return response
            except Exception as e:
                logger.warning(f"Render path template read failed: {e}")
    
    # Fallback: return simple HTML if template not found
    logger.error(f"Could not find index.html template. Tried: {template_path}, {alt_template_path}, {render_paths}")
    logger.info(f"Template folder: {current_app.template_folder}")
    logger.info(f"Root path: {current_app.root_path}")
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Analysis Platform</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; background: #1a1a1a; color: #fff; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #4CAF50; }
            .error { color: #f44336; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Stock Analysis Platform</h1>
            <p>Application is running successfully! 🎉</p>
            <p class="error">Note: Template file not found. This is a fallback page.</p>
            <p>Please check the logs for template path information.</p>
        </div>
    </body>
    </html>
    """, 200


@bp.route('/api/stock/<ticker>')
@cache_if_available(CACHE_TIMEOUTS['yfinance'])
def get_stock(ticker):
    """Get stock data with technical indicators and metrics"""
    period = request.args.get('period', '1y')
    
    try:
        logger.info(f"Fetching stock data for {ticker.upper()} with period {period}")
        data = get_stock_data(ticker.upper(), period)
        
        if data is None:
            # Check if it's an intraday timeframe that might not be available
            intraday_timeframes = ['1m', '5m', '15m', '1h', '4h']
            if period in intraday_timeframes:
                raise NotFoundError(
                    f'Intraday data ({period}) may not be available for this stock. Try a different timeframe or stock like AAPL, MSFT, or TSLA.',
                    {'ticker': ticker.upper(), 'period': period, 'type': 'intraday'}
                )
            else:
                raise NotFoundError(
                    'Stock not found or data unavailable',
                    {'ticker': ticker.upper(), 'period': period}
                )
    except NotFoundError:
        raise  # Re-raise NotFoundError to be handled by error handler
    except Exception as e:
        logger.exception(f"Error fetching stock data for {ticker.upper()}: {str(e)}")
        raise ExternalAPIError(
            f'Error fetching stock data: {str(e)}',
            service='yfinance',
            details={'ticker': ticker.upper(), 'period': period}
        )
    
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
    logger.debug(f"Calling get_earnings_qoq for {ticker.upper()}")
    earnings_qoq = get_earnings_qoq(ticker.upper())
    if earnings_qoq:
        logger.debug(f"Earnings QoQ data retrieved for {ticker.upper()}")
    
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
    
    logger.info(f"Successfully prepared stock data response for {ticker.upper()}")
    return jsonify(response_data)

