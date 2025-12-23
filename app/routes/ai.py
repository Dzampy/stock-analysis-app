"""AI/ML routes"""
from flask import Blueprint, jsonify, request, make_response
from app.services.yfinance_service import get_stock_data
from app.services.ai_service import extract_text_from_pdf, analyze_earnings_call_with_ai, analyze_news_impact_with_ai
from app.analysis.factor import (
    calculate_factor_scores,
    calculate_factor_attribution,
    calculate_factor_rotation,
    calculate_factor_momentum,
    calculate_factor_correlation,
    calculate_optimal_factor_mix,
    calculate_factor_sensitivity,
    calculate_fair_value
)
from app.services.ml_service import get_prediction_history, run_backtest
from app.utils.json_utils import clean_for_json
from app.utils.logger import logger
from app.utils.error_handler import NotFoundError, ExternalAPIError
from app.config import CACHE_TIMEOUTS

bp = Blueprint('ai', __name__)

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


@bp.route('/api/ai-recommendations/<ticker>')
@cache_if_available(CACHE_TIMEOUTS['ml_predictions'])
def get_ai_recommendations(ticker):
    """Get AI-powered stock recommendations"""
    try:
        from app.services.ml_service import generate_ai_recommendations
        
        logger.info(f"Starting AI recommendations request for {ticker.upper()}")
        recommendations = generate_ai_recommendations(ticker.upper())
        if recommendations is None:
            raise NotFoundError('Could not generate recommendations', {'ticker': ticker.upper()})
        
        # Debug: Check chart_data in recommendations
        logger.debug(f"Recommendations keys: {list(recommendations.keys())}")
        if 'chart_data' in recommendations:
            cd = recommendations['chart_data']
            logger.debug(f"Chart data type: {type(cd)}")
            if isinstance(cd, dict) and 'dates' in cd:
                logger.debug(f"Chart data dates length: {len(cd['dates']) if isinstance(cd['dates'], list) else 'not a list'}")
        
        # Always fetch chart_data directly (more reliable)
        chart_data = None
        try:
            stock_data = get_stock_data(ticker.upper(), '1y')
            if stock_data and not stock_data['history'].empty:
                df = stock_data['history']
                dates = df.index.strftime('%Y-%m-%d').tolist()
                chart_data = {
                    'dates': dates,
                    'open': [float(x) for x in df['Open'].round(2).fillna(0).tolist()],
                    'high': [float(x) for x in df['High'].round(2).fillna(0).tolist()],
                    'low': [float(x) for x in df['Low'].round(2).fillna(0).tolist()],
                    'close': [float(x) for x in df['Close'].round(2).fillna(0).tolist()],
                    'volume': [int(x) for x in df['Volume'].fillna(0).astype(int).tolist()],
                }
                logger.debug(f"Fetched chart_data: {len(chart_data['dates'])} dates")
        except Exception as e:
            logger.warning(f"Error fetching chart_data: {e}")
            chart_data = {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        # Clean recommendations - chart_data should be preserved by clean_for_json
        # Extract chart_data BEFORE clean_for_json to preserve it
        chart_data_from_rec = recommendations.get('chart_data')
        logger.debug(f"Chart data from recommendations exists: {chart_data_from_rec is not None}")
        if chart_data_from_rec:
            logger.debug(f"Chart data from rec type: {type(chart_data_from_rec)}, has dates: {'dates' in chart_data_from_rec if isinstance(chart_data_from_rec, dict) else False}")
        
        cleaned = clean_for_json(recommendations)
        
        # ALWAYS add chart_data to cleaned - use from recommendations if available
        if chart_data_from_rec and isinstance(chart_data_from_rec, dict) and 'dates' in chart_data_from_rec:
            cleaned['chart_data'] = {
                'dates': list(chart_data_from_rec.get('dates', [])),
                'open': [float(x) for x in chart_data_from_rec.get('open', [])],
                'high': [float(x) for x in chart_data_from_rec.get('high', [])],
                'low': [float(x) for x in chart_data_from_rec.get('low', [])],
                'close': [float(x) for x in chart_data_from_rec.get('close', [])],
                'volume': [int(x) for x in chart_data_from_rec.get('volume', [])],
            }
            logger.debug(f"Added chart_data from recommendations: {len(cleaned['chart_data']['dates'])} dates")
        elif chart_data and isinstance(chart_data, dict) and 'dates' in chart_data:
            cleaned['chart_data'] = {
                'dates': list(chart_data.get('dates', [])),
                'open': [float(x) for x in chart_data.get('open', [])],
                'high': [float(x) for x in chart_data.get('high', [])],
                'low': [float(x) for x in chart_data.get('low', [])],
                'close': [float(x) for x in chart_data.get('close', [])],
                'volume': [int(x) for x in chart_data.get('volume', [])],
            }
            logger.debug(f"Added chart_data from fetched: {len(cleaned['chart_data']['dates'])} dates")
        else:
            cleaned['chart_data'] = {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
            logger.warning("No chart_data available, using empty")
        
        # Final verification - ensure chart_data is in cleaned
        if 'chart_data' not in cleaned or not cleaned.get('chart_data') or len(cleaned.get('chart_data', {}).get('dates', [])) == 0:
            logger.warning("Chart_data missing or empty! Re-adding from chart_data_from_rec...")
            if chart_data_from_rec and isinstance(chart_data_from_rec, dict) and 'dates' in chart_data_from_rec:
                cleaned['chart_data'] = {
                    'dates': list(chart_data_from_rec.get('dates', [])),
                    'open': [float(x) for x in chart_data_from_rec.get('open', [])],
                    'high': [float(x) for x in chart_data_from_rec.get('high', [])],
                    'low': [float(x) for x in chart_data_from_rec.get('low', [])],
                    'close': [float(x) for x in chart_data_from_rec.get('close', [])],
                    'volume': [int(x) for x in chart_data_from_rec.get('volume', [])],
                }
                logger.debug(f"Re-added chart_data: {len(cleaned['chart_data']['dates'])} dates")
            else:
                cleaned['chart_data'] = {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        dates_len = len(cleaned.get('chart_data', {}).get('dates', []))
        logger.debug(f"Returning with {dates_len} dates, keys: {list(cleaned.keys())}")
        
        # Use json.dumps with make_response
        import json
        response_json = json.dumps(cleaned, default=str)
        parsed = json.loads(response_json)
        if 'chart_data' not in parsed:
            logger.warning("Chart_data missing in JSON! Re-adding...")
            parsed['chart_data'] = cleaned.get('chart_data', {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
            response_json = json.dumps(parsed, default=str)
        
        response = make_response(response_json)
        response.headers['Content-Type'] = 'application/json'
        logger.info(f"Successfully prepared AI recommendations for {ticker.upper()}")
        return response
    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Error in AI recommendations endpoint for {ticker}")
        raise ExternalAPIError('Failed to generate AI recommendations', service='ml_service')


@bp.route('/api/factor-analysis/<ticker>')
def get_factor_analysis(ticker):
    """Get comprehensive factor analysis (Value, Growth, Momentum, Quality) for a stock"""
    try:
        import yfinance as yf
        import time
        
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        time.sleep(0.3)
        info = stock.info
        
        if not info or 'symbol' not in info:
            return jsonify({'error': 'Stock not found'}), 404
        
        # Get historical data for momentum calculation
        hist = stock.history(period='1y')
        
        # Calculate current factor scores
        factor_data = calculate_factor_scores(ticker, info, hist)
        
        if factor_data is None:
            return jsonify({'error': 'Could not calculate factor scores'}), 500
        
        
        # Add factor attribution
        factor_data['attribution'] = calculate_factor_attribution(ticker, info, hist)
        
        # Calculate top contributors for each factor
        top_contributors = {}
        for factor_name, attributions in factor_data['attribution'].items():
            if attributions:
                sorted_contributors = sorted(
                    attributions.items(),
                    key=lambda x: x[1].get('contribution', 0),
                    reverse=True
                )
                top_contributors[factor_name] = [
                    {
                        'metric': metric,
                        'value': data['value'],
                        'contribution': data['contribution'],
                        'status': data['status']
                    }
                    for metric, data in sorted_contributors[:3]  # Top 3 contributors
                ]
        factor_data['top_contributors'] = top_contributors
        
        # Calculate factor rotation (over time)
        factor_rotation = calculate_factor_rotation(ticker)
        if factor_rotation:
            factor_data['rotation'] = factor_rotation
            
            # Calculate factor momentum (if we have previous data)
            if factor_rotation and len(factor_rotation) >= 2:
                current_period = factor_rotation[-1]
                previous_period = factor_rotation[0] if len(factor_rotation) >= 2 else factor_rotation[-1]
                
                current_scores = {
                    'value': current_period.get('value', 0),
                    'growth': current_period.get('growth', 0),
                    'momentum': current_period.get('momentum', 0),
                    'quality': current_period.get('quality', 0)
                }
                previous_scores = {
                    'value': previous_period.get('value', 0),
                    'growth': previous_period.get('growth', 0),
                    'momentum': previous_period.get('momentum', 0),
                    'quality': previous_period.get('quality', 0)
                }
                
                factor_momentum = calculate_factor_momentum(current_scores, previous_scores)
                if factor_momentum:
                    factor_data['momentum_analysis'] = factor_momentum
        
        # Calculate factor correlation
        if 'rotation' in factor_data:
            factor_correlation = calculate_factor_correlation(factor_data['rotation'])
            if factor_correlation:
                factor_data['correlation_matrix'] = factor_correlation
        
        # Calculate optimal factor mix
        optimal_mix = calculate_optimal_factor_mix(ticker, info, hist)
        if optimal_mix:
            factor_data['optimal_mix'] = optimal_mix
        
        # Calculate factor sensitivity
        factor_sensitivity = calculate_factor_sensitivity(ticker, info, hist)
        if factor_sensitivity:
            factor_data['sensitivity_analysis'] = factor_sensitivity
        
        # Calculate fair value
        fair_value_data = calculate_fair_value(ticker, info, hist)
        if fair_value_data:
            factor_data['fair_value'] = fair_value_data
        
        return jsonify(clean_for_json(factor_data))
    
    except Exception as e:
        logger.exception(f"Error in factor analysis endpoint for {ticker}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to get factor analysis: {str(e)}'}), 500


@bp.route('/api/ml-prediction-history/<ticker>')
def get_ml_prediction_history(ticker):
    """Get ML prediction history for a ticker"""
    try:
        if not get_prediction_history:
            return jsonify({'error': 'ML prediction history service not available'}), 500
        
        days = request.args.get('days', 30, type=int)
        history = get_prediction_history(ticker.upper(), days)
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/backtest/<ticker>')
def get_backtest_results(ticker):
    """Get backtest results for ML predictions"""
    try:
        from app.services.ml_service import run_backtest
        
        ticker = ticker.upper()
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)
        
        logger.info(f"Running backtest for {ticker} from {start_date} to {end_date}")
        
        result = run_backtest(ticker, start_date=start_date, end_date=end_date)
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error')
            return jsonify({'error': error_msg}), 400
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        logger.exception(f"Error in backtest endpoint for {ticker}: {e}")
        return jsonify({'error': f'Failed to run backtest: {str(e)}'}), 500


@bp.route('/api/ml-score-history/<ticker>')
def get_ml_score_history(ticker):
    """Get ML prediction score history for a ticker"""
    try:
        if not get_prediction_history:
            return jsonify({'error': 'ML prediction history service not available'}), 500
        
        days = request.args.get('days', 30, type=int)
        history = get_prediction_history(ticker.upper(), days)
        
        # Extract score history
        score_history = []
        for entry in history:
            if 'score' in entry:
                score_history.append({
                    'date': entry.get('date'),
                    'score': entry.get('score'),
                    'current_price': entry.get('current_price')
                })
        
        return jsonify({'score_history': score_history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ai-investment-thesis/<ticker>')
def get_ai_investment_thesis(ticker):
    """Get AI-generated investment thesis for a stock"""
    try:
        from app.services.ai_service import generate_investment_thesis_with_ai
        
        thesis = generate_investment_thesis_with_ai(ticker.upper())
        
        if not thesis['success']:
            return jsonify({'error': thesis.get('error', 'Failed to generate investment thesis')}), 500
        
        return jsonify(clean_for_json(thesis))
    except Exception as e:
        logger.exception(f"Error in get_ai_investment_thesis endpoint for {ticker}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@bp.route('/api/clear-ml-cache', methods=['POST'])
def clear_ml_cache():
    """Clear ML model cache - useful when model structure changes"""
    from app.services.ml_service import clear_ml_cache as ml_clear_cache
    result = ml_clear_cache()
    return jsonify(result)


@bp.route('/api/analyze-earnings-call', methods=['POST'])
def analyze_earnings_call():
    """Upload and analyze earnings call presentation PDF"""
    try:
        import os
        from app.config import GEMINI_AVAILABLE
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Google Gemini API key not configured'}), 500
        
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        ticker = request.form.get('ticker', '').strip().upper()
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Check file size (max 10MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'File size exceeds 10MB limit'}), 400
        
        # Extract text from PDF
        extraction_result = extract_text_from_pdf(file)
        
        if not extraction_result['success']:
            return jsonify({'error': f'Failed to extract text: {extraction_result.get("error", "Unknown error")}'}), 500
        
        extracted_text = extraction_result['text']
        
        if len(extracted_text.strip()) < 100:
            return jsonify({'error': 'PDF appears to be empty or contains no extractable text'}), 400
        
        # Analyze with AI
        analysis_result = analyze_earnings_call_with_ai(extracted_text, ticker)
        
        if not analysis_result['success']:
            return jsonify({'error': f'AI analysis failed: {analysis_result.get("error", "Unknown error")}'}), 500
        
        # Return results
        return jsonify(clean_for_json({
            'success': True,
            'ticker': ticker,
            'pages_extracted': extraction_result['pages'],
            'text_length': len(extracted_text),
            'ai_summary': analysis_result['summary'],
            'structured_data': analysis_result['structured_data'],
            'key_metrics': analysis_result.get('key_metrics', {}),
            'highlights': analysis_result.get('highlights', []),
            'risks': analysis_result.get('risks', [])
        }))
        
    except Exception as e:
        logger.exception(f"Error in analyze-earnings-call endpoint")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

