"""News routes"""
from flask import Blueprint, jsonify, request
from app.services.news_service import get_stock_news
from app.services.ai_service import analyze_news_impact_with_ai
from app.utils.json_utils import clean_for_json
from app.utils.logger import logger
from app.utils.error_handler import ValidationError, ExternalAPIError

bp = Blueprint('news', __name__)


@bp.route('/api/news/<ticker>')
def get_news_for_ticker(ticker):
    """Get news with impact analysis for a ticker"""
    try:
        from app.services.news_service import calculate_historical_news_impact_patterns
        
        ticker = ticker.upper()
        news_list = get_stock_news(ticker, max_news=20)
        
        if not news_list:
            return jsonify({'news': [], 'historical_patterns': None})
        
        # Calculate historical patterns
        historical_patterns = calculate_historical_news_impact_patterns(ticker, news_list)
        
        return jsonify(clean_for_json({
            'news': news_list,
            'historical_patterns': historical_patterns
        }))
        
    except Exception as e:
        logger.exception(f"Error in news endpoint for {ticker}")
        raise ExternalAPIError('Failed to fetch news', service='news_service')


@bp.route('/api/analyze-news-impact', methods=['POST'])
def analyze_news_impact():
    """Analyze how a news article might impact stock price using AI"""
    try:
        from app.config import GEMINI_AVAILABLE
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Google Gemini API key not configured'}), 500
        
        
        data = request.json
        news_title = data.get('title', '')
        news_summary = data.get('summary', '')
        news_content = data.get('content', '')
        ticker = data.get('ticker', '')
        
        if not news_title and not news_summary:
            raise ValidationError('News title or summary required', {'provided': {'title': bool(news_title), 'summary': bool(news_summary)}})
        
        # Combine news content
        news_text = f"{news_title}\n\n{news_summary}\n\n{news_content}" if news_content else f"{news_title}\n\n{news_summary}"
        
        # Analyze with AI
        analysis_result = analyze_news_impact_with_ai(news_text, ticker)
        
        if not analysis_result['success']:
            return jsonify({'error': analysis_result.get('error', 'AI analysis failed')}), 500
        
        return jsonify(clean_for_json(analysis_result))
        
    except ValidationError:
        raise
    except Exception as e:
        logger.exception(f"Error in analyze-news-impact endpoint")
        raise ExternalAPIError('Failed to analyze news impact', service='ai_service')


@bp.route('/api/social-sentiment/<ticker>')
def get_social_sentiment(ticker):
    """Get aggregated social sentiment for a stock ticker"""
    try:
        from app.services.sentiment_service import aggregate_social_sentiment, analyze_social_topics_with_ai
        
        days = int(request.args.get('days', 7))
        
        # Get aggregated sentiment
        aggregated = aggregate_social_sentiment(ticker.upper(), days)
        
        # Get AI topic analysis
        raw_data = aggregated.get('raw_data', {})
        ai_topics = analyze_social_topics_with_ai(raw_data, ticker.upper())
        
        result = {
            'ticker': ticker.upper(),
            'overall_sentiment': aggregated.get('overall_sentiment', 'neutral'),
            'sentiment_score': aggregated.get('sentiment_score', 50.0),
            'trends': aggregated.get('trends', {}),
            'key_topics': aggregated.get('key_topics', []),
            'platform_breakdown': aggregated.get('platform_breakdown', {}),
            'raw_data': raw_data,
            'ai_analysis': ai_topics if ai_topics.get('success') else {}
        }
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        logger.exception(f"Error in get_social_sentiment endpoint for {ticker}")
        raise ExternalAPIError('Failed to fetch social sentiment', service='sentiment_service')


@bp.route('/api/social-sentiment/watchlist')
def get_watchlist_social_sentiment():
    """Get social sentiment for all tickers in watchlist"""
    try:
        import time
        from app.services.sentiment_service import aggregate_social_sentiment
        
        if not aggregate_social_sentiment:
            return jsonify({'error': 'Social sentiment service not available'}), 500
        
        # Get watchlist from request (or could be from localStorage on frontend)
        watchlist_data = request.json.get('tickers', []) if request.json else []
        
        if not watchlist_data:
            # Try to get from query params as fallback
            tickers_str = request.args.get('tickers', '')
            if tickers_str:
                watchlist_data = [t.strip().upper() for t in tickers_str.split(',')]
        
        if not watchlist_data:
            return jsonify({'error': 'No tickers provided'}), 400
        
        results = []
        days = int(request.args.get('days', 7))
        
        for ticker in watchlist_data[:20]:  # Limit to 20 tickers
            try:
                aggregated = aggregate_social_sentiment(ticker, days)
                results.append({
                    'ticker': ticker,
                    'sentiment_score': aggregated.get('sentiment_score', 50.0),
                    'overall_sentiment': aggregated.get('overall_sentiment', 'neutral'),
                    'mention_count': (
                        aggregated.get('platform_breakdown', {}).get('reddit', {}).get('mention_count', 0) +
                        aggregated.get('platform_breakdown', {}).get('twitter', {}).get('mention_count', 0) +
                        aggregated.get('platform_breakdown', {}).get('stocktwits', {}).get('watchlist_count', 0)
                    )
                })
                time.sleep(0.5)  # Rate limiting between tickers
            except Exception as e:
                logger.warning(f"Error processing ticker {ticker}: {str(e)}")
                continue
        
        # Sort by mention count (most discussed first)
        results.sort(key=lambda x: x['mention_count'], reverse=True)
        
        return jsonify(clean_for_json({
            'tickers': results,
            'total_count': len(results)
        }))
        
    except Exception as e:
        logger.exception(f"Error in get_watchlist_social_sentiment endpoint")
        raise ExternalAPIError('Failed to fetch watchlist social sentiment', service='sentiment_service')


@bp.route('/api/economic-event-explanation')
def get_economic_event_explanation():
    """Get AI explanation of economic event"""
    try:
        from app.config import GEMINI_AVAILABLE
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Google Gemini API key not configured'}), 500
        
        from app.services.ai_service import explain_economic_event_with_ai
        
        event_name = request.args.get('event', '')
        event_description = request.args.get('description', '')
        
        if not event_name:
            raise ValidationError('Event name required', {'provided': {'event': event_name}})
        
        explanation = explain_economic_event_with_ai(event_name, event_description)
        
        if not explanation.get('success'):
            return jsonify({'error': explanation.get('error', 'Failed to generate explanation')}), 500
        
        return jsonify(clean_for_json(explanation))
        
    except ValidationError:
        raise
    except Exception as e:
        logger.exception(f"Error in economic-event-explanation endpoint")
        raise ExternalAPIError('Failed to generate economic event explanation', service='ai_service')

