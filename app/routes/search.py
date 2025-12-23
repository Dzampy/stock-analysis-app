"""Search routes"""
from flask import Blueprint, jsonify, request
import yfinance as yf
from app.utils.json_utils import clean_for_json
from app.utils.logger import logger
from app.utils.error_handler import ExternalAPIError
from app.config import CACHE_TIMEOUTS

bp = Blueprint('search', __name__)

# Import cache for route caching
try:
    from app import cache
    CACHE_AVAILABLE = True
except (ImportError, RuntimeError):
    CACHE_AVAILABLE = False
    cache = None


@bp.route('/api/search/<query>')
@cache.cached(timeout=600, unless=lambda: not CACHE_AVAILABLE)  # 10 min cache for search
def search_stocks(query):
    """Advanced stock search with company name matching and fuzzy search"""
    import json
    import os
    import time
    logger.debug(f"Search stocks called with query: {query}")
    try:
        query = query.strip().upper()
        logger.debug(f"Query after processing: {query}")
        if not query or len(query) < 1:
            return jsonify({'results': [], '_version': 'v3_empty'})
        
        results = []
        query_lower = query.lower()
        
        # Popular tickers database (can be expanded or loaded from file)
        from app.services.screener_service import get_popular_tickers
        popular_tickers = get_popular_tickers()
        
        # First, try exact ticker match
        if query in popular_tickers:
            try:
                stock = yf.Ticker(query)
                info = stock.info
                if info and 'symbol' in info:
                    results.append({
                        'ticker': info['symbol'],
                        'name': info.get('longName', info.get('shortName', query)),
                        'exchange': info.get('exchange', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'matchType': 'exact_ticker',
                        'score': 100
                    })
            except Exception:
                pass
        
        # Search through popular tickers for name matches
        for ticker in popular_tickers:
            if len(results) >= 15:  # Limit results
                break
            
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if not info or 'symbol' not in info:
                    continue
                
                ticker_symbol = info['symbol']
                long_name = info.get('longName', '')
                short_name = info.get('shortName', '')
                company_name = long_name or short_name or ''
                
                if not company_name:
                    continue
                
                company_name_lower = company_name.lower()
                score = 0
                match_type = None
                
                # Exact ticker match (already handled above)
                if ticker_symbol.upper() == query:
                    continue
                
                # Check if query matches ticker
                if query_lower in ticker_symbol.lower():
                    score = 80
                    match_type = 'ticker_partial'
                
                # Check if query matches company name (exact or partial)
                elif query_lower in company_name_lower:
                    # Exact match gets higher score
                    if company_name_lower == query_lower:
                        score = 95
                        match_type = 'name_exact'
                    elif company_name_lower.startswith(query_lower):
                        score = 85
                        match_type = 'name_starts_with'
                    else:
                        score = 70
                        match_type = 'name_contains'
                
                # Fuzzy matching - check if words match
                elif len(query) >= 3:
                    query_words = query_lower.split()
                    name_words = company_name_lower.split()
                    matching_words = sum(1 for qw in query_words if any(qw in nw or nw.startswith(qw) for nw in name_words))
                    if matching_words > 0:
                        score = 50 + (matching_words * 10)
                        match_type = 'fuzzy'
                
                if score > 0:
                    results.append({
                        'ticker': ticker_symbol,
                        'name': company_name,
                        'exchange': info.get('exchange', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'matchType': match_type,
                        'score': score
                    })
            
            except Exception as e:
                # Skip tickers that fail to load
                continue
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates and limit to top 15
        seen_tickers = set()
        unique_results = []
        for result in results:
            if result['ticker'] not in seen_tickers:
                seen_tickers.add(result['ticker'])
                unique_results.append(result)
                if len(unique_results) >= 15:
                    break
        
        # Clean up response (remove internal fields)
        for result in unique_results:
            result.pop('matchType', None)
            result.pop('score', None)
        
        cleaned = clean_for_json({'results': unique_results})
        cleaned['_version'] = 'v3'
        return jsonify(cleaned)
        
    except Exception as e:
        logger.exception(f"Error in search endpoint for query: {query}")
        raise ExternalAPIError('Search failed', service='search')


@bp.route('/api/search-ticker')
def search_ticker():
    """Search for stock tickers by symbol or company name"""
    try:
        query = request.args.get('query', '').strip().upper()
        if not query or len(query) < 1:
            return jsonify({'results': []})
        
        results = []
        
        # Try to get info for the query (might be a ticker)
        try:
            stock = yf.Ticker(query)
            info = stock.info
            if info and 'symbol' in info:
                results.append({
                    'ticker': info['symbol'],
                    'name': info.get('longName', info.get('shortName', query)),
                    'exchange': info.get('exchange', 'N/A')
                })
        except Exception as e:
            logger.warning(f"Error searching for ticker {query}: {str(e)}")
            pass
        
        # If query is longer, try partial matches (simple approach)
        # For better search, would need a stock database
        # For now, return direct match if found
        
        return jsonify(clean_for_json({'results': results[:10]}))
        
    except Exception as e:
        logger.exception(f"Error in search-ticker endpoint for query: {query}")
        raise ExternalAPIError('Failed to search ticker', service='search')


