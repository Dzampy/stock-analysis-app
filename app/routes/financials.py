"""Financials routes"""
from flask import Blueprint, jsonify
from app.services.finviz_service import get_quarterly_estimates_from_finviz
from app.services.yfinance_service import get_financials_data
from app.analysis.fundamental import calculate_financials_score, get_peer_comparison_data
from app.utils.json_utils import clean_for_json
from app.utils.logger import logger
from app.utils.error_handler import NotFoundError, ExternalAPIError
from app.config import CACHE_TIMEOUTS

bp = Blueprint('financials', __name__)

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


@bp.route('/api/financials/<ticker>')
@cache_if_available(CACHE_TIMEOUTS['financials'])
def get_financials(ticker):
    """Get comprehensive financial data for Financials tab"""
    import time as time_module
    start_time = time_module.time()
    logger.info(f"Financials endpoint called for {ticker}")
    
    try:
        ticker_upper = ticker.upper()
        logger.debug(f"Processing ticker: {ticker_upper}")
        
        # Try to get financials data with better error handling
        try:
            logger.debug(f"Calling get_financials_data for {ticker_upper}")
            financials = get_financials_data(ticker_upper)
            elapsed = time_module.time() - start_time
            logger.info(f"get_financials_data completed in {elapsed:.2f}s for {ticker_upper}")
        except Exception as fetch_error:
            elapsed = time_module.time() - start_time
            logger.exception(f"Failed to fetch financials data for {ticker_upper} after {elapsed:.2f}s")
            raise ExternalAPIError(
                'Financial data not available',
                service='yfinance',
                details={'ticker': ticker_upper, 'elapsed_seconds': round(elapsed, 2)}
            )
        
        if financials is None:
            elapsed = time_module.time() - start_time
            logger.warning(f"get_financials_data returned None for {ticker_upper} after {elapsed:.2f}s")
            raise NotFoundError(
                'Unable to fetch financial data. The ticker may not exist or data may be temporarily unavailable.',
                {'ticker': ticker_upper, 'elapsed_seconds': round(elapsed, 2)}
            )
        
        # Add peer comparison data (optional, don't fail if it doesn't work)
        try:
            industry_category = financials.get('industry_category', '')
            sector = financials.get('sector', '')
            if industry_category or sector:
                peer_comparison = get_peer_comparison_data(
                    ticker_upper, 
                    industry_category, 
                    sector, 
                    limit=4
                )
                if peer_comparison:
                    financials['peer_comparison'] = peer_comparison
                    logger.info(f"Added {len(peer_comparison)} peers for {ticker_upper}")
        except Exception as peer_error:
            logger.warning(f"Failed to get peer comparison for {ticker_upper}: {str(peer_error)}")
            # Don't fail the whole request if peer comparison fails
        
        # Add sector averages (optional, don't fail if it doesn't work)
        try:
            sector = financials.get('sector', '')
            industry = financials.get('industry', '')
            if sector and sector != 'N/A':
                from app.services.sector_service import get_sector_averages
                sector_averages = get_sector_averages(sector, industry)
                if sector_averages:
                    financials['sector_averages'] = sector_averages
                    logger.info(f"Added sector averages for {ticker_upper} (sector: {sector})")
        except Exception as sector_error:
            logger.warning(f"Failed to get sector averages for {ticker_upper}: {str(sector_error)}")
            # Don't fail the whole request if sector averages fails
        
        logger.info(f"Successfully prepared financials data for {ticker_upper}")
        return jsonify(clean_for_json(financials))
        
    except (NotFoundError, ExternalAPIError):
        raise  # Re-raise to be handled by error handler
    except Exception as e:
        logger.exception(f"Error in financials endpoint for {ticker}")
        raise ExternalAPIError(
            'Failed to get financials',
            service='financials',
            details={'ticker': ticker.upper() if ticker else 'unknown'}
        )

