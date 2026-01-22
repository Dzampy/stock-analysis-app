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


@bp.route('/api/financials/<ticker>/advanced')
@cache_if_available(CACHE_TIMEOUTS['financials'])
def get_financials_advanced(ticker):
    """Get advanced financial analyses (loaded asynchronously for better performance)"""
    import time as time_module
    start_time = time_module.time()
    logger.info(f"Financials advanced endpoint called for {ticker}")
    
    try:
        ticker_upper = ticker.upper()
        
        # Import advanced analysis functions
        from app.analysis.fundamental import (
            get_cash_flow_analysis,
            get_profitability_analysis,
            get_balance_sheet_health,
            get_management_guidance_tracking,
            get_segment_breakdown
        )
        
        advanced_data = {}
        
        # 1. Cash Flow Statement Analysis
        try:
            cash_flow_analysis = get_cash_flow_analysis(ticker_upper)
            if cash_flow_analysis:
                advanced_data['cash_flow_analysis'] = cash_flow_analysis
        except Exception as e:
            logger.warning(f"Failed to get cash flow analysis for {ticker_upper}: {str(e)}")
        
        # 2. Profitability Deep Dive
        try:
            # Need basic financials for profitability analysis
            from app.services.yfinance_service import get_financials_data
            basic_financials = get_financials_data(ticker_upper)
            if basic_financials:
                profitability_analysis = get_profitability_analysis(ticker_upper, basic_financials)
                if profitability_analysis:
                    advanced_data['profitability_analysis'] = profitability_analysis
        except Exception as e:
            logger.warning(f"Failed to get profitability analysis for {ticker_upper}: {str(e)}")
        
        # 3. Balance Sheet Health Score
        try:
            balance_sheet_health = get_balance_sheet_health(ticker_upper)
            if balance_sheet_health:
                advanced_data['balance_sheet_health'] = balance_sheet_health
        except Exception as e:
            logger.warning(f"Failed to get balance sheet health for {ticker_upper}: {str(e)}")
        
        # 4. Management Guidance Tracking
        try:
            guidance_tracking = get_management_guidance_tracking(ticker_upper)
            if guidance_tracking:
                advanced_data['management_guidance'] = guidance_tracking
        except Exception as e:
            logger.warning(f"Failed to get management guidance for {ticker_upper}: {str(e)}")
        
        # 5. Segment/Geography Breakdown
        try:
            segment_breakdown = get_segment_breakdown(ticker_upper)
            if segment_breakdown:
                advanced_data['segment_breakdown'] = segment_breakdown
        except Exception as e:
            logger.warning(f"Failed to get segment breakdown for {ticker_upper}: {str(e)}")
        
        elapsed = time_module.time() - start_time
        logger.info(f"Advanced financials data prepared for {ticker_upper} in {elapsed:.2f}s")
        
        return jsonify(clean_for_json(advanced_data))
        
    except Exception as e:
        logger.exception(f"Error in financials advanced endpoint for {ticker}")
        return jsonify({'error': 'Failed to get advanced financials'}), 500

