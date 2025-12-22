"""Financials routes"""
from flask import Blueprint, jsonify
from app.services.finviz_service import get_quarterly_estimates_from_finviz
from app.services.yfinance_service import get_financials_data
from app.analysis.fundamental import calculate_financials_score, get_peer_comparison_data
from app.utils.json_utils import clean_for_json

bp = Blueprint('financials', __name__)


@bp.route('/api/financials/<ticker>')
def get_financials(ticker):
    """Get comprehensive financial data for Financials tab"""
    import time as time_module
    start_time = time_module.time()
    print(f"[DEBUG] /api/financials/{ticker} called")
    
    try:
        ticker_upper = ticker.upper()
        print(f"[DEBUG] Processing ticker: {ticker_upper}")
        
        # Try to get financials data with better error handling
        try:
            print(f"[DEBUG] Calling get_financials_data for {ticker_upper}")
            financials = get_financials_data(ticker_upper)
            elapsed = time_module.time() - start_time
            print(f"[DEBUG] get_financials_data completed in {elapsed:.2f}s for {ticker_upper}")
        except Exception as fetch_error:
            elapsed = time_module.time() - start_time
            print(f"[ERROR] Failed to fetch financials data for {ticker_upper} after {elapsed:.2f}s: {str(fetch_error)}")
            import traceback
            traceback.print_exc()
            # Return a more informative error
            return jsonify({
                'error': 'Financial data not available',
                'details': str(fetch_error),
                'ticker': ticker_upper,
                'elapsed_seconds': round(elapsed, 2)
            }), 500
        
        if financials is None:
            elapsed = time_module.time() - start_time
            print(f"[WARNING] get_financials_data returned None for {ticker_upper} after {elapsed:.2f}s")
            return jsonify({
                'error': 'Financial data not available',
                'ticker': ticker_upper,
                'message': 'Unable to fetch financial data. The ticker may not exist or data may be temporarily unavailable.',
                'elapsed_seconds': round(elapsed, 2)
            }), 404
        
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
                    print(f"[FINANCIALS] Added {len(peer_comparison)} peers for {ticker_upper}")
        except Exception as peer_error:
            print(f"[WARNING] Failed to get peer comparison for {ticker_upper}: {str(peer_error)}")
            # Don't fail the whole request if peer comparison fails
        
        return jsonify(clean_for_json(financials))
        
    except Exception as e:
        print(f"[ERROR] Error in financials endpoint for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to get financials',
            'details': str(e),
            'ticker': ticker.upper() if ticker else 'unknown'
        }), 500

