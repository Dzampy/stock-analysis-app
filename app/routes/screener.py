"""Stock screener routes"""
from flask import Blueprint, jsonify, request
from app.utils.json_utils import clean_for_json
from app.services.screener_service import run_stock_screener
from app.utils.logger import logger
from app.utils.error_handler import ValidationError, ExternalAPIError

bp = Blueprint('screener', __name__)


@bp.route('/api/screener', methods=['POST'])
def run_screener():
    """Run stock screener with filters and pagination"""
    try:
        filters = request.get_json() or {}
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Validate pagination
        page = max(1, page)
        per_page = max(1, min(100, per_page))  # Limit per_page to 100
        
        results = run_stock_screener(filters)
        total = len(results)
        
        # Apply pagination
        offset = (page - 1) * per_page
        paginated_results = results[offset:offset + per_page]
        
        return jsonify(clean_for_json({
            'results': paginated_results,
            'count': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page if per_page > 0 else 0
        }))
    
    except Exception as e:
        logger.exception(f"Error in screener endpoint")
        raise ExternalAPIError('Failed to run screener', service='screener_service')


