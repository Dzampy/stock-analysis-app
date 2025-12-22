"""Stock screener routes"""
from flask import Blueprint, jsonify, request
from app.utils.json_utils import clean_for_json
from app.services.screener_service import run_stock_screener
from app.utils.logger import logger
from app.utils.error_handler import ValidationError, ExternalAPIError

bp = Blueprint('screener', __name__)


@bp.route('/api/screener', methods=['POST'])
def run_screener():
    """Run stock screener with filters"""
    try:
        filters = request.get_json()
        if not filters:
            return jsonify({'error': 'No filters provided'}), 400
        
        results = run_stock_screener(filters)
        
        return jsonify(clean_for_json({
            'results': results,
            'count': len(results)
        }))
    
    except Exception as e:
        logger.exception(f"Error in screener endpoint")
        raise ExternalAPIError('Failed to run screener', service='screener_service')


