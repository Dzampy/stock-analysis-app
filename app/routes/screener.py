"""Stock screener routes"""
from flask import Blueprint, jsonify, request
from app.utils.json_utils import clean_for_json
from app.services.screener_service import run_stock_screener

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
        print(f"Error in screener endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to run screener: {str(e)}'}), 500


