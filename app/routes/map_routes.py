"""Map routes - Custom treemap data (Finviz-style)"""
from flask import Blueprint, jsonify, request

from app.services.map_service import get_map_data
from app.utils.logger import logger

bp = Blueprint('map', __name__)

MAX_TICKERS = 80


@bp.route('/api/map/data', methods=['POST'])
def map_data():
    """
    POST body: { "tickers": ["AAPL","MSFT",...], "period": "1d"|"5d"|"1m" }
    Returns: { "items": [ { "ticker", "name", "market_cap", "change_pct" }, ... ] }
    """
    try:
        data = request.get_json() or {}
        raw = data.get('tickers') or []
        period = (data.get('period') or '1d').lower().strip()
        if period not in ('1d', '5d', '1m'):
            period = '1d'

        # Normalize to list of strings
        if isinstance(raw, str):
            raw = [s.strip().upper() for s in raw.replace(',', ' ').split() if s.strip()]
        else:
            raw = [str(s).strip().upper() for s in raw if s and str(s).strip()]

        # Dedupe and limit
        tickers = list(dict.fromkeys(raw))[:MAX_TICKERS]

        if not tickers:
            return jsonify({'error': 'At least one ticker is required', 'items': []}), 400

        items = get_map_data(tickers, period)
        return jsonify({'items': items})
    except Exception as e:
        logger.exception(f"map_data error: {e}")
        return jsonify({'error': 'Failed to load map data', 'items': []}), 500
