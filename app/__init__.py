"""
Stock Analysis Platform - Flask Application (package app used by gunicorn app:app)
"""
from flask import Flask
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Module-level cache and limiter so "from app import cache" works in routes
cache = Cache()
limiter = Limiter(key_func=get_remote_address, default_limits=['100 per minute'],
                  storage_uri=os.environ.get('CACHE_REDIS_URL') or 'memory://')


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    # Cache (Redis if CACHE_REDIS_URL, else simple)
    from app.config import CACHE_CONFIG
    try:
        cache.init_app(app, config=CACHE_CONFIG)
    except Exception:
        cache.init_app(app, config={'CACHE_TYPE': 'null'})

    # Rate limiting
    limiter.init_app(app)

    @app.errorhandler(429)
    def ratelimit_handler(e):
        from flask import jsonify
        return jsonify({'error': 'Too many requests', 'message': 'Rate limit exceeded.', 'retry_after': 60}), 429, {'Retry-After': '60'}

    # Register blueprints
    from app.routes import stock, financials, ai, news, analyst, screener, portfolio, search

    app.register_blueprint(stock.bp)
    app.register_blueprint(financials.bp)
    app.register_blueprint(ai.bp)
    app.register_blueprint(news.bp)
    app.register_blueprint(analyst.bp)
    app.register_blueprint(screener.bp)
    app.register_blueprint(portfolio.bp)
    app.register_blueprint(search.bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500

    return app


# Create app instance (used by gunicorn app:app when the app package is loaded)
app = create_app()

