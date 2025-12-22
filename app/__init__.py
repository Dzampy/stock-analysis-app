"""
Stock Analysis Platform - Flask Application
"""
from flask import Flask
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
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

# Create app instance
app = create_app()

