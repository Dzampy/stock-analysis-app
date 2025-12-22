from flask import Flask
import os
from dotenv import load_dotenv
from app.utils.logger import logger
from app.utils.error_handler import register_error_handlers

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Register error handlers
register_error_handlers(app)

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

logger.info("Flask application initialized")

# SEC API configuration
SEC_API_KEY = os.getenv('SEC_API_KEY')

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_AVAILABLE = GEMINI_API_KEY is not None

if not GEMINI_AVAILABLE:
    logger.warning("Google Gemini API key not found. Earnings call analysis will not be available.")
    logger.info("Get your free API key at: https://makersuite.google.com/app/apikey")

# Reddit API configuration (optional)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockAnalysisTool/1.0')
REDDIT_AVAILABLE = REDDIT_CLIENT_ID is not None and REDDIT_CLIENT_SECRET is not None

if not REDDIT_AVAILABLE:
    logger.warning("Reddit API credentials not found. Reddit sentiment will use web scraping fallback.")

if __name__ == '__main__':
    # Run without reloader to avoid cache issues
    import os
    import sys
    os.environ['FLASK_ENV'] = 'production'
    logger.info(f"Starting Flask server with Python {sys.version}")
    logger.info(f"App file: {__file__}")
    
    # Get port from environment variable (for Render, Railway, etc.) or default to 5001
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '127.0.0.1')
    # Use 0.0.0.0 for production (Render, Railway, etc.)
    if os.environ.get('RENDER') or os.environ.get('RAILWAY') or os.environ.get('FLY') or os.environ.get('PORT'):
        host = '0.0.0.0'
    logger.info(f"Starting server on {host}:{port}")
    app.run(debug=False, host=host, port=port, use_reloader=False, threaded=True)

