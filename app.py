from flask import Flask
import os
from pathlib import Path
from dotenv import load_dotenv
from app.utils.logger import logger
from app.utils.error_handler import register_error_handlers

# Load environment variables
load_dotenv()

# Get absolute path to templates folder (relative to app.py location)
# app.py is in root, so templates are in root/templates
base_dir = Path(__file__).parent.absolute()
template_dir = base_dir / 'templates'
static_dir = base_dir / 'static'

# Log for debugging
logger.info(f"=== FLASK INITIALIZATION ===")
logger.info(f"Base directory (app.py location): {base_dir}")
logger.info(f"Template directory: {template_dir}")
logger.info(f"Template exists: {template_dir.exists()}")
if template_dir.exists():
    template_files = list(template_dir.glob('*.html'))
    logger.info(f"Template files found: {[f.name for f in template_files]}")
    logger.info(f"index.html exists: {(template_dir / 'index.html').exists()}")

# Ensure directories exist (but don't fail if we can't create them)
try:
    template_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
except Exception as e:
    logger.warning(f"Could not create directories: {e}")

# Flask root_path defaults to the package directory (app/), but we need it to be the project root
# IMPORTANT: Use absolute path for template_folder to ensure Flask finds it regardless of root_path
# Convert to absolute path explicitly
template_abs_path = str(template_dir.absolute())
static_abs_path = str(static_dir.absolute())

app = Flask(__name__, 
            root_path=str(base_dir),
            template_folder=template_abs_path, 
            static_folder=static_abs_path)

# Verify paths after Flask initialization
logger.info(f"Flask root_path: {app.root_path}")
logger.info(f"Flask template_folder: {app.template_folder}")
logger.info(f"Is template_folder absolute: {Path(app.template_folder).is_absolute()}")
logger.info(f"Expected template path: {Path(app.template_folder) / 'index.html'}")
logger.info(f"Template path exists: {(Path(app.template_folder) / 'index.html').exists()}")
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

