"""
Centralized logging configuration
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os


def setup_logger(name: str = 'stock_analysis', log_level: str = None) -> logging.Logger:
    """
    Setup and configure application logger
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or default to INFO
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout) - always available
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handlers - only if we can write to disk (not on Render/Heroku)
    # Check if we're in a cloud environment
    is_cloud = os.getenv('RENDER') or os.getenv('HEROKU') or os.getenv('RAILWAY') or os.getenv('FLY')
    
    if not is_cloud:
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # File handler (rotating)
            file_handler = RotatingFileHandler(
                log_dir / 'app.log',
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            
            # Error file handler (errors only)
            error_handler = RotatingFileHandler(
                log_dir / 'errors.log',
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_handler)
        except (OSError, PermissionError) as e:
            # If we can't create log files, just use console logging
            logger.warning(f"Could not create log files: {e}. Using console logging only.")
    
    return logger


# Create default logger instance
logger = setup_logger()

