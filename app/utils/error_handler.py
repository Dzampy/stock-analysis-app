"""
Centralized error handling and standardized error responses
"""
from flask import jsonify, request
from typing import Dict, Optional
import traceback
from app.utils.logger import logger


class AppError(Exception):
    """Base application error"""
    def __init__(self, message: str, status_code: int = 500, error_code: str = None, details: Dict = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or f"ERR_{status_code}"
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppError):
    """Validation error (400)"""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, 400, "VALIDATION_ERROR", details)


class NotFoundError(AppError):
    """Not found error (404)"""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, 404, "NOT_FOUND", details)


class ExternalAPIError(AppError):
    """External API error (502)"""
    def __init__(self, message: str, service: str = None, details: Dict = None):
        details = details or {}
        if service:
            details['service'] = service
        super().__init__(message, 502, "EXTERNAL_API_ERROR", details)


class RateLimitError(AppError):
    """Rate limit error (429)"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, details: Dict = None):
        details = details or {}
        if retry_after:
            details['retry_after'] = retry_after
        super().__init__(message, 429, "RATE_LIMIT_ERROR", details)


def create_error_response(
    message: str,
    status_code: int = 500,
    error_code: str = None,
    details: Dict = None,
    include_traceback: bool = False
) -> tuple:
    """
    Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        error_code: Application error code
        details: Additional error details
        include_traceback: Whether to include traceback (only in debug mode)
        
    Returns:
        Tuple of (JSON response, status_code)
    """
    error_code = error_code or f"ERR_{status_code}"
    details = details or {}
    
    response = {
        'success': False,
        'error': {
            'message': message,
            'code': error_code,
            'status_code': status_code
        }
    }
    
    # Add details if provided
    if details:
        response['error']['details'] = details
    
    # Include traceback only in debug mode
    if include_traceback:
        import traceback as tb
        response['error']['traceback'] = tb.format_exc()
    
    # Log error
    logger.error(
        f"Error {error_code}: {message}",
        extra={
            'status_code': status_code,
            'error_code': error_code,
            'details': details,
            'path': request.path if request else None,
            'method': request.method if request else None
        }
    )
    
    return jsonify(response), status_code


def handle_app_error(error: AppError) -> tuple:
    """Handle AppError exceptions"""
    return create_error_response(
        error.message,
        error.status_code,
        error.error_code,
        error.details,
        include_traceback=False
    )


def handle_generic_error(error: Exception) -> tuple:
    """Handle generic exceptions"""
    error_message = str(error) or "An unexpected error occurred"
    
    # Safely get request info
    try:
        path = request.path if request else None
        method = request.method if request else None
    except:
        path = None
        method = None
    
    # Log full traceback
    logger.exception(
        f"Unhandled exception: {error_message}",
        extra={
            'exception_type': type(error).__name__,
            'path': path,
            'method': method
        }
    )
    
    # Don't expose internal errors in production
    import os
    is_debug = os.getenv('FLASK_ENV') == 'development' or os.getenv('DEBUG') == 'True'
    
    return create_error_response(
        error_message if is_debug else "An internal server error occurred",
        500,
        "INTERNAL_ERROR",
        {'exception_type': type(error).__name__} if is_debug else {},
        include_traceback=is_debug
    )


def register_error_handlers(app):
    """
    Register error handlers with Flask app
    
    Args:
        app: Flask application instance
    """
    @app.errorhandler(AppError)
    def handle_app_error_handler(error: AppError):
        return handle_app_error(error)
    
    @app.errorhandler(404)
    def handle_not_found(error):
        try:
            path = request.path if request else 'unknown'
        except:
            path = 'unknown'
        return create_error_response(
            f"Endpoint not found: {path}",
            404,
            "NOT_FOUND",
            {'path': path}
        )
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        try:
            method = request.method if request else 'unknown'
            path = request.path if request else 'unknown'
        except:
            method = 'unknown'
            path = 'unknown'
        return create_error_response(
            f"Method not allowed: {method}",
            405,
            "METHOD_NOT_ALLOWED",
            {'method': method, 'path': path}
        )
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        return handle_generic_error(error)
    
    @app.errorhandler(Exception)
    def handle_all_exceptions(error):
        return handle_generic_error(error)

