# Testing Guide

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=app --cov-report=html --cov-report=term
```

### Run specific test file
```bash
pytest tests/test_utils.py
```

### Run specific test
```bash
pytest tests/test_utils.py::TestValidators::test_validate_ticker_valid
```

### Run by marker
```bash
pytest -m unit
pytest -m integration
```

## Test Structure

- `test_utils.py` - Unit tests for utility functions (validators, formatters)
- `test_services.py` - Unit tests for service functions
- `test_routes.py` - Integration tests for API routes
- `test_error_handler.py` - Unit tests for error handling

## Coverage Goals

- Target: 70%+ coverage
- Critical paths: 90%+ coverage
- Services: 80%+ coverage
- Routes: 70%+ coverage

## Adding New Tests

1. Create test file in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use unittest or pytest
4. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
5. Mock external dependencies (yfinance, APIs, etc.)




