# Testing and Error Handling Improvements

## ✅ Implementované vylepšení

### 1. Logging systém
- **Soubor**: `app/utils/logger.py`
- Centralizovaný logging s rotujícími log soubory
- Oddělené logy pro errors (`logs/errors.log`)
- Konfigurovatelná úroveň logování přes `LOG_LEVEL` environment variable
- Nahrazuje všechny `print()` statements

### 2. Error Handling
- **Soubor**: `app/utils/error_handler.py`
- Centralizovaný error handler pro Flask
- Standardizované error responses
- Custom exception classes:
  - `AppError` - base error
  - `ValidationError` - 400
  - `NotFoundError` - 404
  - `ExternalAPIError` - 502
  - `RateLimitError` - 429

### 3. Testování
- **Adresář**: `tests/`
- Unit testy pro utility funkce (`test_utils.py`)
- Unit testy pro services (`test_services.py`)
- Integration testy pro routes (`test_routes.py`)
- Testy pro error handling (`test_error_handler.py`)
- Pytest konfigurace (`pytest.ini`)
- Coverage tracking support

## 📋 Jak používat

### Logging
```python
from app.utils.logger import logger

logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
logger.exception("Exception occurred", exc_info=True)
```

### Error Handling
```python
from app.utils.error_handler import ValidationError, NotFoundError

# V route:
if not validate_ticker(ticker):
    raise ValidationError("Invalid ticker format", {'ticker': ticker})

if stock_not_found:
    raise NotFoundError("Stock not found", {'ticker': ticker})
```

### Spuštění testů
```bash
# Všechny testy
pytest

# S coverage
pytest --cov=app --cov-report=html

# Konkrétní test
pytest tests/test_utils.py
```

## 🎯 Další kroky

1. **Migrace print() na logger** - postupně nahradit všechny `print()` v celé aplikaci
2. **Rozšíření testů** - přidat více testů pro services a routes
3. **Error tracking** - integrovat Sentry nebo podobný nástroj
4. **Monitoring** - přidat health checks a metrics

