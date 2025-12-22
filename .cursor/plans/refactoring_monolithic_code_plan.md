---
name: Refactoring Monolithic Code to Modular Architecture
overview: Rozdělení monolitického kódu (app.py 4549 řádků, index.html 8686 řádků) na modulární architekturu s jasnou separací concerns.
todos: []
---

# Refactoring: Monolitický kód → Modulární architektura

## Cíl
Rozdělit monolitický kód na udržovatelné moduly s jasnou separací concerns, zachováním veškeré funkcionality a umožněním postupného refaktoringu.

## Nová struktura projektu

```
untitled folder/
├── app.py                    # Main entry point, minimální kód
├── config.py                 # Konfigurace aplikace
├── requirements.txt
├── .env
├── README.md
│
├── backend/
│   ├── __init__.py
│   │
│   ├── routes/               # Flask Blueprints (API endpoints)
│   │   ├── __init__.py
│   │   ├── stock_routes.py
│   │   ├── financials_routes.py
│   │   ├── news_routes.py
│   │   ├── analyst_routes.py
│   │   ├── insider_routes.py
│   │   ├── portfolio_routes.py
│   │   ├── economic_routes.py
│   │   ├── ai_routes.py
│   │   └── search_routes.py
│   │
│   ├── services/            # Business logika
│   │   ├── __init__.py
│   │   ├── stock_service.py
│   │   ├── financials_service.py
│   │   ├── news_service.py
│   │   ├── analyst_service.py
│   │   ├── insider_service.py
│   │   ├── portfolio_service.py
│   │   ├── economic_service.py
│   │   └── ai_service.py
│   │
│   ├── data_sources/        # Externí API integrace
│   │   ├── __init__.py
│   │   ├── yfinance_client.py
│   │   ├── sec_api_client.py
│   │   ├── finviz_client.py
│   │   ├── marketbeat_client.py
│   │   └── investing_client.py
│   │
│   ├── ml/                  # Machine Learning modely
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   ├── price_predictor.py
│   │   ├── trend_classifier.py
│   │   └── risk_scorer.py
│   │
│   ├── utils/               # Helper funkce
│   │   ├── __init__.py
│   │   ├── json_utils.py
│   │   ├── date_utils.py
│   │   ├── sentiment_utils.py
│   │   └── validators.py
│   │
│   └── models/              # Data modely (volitelné)
│       ├── __init__.py
│       ├── stock_data.py
│       └── portfolio_data.py
│
├── static/                  # Frontend assets
│   ├── css/
│   │   ├── main.css
│   │   ├── charts.css
│   │   └── components.css
│   │
│   └── js/
│       ├── main.js
│       ├── charts/
│       │   ├── price-chart.js
│       │   ├── candlestick-chart.js
│       │   ├── rsi-chart.js
│       │   └── macd-chart.js
│       ├── services/
│       │   ├── api-client.js
│       │   ├── storage-service.js
│       │   └── notification-service.js
│       ├── components/
│       │   ├── stock-analysis.js
│       │   ├── financials.js
│       │   ├── news.js
│       │   ├── portfolio.js
│       │   └── watchlist.js
│       └── utils/
│           ├── formatters.js
│           ├── validators.js
│           └── ui-helpers.js
│
└── templates/
    ├── index.html           # Hlavní template (minimální)
    ├── components/          # HTML komponenty (volitelné)
    │   ├── header.html
    │   ├── navigation.html
    │   └── footer.html
    └── partials/            # Partial templates
        ├── stock-analysis.html
        ├── financials.html
        └── portfolio.html
```

## Krok 1: Backend Refactoring

### 1.1 Vytvoření základní struktury

**Soubory k vytvoření:**
- `backend/__init__.py` - inicializace backend modulu
- `config.py` - centralizovaná konfigurace
- `backend/utils/__init__.py`
- `backend/services/__init__.py`
- `backend/routes/__init__.py`

**Úkoly:**
1. Vytvořit složkovou strukturu
2. Přesunout konstanty do `config.py`
3. Vytvořit `backend/__init__.py` s factory funkcí pro Flask app

### 1.2 Rozdělení utility funkcí

**Soubor:** `backend/utils/json_utils.py`
- `clean_for_json(data)` - z `app.py` řádek 47

**Soubor:** `backend/utils/date_utils.py`
- `normalize_date(date_str)` - z `app.py` řádek 1109

**Soubor:** `backend/utils/sentiment_utils.py`
- `analyze_sentiment(text)` - z `app.py` řádek 464

**Soubor:** `backend/utils/validators.py`
- `validate_ticker(ticker)` - nová funkce
- `validate_period(period)` - nová funkce

### 1.3 Rozdělení data sources

**Soubor:** `backend/data_sources/yfinance_client.py`
- `get_stock_data(ticker, period)` - z `app.py` řádek 67
- `get_stock_info(ticker)` - extrahovat z `get_stock_data`
- `get_earnings_data(ticker)` - z `app.py` řádek 244
- `get_financials_data(ticker)` - z `app.py` řádek 491
- `get_stock_news(ticker, max_news)` - z `app.py` řádek 1152
- `get_ownership_data(ticker)` - extrahovat z insider trading

**Soubor:** `backend/data_sources/sec_api_client.py`
- `get_sec_insider_trading(ticker)` - z `app.py` řádek 3155
- `get_sec_ownership(ticker)` - pokud existuje

**Soubor:** `backend/data_sources/finviz_client.py`
- `get_finviz_analyst_ratings(ticker)` - z `app.py` řádek 2348
- `get_finviz_insider_trading(ticker)` - z `app.py` řádek 2908

**Soubor:** `backend/data_sources/marketbeat_client.py`
- `get_marketbeat_analyst_ratings(ticker)` - z `app.py` řádek 2544
- `get_marketbeat_insider_trading(ticker)` - z `app.py` řádek 2772

**Soubor:** `backend/data_sources/investing_client.py`
- `get_economic_calendar()` - z `app.py` řádek 3505
- `generate_economic_calendar_sample_data()` - z `app.py` řádek 3864

### 1.4 Rozdělení calculation services

**Soubor:** `backend/services/stock_service.py`
- `calculate_technical_indicators(df)` - z `app.py` řádek 171
- `calculate_metrics(df, info)` - z `app.py` řádek 199
- `get_stock_analysis(ticker, period)` - orchestrace stock analýzy

**Soubor:** `backend/services/financials_service.py`
- `analyze_financials(ticker)` - z `app.py` řádek 491 (logika analýzy)
- `detect_company_stage(financials)` - z `app.py`
- `generate_financial_verdict(financials)` - z `app.py`

**Soubor:** `backend/services/news_service.py`
- `get_news_with_sentiment(ticker)` - z `app.py` řádek 1152
- `generate_news_summary(news_list, ticker)` - z `app.py` řádek 974

**Soubor:** `backend/services/analyst_service.py`
- `get_analyst_ratings(ticker)` - orchestrace z různých zdrojů
- `aggregate_analyst_data(ticker)` - z `app.py` řádek 2666

**Soubor:** `backend/services/insider_service.py`
- `get_insider_trading_data(ticker)` - orchestrace z různých zdrojů
- `aggregate_insider_data(ticker)` - z `app.py` řádek 3311

**Soubor:** `backend/services/portfolio_service.py`
- `calculate_portfolio_data(positions)` - z `app.py` řádek 4415

**Soubor:** `backend/services/economic_service.py`
- `get_economic_events()` - z `app.py` řádek 4270
- `get_economic_event_explanation(event_name)` - z `app.py` řádek 4398

### 1.5 Rozdělení ML modulů

**Soubor:** `backend/ml/feature_extractor.py`
- `extract_ml_features(ticker, df, info, indicators, metrics, news_list)` - z `app.py` řádek 1247

**Soubor:** `backend/ml/price_predictor.py`
- `predict_price(features, current_price)` - z `app.py` řádek 1429

**Soubor:** `backend/ml/trend_classifier.py`
- `classify_trend(features)` - z `app.py` řádek 1520

**Soubor:** `backend/ml/risk_scorer.py`
- `calculate_risk_score(features, metrics, info)` - z `app.py` řádek 1629

**Soubor:** `backend/services/ai_service.py`
- `generate_ai_recommendations(ticker)` - z `app.py` řádek 1733

### 1.6 Rozdělení routes (Flask Blueprints)

**Soubor:** `backend/routes/stock_routes.py`
```python
from flask import Blueprint
stock_bp = Blueprint('stock', __name__)

@stock_bp.route('/api/stock/<ticker>')
def get_stock(ticker):
    # Z app.py řádek 1979
    pass
```

**Soubory k vytvoření:**
- `backend/routes/financials_routes.py` - `/api/financials/<ticker>`
- `backend/routes/news_routes.py` - news endpoints (pokud existují)
- `backend/routes/analyst_routes.py` - `/api/analyst-data/<ticker>`
- `backend/routes/insider_routes.py` - `/api/insider-trading/<ticker>`
- `backend/routes/portfolio_routes.py` - `/api/portfolio-data`
- `backend/routes/economic_routes.py` - `/api/economic-calendar`, `/api/economic-event-explanation`
- `backend/routes/ai_routes.py` - `/api/ai-recommendations/<ticker>`
- `backend/routes/search_routes.py` - `/api/search-ticker`
- `backend/routes/earnings_routes.py` - `/api/earnings-calendar`
- `backend/routes/alerts_routes.py` - `/api/alerts-dashboard`

### 1.7 Refaktoring app.py

**Nový app.py:**
```python
from flask import Flask
from backend import create_app
from config import Config

app = create_app(Config)

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5001, use_reloader=False, threaded=True)
```

**Soubor:** `backend/__init__.py`
```python
def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Register blueprints
    from backend.routes import stock_routes, financials_routes, ...
    app.register_blueprint(stock_routes.stock_bp)
    app.register_blueprint(financials_routes.financials_bp)
    # ...
    
    return app
```

## Krok 2: Frontend Refactoring

### 2.1 Rozdělení CSS

**Soubor:** `static/css/main.css`
- Základní styly (root variables, dark mode, layout)
- Z `templates/index.html` <style> sekce (řádky 12-2700 cca)

**Soubor:** `static/css/charts.css`
- Styly pro grafy (Chart.js, Lightweight Charts)

**Soubor:** `static/css/components.css`
- Komponenty (cards, buttons, modals, etc.)

### 2.2 Rozdělení JavaScript - Utilities

**Soubor:** `static/js/utils/formatters.js`
- `formatMarketCap(value)`
- `formatNumber(value)`
- `formatCurrency(value)`
- `formatDate(date)`
- `formatRelativeTime(date)`

**Soubor:** `static/js/utils/validators.js`
- `validateTicker(ticker)`
- `validatePrice(price)`

**Soubor:** `static/js/utils/ui-helpers.js`
- `showLoading()`
- `hideLoading()`
- `showError(message)`
- `hideError()`
- `showToast(message, type)`
- `toggleDarkMode()`

### 2.3 Rozdělení JavaScript - Services

**Soubor:** `static/js/services/api-client.js`
```javascript
class ApiClient {
    async getStock(ticker, period) { }
    async getFinancials(ticker) { }
    async getNews(ticker) { }
    // ... všechny API calls
}
```

**Soubor:** `static/js/services/storage-service.js`
```javascript
class StorageService {
    getWatchlist() { }
    saveWatchlist(watchlist) { }
    getPriceAlerts() { }
    savePriceAlerts(alerts) { }
    // ... localStorage operations
}
```

**Soubor:** `static/js/services/notification-service.js`
```javascript
class NotificationService {
    requestPermission() { }
    showNotification(title, body) { }
    checkPriceAlerts() { }
}
```

### 2.4 Rozdělení JavaScript - Charts

**Soubor:** `static/js/charts/price-chart.js`
```javascript
class PriceChart {
    constructor(containerId) { }
    create(data) { }
    update(data) { }
    destroy() { }
}
```

**Soubor:** `static/js/charts/candlestick-chart.js`
```javascript
class CandlestickChart {
    constructor(containerId) { }
    create(data) { }
    update(data) { }
    destroy() { }
}
```

**Soubor:** `static/js/charts/rsi-chart.js`
- `createRSIChart(data)`

**Soubor:** `static/js/charts/macd-chart.js`
- `createMACDChart(data)`

### 2.5 Rozdělení JavaScript - Components

**Soubor:** `static/js/components/stock-analysis.js`
```javascript
class StockAnalysis {
    constructor() { }
    async load(ticker, period) { }
    display(data) { }
    handleTimeframeChange(period) { }
}
```

**Soubor:** `static/js/components/financials.js`
```javascript
class Financials {
    constructor() { }
    async load(ticker) { }
    display(data) { }
}
```

**Soubor:** `static/js/components/news.js`
```javascript
class News {
    constructor() { }
    async load(ticker) { }
    display(newsList) { }
}
```

**Soubor:** `static/js/components/portfolio.js`
```javascript
class Portfolio {
    constructor() { }
    async load() { }
    async addPosition(position) { }
    async calculate() { }
}
```

**Soubor:** `static/js/components/watchlist.js`
```javascript
class Watchlist {
    constructor() { }
    load() { }
    add(ticker) { }
    remove(ticker) { }
    display() { }
}
```

### 2.6 Hlavní JavaScript soubor

**Soubor:** `static/js/main.js`
```javascript
// Import všech modulů
import { ApiClient } from './services/api-client.js';
import { StorageService } from './services/storage-service.js';
import { StockAnalysis } from './components/stock-analysis.js';
// ...

// Inicializace
document.addEventListener('DOMContentLoaded', () => {
    // Setup aplikace
});
```

### 2.7 Refaktoring index.html

**Nový index.html:**
- Minimální HTML struktura
- Link na externí CSS soubory
- Script tagy pro JS moduly
- Základní layout s placeholder divy pro komponenty

## Krok 3: Postupná migrace

### Fáze 1: Backend utilities (Low Risk)
1. Vytvořit `backend/utils/` moduly
2. Přesunout utility funkce
3. Aktualizovat importy v `app.py`
4. Testovat

### Fáze 2: Backend data sources (Medium Risk)
1. Vytvořit `backend/data_sources/` moduly
2. Přesunout data fetching funkce
3. Aktualizovat services, které je používají
4. Testovat

### Fáze 3: Backend services (Medium Risk)
1. Vytvořit `backend/services/` moduly
2. Přesunout business logiku
3. Aktualizovat routes
4. Testovat

### Fáze 4: Backend routes (High Risk)
1. Vytvořit Blueprints
2. Přesunout routes
3. Registrovat v `app.py`
4. Testovat všechny endpoints

### Fáze 5: Frontend CSS (Low Risk)
1. Extrahovat CSS do souborů
2. Linknout v HTML
3. Testovat styling

### Fáze 6: Frontend JS utilities (Low Risk)
1. Vytvořit utility moduly
2. Přesunout funkce
3. Aktualizovat importy
4. Testovat

### Fáze 7: Frontend JS services (Medium Risk)
1. Vytvořit service moduly
2. Refaktorovat API calls
3. Testovat

### Fáze 8: Frontend JS components (High Risk)
1. Vytvořit component moduly
2. Refaktorovat display funkce
3. Testovat všechny sekce

### Fáze 9: Frontend charts (High Risk)
1. Vytvořit chart moduly
2. Refaktorovat chart creation
3. Testovat všechny grafy

## Důležité poznámky

### Zachování kompatibility
- Během migrace udržovat starý kód funkční
- Postupně přepínat na nové moduly
- Možnost rollbacku

### Testování
- Po každé fázi testovat všechny funkce
- Zkontrolovat všechny endpoints
- Zkontrolovat všechny UI sekce

### Dokumentace
- Přidat docstrings do všech funkcí
- Dokumentovat API endpoints
- Dokumentovat component API

### Dependencies
- Zkontrolovat, že všechny importy fungují
- Aktualizovat requirements.txt pokud je potřeba
- Zkontrolovat frontend dependencies (CDN vs npm)

## Výhody nové architektury

1. **Udržovatelnost**: Každý modul má jasnou zodpovědnost
2. **Testovatelnost**: Moduly lze testovat samostatně
3. **Škálovatelnost**: Snadné přidávání nových features
4. **Znovupoužitelnost**: Moduly lze použít v jiných projektech
5. **Čitelnost**: Menší soubory jsou snadněji čitelné
6. **Týmová práce**: Různí vývojáři mohou pracovat na různých modulech

## Rizika a mitigace

**Riziko**: Breaking changes během migrace
**Mitigace**: Postupná migrace, testování po každé fázi

**Riziko**: Ztráta funkcionality
**Mitigace**: Detailní testování, checklist funkcí

**Riziko**: Performance regrese
**Mitigace**: Benchmark před a po, optimalizace importů

## Checklist před začátkem

- [ ] Vytvořit git branch pro refactoring
- [ ] Zálohovat současný kód
- [ ] Vytvořit test suite (pokud neexistuje)
- [ ] Dokumentovat současnou funkcionalitu
- [ ] Identifikovat všechny dependencies

## Odhadovaný čas

- Backend refactoring: 8-12 hodin
- Frontend refactoring: 12-16 hodin
- Testování a debugging: 4-6 hodin
- **Celkem: 24-34 hodin**











