# Jak funguje ML v AI Recommendations - Detailní průvodce

## 📋 Přehled

Když klikneš na "AI Recommendations" pro nějaký ticker (např. AAPL), aplikace:

1. **Získá data** o akcii
2. **Natrénuje ML model** (pokud ještě není v cache)
3. **Předpoví budoucí cenu** pomocí ML modelu
4. **Použije predikci** pro doporučení

---

## 🔄 Krok po kroku

### 1️⃣ Uživatel klikne na "AI Recommendations" pro AAPL

```
Frontend → GET /api/ai-recommendations/AAPL
```

### 2️⃣ Zavolá se `generate_ai_recommendations('AAPL')`

Tato funkce v `app/services/ml_service.py` (řádek 901):

```python
def generate_ai_recommendations(ticker):
    # Získá stock data
    stock_data = get_stock_data(ticker, '1y')
    df = stock_data['history']  # Historie cen
    info = stock_data['info']   # Fundamentální data
    
    # Vypočítá technické indikátory
    indicators = calculate_technical_indicators(df)  # RSI, MACD, SMA, ...
    
    # Vypočítá metriky
    metrics = calculate_metrics(df, info)  # P/E, ROE, atd.
    
    # Získá news sentiment
    news_list = get_stock_news(ticker)
```

### 3️⃣ Extrahuje se ML features

```python
# Vytvoří 38+ features pro ML model
ml_features = extract_ml_features(ticker, df, info, indicators, metrics, news_list)
# Features zahrnují:
# - RSI, MACD, SMA ratios
# - Price momentum
# - Volatility
# - News sentiment
# - Fundamentální metriky
```

### 4️⃣ Zavolá se `predict_price()` - TOTO JE KLÍČOVÉ ML VOLÁNÍ

```python
price_prediction = predict_price(ml_features, current_price, df)
```

### 5️⃣ `predict_price()` zkontroluje cache

```python
# Pokud model už existuje v cache → použije ho (rychlé!)
if cache_key in _model_cache:
    model = _model_cache[cache_key]
    scaler = _scaler_cache[cache_key]
else:
    # Pokud ne → NATRÉNUJE NOVÝ MODEL
    model, scaler = _train_random_forest_model(ticker, features, current_price, df)
```

### 6️⃣ `_train_random_forest_model()` natrénuje model

Tato funkce:

```python
# a) Stáhne 2+ let historických dat
df = _download_extended_historical_data(ticker, years=2)  # ~500+ dní

# b) Pro každý den v historii:
for i in range(60, len(df) - 1):  # Začne po 60 dnech (lookback)
    # Vypočítá features pro tento den
    features = _extract_historical_features(df, i)
    # Cílová hodnota = cena následujícího dne
    target = df['Close'].iloc[i + 1]
    
    # Přidá do trénovací sady
    X_train.append(features)
    y_train.append(target)

# c) Trénuje Random Forest model
model = RandomForestRegressor(
    n_estimators=200,      # 200 rozhodovacích stromů
    max_depth=15,
    min_samples_split=10,
    ...
)

# d) Použije TimeSeriesSplit cross-validation
# (rozdělí data na train/test podle času - důležité pro časové řady!)

# e) Hyperparameter tuning - zkouší různé parametry
# Vybere ty, které dávají nejlepší R² score

# f) Natrénuje finální model
model.fit(X_train_scaled, y_train)

# g) Uloží do cache
_model_cache[cache_key] = model
```

### 7️⃣ Model se použije pro predikci

```python
# Vezme aktuální features (RSI, MACD, momentum, ...)
X = np.array([[features.get(name, 0.0) for name in feature_names]])

# Projde všemi 200 stromy v Random Forest
tree_predictions = []
for tree in model.estimators_:
    pred = tree.predict(X_scaled)[0]
    tree_predictions.append(pred)

# Zprůměruje predikce (ensemble průměr)
next_day_prediction = np.mean(tree_predictions)
```

### 8️⃣ Vypočítá predikce pro různé timeframy

```python
# Model předpovídá jen "next day", ale potřebujeme 1m, 3m, 6m, 12m
# Použije compounding:
annualized_return = (next_day_prediction - current_price) / current_price * 252

# Pro 1 měsíc (21 trading days):
daily_return = annualized_return / 252
compounded_return = (1 + daily_return) ** 21 - 1
predicted_price_1m = current_price * (1 + compounded_return)

# Stejně pro 3m, 6m, 12m...
```

### 9️⃣ Výsledek se použije v AI Recommendations

```python
# ML predikce ovlivní:
expected_return_6m = price_prediction['expected_returns']['6m']

# Upraví technical_score:
if expected_return_6m > 20:
    technical_score += 15
    reasons.append(f"ML model predicts strong 6-month return (+{expected_return_6m:.1f}%)")

# Použije se pro Entry Price:
entry_price = price_prediction['predictions']['1m']['price']

# Použije se pro Take Profit levels:
tp1_price = price_prediction['predictions']['3m']['price']
tp2_price = price_prediction['predictions']['6m']['price']
tp3_price = price_prediction['predictions']['12m']['price']

# Použije se pro Position Sizing:
ml_confidence = price_prediction['predictions']['6m']['confidence']
# Čím vyšší confidence, tím větší position size
```

---

## 🎯 Co je důležité pochopit:

### ✅ ML model je skutečný:
- Trénuje se na 2+ let historických dat
- Používá Random Forest (200 rozhodovacích stromů)
- Používá TimeSeriesSplit cross-validation
- Vybere nejlepší hyperparametry podle R² score

### ✅ Model se cachuje:
- Poprvé pro ticker → trénování trvá ~5-10 sekund
- Podruhé → použije cached model (okamžité)

### ✅ Model ovlivňuje doporučení:
- Entry price (vstupní cena)
- Take Profit levels (cílové ceny)
- Position sizing (velikost pozice)
- Technical score (celkové skóre)
- Reasons/Warnings (důvody doporučení)

---

## 🔍 Jak poznat, že se používá ML?

V logu uvidíš:
```
INFO - Training new ML model for AAPL
INFO - Model trained successfully. Training R² score: 0.9871
```

V kódu:
```python
if price_prediction['model_used'] == 'random_forest':
    # ✅ Používá skutečný ML model
elif price_prediction['model_used'] == 'momentum_estimate':
    # ⚠️ Používá fallback (není ML)
```

---

## 📊 Příklad výstupu:

```python
{
    'model_used': 'random_forest',  # ✅ Skutečný ML
    'predictions': {
        '1m': {'price': 280.50, 'confidence': 0.66},
        '3m': {'price': 290.20, 'confidence': 0.55},
        '6m': {'price': 310.80, 'confidence': 0.44},
        '12m': {'price': 340.50, 'confidence': 0.30}
    },
    'expected_returns': {
        '1m': 3.4,
        '3m': 6.9,
        '6m': 14.5,
        '12m': 25.5
    },
    'feature_importance': {
        'rsi': 0.15,
        'price_momentum_30d': 0.12,
        'macd_diff': 0.10,
        ...
    }
}
```

---

## 🆚 Rozdíl oproti Backtestu:

| Backtest | AI Recommendations |
|----------|-------------------|
| Testuje model na minulých datech | Předpovídá budoucí ceny |
| Natrénuje model mnohokrát (walk-forward) | Natrénuje model jednou |
| Porovnává s realitou | Používá se pro investiční rozhodnutí |
| Vypočítá metriky (R², MAE, ...) | Ovlivní doporučení a score |

**Ale oba používají STEJNOU funkci `_train_random_forest_model()`!**

