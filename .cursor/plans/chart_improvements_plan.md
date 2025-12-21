# Vylepšení grafů v Stock Analysis Section

## Přehled

Implementace čtyř hlavních vylepšení pro price chart:
1. Benchmark comparison (S&P 500, NASDAQ)
2. Support/Resistance levels (automatická detekce)
3. Volume overlay (zobrazení objemu)
4. Chart annotations (poznámky na graf)

## 1. Benchmark Comparison

### Implementace
- Přidat toggle button pro zapnutí/vypnutí benchmark comparison
- Načíst data pro S&P 500 (^GSPC) nebo NASDAQ (^IXIC) z yfinance
- Normalizovat data na stejný start point (100% = start date)
- Přidat jako další dataset do Chart.js line chart
- Přidat jako další line series do Lightweight Charts candlestick chart
- Zobrazit v legendě s možností zapnout/vypnout

### Backend změny
- Rozšířit `/api/stock/<ticker>` endpoint o volitelný parametr `benchmark` (sp500/nasdaq)
- Vrátit normalizovaná benchmark data spolu se stock daty

### Frontend změny
- Přidat toggle button v chart controls
- Upravit `createPriceChart()` a `createCandlestickChart()` pro přidání benchmark datasetu
- Přidat styling pro benchmark line (jiná barva, přerušovaná čára)

## 2. Support/Resistance Levels

### Implementace
- Algoritmus pro detekci support/resistance:
  - Najít lokální minima (support) a maxima (resistance)
  - Použít rolling window pro detekci významných úrovní
  - Filtrovat podle frekvence dotyků (více dotyků = silnější úroveň)
- Zobrazit jako horizontální čáry na grafu
- Použít Chart.js annotation plugin nebo vlastní implementaci
- Pro Lightweight Charts použít `createPriceLine()` nebo horizontální čáry

### Backend změny
- Přidat endpoint `/api/support-resistance/<ticker>` nebo rozšířit existující endpoint
- Implementovat algoritmus pro detekci S/R levels v Pythonu

### Frontend změny
- Přidat toggle pro zapnutí/vypnutí S/R levels
- Zobrazit úrovně jako horizontální čáry s popisky (Support $XXX, Resistance $XXX)
- Barevné rozlišení (zelená = support, červená = resistance)

## 3. Volume Overlay

### Implementace
- Volume data jsou již dostupná v chart_data
- Pro Chart.js: přidat volume jako bar chart na sekundární Y-axis
- Pro Lightweight Charts: použít `addHistogramSeries()` nebo `addVolumeSeries()`
- Zobrazit volume pod price chartem nebo jako overlay
- Možnost přepnout mezi overlay a separate chart

### Frontend změny
- Přidat toggle pro volume display
- Upravit chart layout pro zobrazení volume
- Styling volume bars (barva podle price direction)

## 4. Chart Annotations

### Implementace
- Přidat toolbar s tlačítkem "Add Annotation"
- Po kliknutí na graf umožnit přidat poznámku
- Uložit annotations do localStorage (keyed by ticker + timeframe)
- Zobrazit annotations jako značky na grafu s tooltipem
- Možnost editovat/smazat annotations

### Frontend změny
- Přidat annotation mode toggle
- Modal pro přidání/editaci annotation
- Zobrazit annotations jako značky na grafu
- Pro Chart.js použít annotation plugin nebo vlastní implementaci
- Pro Lightweight Charts použít markers nebo price lines s popisky

## Technické detaily

### Chart.js (Line Chart)
- Použít Chart.js annotation plugin pro S/R levels a annotations
- Přidat sekundární Y-axis pro volume
- Benchmark jako další dataset s normalizovanými hodnotami

### Lightweight Charts (Candlestick Chart)
- Benchmark jako další line series pomocí `addLineSeries()`
- S/R levels pomocí `createPriceLine()` nebo horizontálních čar
- Volume pomocí `addHistogramSeries()` nebo `addVolumeSeries()`
- Annotations pomocí markers nebo price lines s popisky

### Data normalizace pro benchmark
```javascript
// Normalize benchmark data to start at same point as stock
const stockStartPrice = stockData[0].close;
const benchmarkStartPrice = benchmarkData[0].close;
const normalizedBenchmark = benchmarkData.map(d => ({
    ...d,
    normalized: (d.close / benchmarkStartPrice) * stockStartPrice
}));
```

### Support/Resistance algoritmus
1. Najít lokální minima a maxima v rolling window (např. 20 dní)
2. Seskupit podobné úrovně (tolerance ±2%)
3. Spočítat počet dotyků každé úrovně
4. Vybrat top 3-5 support a resistance úrovní
5. Filtrovat podle síly (více dotyků = silnější)

## Implementační pořadí

1. **Volume Overlay** - nejjednodušší, data už jsou dostupná
2. **Benchmark Comparison** - vyžaduje backend změny, ale jasná implementace
3. **Support/Resistance Levels** - vyžaduje algoritmus, střední složitost
4. **Chart Annotations** - nejkomplexnější, vyžaduje UI pro správu annotations

## Soubory k úpravě

### Backend (`app.py`)
- Rozšířit `/api/stock/<ticker>` endpoint
- Přidat funkci pro detekci support/resistance levels
- Přidat funkci pro načtení benchmark dat

### Frontend (`templates/index.html`)
- Upravit `createPriceChart()` funkci
- Upravit `createCandlestickChart()` funkci
- Přidat UI controls pro nové funkce
- Přidat funkce pro správu annotations
- Přidat CSS pro nové elementy



