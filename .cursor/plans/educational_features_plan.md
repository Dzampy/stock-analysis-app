---
name: Educational Features - Vzdělávací sekce pro začátečníky
overview: Implementace komplexní vzdělávací sekce pro začátečníky s vysvětlením metrik, investing basics guide, glossary a "Why This Matters" sekcemi.
todos: []
---

# Educational Features - Vzdělávací sekce pro začátečníky

## Přehled

Přidání komplexní vzdělávací sekce pro začátečníky, která pomůže uživatelům lépe pochopit finanční metriky, základní investiční koncepty a proč jsou určité metriky důležité.

## Problém

Začátečníci mohou být zahlceni množstvím metrik a dat bez pochopení, co znamenají a proč jsou důležité. Tooltips existují, ale chybí komplexní vzdělávací sekce.

## Řešení

Vytvořit novou sekci "🎓 Learn" v navigaci s následujícími podsekcemi:
1. **Metric Explanations** - detailní vysvětlení všech metrik
2. **Investing Basics Guide** - základní investiční koncepty
3. **Glossary** - slovník finančních termínů
4. **Why This Matters** - vysvětlení důležitosti metrik v kontextu

## Implementace

### 1. Nová sekce v navigaci

**Soubor:** `templates/index.html` (řádek ~4412)

Přidat novou navigační skupinu "Learn" do sidebar navigace:
- Ikona: 🎓
- Umístění: Po Research skupině nebo na konci navigace
- Podsekce:
  - Metric Explanations
  - Investing Basics
  - Glossary
  - Why This Matters

### 2. Metric Explanations sekce

**Soubor:** `templates/index.html` (nová sekce)

Struktura:
- **Technical Metrics:**
  - RSI (Relative Strength Index) - co to je, jak se počítá, co znamená oversold/overbought
  - MACD (Moving Average Convergence Divergence) - co to je, jak se interpretuje
  - Bollinger Bands - co to je, co znamená když se cena dotkne pásem
  - Support/Resistance - co to jsou, proč jsou důležité
  - Moving Averages (SMA, EMA) - co to je, jak se používají
  
- **Fundamental Metrics:**
  - P/E Ratio - co to je, jak se interpretuje, co je dobrá hodnota
  - P/B Ratio - co to je, kdy je relevantní
  - P/S Ratio - co to je, kdy se používá
  - Market Cap - co to je, kategorie (large/mid/small cap)
  - EPS (Earnings Per Share) - co to je, jak se počítá
  - Revenue Growth - co to znamená, proč je důležité
  - Profit Margin - co to je, jaký je rozdíl mezi gross/operating/net margin
  - ROE/ROA/ROIC - co to jsou, jak se interpretují
  - Debt-to-Equity - co to je, co je zdravá hodnota
  - Current Ratio - co to je, co znamená >1 nebo <1
  - FCF (Free Cash Flow) - co to je, proč je důležité
  
- **Advanced Metrics:**
  - Beta - co to je, co znamená >1 nebo <1
  - Volatility - co to je, jak se měří
  - Dividend Yield - co to je, jak se počítá
  - PEG Ratio - co to je, kdy se používá
  - EV/EBITDA - co to je, kdy se používá

Každá metrika by měla obsahovat:
- Definici (co to je)
- Vzorec (jak se počítá)
- Interpretaci (co znamená vysoká/nízká hodnota)
- Příklady (konkrétní čísla s vysvětlením)
- Tipy (kdy použít, na co si dát pozor)

### 3. Investing Basics Guide

**Soubor:** `templates/index.html` (nová sekce)

Sekce s následujícími kapitolami:

- **Getting Started:**
  - Co jsou akcie a jak fungují
  - Typy investování (buy & hold, trading, swing trading)
  - Jak začít investovat (brokerage account, minimums)
  
- **Investment Strategies:**
  - Value Investing - co to je, jak se používá
  - Growth Investing - co to je, jak se identifikují growth stocks
  - Dividend Investing - co to je, pro koho je vhodné
  - Index Investing - co to je, výhody/nevýhody
  
- **Risk Management:**
  - Co je riziko v investování
  - Diversifikace - co to je, proč je důležitá
  - Position Sizing - jak velká pozice by měla být
  - Stop Loss - co to je, kdy použít
  
- **Reading Financial Statements:**
  - Income Statement - co obsahuje, jak se čte
  - Balance Sheet - co obsahuje, jak se čte
  - Cash Flow Statement - co obsahuje, proč je důležitý

### 4. Glossary (Slovník)

**Soubor:** `templates/index.html` (nová sekce)

Interaktivní slovník s:
- Vyhledáváním termínů
- Kategorizací (Technical, Fundamental, Options, atd.)
- A-Z seznamem termínů
- Každý termín má:
  - Definici
  - Příklady použití
  - Související termíny
  - Link na relevantní metriku v Metric Explanations

Klíčové termíny:
- Aktiva, Pasiva, Equity
- Bull market, Bear market
- Call/Put options
- Market order, Limit order
- Earnings, Revenue, Profit
- Volatility, Beta, Alpha
- Dividend, Stock split
- Market cap, Enterprise value
- Atd. (50+ termínů)

### 5. Why This Matters sekce

**Soubor:** `templates/index.html` (nová sekce)

Kontextové vysvětlení, proč jsou určité metriky důležité:

- **Pro Value Investors:**
  - Proč je důležitý P/E ratio
  - Proč je důležitá debt-to-equity
  - Proč je důležitý book value
  
- **Pro Growth Investors:**
  - Proč je důležitý revenue growth
  - Proč je důležitý EPS growth
  - Proč je důležitý PEG ratio
  
- **Pro Dividend Investors:**
  - Proč je důležitý dividend yield
  - Proč je důležitý payout ratio
  - Proč je důležitý dividend history
  
- **Pro Risk Management:**
  - Proč je důležitý beta
  - Proč je důležitá volatility
  - Proč je důležitý current ratio
  
- **Pro Fundamental Analysis:**
  - Proč je důležitý cash flow
  - Proč je důležitý ROE/ROIC
  - Proč je důležitý working capital

### 6. Design & UX

- **Layout:**
  - Tabs nebo accordion pro jednotlivé podsekce
  - Search bar pro rychlé vyhledávání v Glossary a Metric Explanations
  - Breadcrumbs pro navigaci
  - "Back to Analysis" button pro rychlý návrat
  
- **Styling:**
  - Konzistentní s existujícím designem
  - Použití karet pro jednotlivé metriky/termíny
  - Ikony pro vizuální rozlišení kategorií
  - Highlighting důležitých informací
  
- **Interaktivita:**
  - Expandable sekce pro každou metriku/termín
  - Linky mezi souvisejícími termíny/metrikami
  - "Learn More" odkazy na relevantní sekce
  - Možnost označit jako "Přečteno" (localStorage)

## Technické detaily

### Frontend

- **Nová sekce:** `educationSection` v HTML
- **Funkce:** `loadEducationSection()` pro načítání obsahu
- **Routing:** Přidat do `navigateToSection()` funkce
- **Data:** Statický obsah (možno v budoucnu přesunout do JSON/backend)

### Struktura souborů

```
templates/index.html
  - Nová sekce <div id="educationSection">
    - Tabs/Accordion pro navigaci mezi podsekcemi
    - Metric Explanations content
    - Investing Basics content
    - Glossary content
    - Why This Matters content
```

## Očekávaný výsledek

Začátečníci budou mít přístup ke komplexní vzdělávací sekci, která jim pomůže:
1. Pochopit, co jednotlivé metriky znamenají
2. Naučit se základní investiční koncepty
3. Najít definice finančních termínů
4. Pochopit, proč jsou určité metriky důležité pro jejich investiční strategii

## Priorita implementace

**Priorita:** STŘEDNÍ - užitečné pro začátečníky, ale ne kritické pro základní funkcionalitu

**Doporučený postup:**
1. Nejdříve Glossary (nejjednodušší, nejrychlejší)
2. Pak Metric Explanations (rozšíření existujících tooltips)
3. Pak Investing Basics Guide
4. Nakonec Why This Matters sekce

