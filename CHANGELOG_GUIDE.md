# Changelog / Updates Guide

Tento dokument popisuje, jak přidávat nové updaty do systému verzí aplikace.

## Jak přidat nový update

### 1. Aktualizuj verzi

Upravte `VERSION.txt` a `app/utils/updates.py`:

**VERSION.txt:**
```
2.1.1  # nebo vyšší
```

**app/utils/updates.py:**
```python
CURRENT_VERSION = "2.1.1"  # Aktualizuj zde také
```

### 2. Přidej nový update do seznamu

V souboru `app/utils/updates.py` přidej nový entry na **ZAČÁTEK** seznamu `UPDATES` (nejnovější první):

```python
UPDATES = [
    {
        "version": "2.1.1",  # Verze této změny
        "date": "2025-01-28",  # Datum ve formátu YYYY-MM-DD
        "title": "Název updatu",  # Krátký, výstižný název
        "type": "feature",  # "feature", "fix", nebo "improvement"
        "description": "Krátký popis změny",  # 1-2 věty
        "details": [  # Seznam detailů (volitelné)
            "Detail 1",
            "Detail 2",
            "Detail 3"
        ],
        "icon": "🎯"  # Emoji ikona (volitelné)
    },
    # ... předchozí updaty
]
```

### 3. Typy updatů

- **`feature`** - Nová funkce (zelená barva)
- **`fix`** - Oprava bugu (červená barva)
- **`improvement`** - Vylepšení existující funkce (modrá barva)

### 4. Příklad kompletního updatu

```python
{
    "version": "2.2.0",
    "date": "2025-02-01",
    "title": "Nový Financials Tab",
    "type": "feature",
    "description": "Přidána kompletní finanční analýza s executive snapshot, income statement, cash flow a balance sheet",
    "details": [
        "Executive snapshot s klíčovými metrikami",
        "Quarterly a annual income statement",
        "Cash flow analysis s trendem",
        "Balance sheet overview",
        "Automatická detekce red flags",
        "Sector comparison a industry ranking"
    ],
    "icon": "📊"
}
```

### 5. Commit a push

```bash
git add VERSION.txt app/utils/updates.py
git commit -m "Update: verze 2.1.1 - [krátký popis]"
git push
```

## Zobrazení na webu

Updaty se automaticky zobrazují na hlavní stránce aplikace v sekci **"Latest Updates"**:
- Zobrazuje se aktuální verze
- Seznam posledních 5 updatů (default)
- Každý update má ikonu, verzi, datum a detaily
- Sekce je collapsible (lze skrýt/zobrazit)

## API Endpoint

Updaty jsou dostupné přes API endpoint:
```
GET /api/updates?limit=5
```

Response:
```json
{
    "success": true,
    "version": "2.1.0",
    "updates": [
        {
            "version": "2.1.0",
            "date": "2025-01-27",
            "title": "...",
            "type": "feature",
            "description": "...",
            "details": [...],
            "icon": "⚡"
        }
    ]
}
```

## Best Practices

1. **Aktualizuj verzi podle významu změny:**
   - `MAJOR.MINOR.PATCH` (např. 2.1.0)
   - MAJOR: breaking changes
   - MINOR: nové funkce
   - PATCH: opravy bugů

2. **Přidávej updaty pravidelně:**
   - Po každé větší změně
   - Po skupině souvisejících změn
   - Minimálně 1x měsíčně (pokud jsou změny)

3. **Buď specifický:**
   - Pojmenuj jasně, co bylo změněno
   - Přidej detaily pro větší změny
   - Použij vhodnou ikonu

4. **Datum:**
   - Použij datum, kdy byla změna commitnuta do main branchu
   - Formát: YYYY-MM-DD

