# 📊 Stock Analysis Platform

Komplexní platforma pro analýzu akcií s AI doporučeními, technickou analýzou, fundamentálními daty a ML predikcemi.

## 🚀 Rychlý start

### Lokální spuštění

1. **Nainstalujte závislosti:**
```bash
pip install -r requirements.txt
```

2. **Nastavte environment variables (volitelné):**
Vytvořte soubor `.env`:
```
GEMINI_API_KEY=your_gemini_api_key
SEC_API_KEY=your_sec_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

3. **Spusťte aplikaci:**
```bash
python app.py
```

4. **Otevřete v prohlížeči:**
```
http://localhost:5001
```

## 🌐 Nasazení na web

Pro instrukce k nasazení na Render.com (zdarma) viz [DEPLOYMENT.md](DEPLOYMENT.md)

## ✨ Funkce

- 📈 **Technická analýza**: RSI, MACD, Bollinger Bands, Support/Resistance
- 🤖 **AI doporučení**: ML-based price predictions, entry/exit points
- 💰 **Fundamentální analýza**: Financials, cash flow, profitability
- 📰 **News sentiment**: AI-powered news analysis
- 🔍 **Stock screener**: Filtrování akcií podle kritérií
- 📊 **Factor analysis**: Value, Growth, Momentum, Quality scores
- 🐋 **Institutional tracking**: Ownership, flow, whale watching

## 📋 Požadavky

- Python 3.11+
- Všechny závislosti jsou v `requirements.txt`

## 📝 Poznámky

- API klíče jsou volitelné - aplikace funguje i bez nich (některé funkce nebudou dostupné)
- První spuštění může trvat déle kvůli stahování ML modelů
- Pro produkci použijte gunicorn (viz `Procfile`)

## 📄 Licence

MIT
