# 🚀 Nasazení na Render.com (ZDARMA)

## 📋 Požadavky

1. **GitHub účet** (zdarma)
2. **Render.com účet** (zdarma)
3. **API klíče** (volitelné):
   - Google Gemini API (pro AI analýzy)
   - SEC API (pro SEC data)
   - Reddit API (pro sentiment)

## 🔧 Krok 1: Příprava projektu

### 1.1 Vytvořte GitHub repository

```bash
# Pokud ještě nemáte git repository
git init
git add .
git commit -m "Initial commit"
git branch -M main

# Vytvořte nový repository na GitHub.com a pak:
git remote add origin https://github.com/VASE_USERNAME/VASE_REPO_NAME.git
git push -u origin main
```

### 1.2 Zkontrolujte soubory

Ujistěte se, že máte tyto soubory:
- ✅ `requirements.txt` - Python závislosti
- ✅ `Procfile` - instrukce pro spuštění
- ✅ `runtime.txt` - verze Pythonu (volitelné)
- ✅ `render.yaml` - konfigurace pro Render (volitelné)

## 🌐 Krok 2: Nasazení na Render.com

### 2.1 Vytvořte účet na Render.com

1. Jděte na https://render.com
2. Klikněte na "Get Started for Free"
3. Přihlaste se pomocí GitHub účtu

### 2.2 Vytvořte nový Web Service

1. V Render dashboard klikněte na **"New +"** → **"Web Service"**
2. Vyberte vaše GitHub repository
3. Vyplňte:
   - **Name**: `stock-analysis-app` (nebo jakýkoliv název)
   - **Region**: Vyberte nejbližší (např. Frankfurt)
   - **Branch**: `main`
   - **Root Directory**: (nechte prázdné)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`

### 2.3 Nastavte Environment Variables

V sekci **"Environment"** přidejte:

```
GEMINI_API_KEY=your_gemini_api_key_here
SEC_API_KEY=your_sec_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

**Poznámka**: Tyto klíče jsou volitelné - aplikace bude fungovat i bez nich (jen některé funkce nebudou dostupné).

### 2.4 Upravte app.py pro Render

Ujistěte se, že na konci `app.py` máte:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### 2.5 Spusťte nasazení

1. Klikněte na **"Create Web Service"**
2. Render začne automaticky buildovat a nasazovat aplikaci
3. Počkejte 5-10 minut na první build
4. Aplikace bude dostupná na: `https://stock-analysis-app.onrender.com` (nebo váš název)

## 🔍 Krok 3: Ověření

1. Otevřete URL vaší aplikace
2. Zkuste načíst nějakou akcii (např. AAPL)
3. Zkontrolujte logy v Render dashboardu pro případné chyby

## ⚙️ Krok 4: Aktualizace kódu

Při každém push do GitHubu se aplikace automaticky přebuildí a nasadí:

```bash
git add .
git commit -m "Update"
git push
```

## 💰 Free Tier Limity na Render.com

- ✅ **750 hodin měsíčně** (dostatečné pro malou aplikaci)
- ✅ **Automatické SSL** (HTTPS)
- ✅ **Automatické nasazení** z GitHubu
- ⚠️ **Aplikace usne po 15 minutách nečinnosti** (první request může trvat 30-60s)
- ⚠️ **512 MB RAM**

## 🚨 Řešení problémů

### Aplikace se nespustí

1. Zkontrolujte logy v Render dashboardu
2. Ověřte, že `requirements.txt` obsahuje všechny závislosti
3. Zkontrolujte, že `Procfile` má správný formát

### Aplikace usne po nečinnosti

- To je normální na free tieru
- První request po probuzení může trvat 30-60 sekund
- Pro produkci zvažte upgrade na paid tier ($7/měsíc)

### Chyby s API klíči

- Aplikace funguje i bez API klíčů
- Některé funkce (AI analýzy, Reddit sentiment) nebudou dostupné
- To je v pořádku pro základní funkcionalitu

## 📝 Alternativní hosting služby

### Railway.app
- Podobné jako Render
- Free tier: $5 kreditu měsíčně
- Snadné nasazení

### Fly.io
- Free tier: 3 shared-cpu VMs
- Dobré pro Python aplikace
- Trochu složitější setup

### PythonAnywhere
- Specificky pro Python
- Free tier: 1 web app
- Omezení na 1 request za sekundu

## 🎯 Doporučení

**Pro začátek**: Použijte **Render.com** - je to nejjednodušší a má dobrý free tier.

**Pro produkci**: Zvažte upgrade na Render paid tier ($7/měsíc) nebo Railway.app pro lepší výkon.

