# 🚀 Rychlé nasazení na Render.com

## ✅ Co je už připravené:
- ✅ `Procfile` - správný gunicorn příkaz
- ✅ `render.yaml` - konfigurace pro Render
- ✅ `requirements.txt` - všechny závislosti
- ✅ `runtime.txt` - Python 3.11.0
- ✅ `app/__init__.py` - Flask app s `template_folder` a `static_folder` v kořeni projektu (pro Render se používá `gunicorn app:app`)

## 📋 Krok 1: Commit a Push na GitHub

```bash
cd "/Users/davidlangr/untitled folder"

# Přidejte všechny změny
git add .

# Commit
git commit -m "Fix ML implementation: 2+ years data, backtesting, remove fake predictions"

# Push na GitHub
git push origin main
```

**Pokud nemáte GitHub repo:**
```bash
# Vytvořte nový repo na https://github.com/new
# Pak:
git remote add origin https://github.com/VASE_USERNAME/VASE_REPO_NAME.git
git push -u origin main
```

## 🌐 Krok 2: Nasazení na Render.com

### 2.1 Vytvořte účet
1. Jděte na https://render.com
2. Klikněte **"Get Started for Free"**
3. Přihlaste se pomocí **GitHub účtu**

### 2.2 Vytvořte Web Service
1. V Render dashboard klikněte **"New +"** → **"Web Service"**
2. Vyberte vaše **GitHub repository**
3. Vyplňte:
   - **Name**: `stock-analysis-app` (nebo jakýkoliv název)
   - **Region**: `Frankfurt` (nebo nejbližší)
   - **Branch**: `main`
   - **Root Directory**: (nechte prázdné)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120`

### 2.3 Environment Variables (volitelné)
V sekci **"Environment"** přidejte:

```
GEMINI_API_KEY=your_gemini_api_key_here
SEC_API_KEY=your_sec_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

**Poznámka**: Aplikace funguje i bez těchto klíčů (jen některé AI funkce nebudou dostupné).

### 2.4 Spusťte nasazení
1. Klikněte **"Create Web Service"**
2. Render začne automaticky buildovat
3. Počkejte **5-10 minut** na první build
4. Aplikace bude dostupná na: `https://stock-analysis-app.onrender.com`

## ✅ Krok 3: Ověření

1. Otevřete URL vaší aplikace
2. Zkuste načíst akcii (např. AAPL, TSLA, MSFT)
3. Zkontrolujte logy v Render dashboardu

## 🔄 Automatické nasazení

Při každém push do GitHubu se aplikace automaticky přebuildí:

```bash
git add .
git commit -m "Update"
git push
```

## ⚠️ Důležité poznámky

### Free Tier limity:
- ✅ 750 hodin měsíčně (dostatečné)
- ✅ Automatické SSL (HTTPS)
- ⚠️ Aplikace usne po 15 minutách nečinnosti
- ⚠️ První request po probuzení může trvat 30-60s
- ⚠️ 512 MB RAM

### Pro produkci:
- Zvažte upgrade na **Render paid tier** ($7/měsíc) pro:
  - Žádné usínání
  - Více RAM
  - Rychlejší response times

## 🐛 Řešení problémů

### Build selže:
- Zkontrolujte logy v Render dashboardu
- Ověřte, že `requirements.txt` obsahuje všechny závislosti
- Zkontrolujte, že Python verze v `runtime.txt` je podporovaná

### Aplikace se nespustí:
- Zkontrolujte logy
- Ověřte, že `Procfile` má správný formát
- Zkontrolujte, že `app.py` exportuje `app` objekt

### Pomalé načítání:
- To je normální na free tieru (usínání po nečinnosti)
- První request po probuzení trvá déle
- Pro lepší výkon zvažte paid tier

## 📞 Podpora

- Render dokumentace: https://render.com/docs
- Render status: https://status.render.com



