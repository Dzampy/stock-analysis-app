# 📤 Jak nahrát projekt na GitHub

## Krok 1: Vytvořte GitHub repository

1. Jděte na **https://github.com** a přihlaste se (nebo vytvořte účet)
2. Klikněte na **"+"** v pravém horním rohu → **"New repository"**
3. Vyplňte:
   - **Repository name**: `stock-analysis-app` (nebo jakýkoliv název)
   - **Description**: "Stock Analysis Platform with AI recommendations"
   - **Visibility**: Vyberte **Public** (zdarma) nebo **Private**
   - **NEPIŠTE** žádné README, .gitignore nebo license (už je máme)
4. Klikněte na **"Create repository"**

## Krok 2: Zkopírujte URL vašeho repository

Po vytvoření uvidíte URL, například:
```
https://github.com/VASE_USERNAME/stock-analysis-app.git
```
**Zkopírujte si tuto URL!**

## Krok 3: Spusťte tyto příkazy v terminálu

Otevřete terminál v adresáři projektu a spusťte:

```bash
# 1. Inicializujte git repository
git init

# 2. Přidejte všechny soubory
git add .

# 3. Vytvořte první commit
git commit -m "Initial commit - Stock Analysis Platform"

# 4. Přejmenujte hlavní branch na 'main'
git branch -M main

# 5. Přidejte GitHub repository jako remote (Nahraďte URL vaším!)
git remote add origin https://github.com/VASE_USERNAME/stock-analysis-app.git

# 6. Nahrajte kód na GitHub
git push -u origin main
```

## ⚠️ Pokud máte problém s autentizací:

### Možnost A: Personal Access Token (doporučeno)

1. Jděte na GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Klikněte **"Generate new token (classic)"**
3. Vyplňte:
   - **Note**: "Stock Analysis App"
   - **Expiration**: Vyberte dobu (např. 90 days)
   - **Scopes**: Zaškrtněte **"repo"** (všechny podsekce)
4. Klikněte **"Generate token"**
5. **Zkopírujte si token** (zobrazí se jen jednou!)

Při `git push` použijte:
- **Username**: vaše GitHub username
- **Password**: vložte token (ne heslo!)

### Možnost B: GitHub CLI

```bash
# Nainstalujte GitHub CLI
brew install gh

# Přihlaste se
gh auth login

# Pak můžete použít normální git push
```

## ✅ Ověření

Po úspěšném push:
1. Jděte na vaše GitHub repository
2. Měli byste vidět všechny soubory
3. URL bude: `https://github.com/VASE_USERNAME/stock-analysis-app`

## 🔄 Aktualizace kódu v budoucnu

Když uděláte změny a chcete je nahrát:

```bash
git add .
git commit -m "Popis změn"
git push
```

## 🆘 Řešení problémů

### "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/VASE_USERNAME/stock-analysis-app.git
```

### "Permission denied"
- Zkontrolujte, že máte správné oprávnění k repository
- Použijte Personal Access Token místo hesla

### "Large files"
Pokud máte velké soubory (>100MB), možná budete muset použít Git LFS nebo je přidat do .gitignore

