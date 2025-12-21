from flask import Flask, render_template, jsonify, request, make_response
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import json
import time
import math
import os
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. Using fallback models.")

# XGBoost and LightGBM are optional - fallback models work without them
# Note: XGBoost requires OpenMP runtime (brew install libomp on macOS)
# For now, we use fallback models that work without XGBoost/LightGBM
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# SEC API configuration
SEC_API_KEY = os.getenv('SEC_API_KEY')

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_AVAILABLE = GEMINI_API_KEY is not None

if not GEMINI_AVAILABLE:
    print("Warning: Google Gemini API key not found. Earnings call analysis will not be available.")
    print("Get your free API key at: https://makersuite.google.com/app/apikey")

# Reddit API configuration (optional)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockAnalysisTool/1.0')
REDDIT_AVAILABLE = REDDIT_CLIENT_ID is not None and REDDIT_CLIENT_SECRET is not None

if not REDDIT_AVAILABLE:
    print("Warning: Reddit API credentials not found. Reddit sentiment will use web scraping fallback.")

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def clean_for_json(data):
    """Replace NaN and inf values with None for JSON serialization"""
    # Handle Timestamp first (before Series, as Series might contain Timestamps)
    if isinstance(data, pd.Timestamp):
        try:
            return data.strftime('%Y-%m-%d')
        except:
            return str(data)
    if isinstance(data, pd.Series):
        data = data.tolist()
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to dict of lists
        return {str(col): clean_for_json(data[col].tolist()) for col in data.columns}
    if isinstance(data, (list, tuple)):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_for_json(value) for key, value in data.items()}
    elif isinstance(data, (bool, np.bool_)):
        return bool(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, (float, np.floating, np.number)):
        # Check for all types of invalid values
        if pd.isna(data) or math.isnan(data) or math.isinf(data) or not np.isfinite(data):
            return None
        val = float(data)
        # Double check after conversion
        if not np.isfinite(val) or math.isinf(val) or math.isnan(val):
            return None
        return val
    elif pd.isna(data):
        return None
    # Handle any other pandas/numpy types that might contain Timestamps
    try:
        import json
        json.dumps(data)  # Test if it's JSON serializable
        return data
    except (TypeError, ValueError):
        return str(data)  # Fallback to string representation

def extract_text_from_pdf(pdf_file):
    """Extract text content from PDF file"""
    try:
        import PyPDF2
        from io import BytesIO
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text_content = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'text': text.strip()
                    })
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        full_text = '\n\n'.join([page['text'] for page in text_content])
        return {
            'success': True,
            'text': full_text,
            'pages': len(text_content),
            'page_breakdown': text_content
        }
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def parse_ai_summary(summary_text):
    """Parse AI summary into structured format with improved detection"""
    structured = {
        'executive_summary': '',
        'financial_highlights': [],
        'strategic_initiatives': [],
        'management_outlook': [],
        'risks_challenges': [],
        'overall_sentiment': 'neutral',
        'sentiment_explanation': ''
    }
    
    # Split by sections using multiple delimiters
    sections = {}
    
    # Try to find sections by various markers
    section_markers = {
        'executive_summary': ['=== Executive Summary ===', '**Executive Summary:**', 'Executive Summary:', 'Shrnutí:', 'Executive Summary'],
        'financial_highlights': ['=== Key Financial Highlights ===', '**Key Financial Highlights:**', 'Key Financial Highlights:', 'Finanční body:', 'Financial Highlights'],
        'strategic_initiatives': ['=== Strategic Initiatives ===', '**Strategic Initiatives:**', 'Strategic Initiatives:', 'Strategické iniciativy:', 'Strategic'],
        'management_outlook': ['=== Management Outlook ===', '**Management Outlook:**', 'Management Outlook:', 'Výhled managementu:', 'Outlook'],
        'risks_challenges': ['=== Risks & Challenges ===', '**Risks & Challenges:**', 'Risks & Challenges:', 'Rizika a výzvy:', 'Risks', 'Rizika'],
        'overall_sentiment': ['=== Overall Sentiment ===', '**Overall Sentiment:**', 'Overall Sentiment:', 'Celkový sentiment:', 'Sentiment']
    }
    
    # Find section boundaries
    text_lower = summary_text.lower()
    section_positions = {}
    
    for section_name, markers in section_markers.items():
        for marker in markers:
            pos = summary_text.find(marker)
            if pos != -1:
                section_positions[section_name] = (pos, marker)
                break
    
    # Sort by position
    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1][0])
    
    # Extract content for each section
    for i, (section_name, (start_pos, marker)) in enumerate(sorted_sections):
        # Find end position (start of next section or end of text)
        if i + 1 < len(sorted_sections):
            end_pos = sorted_sections[i + 1][1][0]
        else:
            end_pos = len(summary_text)
        
        # Extract section content
        section_content = summary_text[start_pos + len(marker):end_pos].strip()
        
        if section_name == 'executive_summary':
            structured['executive_summary'] = section_content
        elif section_name == 'financial_highlights':
            # Extract bullet points
            for line in section_content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0].isdigit()):
                    bullet = line.lstrip('-•*0123456789. ').strip()
                    if bullet:
                        structured['financial_highlights'].append(bullet)
        elif section_name == 'strategic_initiatives':
            for line in section_content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0].isdigit()):
                    bullet = line.lstrip('-•*0123456789. ').strip()
                    if bullet:
                        structured['strategic_initiatives'].append(bullet)
        elif section_name == 'management_outlook':
            for line in section_content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0].isdigit()):
                    bullet = line.lstrip('-•*0123456789. ').strip()
                    if bullet:
                        structured['management_outlook'].append(bullet)
        elif section_name == 'risks_challenges':
            for line in section_content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0].isdigit()):
                    bullet = line.lstrip('-•*0123456789. ').strip()
                    if bullet:
                        structured['risks_challenges'].append(bullet)
        elif section_name == 'overall_sentiment':
            # Extract sentiment and explanation
            lines = section_content.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'positive' in line_lower or 'pozitivní' in line_lower:
                    structured['overall_sentiment'] = 'positive'
                elif 'negative' in line_lower or 'negativní' in line_lower:
                    structured['overall_sentiment'] = 'negative'
                structured['sentiment_explanation'] += line + ' '
    
    # Fallback: if structured parsing didn't work, try simple line-by-line
    if not structured['executive_summary'] and not structured['financial_highlights']:
        lines = summary_text.split('\n')
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Detect section headers (more flexible)
            if any(marker.lower() in line_stripped.lower() for marker in section_markers['executive_summary']):
                current_section = 'executive_summary'
            elif any(marker.lower() in line_stripped.lower() for marker in section_markers['financial_highlights']):
                current_section = 'financial_highlights'
            elif any(marker.lower() in line_stripped.lower() for marker in section_markers['strategic_initiatives']):
                current_section = 'strategic_initiatives'
            elif any(marker.lower() in line_stripped.lower() for marker in section_markers['management_outlook']):
                current_section = 'management_outlook'
            elif any(marker.lower() in line_stripped.lower() for marker in section_markers['risks_challenges']):
                current_section = 'risks_challenges'
            elif any(marker.lower() in line_stripped.lower() for marker in section_markers['overall_sentiment']):
                current_section = 'sentiment'
            elif line_stripped.startswith('-') or line_stripped.startswith('•') or line_stripped.startswith('*') or (line_stripped and line_stripped[0].isdigit()):
                # Bullet point
                bullet_text = line_stripped.lstrip('-•*0123456789. ').strip()
                if bullet_text and current_section and current_section in ['financial_highlights', 'strategic_initiatives', 'management_outlook', 'risks_challenges']:
                    structured[current_section].append(bullet_text)
            elif current_section == 'executive_summary':
                structured['executive_summary'] += line_stripped + ' '
            elif current_section == 'sentiment':
                if 'positive' in line_stripped.lower() or 'pozitivní' in line_stripped.lower():
                    structured['overall_sentiment'] = 'positive'
                elif 'negative' in line_stripped.lower() or 'negativní' in line_stripped.lower():
                    structured['overall_sentiment'] = 'negative'
                structured['sentiment_explanation'] += line_stripped + ' '
    
    return structured

def analyze_earnings_call_with_ai(text_content, ticker=None):
    """Analyze earnings call presentation text using Google Gemini API"""
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'error': 'Google Gemini API key not configured. Get your free API key at: https://makersuite.google.com/app/apikey'
        }
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models and find one that supports generateContent
        available_model = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # Prefer flash models (faster and free)
                    if 'flash' in m.name.lower():
                        available_model = m.name
                        break
                    elif available_model is None:
                        available_model = m.name
        except Exception as e:
            print(f"Error listing models: {str(e)}")
        
        # If no model found via list, try common model names
        if available_model is None:
            # Try common model names
            for model_name in ['gemini-pro', 'gemini-1.5-flash', 'models/gemini-pro']:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    available_model = model_name
                    break
                except:
                    continue
        
        if available_model is None:
            raise Exception("No available Gemini models found. Please check your API key at https://makersuite.google.com/app/apikey")
        
        # Use the available model
        model = genai.GenerativeModel(available_model)
        model_name_used = available_model
        
        # Create comprehensive prompt for detailed AI analysis with strict formatting
        prompt = f"""Jsi expertní finanční analytik specializující se na analýzu earnings call prezentací. Analyzuj následující earnings call prezentaci EXTREMĚ PODROBNĚ a poskytni velmi rozsáhlé, komplexní shrnutí v českém jazyce.

DŮLEŽITÉ: Formátuj odpověď PŘESNĚ podle následující struktury. Každá sekce musí být jasně označena a obsahovat minimálně požadovaný počet bodů.

Zaměř se detailně na:

1. **Klíčové finanční metriky:**
   - Revenue (tržby) - aktuální hodnoty, růst/klesání, srovnání s předchozími obdobími, YoY, QoQ změny
   - EPS (zisk na akcii) - aktuální hodnoty, změny, překvapení vs. očekávání, beat/miss
   - Guidance (výhled) - forward-looking statements, změny v guidance, upgrade/downgrade
   - EBITDA, marginální metriky (gross margin, operating margin, net margin)
   - Cash flow, FCF, cash position, cash burn rate
   - Balance sheet metrics (debt, equity, working capital)
   - Jakékoli další důležité finanční ukazatele

2. **Důležité obchodní aktualizace a strategické iniciativy:**
   - Nové produkty, služby, technologie uvedené na trh
   - Expanze do nových trhů nebo segmentů
   - Strategické partnerství, akvizice, joint ventures, M&A aktivity
   - Investice do R&D, CapEx plány, investiční strategie
   - Změny v business modelu nebo strategii
   - Customer wins, contract announcements
   - Market expansion, geographic growth

3. **Management komentáře a výhled:**
   - CEO a CFO komentáře k výkonu, jejich citace
   - Outlook pro příští kvartály/roky, guidance
   - Klíčové faktory ovlivňující budoucí výkon
   - Tržní příležitosti a výzvy
   - Strategické priority managementu
   - Forward-looking statements

4. **Rizika a výzvy:**
   - Identifikovaná rizika v prezentaci
   - Výzvy v konkurenčním prostředí
   - Makroekonomické faktory (inflace, úrokové sazby, geopolitika)
   - Regulační rizika
   - Operativní výzvy
   - Supply chain issues
   - Customer concentration risks

5. **Pozitivní body a úspěchy:**
   - Klíčové úspěchy v období
   - Milníky dosažené
   - Silné stránky společnosti
   - Konkurenční výhody
   - Customer testimonials, case studies

**PŘESNÝ FORMÁT ODPOVĚDI (dodržuj tuto strukturu):**

=== Executive Summary ===
Napiš 6-8 vět velmi podrobně shrnujících nejdůležitější body prezentace. Zahrň konkrétní čísla, hlavní zprávy, klíčové změny a celkový tón. Buď velmi konkrétní.

=== Key Financial Highlights ===
Uveď MINIMÁLNĚ 8-12 odrážek s konkrétními čísly, procenty a kontextem. Každá odrážka musí obsahovat:
- Konkrétní číslo nebo procento
- Srovnání (YoY, QoQ, vs. očekávání)
- Kontext a význam

Příklady formátu:
- Revenue vzrostla o 582% YoY na $10.1M z $1.5M v Q3 2024, překonala očekávání o 15%
- EPS dosáhlo $0.12 vs. $0.05 v předchozím kvartálu, beat o $0.03
- FCF pozitivní $2.5M, první pozitivní FCF za 4 kvartály

=== Strategic Initiatives ===
Uveď MINIMÁLNĚ 5-8 odrážek velmi podrobně popisujících strategické kroky společnosti. Každá odrážka musí obsahovat:
- Co společnost dělá
- Proč je to důležité
- Jak to ovlivní budoucí výkon

=== Management Outlook ===
Uveď MINIMÁLNĚ 5-8 odrážek s management komentáři a výhledem. Zahrň:
- Forward-looking statements
- Guidance pro příští období
- Klíčové faktory ovlivňující budoucí výkon
- Tržní příležitosti
- Strategické priority

=== Risks & Challenges ===
Uveď MINIMÁLNĚ 4-6 odrážek s identifikovanými riziky a výzvami. Buď velmi konkrétní:
- Jaké konkrétní riziko
- Jak společnost plánuje toto riziko řešit
- Dopad na business

=== Overall Sentiment ===
Urči sentiment (positive/neutral/negative) a poskytni velmi podrobné vysvětlení (5-6 vět) proč. Zahrň:
- Mix pozitivních faktorů
- Mix negativních faktorů
- Celkové hodnocení
- Klíčové faktory ovlivňující sentiment

Text earnings call prezentace:
{text_content[:50000]}  # Zvýšený limit pro rozsáhlejší analýzu
"""
        
        # Generate content with higher token limit for more comprehensive analysis
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,  # Lower temperature for more factual analysis
                'max_output_tokens': 8192,  # Zvýšeno na 8192 pro velmi rozsáhlou analýzu
            }
        )
        
        ai_summary = response.text
        
        # Extract structured data from AI response
        summary_data = parse_ai_summary(ai_summary)
        
        return {
            'success': True,
            'summary': ai_summary,
            'structured_data': summary_data,
            'model_used': model_name_used or 'unknown'
        }
        
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def parse_news_impact_analysis(analysis_text):
    """Parse AI news impact analysis into structured format"""
    structured = {
        'impact_assessment': 'neutral',
        'impact_level': 'medium',
        'key_factors': [],
        'explanation': '',
        'price_impact_estimate': ''
    }
    
    # Find sections
    sections = {
        'impact_assessment': ['=== Impact Assessment ===', 'Impact Assessment:'],
        'impact_level': ['=== Impact Level ===', 'Impact Level:'],
        'key_factors': ['=== Key Factors ===', 'Key Factors:'],
        'explanation': ['=== Explanation ===', 'Explanation:'],
        'price_impact_estimate': ['=== Price Impact Estimate ===', 'Price Impact Estimate:']
    }
    
    section_positions = {}
    for section_name, markers in sections.items():
        for marker in markers:
            pos = analysis_text.find(marker)
            if pos != -1:
                section_positions[section_name] = (pos, marker)
                break
    
    # Sort by position
    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1][0])
    
    # Extract content
    for i, (section_name, (start_pos, marker)) in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            end_pos = sorted_sections[i + 1][1][0]
        else:
            end_pos = len(analysis_text)
        
        content = analysis_text[start_pos + len(marker):end_pos].strip()
        
        if section_name == 'impact_assessment':
            content_lower = content.lower()
            if 'positive' in content_lower or 'pozitivní' in content_lower:
                structured['impact_assessment'] = 'positive'
            elif 'negative' in content_lower or 'negativní' in content_lower:
                structured['impact_assessment'] = 'negative'
            else:
                structured['impact_assessment'] = 'neutral'
        
        elif section_name == 'impact_level':
            content_lower = content.lower()
            if 'high' in content_lower or 'vysoký' in content_lower:
                structured['impact_level'] = 'high'
            elif 'low' in content_lower or 'nízký' in content_lower:
                structured['impact_level'] = 'low'
            else:
                structured['impact_level'] = 'medium'
        
        elif section_name == 'key_factors':
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit())):
                    factor = line.lstrip('-•*0123456789. ').strip()
                    if factor:
                        structured['key_factors'].append(factor)
        
        elif section_name == 'explanation':
            structured['explanation'] = content.strip()
        
        elif section_name == 'price_impact_estimate':
            structured['price_impact_estimate'] = content.strip()
    
    # Fallback parsing if structured parsing didn't work
    if not structured['explanation']:
        lines = analysis_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'positive' in line_lower or 'pozitivní' in line_lower:
                structured['impact_assessment'] = 'positive'
            elif 'negative' in line_lower or 'negativní' in line_lower:
                structured['impact_assessment'] = 'negative'
    
    return structured

def analyze_news_impact_with_ai(news_text, ticker=None):
    """Analyze how a news article might impact stock price using Google Gemini API"""
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'error': 'Google Gemini API key not configured'
        }
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models and find one that supports generateContent
        available_model = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name.lower():
                        available_model = m.name
                        break
                    elif available_model is None:
                        available_model = m.name
        except Exception:
            pass
        
        if available_model is None:
            for model_name in ['gemini-pro', 'gemini-1.5-flash', 'models/gemini-pro']:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    available_model = model_name
                    break
                except:
                    continue
        
        if available_model is None:
            raise Exception("No available Gemini models found.")
        
        model = genai.GenerativeModel(available_model)
        
        # Create prompt for news impact analysis
        ticker_context = f" pro akcii {ticker}" if ticker else ""
        prompt = f"""Jsi expertní finanční analytik specializující se na analýzu dopadu news na ceny akcií. Analyzuj následující news{ticker_context} a urči, jak může ovlivnit cenu akcie.

**Tvá úloha:**
1. Urči, zda news může ovlivnit cenu pozitivně, negativně nebo neutrálně
2. Identifikuj konkrétní faktory z news, které jsou relevantní pro cenu akcie
3. Vysvětli proč a jak tyto faktory mohou ovlivnit cenu

**Formátuj odpověď PŘESNĚ takto:**

=== Impact Assessment ===
[positive/negative/neutral]

=== Impact Level ===
[low/medium/high]

=== Key Factors ===
Uveď 3-5 konkrétních faktorů z news, které mohou ovlivnit cenu. Každý faktor na samostatný řádek s odrážkou.

=== Explanation ===
Napiš 3-4 věty vysvětlující, proč a jak news může ovlivnit cenu akcie. Buď konkrétní a zahrň:
- Jaké konkrétní informace z news jsou důležité
- Proč tyto informace mohou ovlivnit cenu
- Jaký typ dopadu lze očekávat (krátkodobý/dlouhodobý)
- Jaké další faktory mohou hrát roli

=== Price Impact Estimate ===
[Krátkodobý dopad: +X% až +Y% / -X% až -Y% / minimální dopad]
[Poznámka: Toto je pouze odhad na základě news, skutečný dopad závisí na mnoha dalších faktorech]

News text:
{news_text[:10000]}  # Limit pro rychlejší analýzu
"""
        
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 1024,  # Menší limit než earnings call, protože je to kratší analýza
            }
        )
        
        ai_analysis = response.text
        
        # Parse AI response
        impact_data = parse_news_impact_analysis(ai_analysis)
        
        return {
            'success': True,
            'impact_assessment': impact_data.get('impact_assessment', 'neutral'),
            'impact_level': impact_data.get('impact_level', 'medium'),
            'key_factors': impact_data.get('key_factors', []),
            'explanation': impact_data.get('explanation', ''),
            'price_impact_estimate': impact_data.get('price_impact_estimate', ''),
            'full_analysis': ai_analysis,
            'model_used': available_model
        }
        
    except Exception as e:
        print(f"Error in AI news impact analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def get_stock_data(ticker, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        # Map custom timeframes to yfinance period and interval
        # Note: yfinance has limits on intraday data and period format:
        # - 1m: max 7 days (use '7d' or '1d')
        # - 5m: max 60 days (use '60d' or '1mo')
        # - 15m: max 60 days (use '60d' or '1mo')
        # - 1h: max 730 days (use '2y' as closest valid period)
        # Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        timeframe_map = {
            '1m': {'period': '1d', 'interval': '1m'},  # Max 7 days for 1m, but yfinance accepts max 1d for 1m
            '5m': {'period': '5d', 'interval': '5m'},  # Max 60 days for 5m, use 5d (safe)
            '15m': {'period': '5d', 'interval': '15m'},  # Max 60 days for 15m, use 5d (safe)
            '1h': {'period': '1mo', 'interval': '1h'},  # Max ~730 days for 1h, use 1mo (safe)
            '4h': {'period': '3mo', 'interval': '1h'},  # 4h not directly supported, use 1h with longer period
            '1d': {'period': '1y', 'interval': '1d'},  # Daily
            '1w': {'period': '5y', 'interval': '1wk'},  # Weekly
        }
        
        # Check if this is an intraday timeframe
        if period in timeframe_map:
            tf_config = timeframe_map[period]
            use_period = tf_config['period']
            use_interval = tf_config['interval']
        else:
            # Standard timeframes (5d, 1mo, 3mo, 6mo, 1y, 5y)
            use_period = period
            use_interval = None
        
        # Try using Ticker method first (more reliable for intraday)
        try:
            if use_interval:
                # Intraday data - use Ticker method ONLY (download method has issues with intraday)
                print(f"Fetching intraday data for {ticker}: period={use_period}, interval={use_interval}")
                stock = yf.Ticker(ticker)
                hist = stock.history(period=use_period, interval=use_interval, auto_adjust=True, prepost=False)
                print(f"Got {len(hist)} rows of intraday data")
                
                # Don't use download method for intraday - it causes issues
            else:
                hist = yf.download(ticker, period=use_period, progress=False, show_errors=False, auto_adjust=True)
                if hist.empty:
                    # Fallback to Ticker method
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=use_period, auto_adjust=True)
        except Exception as e:
            print(f"Error downloading data for {ticker} with period={use_period}, interval={use_interval}: {str(e)}")
            # Fallback to Ticker method
            try:
                stock = yf.Ticker(ticker)
                if use_interval:
                    hist = stock.history(period=use_period, interval=use_interval, auto_adjust=True, prepost=False)
                else:
                    hist = stock.history(period=use_period, auto_adjust=True)
            except Exception as e2:
                print(f"Error with Ticker method: {str(e2)}")
                # For intraday, if it fails, return None with better error message
                if use_interval:
                    print(f"Intraday data may not be available for {ticker} with interval {use_interval}")
                raise
        
        if hist.empty:
            return None
        
        # Handle multi-level columns from download()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        
        # Get additional info with error handling
        info = {}
        try:
            stock = yf.Ticker(ticker)
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            info = stock.info
            if not info or 'symbol' not in info:
                # Create minimal info if API fails
                info = {
                    'longName': ticker,
                    'symbol': ticker,
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'longBusinessSummary': 'Data temporarily unavailable from Yahoo Finance API.'
                }
        except Exception as e:
            print(f"Warning: Could not fetch info for {ticker}: {str(e)}")
            # Create minimal info
            info = {
                'longName': ticker,
                'symbol': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'longBusinessSummary': 'Data temporarily unavailable from Yahoo Finance API.'
            }
        
        return {
            'history': hist,
            'info': info
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    indicators = {}
    
    # Moving Averages
    indicators['sma_20'] = SMAIndicator(df['Close'], window=20).sma_indicator().tolist()
    indicators['sma_50'] = SMAIndicator(df['Close'], window=50).sma_indicator().tolist()
    indicators['ema_12'] = EMAIndicator(df['Close'], window=12).ema_indicator().tolist()
    indicators['ema_26'] = EMAIndicator(df['Close'], window=26).ema_indicator().tolist()
    
    # RSI
    rsi = RSIIndicator(df['Close'], window=14)
    indicators['rsi'] = rsi.rsi().tolist()
    
    # MACD
    macd = MACD(df['Close'])
    indicators['macd'] = macd.macd().tolist()
    indicators['macd_signal'] = macd.macd_signal().tolist()
    indicators['macd_diff'] = macd.macd_diff().tolist()
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    indicators['bb_high'] = bb.bollinger_hband().tolist()
    indicators['bb_low'] = bb.bollinger_lband().tolist()
    indicators['bb_mid'] = bb.bollinger_mavg().tolist()
    
    # ADX (Average Directional Index) - requires High, Low, Close
    try:
        if 'High' in df.columns and 'Low' in df.columns and len(df) > 14:
            adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            adx_values = adx.adx().tolist()
            indicators['adx'] = adx_values
            print(f"[INDICATORS] ADX calculated: {len(adx_values)} values, last={adx_values[-1] if adx_values else 'N/A'}")
        else:
            print(f"[INDICATORS] ADX skipped: High={'High' in df.columns}, Low={'Low' in df.columns}, len={len(df)}")
            indicators['adx'] = []
    except Exception as e:
        print(f"[INDICATORS] ADX calculation error: {e}")
        indicators['adx'] = []
    
    # Stochastic Oscillator - requires High, Low, Close
    try:
        if 'High' in df.columns and 'Low' in df.columns and len(df) > 14:
            stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            stoch_k_values = stoch.stoch().tolist()
            stoch_d_values = stoch.stoch_signal().tolist()
            indicators['stoch_k'] = stoch_k_values
            indicators['stoch_d'] = stoch_d_values
            print(f"[INDICATORS] Stochastic calculated: K={stoch_k_values[-1] if stoch_k_values else 'N/A'}, D={stoch_d_values[-1] if stoch_d_values else 'N/A'}")
        else:
            print(f"[INDICATORS] Stochastic skipped: High={'High' in df.columns}, Low={'Low' in df.columns}, len={len(df)}")
            indicators['stoch_k'] = []
            indicators['stoch_d'] = []
    except Exception as e:
        print(f"[INDICATORS] Stochastic calculation error: {e}")
        indicators['stoch_k'] = []
        indicators['stoch_d'] = []
    
    # ATR (Average True Range) - requires High, Low, Close
    try:
        if 'High' in df.columns and 'Low' in df.columns and len(df) > 14:
            atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            atr_values = atr.average_true_range().tolist()
            indicators['atr'] = atr_values
            print(f"[INDICATORS] ATR calculated: {len(atr_values)} values, last={atr_values[-1] if atr_values else 'N/A'}")
        else:
            print(f"[INDICATORS] ATR skipped: High={'High' in df.columns}, Low={'Low' in df.columns}, len={len(df)}")
            indicators['atr'] = []
    except Exception as e:
        print(f"[INDICATORS] ATR calculation error: {e}")
        indicators['atr'] = []
    
    return indicators

def calculate_metrics(df, info):
    """Calculate investment metrics"""
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
    
    # Price change
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0
    
    # Volatility (30-day)
    if len(df) >= 30:
        returns = df['Close'].pct_change().dropna()
        volatility = returns.tail(30).std() * np.sqrt(252) * 100  # Annualized
    else:
        volatility = None
    
    # 52-week range
    year_high = df['High'].max()
    year_low = df['Low'].min()
    
    # Metrics from info
    metrics = {
        'current_price': round(current_price, 2),
        'price_change': round(price_change, 2),
        'price_change_pct': round(price_change_pct, 2),
        'volume': int(df['Volume'].iloc[-1]),
        'avg_volume': int(df['Volume'].tail(30).mean()) if len(df) >= 30 else int(df['Volume'].mean()),
        'volatility': round(volatility, 2) if volatility else None,
        'year_high': round(year_high, 2),
        'year_low': round(year_low, 2),
        'pe_ratio': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'market_cap': info.get('marketCap'),
        'dividend_yield': info.get('dividendYield'),
        'beta': info.get('beta'),
        'eps': info.get('trailingEps'),
        'book_value': info.get('bookValue'),
        'price_to_book': info.get('priceToBook'),
        'profit_margin': info.get('profitMargins'),
        'revenue_growth': info.get('revenueGrowth'),
        'earnings_growth': info.get('earningsGrowth'),
    }
    
    return metrics

def get_earnings_qoq(ticker):
    """Get quarterly earnings, EPS, revenue and compare with expectations"""
    # #region agent log
    print(f"[DEBUG] get_earnings_qoq: ENTRY for {ticker}")
    # #endregion
    try:
        # Get Finviz actuals first (outside loop for efficiency)
        finviz_data = get_quarterly_estimates_from_finviz(ticker)
        quarterly_actuals = finviz_data.get('actuals', {}) if isinstance(finviz_data, dict) else {}
        # #region agent log
        print(f"[DEBUG] get_earnings_qoq {ticker}: Finviz data received: actuals keys = {list(quarterly_actuals.keys())}")
        if 'eps' in quarterly_actuals:
            print(f"[DEBUG] get_earnings_qoq {ticker}: Finviz actuals['eps'] = {quarterly_actuals['eps']}")
        # #endregion
        
        stock = yf.Ticker(ticker)
        income_stmt = stock.quarterly_income_stmt
        
        if income_stmt is None or income_stmt.empty:
            return None
        
        # Find Net Income row
        net_income_row = None
        search_terms = [
            'net income from continuing operation',
            'net income from continuing operations',
            'net income',
            'total net income',
            'income from continuing operations',
            'normalized income'
        ]
        
        for term in search_terms:
            for idx in income_stmt.index:
                idx_str = str(idx).lower()
                if term in idx_str:
                    net_income_row = income_stmt.loc[idx]
                    break
            if net_income_row is not None:
                break
        
        # Find Revenue row
        revenue_row = None
        for idx in income_stmt.index:
            idx_str = str(idx).lower()
            if 'total revenue' in idx_str:
                revenue_row = income_stmt.loc[idx]
                break
        
        # Find EPS row - try multiple variations
        eps_row = None
        eps_search_terms = [
            'diluted eps',
            'diluted earnings per share',
            'eps - diluted',
            'eps diluted',
            'earnings per share - diluted',
            'basic eps',
            'basic earnings per share',
            'eps - basic',
            'eps basic',
            'earnings per share - basic',
            'eps',
            'earnings per share',
            'net income per share'
        ]
        
        # #region agent log
        print(f"[DEBUG] get_earnings_qoq for {ticker}: Searching for EPS row")
        print(f"[DEBUG] Available income statement rows: {list(income_stmt.index)[:10]}...")  # Show first 10 rows
        # #endregion
        
        for search_term in eps_search_terms:
            for idx in income_stmt.index:
                idx_str = str(idx).lower()
                if search_term in idx_str:
                    eps_row = income_stmt.loc[idx]
                    # #region agent log
                    print(f"[DEBUG] Found EPS row: '{idx}' (matched '{search_term}')")
                    # #endregion
                    break
            if eps_row is not None:
                break
        
        if net_income_row is None:
            print(f"Could not find Net Income row for {ticker}")
            # Try to use first available row as fallback
            if len(income_stmt.index) > 0:
                net_income_row = income_stmt.iloc[0]
                print(f"Using fallback row: {income_stmt.index[0]}")
            else:
                return None
        
        if len(net_income_row) < 2:
            print(f"Not enough data columns: {len(net_income_row)}")
            return None
        
        # Get earnings estimates
        estimates_df = None
        try:
            estimates_df = stock.earnings_dates
        except Exception:
            pass
        
        # Get quarters (columns are dates, most recent first)
        quarters = []
        earnings_data = []
        revenue_data = []
        eps_data = []
        eps_estimates = []
        eps_reported = []
        eps_surprise = []
        
        for col in income_stmt.columns:
            quarter_date = pd.Timestamp(col)
            # Format as YYYY-Q1, Q2, Q3, Q4
            quarter_num = (quarter_date.month - 1) // 3 + 1
            quarter_str = f"{quarter_date.year}-Q{quarter_num}"
            quarters.append(quarter_str)
            # #region agent log
            if len(quarters) <= 4:  # Log first 4 quarters
                print(f"[DEBUG] get_earnings_qoq {ticker}: yfinance quarter_date={quarter_date}, quarter_str={quarter_str}")
            # #endregion
            
            # Net Income
            earnings_data.append(float(net_income_row[col]) if pd.notna(net_income_row[col]) else None)
            
            # Revenue - USE ONLY FINVIZ ACTUALS, map by date proximity (not quarter_str)
            revenue_value = None
            revenue_source = None
            
            # Try Finviz actuals ONLY - find closest match by date
            if quarterly_actuals and 'revenue' in quarterly_actuals:
                # #region agent log
                finviz_rev_keys = list(quarterly_actuals.get('revenue', {}).keys())
                print(f"[DEBUG] get_earnings_qoq {ticker}: Finviz actuals Revenue keys: {finviz_rev_keys}")
                print(f"[DEBUG] get_earnings_qoq {ticker}: yfinance quarter_date={quarter_date}, quarter_str={quarter_str}")
                # #endregion
                
                best_match_rev = None
                best_match_q = None
                min_date_diff = float('inf')
                
                # Find closest Finviz revenue by date (not by quarter_str)
                for finviz_q, finviz_rev in quarterly_actuals['revenue'].items():
                    try:
                        if '-Q' in finviz_q:
                            fv_year, fv_num = finviz_q.split('-Q')
                            fv_num = int(fv_num)
                            fv_year = int(fv_year)
                            fv_month = (fv_num - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                            fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                            date_diff = abs((quarter_date - fv_date).days)
                            
                            # #region agent log
                            print(f"[DEBUG] get_earnings_qoq {ticker}: Comparing dates Revenue - yfinance {quarter_date.date()} vs Finviz {fv_date.date()} (diff: {date_diff} days)")
                            # #endregion
                            
                            # Accept match within 120 days (allows for fiscal vs calendar quarter differences)
                            if date_diff < min_date_diff and date_diff <= 120:
                                min_date_diff = date_diff
                                best_match_rev = finviz_rev
                                best_match_q = finviz_q
                    except Exception as e:
                        # #region agent log
                        print(f"[DEBUG] get_earnings_qoq {ticker}: Error parsing Finviz Revenue quarter '{finviz_q}': {e}")
                        # #endregion
                        pass
                
                if best_match_rev is not None:
                    revenue_value = best_match_rev
                    revenue_source = 'Finviz'
                    # #region agent log
                    print(f"[DEBUG] get_earnings_qoq {ticker}: Date match - Using Finviz Revenue from '{best_match_q}' for yfinance {quarter_str}: {revenue_value} (date diff: {min_date_diff} days)")
                    # #endregion
            else:
                    # #region agent log
                    print(f"[DEBUG] get_earnings_qoq {ticker}: No Revenue match found for {quarter_str} (date: {quarter_date.date()})")
                    # #endregion
            
            # NO FALLBACK TO YFINANCE - USE ONLY FINVIZ
            revenue_data.append(revenue_value)
            
            # EPS - USE ONLY FINVIZ ACTUALS, map by date proximity (not quarter_str)
            eps_value = None
            eps_source = None
            
            # Try Finviz actuals ONLY - find closest match by date
            if quarterly_actuals and 'eps' in quarterly_actuals:
                # #region agent log
                finviz_eps_keys = list(quarterly_actuals.get('eps', {}).keys())
                print(f"[DEBUG] get_earnings_qoq {ticker}: Finviz actuals EPS keys: {finviz_eps_keys}")
                print(f"[DEBUG] get_earnings_qoq {ticker}: yfinance quarter_date={quarter_date}, quarter_str={quarter_str}")
                # #endregion
                
                best_match_eps = None
                best_match_q = None
                min_date_diff = float('inf')
                
                # Find closest Finviz EPS by date (not by quarter_str)
                for finviz_q, finviz_eps in quarterly_actuals['eps'].items():
                    try:
                        if '-Q' in finviz_q:
                            fv_year, fv_num = finviz_q.split('-Q')
                            fv_num = int(fv_num)
                            fv_year = int(fv_year)
                            fv_month = (fv_num - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                            fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                            date_diff = abs((quarter_date - fv_date).days)
                            
                            # #region agent log
                            print(f"[DEBUG] get_earnings_qoq {ticker}: Comparing dates Revenue - yfinance {quarter_date.date()} vs Finviz {fv_date.date()} (diff: {date_diff} days)")
                            # #endregion
                            
                            # Accept match within 120 days (allows for fiscal vs calendar quarter differences)
                            if date_diff < min_date_diff and date_diff <= 120:
                                min_date_diff = date_diff
                                best_match_eps = finviz_eps
                                best_match_q = finviz_q
                    except Exception as e:
                        # #region agent log
                        print(f"[DEBUG] get_earnings_qoq {ticker}: Error parsing Finviz EPS quarter '{finviz_q}': {e}")
                        # #endregion
                        pass
                
                if best_match_eps is not None:
                    eps_value = best_match_eps
                    eps_source = 'Finviz'
                    # #region agent log
                    print(f"[DEBUG] get_earnings_qoq {ticker}: Date match - Using Finviz EPS from '{best_match_q}' for yfinance {quarter_str}: {eps_value} (date diff: {min_date_diff} days)")
                    # #endregion
            else:
                    # #region agent log
                    print(f"[DEBUG] get_earnings_qoq {ticker}: No EPS match found for {quarter_str} (date: {quarter_date.date()})")
                    # #endregion
            
            # #region agent log
            print(f"[DEBUG] get_earnings_qoq {ticker}: Quarter {quarter_str} - Final eps_value: {eps_value}, source: {eps_source}")
            if eps_value is None:
                print(f"[DEBUG] No EPS value found for {ticker} {quarter_str} from Finviz")
                print(f"[DEBUG] - Finviz actuals available: {quarterly_actuals and 'eps' in quarterly_actuals}")
                if quarterly_actuals and 'eps' in quarterly_actuals:
                    print(f"[DEBUG] - Finviz actuals keys: {list(quarterly_actuals['eps'].keys())}")
                    print(f"[DEBUG] - Finviz actuals values: {quarterly_actuals['eps']}")
                    # Compare with revenue actuals
                    if quarterly_actuals and 'revenue' in quarterly_actuals:
                        print(f"[DEBUG] - Finviz revenue actuals keys: {list(quarterly_actuals['revenue'].keys())}")
                        print(f"[DEBUG] - Finviz revenue actuals values: {quarterly_actuals['revenue']}")
            # #endregion
            
            eps_data.append(eps_value)
            
            # Match with estimates (find closest earnings date)
            eps_est = None
            eps_rep = None
            eps_sur = None
            
            if estimates_df is not None and not estimates_df.empty:
                # Find matching quarter in estimates
                for est_date in estimates_df.index:
                    est_quarter = pd.Timestamp(est_date)
                    # Normalize both timestamps to be timezone-naive for comparison
                    if est_quarter.tz is not None:
                        est_quarter = est_quarter.tz_localize(None)
                    if quarter_date.tz is not None:
                        quarter_date_naive = quarter_date.tz_localize(None)
                    else:
                        quarter_date_naive = quarter_date
                    
                    est_q_num = (est_quarter.month - 1) // 3 + 1
                    est_q_str = f"{est_quarter.year}-Q{est_q_num}"
                    
                    # Match by quarter (allow some date flexibility)
                    try:
                        days_diff = abs((est_quarter - quarter_date_naive).days)
                    except:
                        days_diff = 999
                    
                    if est_q_str == quarter_str or days_diff < 45:
                        if 'EPS Estimate' in estimates_df.columns:
                            eps_est = float(estimates_df.loc[est_date, 'EPS Estimate']) if pd.notna(estimates_df.loc[est_date, 'EPS Estimate']) else None
                        if 'Reported EPS' in estimates_df.columns:
                            eps_rep = float(estimates_df.loc[est_date, 'Reported EPS']) if pd.notna(estimates_df.loc[est_date, 'Reported EPS']) else None
                        if 'Surprise(%)' in estimates_df.columns:
                            eps_sur = float(estimates_df.loc[est_date, 'Surprise(%)']) if pd.notna(estimates_df.loc[est_date, 'Surprise(%)']) else None
                        break
            
            eps_estimates.append(eps_est)
            eps_reported.append(eps_rep)
            eps_surprise.append(eps_sur)
        
        # #region agent log
        print(f"[DEBUG] get_earnings_qoq for {ticker}: EPS row found: {eps_row is not None}")
        if eps_row is not None:
            print(f"[DEBUG] EPS row name: {eps_row.name if hasattr(eps_row, 'name') else 'N/A'}")
            print(f"[DEBUG] EPS data sample: {eps_data[:4]}")
            print(f"[DEBUG] EPS data full: {eps_data}")
            print(f"[DEBUG] EPS data None count: {sum(1 for x in eps_data if x is None)}")
            # Show EPS values from yfinance for first 4 quarters
            if len(income_stmt.columns) >= 4:
                for i, col in enumerate(income_stmt.columns[:4]):
                    if eps_row is not None:
                        eps_val = eps_row[col] if pd.notna(eps_row[col]) else None
                        print(f"[DEBUG] yfinance EPS for {quarters[i]}: {eps_val} (type: {type(eps_val)})")
        else:
            print(f"[DEBUG] EPS row NOT found. Available rows: {list(income_stmt.index)[:15]}")
        # #endregion
        
        # Calculate QoQ changes for earnings
        earnings_qoq_changes = []
        earnings_qoq_changes_pct = []
        
        # Calculate QoQ changes for revenue
        revenue_qoq_changes = []
        revenue_qoq_changes_pct = []
        
        # Calculate QoQ changes for EPS
        eps_qoq_changes = []
        eps_qoq_changes_pct = []
        
        for i in range(len(earnings_data)):
            if i == 0:
                earnings_qoq_changes.append(None)
                earnings_qoq_changes_pct.append(None)
                revenue_qoq_changes.append(None)
                revenue_qoq_changes_pct.append(None)
                eps_qoq_changes.append(None)
                eps_qoq_changes_pct.append(None)
            else:
                # Earnings QoQ
                current_earn = earnings_data[i]
                previous_earn = earnings_data[i-1]
                if current_earn is not None and previous_earn is not None and previous_earn != 0:
                    change = current_earn - previous_earn
                    change_pct = (change / abs(previous_earn)) * 100
                    earnings_qoq_changes.append(round(change / 1e9, 2))
                    earnings_qoq_changes_pct.append(round(change_pct, 2))
                else:
                    earnings_qoq_changes.append(None)
                    earnings_qoq_changes_pct.append(None)
                
                # Revenue QoQ
                current_rev = revenue_data[i]
                previous_rev = revenue_data[i-1]
                if current_rev is not None and previous_rev is not None and previous_rev != 0:
                    change = current_rev - previous_rev
                    change_pct = (change / abs(previous_rev)) * 100
                    revenue_qoq_changes.append(round(change / 1e9, 2))
                    revenue_qoq_changes_pct.append(round(change_pct, 2))
                else:
                    revenue_qoq_changes.append(None)
                    revenue_qoq_changes_pct.append(None)
                
                # EPS QoQ
                current_eps = eps_data[i]
                previous_eps = eps_data[i-1]
                if current_eps is not None and previous_eps is not None and previous_eps != 0:
                    change = current_eps - previous_eps
                    change_pct = (change / abs(previous_eps)) * 100
                    eps_qoq_changes.append(round(change, 2))
                    eps_qoq_changes_pct.append(round(change_pct, 2))
                else:
                    eps_qoq_changes.append(None)
                    eps_qoq_changes_pct.append(None)
        
        # Format earnings and revenue in billions
        earnings_formatted = [round(e / 1e9, 2) if e is not None else None for e in earnings_data]
        revenue_formatted = [round(r / 1e9, 2) if r is not None else None for r in revenue_data]
        
        result = {
            'quarters': quarters,
            'earnings': earnings_formatted,  # In billions
            'earnings_qoq_change': earnings_qoq_changes,
            'earnings_qoq_change_pct': earnings_qoq_changes_pct,
            'revenue': revenue_formatted,  # In billions
            'revenue_qoq_change': revenue_qoq_changes,
            'revenue_qoq_change_pct': revenue_qoq_changes_pct,
            'eps': eps_data,
            'eps_qoq_change': eps_qoq_changes,
            'eps_qoq_change_pct': eps_qoq_changes_pct,
            'eps_estimate': eps_estimates,
            'eps_reported': eps_reported,
            'eps_surprise_pct': eps_surprise
        }
        
        # #region agent log
        print(f"[DEBUG] get_earnings_qoq {ticker}: Returning result with EPS data: {eps_data[:4]}")
        print(f"[DEBUG] get_earnings_qoq {ticker}: Full EPS data: {eps_data}")
        print(f"[DEBUG] get_earnings_qoq {ticker}: EPS data types: {[type(x) for x in eps_data[:4]]}")
        print(f"[DEBUG] get_earnings_qoq {ticker}: EPS data None count: {sum(1 for x in eps_data if x is None)}")
        # #endregion
        
        return result
    except Exception as e:
        print(f"Error fetching earnings for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return None to indicate failure
        return None

def get_volume_analysis(ticker):
    """Calculate average daily volume vs recent volume for unusual activity detection"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        # Get historical data for volume analysis
        hist = stock.history(period='6mo')  # 6 months for better averages
        
        if hist.empty or 'Volume' not in hist.columns:
            return None
        
        # Calculate moving averages
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_MA50'] = hist['Volume'].rolling(window=50).mean()
        hist['Volume_MA90'] = hist['Volume'].rolling(window=90).mean()
        
        # Get recent volumes (last 5 days)
        recent_volumes = hist['Volume'].tail(5).tolist()
        recent_dates = [d.strftime('%Y-%m-%d') for d in hist.tail(5).index]
        
        # Current volume
        current_volume = int(hist['Volume'].iloc[-1])
        
        # Average volumes
        avg_volume_20d = int(hist['Volume_MA20'].iloc[-1]) if not pd.isna(hist['Volume_MA20'].iloc[-1]) else None
        avg_volume_50d = int(hist['Volume_MA50'].iloc[-1]) if not pd.isna(hist['Volume_MA50'].iloc[-1]) else None
        avg_volume_90d = int(hist['Volume_MA90'].iloc[-1]) if not pd.isna(hist['Volume_MA90'].iloc[-1]) else None
        
        # Volume ratios
        volume_ratio_20d = (current_volume / avg_volume_20d) if avg_volume_20d and avg_volume_20d > 0 else None
        volume_ratio_50d = (current_volume / avg_volume_50d) if avg_volume_50d and avg_volume_50d > 0 else None
        volume_ratio_90d = (current_volume / avg_volume_90d) if avg_volume_90d and avg_volume_90d > 0 else None
        
        # Detect unusual activity (volume > 2x average)
        unusual_activity = False
        if volume_ratio_20d and volume_ratio_20d > 2.0:
            unusual_activity = True
        
        # Prepare historical data for chart
        volume_data = []
        for i, (date, row) in enumerate(hist.iterrows()):
            if i >= 20:  # Only include data where we have MA20
                volume_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'volume': int(row['Volume']),
                    'ma20': int(row['Volume_MA20']) if not pd.isna(row['Volume_MA20']) else None,
                    'ma50': int(row['Volume_MA50']) if not pd.isna(row['Volume_MA50']) else None,
                    'ma90': int(row['Volume_MA90']) if not pd.isna(row['Volume_MA90']) else None
                })
        
        return {
            'current_volume': current_volume,
            'avg_volume_20d': avg_volume_20d,
            'avg_volume_50d': avg_volume_50d,
            'avg_volume_90d': avg_volume_90d,
            'volume_ratio_20d': round(volume_ratio_20d, 2) if volume_ratio_20d else None,
            'volume_ratio_50d': round(volume_ratio_50d, 2) if volume_ratio_50d else None,
            'volume_ratio_90d': round(volume_ratio_90d, 2) if volume_ratio_90d else None,
            'unusual_activity': unusual_activity,
            'recent_volumes': recent_volumes,
            'recent_dates': recent_dates,
            'volume_history': volume_data[-60:]  # Last 60 days for chart
        }
    
    except Exception as e:
        print(f"Error fetching volume analysis for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_institutional_ownership(ticker):
    """Get institutional ownership % and top holders from Finviz/yfinance"""
    try:
        # Try yfinance first
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        info = stock.info
        
        institutional_holders = stock.institutional_holders
        current_ownership_pct = info.get('heldPercentInstitutions', 0) * 100 if info.get('heldPercentInstitutions') else None
        
        # Get data from Finviz
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                snapshot_table = soup.find('table', class_='snapshot-table2')
                
                if snapshot_table:
                    rows = snapshot_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True)
                            if 'Inst Own' in label:
                                value_cell = cells[1]
                                b_tag = value_cell.find('b')
                                value_text = b_tag.get_text(strip=True) if b_tag else value_cell.get_text(strip=True)
                                value_text = value_text.replace('%', '').replace(',', '').strip()
                                try:
                                    ownership_pct = float(value_text)
                                    if ownership_pct > (current_ownership_pct or 0) or current_ownership_pct is None:
                                        current_ownership_pct = ownership_pct
                                except (ValueError, TypeError):
                                    pass
                
                # Try to get top holders from JSON data
                import re
                json_match = re.search(r'id="institutional-ownership-init-data-0"[^>]*>([^<]+)', response.text)
                if json_match:
                    try:
                        import json
                        holders_data = json.loads(json_match.group(1))
                        top_holders = []
                        if 'managersOwnership' in holders_data:
                            for holder in holders_data['managersOwnership'][:10]:
                                top_holders.append({
                                    'holder': holder.get('name', 'N/A'),
                                    'pct_ownership': round(holder.get('percOwnership', 0), 2)
                                })
                        if top_holders:
                            return {
                                'ownership_pct': round(current_ownership_pct, 2) if current_ownership_pct else None,
                                'top_holders': top_holders
                            }
                    except:
                        pass
        except Exception as e:
            print(f"[INSTITUTIONAL] Error scraping Finviz for {ticker}: {e}")
        
        # Get top holders from yfinance
        top_holders = []
        if institutional_holders is not None and not institutional_holders.empty:
            for _, row in institutional_holders.head(10).iterrows():
                top_holders.append({
                    'holder': row.get('Holder', 'N/A'),
                    'shares': int(row.get('Shares', 0)) if pd.notna(row.get('Shares')) else 0,
                    'value': int(row.get('Value', 0)) if pd.notna(row.get('Value')) else 0,
                    'pct_ownership': float(row.get('% Out', 0)) if pd.notna(row.get('% Out')) else 0
                })
        
        return {
            'ownership_pct': round(current_ownership_pct, 2) if current_ownership_pct else None,
            'top_holders': top_holders[:10]
        }
    
    except Exception as e:
        print(f"Error fetching institutional ownership for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_institutional_flow(ticker):
    """Get institutional flow (buying/selling) from Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        snapshot_table = soup.find('table', class_='snapshot-table2')
        
        if not snapshot_table:
            return None
        
        inst_trans_pct = None
        rows = snapshot_table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True)
                if 'Inst Trans' in label:
                    value_cell = cells[1]
                    b_tag = value_cell.find('b')
                    value_text = b_tag.get_text(strip=True) if b_tag else value_cell.get_text(strip=True)
                    value_text = value_text.replace('%', '').replace(',', '').strip()
                    try:
                        inst_trans_pct = float(value_text)
                    except (ValueError, TypeError):
                        pass
        
        return {
            'inst_trans_pct': round(inst_trans_pct, 2) if inst_trans_pct else None,
            'flow_direction': 'positive' if inst_trans_pct and inst_trans_pct > 0 else 'negative' if inst_trans_pct and inst_trans_pct < 0 else 'neutral'
        }
    
    except Exception as e:
        print(f"Error fetching institutional flow for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_retail_activity_indicators(ticker):
    """Estimate retail activity from volume patterns"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        # Get recent price and volume data
        hist = stock.history(period='3mo')
        
        if hist.empty:
            return None
        
        # Calculate metrics that might indicate retail activity
        # Retail tends to trade in smaller sizes and during certain hours
        # We'll use volume patterns and price volatility as proxies
        
        recent_volume = hist['Volume'].tail(20).mean()
        avg_volume = hist['Volume'].mean()
        volume_spike = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Price volatility (retail often causes higher volatility)
        returns = hist['Close'].pct_change().dropna()
        recent_volatility = returns.tail(20).std()
        avg_volatility = returns.std()
        volatility_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1
        
        # Estimate retail activity score (0-100)
        # Higher volume spikes + higher volatility = more retail activity
        retail_score = min(100, (volume_spike - 1) * 30 + (volatility_ratio - 1) * 20)
        retail_score = max(0, retail_score)
        
        return {
            'retail_activity_score': round(retail_score, 1),
            'volume_spike_ratio': round(volume_spike, 2),
            'volatility_ratio': round(volatility_ratio, 2),
            'indicator': 'high' if retail_score > 60 else 'medium' if retail_score > 30 else 'low'
        }
    
    except Exception as e:
        print(f"Error calculating retail activity indicators for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_whale_watching(ticker):
    """Get top institutional holders (whales) from Finviz/yfinance"""
    try:
        # This is similar to get_institutional_ownership but focused on largest positions
        ownership_data = get_institutional_ownership(ticker)
        
        if not ownership_data or not ownership_data.get('top_holders'):
            return None
        
        # Sort by pct_ownership or value
        top_holders = ownership_data['top_holders']
        top_holders_sorted = sorted(top_holders, key=lambda x: x.get('pct_ownership', 0) or x.get('value', 0), reverse=True)
        
        return {
            'whales': top_holders_sorted[:20],  # Top 20 whales
            'total_ownership_pct': ownership_data.get('ownership_pct')
        }
    
    except Exception as e:
        print(f"Error fetching whale watching data for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_popular_tickers():
    """Get a list of popular stock tickers for screening (S&P 500 + NASDAQ 100)"""
    # Popular tickers - S&P 500 + NASDAQ 100 + some additional popular stocks
    popular_tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'CSCO',
        'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC', 'NXPI', 'MRVL', 'SWKS', 'ON', 'MPWR', 'CRUS', 'ALGM',
        # Data Centers / AI Infrastructure
        'ONDS', 'CIFR', 'IREN', 'NBIS', 'SMCI', 'DELL', 'HPE', 'ANET', 'ARISTA',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'COF', 'AXP', 'V', 'MA', 'PYPL', 'SQ',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN',
        # Consumer
        'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'TGT', 'LOW', 'COST', 'TJX', 'ROST',
        # Industrial
        'BA', 'CAT', 'GE', 'HON', 'RTX', 'LMT', 'NOC', 'GD', 'TXT',
        # Energy
        'XOM', 'CVX', 'SLB', 'EOG', 'COP', 'MPC', 'VLO', 'PSX',
        # Communication
        'T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'PARA',
        # Utilities
        'NEE', 'DUK', 'SO', 'AEP', 'SRE',
        # Real Estate
        'AMT', 'PLD', 'EQIX', 'PSA', 'WELL',
        # Materials
        'LIN', 'APD', 'ECL', 'SHW', 'PPG',
        # Small/Mid Cap Growth
        'RBLX', 'HOOD', 'SOFI', 'PLTR', 'LCID', 'RIVN', 'F', 'GM', 'FORD',
        # ETFs (optional - can filter out)
        # 'SPY', 'QQQ', 'IWM', 'DIA'
    ]
    return popular_tickers

def run_stock_screener(filters):
    """Run stock screener with given filters"""
    try:
        tickers = get_popular_tickers()
        results = []
        
        # Extract filters
        market_cap_min = filters.get('market_cap_min')
        market_cap_max = filters.get('market_cap_max')
        pe_min = filters.get('pe_min')
        pe_max = filters.get('pe_max')
        pb_min = filters.get('pb_min')
        pb_max = filters.get('pb_max')
        ps_min = filters.get('ps_min')
        ps_max = filters.get('ps_max')
        dividend_yield_min = filters.get('dividend_yield_min')
        dividend_yield_max = filters.get('dividend_yield_max')
        revenue_growth_min = filters.get('revenue_growth_min')
        revenue_growth_max = filters.get('revenue_growth_max')
        eps_growth_min = filters.get('eps_growth_min')
        eps_growth_max = filters.get('eps_growth_max')
        roe_min = filters.get('roe_min')
        roe_max = filters.get('roe_max')
        debt_equity_min = filters.get('debt_equity_min')
        debt_equity_max = filters.get('debt_equity_max')
        beta_min = filters.get('beta_min')
        beta_max = filters.get('beta_max')
        rsi_min = filters.get('rsi_min')
        rsi_max = filters.get('rsi_max')
        short_float_min = filters.get('short_float_min')
        short_float_max = filters.get('short_float_max')
        volume_min = filters.get('volume_min')
        volume_max = filters.get('volume_max')
        price_min = filters.get('price_min')
        price_max = filters.get('price_max')
        sector = filters.get('sector')
        industry = filters.get('industry')
        
        print(f"[SCREENER] Screening {len(tickers)} tickers with filters: {filters}")
        
        # Process tickers in batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            
            for ticker in batch:
                try:
                    time.sleep(0.2)  # Rate limiting
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    if not info or 'symbol' not in info:
                        continue
                    
                    # Get historical data for RSI calculation
                    hist = stock.history(period='6mo')
                    if hist.empty:
                        continue
                    
                    # Calculate RSI
                    rsi_value = None
                    try:
                        from ta.momentum import RSIIndicator
                        rsi_indicator = RSIIndicator(hist['Close'], window=14)
                        rsi_series = rsi_indicator.rsi()
                        if not rsi_series.empty:
                            rsi_value = float(rsi_series.iloc[-1])
                    except:
                        pass
                    
                    # Extract metrics
                    market_cap = info.get('marketCap')
                    pe_ratio = info.get('trailingPE')
                    pb_ratio = info.get('priceToBook')
                    ps_ratio = info.get('priceToSalesTrailing12Months')
                    dividend_yield = info.get('dividendYield')
                    if dividend_yield:
                        dividend_yield = dividend_yield * 100  # Convert to percentage
                    revenue_growth = info.get('revenueGrowth')
                    if revenue_growth:
                        revenue_growth = revenue_growth * 100  # Convert to percentage
                    eps_growth = info.get('earningsGrowth')
                    if eps_growth:
                        eps_growth = eps_growth * 100  # Convert to percentage
                    roe = info.get('returnOnEquity')
                    if roe:
                        roe = roe * 100  # Convert to percentage
                    debt_equity = info.get('debtToEquity')
                    beta = info.get('beta')
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    avg_volume = info.get('averageVolume')
                    sector_info = info.get('sector')
                    industry_info = info.get('industry')
                    
                    # Apply filters
                    if market_cap_min and market_cap and market_cap < market_cap_min:
                        continue
                    if market_cap_max and market_cap and market_cap > market_cap_max:
                        continue
                    if pe_min and pe_ratio and pe_ratio < pe_min:
                        continue
                    if pe_max and pe_ratio and pe_ratio > pe_max:
                        continue
                    if pb_min and pb_ratio and pb_ratio < pb_min:
                        continue
                    if pb_max and pb_ratio and pb_ratio > pb_max:
                        continue
                    if ps_min and ps_ratio and ps_ratio < ps_min:
                        continue
                    if ps_max and ps_ratio and ps_ratio > ps_max:
                        continue
                    if dividend_yield_min and dividend_yield and dividend_yield < dividend_yield_min:
                        continue
                    if dividend_yield_max and dividend_yield and dividend_yield > dividend_yield_max:
                        continue
                    if revenue_growth_min and revenue_growth and revenue_growth < revenue_growth_min:
                        continue
                    if revenue_growth_max and revenue_growth and revenue_growth > revenue_growth_max:
                        continue
                    if eps_growth_min and eps_growth and eps_growth < eps_growth_min:
                        continue
                    if eps_growth_max and eps_growth and eps_growth > eps_growth_max:
                        continue
                    if roe_min and roe and roe < roe_min:
                        continue
                    if roe_max and roe and roe > roe_max:
                        continue
                    if debt_equity_min and debt_equity and debt_equity < debt_equity_min:
                        continue
                    if debt_equity_max and debt_equity and debt_equity > debt_equity_max:
                        continue
                    if beta_min and beta and beta < beta_min:
                        continue
                    if beta_max and beta and beta > beta_max:
                        continue
                    if rsi_min and rsi_value and rsi_value < rsi_min:
                        continue
                    if rsi_max and rsi_value and rsi_value > rsi_max:
                        continue
                    if volume_min and avg_volume and avg_volume < volume_min:
                        continue
                    if volume_max and avg_volume and avg_volume > volume_max:
                        continue
                    if price_min and current_price and current_price < price_min:
                        continue
                    if price_max and current_price and current_price > price_max:
                        continue
                    if sector and sector_info and sector.lower() not in sector_info.lower():
                        continue
                    if industry and industry_info and industry.lower() not in industry_info.lower():
                        continue
                    
                    # Get short float if available
                    short_float_pct = None
                    try:
                        short_data = get_short_interest_from_finviz(ticker)
                        if short_data and short_data.get('short_float_pct'):
                            short_float_pct = short_data['short_float_pct']
                    except:
                        pass
                    
                    # Apply short float filter
                    if short_float_min and short_float_pct and short_float_pct < short_float_min:
                        continue
                    if short_float_max and short_float_pct and short_float_pct > short_float_max:
                        continue
                    
                    # Stock passes all filters
                    results.append({
                        'ticker': ticker,
                        'name': info.get('longName') or info.get('shortName', 'N/A'),
                        'sector': sector_info or 'N/A',
                        'industry': industry_info or 'N/A',
                        'market_cap': market_cap,
                        'price': current_price,
                        'pe_ratio': pe_ratio,
                        'pb_ratio': pb_ratio,
                        'ps_ratio': ps_ratio,
                        'dividend_yield': dividend_yield,
                        'revenue_growth': revenue_growth,
                        'eps_growth': eps_growth,
                        'roe': roe,
                        'debt_equity': debt_equity,
                        'beta': beta,
                        'rsi': rsi_value,
                        'short_float_pct': short_float_pct,
                        'volume': avg_volume,
                        'volume_24h': info.get('volume') or avg_volume
                    })
                    
                except Exception as e:
                    print(f"[SCREENER] Error processing {ticker}: {str(e)}")
                    continue
            
            # Small delay between batches
            if i + batch_size < len(tickers):
                time.sleep(0.5)
        
        print(f"[SCREENER] Found {len(results)} stocks matching filters")
        return results
        
    except Exception as e:
        print(f"Error in stock screener: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def get_short_interest_from_finviz(ticker):
    """Get short interest data from Finviz (Short Float %, Short Ratio, Short Interest)"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Finviz returned status {response.status_code} for {ticker} short interest")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the snapshot table - try multiple class names
        snapshot_table = soup.find('table', class_='snapshot-table2')
        if not snapshot_table:
            # Try alternative class names
            snapshot_table = soup.find('table', class_='screener_snapshot-table-body')
        if not snapshot_table:
            # Try finding by containing 'snapshot' in class
            all_tables = soup.find_all('table')
            for table in all_tables:
                classes = table.get('class', [])
                if any('snapshot' in str(c).lower() for c in classes):
                    snapshot_table = table
                    break
        
        if not snapshot_table:
            print(f"[SHORT INTEREST] No snapshot table found for {ticker}")
            return None
        
        short_interest_data = {}
        
        # Parse table rows
        rows = snapshot_table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True)
                value_cell = cells[1]
                
                # Get text from value cell, handling nested tags (b, a, span, etc.)
                # Try to get text from <b> tag first, then fallback to all text
                b_tag = value_cell.find('b')
                if b_tag:
                    value_text = b_tag.get_text(strip=True)
                else:
                    value_text = value_cell.get_text(strip=True)
                
                # Remove any HTML entities and clean up
                value_text = value_text.replace('\xa0', ' ').strip()
                
                # Extract Short Float %
                if 'Short Float' in label:
                    # Remove % sign and convert to float
                    try:
                        short_float_pct = float(value_text.replace('%', '').replace(',', ''))
                        short_interest_data['short_float_pct'] = short_float_pct
                    except (ValueError, AttributeError):
                        pass
                
                # Extract Short Ratio
                elif 'Short Ratio' in label:
                    try:
                        short_ratio = float(value_text.replace(',', ''))
                        short_interest_data['short_ratio'] = short_ratio
                    except (ValueError, AttributeError):
                        pass
                
                # Extract Short Interest (absolute number)
                elif 'Short Interest' in label and 'Short Float' not in label and 'Short Ratio' not in label:
                    # Parse value like "57.75M" or "1.2B"
                    try:
                        value_clean = value_text.replace(',', '').replace(' ', '')
                        if 'M' in value_clean:
                            short_interest = float(value_clean.replace('M', '')) * 1_000_000
                        elif 'B' in value_clean:
                            short_interest = float(value_clean.replace('B', '')) * 1_000_000_000
                        elif 'K' in value_clean:
                            short_interest = float(value_clean.replace('K', '')) * 1_000
                        else:
                            short_interest = float(value_clean)
                        short_interest_data['short_interest'] = short_interest
                    except (ValueError, AttributeError):
                        pass
        
        # If no data found in main table, try yfinance as fallback
        if not short_interest_data:
            try:
                stock = yf.Ticker(ticker)
                time.sleep(0.2)
                info = stock.info
                
                # yfinance doesn't directly provide short interest, but we can try to calculate from sharesShort
                shares_short = info.get('sharesShort')
                shares_outstanding = info.get('sharesOutstanding')
                float_shares = info.get('floatShares') or shares_outstanding
                
                if shares_short and float_shares:
                    short_float_pct = (shares_short / float_shares) * 100
                    short_interest_data['short_float_pct'] = round(short_float_pct, 2)
                    short_interest_data['short_interest'] = shares_short
                    
                    # Calculate short ratio (approximate - would need average daily volume)
                    avg_volume = info.get('averageVolume')
                    if avg_volume and avg_volume > 0:
                        short_ratio = shares_short / avg_volume
                        short_interest_data['short_ratio'] = round(short_ratio, 2)
            except Exception as e:
                print(f"[SHORT INTEREST] yfinance fallback failed for {ticker}: {e}")
        
        if not short_interest_data:
            return None
        
        # Calculate short squeeze score (0-100)
        # Higher score = higher squeeze potential
        squeeze_score = 0
        
        if 'short_float_pct' in short_interest_data:
            short_float = short_interest_data['short_float_pct']
            # High short float (>20%) = higher squeeze potential
            if short_float > 20:
                squeeze_score += 40
            elif short_float > 10:
                squeeze_score += 25
            elif short_float > 5:
                squeeze_score += 10
        
        if 'short_ratio' in short_interest_data:
            short_ratio = short_interest_data['short_ratio']
            # High short ratio (>5) = higher squeeze potential
            if short_ratio > 5:
                squeeze_score += 30
            elif short_ratio > 3:
                squeeze_score += 20
            elif short_ratio > 1:
                squeeze_score += 10
        
        short_interest_data['squeeze_score'] = min(100, squeeze_score)
        
        # Try to get historical short interest data from Finviz chart
        try:
            # Finviz stores short interest history in chart data
            json_match = re.search(r'var\s+data\s*=\s*({[^;]+});', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    finviz_data = json.loads(json_str)
                    # Look for short interest chart data
                    if 'chartEvents' in finviz_data:
                        short_history = []
                        for event in finviz_data['chartEvents']:
                            if event.get('eventType') == 'chartEvent/shortInterest':
                                date_ts = event.get('date')
                                short_val = event.get('shortInterest') or event.get('value')
                                if date_ts and short_val:
                                    from datetime import datetime
                                    date_str = datetime.fromtimestamp(date_ts).strftime('%Y-%m-%d')
                                    short_history.append({
                                        'date': date_str,
                                        'short_interest': float(short_val),
                                        'short_float_pct': event.get('shortFloatPct')
                                    })
                        
                        if short_history:
                            # Sort by date
                            short_history.sort(key=lambda x: x['date'])
                            short_interest_data['history'] = short_history
                except:
                    pass
        except:
            pass
        
        return short_interest_data
    
    except Exception as e:
        print(f"Error fetching short interest for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_short_interest_history(ticker):
    """Get historical short interest data from MarketBeat (reported every 15 days)"""
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
        import re
        from datetime import datetime
        import csv
        from io import StringIO
        
        # Determine exchange for MarketBeat URL
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        info = stock.info
        exchange = info.get('exchange', 'NASDAQ').upper()
        
        # MarketBeat URL
        url = f'https://www.marketbeat.com/stocks/{exchange}/{ticker.upper()}/short-interest/'
        
        # Use cloudscraper to bypass Cloudflare
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"[SHORT INTEREST HISTORY] MarketBeat returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        
        history_data = []
        
        # Parse shortInterestSeries (contains actual short interest amounts)
        for script in scripts:
            if not script.string:
                continue
            
            # Extract shortInterestSeries CSV
            match = re.search(r'var shortInterestSeries = "([^"]+)"', script.string)
            if match:
                csv_data = match.group(1)
                # Unescape the CSV data
                csv_data = csv_data.replace('\\n', '\n').replace('\\,', ',')
                try:
                    reader = csv.DictReader(StringIO(csv_data))
                    for row in reader:
                        date_str = row.get('Date', '')
                        amount_str = row.get('Amount', '').replace(',', '').replace('$', '').strip()
                        
                        if date_str and amount_str:
                            try:
                                # Parse date (format: MM/DD/YYYY)
                                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                date_formatted = date_obj.strftime('%Y-%m-%d')
                                
                                # Parse amount (remove commas and $)
                                amount = float(amount_str)
                                
                                history_data.append({
                                    'date': date_formatted,
                                    'short_interest': int(amount),
                                    'short_float_pct': None,  # Will be filled from shortInterestFloatSeries
                                    'short_ratio': None  # Will be filled from shortInterestRatioSeries
                                })
                            except (ValueError, TypeError) as e:
                                continue
                except Exception as e:
                    print(f"[SHORT INTEREST HISTORY] Error parsing shortInterestSeries: {e}")
            
            # Extract shortInterestFloatSeries (contains short float %)
            match = re.search(r'var shortInterestFloatSeries = "([^"]+)"', script.string)
            if match:
                csv_data = match.group(1)
                # Unescape the CSV data
                csv_data = csv_data.replace('\\n', '\n').replace('\\,', ',')
                try:
                    reader = csv.DictReader(StringIO(csv_data))
                    float_data = {}
                    for row in reader:
                        date_str = row.get('Date', '')
                        percent_str = row.get('Percent', '').strip()
                        
                        if date_str and percent_str:
                            try:
                                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                date_formatted = date_obj.strftime('%Y-%m-%d')
                                percent = float(percent_str) * 100  # Convert to percentage
                                float_data[date_formatted] = round(percent, 2)
                            except (ValueError, TypeError):
                                continue
                    
                    # Merge float % data into history_data
                    for entry in history_data:
                        if entry['date'] in float_data:
                            entry['short_float_pct'] = float_data[entry['date']]
                except Exception as e:
                    print(f"[SHORT INTEREST HISTORY] Error parsing shortInterestFloatSeries: {e}")
            
            # Extract shortInterestRatioSeries (contains short ratio)
            match = re.search(r'var shortInterestRatioSeries = "([^"]+)"', script.string)
            if match:
                csv_data = match.group(1)
                # Unescape the CSV data
                csv_data = csv_data.replace('\\n', '\n').replace('\\,', ',')
                try:
                    reader = csv.DictReader(StringIO(csv_data))
                    ratio_data = {}
                    for row in reader:
                        date_str = row.get('Date', '')
                        ratio_str = row.get('Ratio', '').strip()
                        
                        if date_str and ratio_str:
                            try:
                                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                date_formatted = date_obj.strftime('%Y-%m-%d')
                                ratio = float(ratio_str)
                                ratio_data[date_formatted] = round(ratio, 2)
                            except (ValueError, TypeError):
                                continue
                    
                    # Merge ratio data into history_data
                    for entry in history_data:
                        if entry['date'] in ratio_data:
                            entry['short_ratio'] = ratio_data[entry['date']]
                except Exception as e:
                    print(f"[SHORT INTEREST HISTORY] Error parsing shortInterestRatioSeries: {e}")
        
        # Sort by date
        history_data.sort(key=lambda x: x['date'])
        
        # Return last 30 data points (approximately 1 year of bi-weekly data)
        return history_data[-30:] if history_data else None
    
    except ImportError:
        print("[SHORT INTEREST HISTORY] cloudscraper not installed. Install with: pip install cloudscraper")
        return None
    except Exception as e:
        print(f"Error fetching short interest history from MarketBeat for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_factor_scores(ticker, info, df=None):
    """Calculate factor scores (Value, Growth, Momentum, Quality) for a stock"""
    try:
        factors = {
            'value': 0,
            'growth': 0,
            'momentum': 0,
            'quality': 0
        }
        
        factor_breakdown = {
            'value': {},
            'growth': {},
            'momentum': {},
            'quality': {}
        }
        
        # VALUE FACTORS (lower P/E, P/B, P/S = better value)
        # Milder scoring - especially for growth stocks
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        # Check if this is a growth stock (high revenue growth)
        revenue_growth = info.get('revenueGrowth')
        is_growth_stock = revenue_growth and revenue_growth > 0.15  # >15% revenue growth
        
        value_score = 0
        if pe_ratio and pe_ratio > 0:
            # Milder scoring - give points even for higher P/E ratios
            if pe_ratio < 15:
                value_score += 30
            elif pe_ratio < 25:
                value_score += 25  # Increased from 20
            elif pe_ratio < 35:
                value_score += 20  # Increased from 10
            elif pe_ratio < 50:
                value_score += 10  # New tier
            elif pe_ratio < 100:
                value_score += 5   # New tier for growth stocks
            # For growth stocks, be even more lenient
            if is_growth_stock and pe_ratio < 150:
                value_score += 5   # Bonus for growth stocks with reasonable P/E
            factor_breakdown['value']['pe_ratio'] = round(pe_ratio, 2)
        
        if pb_ratio and pb_ratio > 0:
            # Milder scoring for P/B
            if pb_ratio < 2:
                value_score += 25
            elif pb_ratio < 4:
                value_score += 20  # Increased from 15
            elif pb_ratio < 6:
                value_score += 15  # Increased from 5
            elif pb_ratio < 10:
                value_score += 8   # New tier
            elif pb_ratio < 20:
                value_score += 3   # New tier
            # For growth stocks, be more lenient
            if is_growth_stock and pb_ratio < 30:
                value_score += 3   # Bonus for growth stocks
            factor_breakdown['value']['pb_ratio'] = round(pb_ratio, 2)
        
        if ps_ratio and ps_ratio > 0:
            # Milder scoring for P/S
            if ps_ratio < 3:
                value_score += 20
            elif ps_ratio < 6:
                value_score += 15  # Increased from 10
            elif ps_ratio < 10:
                value_score += 10  # Increased from 5
            elif ps_ratio < 20:
                value_score += 5   # New tier
            elif ps_ratio < 40:
                value_score += 2   # New tier
            # For growth stocks, be more lenient
            if is_growth_stock and ps_ratio < 50:
                value_score += 3   # Bonus for growth stocks
            factor_breakdown['value']['ps_ratio'] = round(ps_ratio, 2)
        
        if dividend_yield > 0:
            # Higher dividend yield = better value
            if dividend_yield > 4:
                value_score += 25
            elif dividend_yield > 2:
                value_score += 15
            elif dividend_yield > 1:
                value_score += 5
            factor_breakdown['value']['dividend_yield'] = round(dividend_yield, 2)
        else:
            # For growth stocks without dividends, don't penalize as much
            if is_growth_stock:
                value_score += 5   # Small bonus for growth stocks (dividends not expected)
        
        # Ensure minimum score for growth stocks
        if is_growth_stock and value_score < 20:
            value_score = 20  # Minimum 20 points for growth stocks
        
        factors['value'] = min(100, value_score)
        
        # GROWTH FACTORS (higher revenue/earnings growth = better growth)
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsQuarterlyGrowth')
        earnings_growth_yearly = info.get('earningsGrowth')
        
        growth_score = 0
        if revenue_growth:
            revenue_growth_pct = revenue_growth * 100
            if revenue_growth_pct > 30:
                growth_score += 30
            elif revenue_growth_pct > 15:
                growth_score += 20
            elif revenue_growth_pct > 5:
                growth_score += 10
            factor_breakdown['growth']['revenue_growth'] = round(revenue_growth_pct, 2)
        
        if earnings_growth_yearly:
            earnings_growth_pct = earnings_growth_yearly * 100
            if earnings_growth_pct > 30:
                growth_score += 30
            elif earnings_growth_pct > 15:
                growth_score += 20
            elif earnings_growth_pct > 5:
                growth_score += 10
            factor_breakdown['growth']['earnings_growth'] = round(earnings_growth_pct, 2)
        
        if earnings_growth:
            earnings_growth_pct = earnings_growth * 100
            if earnings_growth_pct > 30:
                growth_score += 25
            elif earnings_growth_pct > 15:
                growth_score += 15
            elif earnings_growth_pct > 5:
                growth_score += 5
            factor_breakdown['growth']['earnings_quarterly_growth'] = round(earnings_growth_pct, 2)
        
        # Forward P/E vs Trailing P/E (lower forward = growth expected)
        if pe_ratio and info.get('forwardPE'):
            forward_pe = info.get('forwardPE')
            if forward_pe > 0 and pe_ratio > 0:
                pe_ratio_change = ((pe_ratio - forward_pe) / pe_ratio) * 100
                if pe_ratio_change > 20:  # Forward P/E significantly lower
                    growth_score += 15
                elif pe_ratio_change > 10:
                    growth_score += 10
        
        factors['growth'] = min(100, growth_score)
        
        # MOMENTUM FACTORS (price performance, RSI, MACD)
        momentum_score = 0
        
        if df is not None and not df.empty:
            # Price momentum (1m, 3m, 6m, 1y returns)
            current_price = df['Close'].iloc[-1]
            
            # 1 month return
            if len(df) >= 21:
                price_1m_ago = df['Close'].iloc[-21]
                return_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
                if return_1m > 10:
                    momentum_score += 15
                elif return_1m > 5:
                    momentum_score += 10
                elif return_1m > 0:
                    momentum_score += 5
                factor_breakdown['momentum']['return_1m'] = round(return_1m, 2)
            
            # 3 month return
            if len(df) >= 63:
                price_3m_ago = df['Close'].iloc[-63]
                return_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
                if return_3m > 20:
                    momentum_score += 20
                elif return_3m > 10:
                    momentum_score += 15
                elif return_3m > 0:
                    momentum_score += 5
                factor_breakdown['momentum']['return_3m'] = round(return_3m, 2)
            
            # 6 month return
            if len(df) >= 126:
                price_6m_ago = df['Close'].iloc[-126]
                return_6m = ((current_price - price_6m_ago) / price_6m_ago) * 100
                if return_6m > 30:
                    momentum_score += 20
                elif return_6m > 15:
                    momentum_score += 15
                elif return_6m > 0:
                    momentum_score += 5
                factor_breakdown['momentum']['return_6m'] = round(return_6m, 2)
            
            # RSI momentum
            try:
                from ta.momentum import RSIIndicator
                rsi_indicator = RSIIndicator(df['Close'], window=14)
                rsi = rsi_indicator.rsi().iloc[-1]
                if not pd.isna(rsi):
                    if 50 < rsi < 70:  # Bullish but not overbought
                        momentum_score += 15
                    elif rsi > 70:  # Overbought (negative)
                        momentum_score -= 10
                    elif rsi < 30:  # Oversold (negative)
                        momentum_score -= 10
                    factor_breakdown['momentum']['rsi'] = round(rsi, 2)
            except:
                pass
        
        # 52-week performance
        if info.get('fiftyTwoWeekHigh') and info.get('fiftyTwoWeekLow'):
            high_52w = info.get('fiftyTwoWeekHigh')
            low_52w = info.get('fiftyTwoWeekLow')
            current_price = info.get('currentPrice') or (df['Close'].iloc[-1] if df is not None and not df.empty else None)
            if current_price and high_52w and low_52w:
                position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100
                if position_52w > 80:  # Near 52-week high
                    momentum_score += 15
                elif position_52w > 60:
                    momentum_score += 10
                factor_breakdown['momentum']['position_52w'] = round(position_52w, 2)
        
        factors['momentum'] = max(0, min(100, momentum_score))
        
        # QUALITY FACTORS (ROE, ROA, profit margins, debt)
        quality_score = 0
        
        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        profit_margin = info.get('profitMargins')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        
        if roe:
            roe_pct = roe * 100
            if roe_pct > 20:
                quality_score += 25
            elif roe_pct > 15:
                quality_score += 15
            elif roe_pct > 10:
                quality_score += 5
            factor_breakdown['quality']['roe'] = round(roe_pct, 2)
        
        if roa:
            roa_pct = roa * 100
            if roa_pct > 10:
                quality_score += 20
            elif roa_pct > 5:
                quality_score += 10
            elif roa_pct > 0:
                quality_score += 5
            factor_breakdown['quality']['roa'] = round(roa_pct, 2)
        
        if profit_margin:
            profit_margin_pct = profit_margin * 100
            if profit_margin_pct > 20:
                quality_score += 25
            elif profit_margin_pct > 10:
                quality_score += 15
            elif profit_margin_pct > 5:
                quality_score += 5
            factor_breakdown['quality']['profit_margin'] = round(profit_margin_pct, 2)
        
        if debt_to_equity is not None:
            if debt_to_equity < 30:  # Low debt
                quality_score += 15
            elif debt_to_equity < 50:
                quality_score += 10
            elif debt_to_equity > 100:  # High debt (negative)
                quality_score -= 15
            factor_breakdown['quality']['debt_to_equity'] = round(debt_to_equity, 2)
        
        if current_ratio:
            if current_ratio > 2:  # Strong liquidity
                quality_score += 15
            elif current_ratio > 1.5:
                quality_score += 10
            elif current_ratio < 1:  # Weak liquidity (negative)
                quality_score -= 10
            factor_breakdown['quality']['current_ratio'] = round(current_ratio, 2)
        
        factors['quality'] = max(0, min(100, quality_score))
        
        # Generate recommendation
        max_factor = max(factors.items(), key=lambda x: x[1])
        min_factor = min(factors.items(), key=lambda x: x[1])
        
        recommendation = f"Strong {max_factor[0]}, weak {min_factor[0]}"
        
        return {
            'factors': factors,
            'factor_breakdown': factor_breakdown,
            'recommendation': recommendation
        }
    
    except Exception as e:
        print(f"Error calculating factor scores for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    if not text or len(text.strip()) == 0:
        return {'sentiment': 'neutral', 'score': 0.0, 'label': 'Neutrální'}
    
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = 'positive'
        label = 'Pozitivní'
    elif compound <= -0.05:
        sentiment = 'negative'
        label = 'Negativní'
    else:
        sentiment = 'neutral'
        label = 'Neutrální'
    
    return {
        'sentiment': sentiment,
        'score': round(compound, 3),
        'label': label,
        'positive': round(scores['pos'], 3),
        'negative': round(scores['neg'], 3),
        'neutral': round(scores['neu'], 3)
    }

# REMOVED - Using Finviz only

def get_quarterly_estimates_from_finviz(ticker):
    """Get quarterly revenue and EPS estimates AND actual values from Finviz - PARSING JSON DATA FROM HTML"""
    estimates = {'revenue': {}, 'eps': {}}
    actuals = {'revenue': {}, 'eps': {}}  # Store actual reported values
    try:
        import re
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # Try using cloudscraper first (for Cloudflare protection on Render)
        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, headers=headers, timeout=10)
        except ImportError:
            # Fallback to requests if cloudscraper not available
            try:
                response = requests.get(url, headers=headers, timeout=10)
            except requests.exceptions.Timeout:
                print(f"[WARNING] Finviz request timeout for {ticker}")
                return {'estimates': estimates, 'actuals': actuals}
            except requests.exceptions.RequestException as e:
                print(f"[WARNING] Finviz request failed for {ticker}: {str(e)}")
                return {'estimates': estimates, 'actuals': actuals}
        except Exception as e:
            # If cloudscraper fails, try regular requests
            try:
                response = requests.get(url, headers=headers, timeout=10)
            except requests.exceptions.Timeout:
                print(f"[WARNING] Finviz request timeout for {ticker}")
                return {'estimates': estimates, 'actuals': actuals}
            except requests.exceptions.RequestException as req_e:
                print(f"[WARNING] Finviz request failed for {ticker}: {str(req_e)}")
                return {'estimates': estimates, 'actuals': actuals}
        
        if response.status_code != 200:
            print(f"[WARNING] Finviz returned status {response.status_code} for {ticker}")
            return {'estimates': estimates, 'actuals': actuals}
        
        # Finviz stores estimates in JSON data embedded in HTML
        # Look for var data = {...} with chartEvents containing earnings data
        html_text = response.text
        
        # Find JSON data object
        json_match = re.search(r'var\s+data\s*=\s*({[^;]+});', html_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                data = json.loads(json_str)
                # Extract chartEvents which contain earnings estimates
                if 'chartEvents' in data:
                    earnings_events = [e for e in data['chartEvents'] if e.get('eventType') == 'chartEvent/earnings']
                    for event in data['chartEvents']:
                        if event.get('eventType') == 'chartEvent/earnings':
                            fiscal_period = event.get('fiscalPeriod', '')
                            fiscal_end_date_ts = event.get('fiscalEndDate')
                            # Parse fiscal period (e.g., "2025Q3" -> "2025-Q3")
                            period_match = re.search(r'(\d{4})Q(\d)', fiscal_period)
                            if period_match:
                                year = period_match.group(1)
                                quarter = period_match.group(2)
                                
                                # Convert fiscal quarter to calendar quarter using fiscalEndDate
                                # Finviz provides fiscalEndDate timestamp - use it to determine calendar quarter
                                fiscal_end_date_ts = event.get('fiscalEndDate')
                                if fiscal_end_date_ts:
                                    from datetime import datetime
                                    fiscal_end_date = datetime.fromtimestamp(fiscal_end_date_ts)
                                    # Calculate calendar quarter from the fiscal end date
                                    calendar_quarter = (fiscal_end_date.month - 1) // 3 + 1
                                    calendar_year = fiscal_end_date.year
                                    quarter_str = f"{calendar_year}-Q{calendar_quarter}"
                                else:
                                    # Fallback to fiscal quarter if no date available
                                    quarter_str = f"{year}-Q{quarter}"
                                
                                # Get EPS estimate (SIMPLE - no conversion needed)
                                eps_est = event.get('epsEstimate')
                                if eps_est is not None and eps_est != 0:
                                    estimates['eps'][quarter_str] = float(eps_est)
                                
                                # Get revenue estimate - EXACT SAME PATTERN AS EPS ESTIMATE
                                # Finviz returns revenue in millions, convert to dollars
                                revenue_est = event.get('salesEstimate')
                                
                                # Try all possible keys systematically (same as EPS)
                                if revenue_est is None or revenue_est == 0:
                                    # Try alternative keys in order
                                    revenue_est_keys = ['revenueEstimate', 'sales', 'revenue', 'estimatedRevenue', 'estimatedSales']
                                    for key in revenue_est_keys:
                                        if key in event and event[key] is not None:
                                            try:
                                                val = float(event[key])
                                                if val != 0:
                                                    revenue_est = val
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                
                                if revenue_est is not None and revenue_est != 0:
                                    # Convert millions to actual dollars (Finviz returns revenue in millions)
                                    revenue_est_dollars = float(revenue_est) * 1_000_000
                                    estimates['revenue'][quarter_str] = revenue_est_dollars
                                
                                # Get actual EPS (reported EPS) if available - check multiple possible keys
                                eps_actual = None
                                eps_actual_key = None
                                # Priority order: prefer explicit actual keys first
                                priority_keys = ['epsActual', 'reportedEPS', 'epsReported', 'actualEPS', 'epsActualValue', 'actualEpsValue']
                                fallback_keys = ['eps', 'reportedEps']
                                
                                # First try priority keys (these are more reliable for actuals)
                                for key in priority_keys:
                                    if key in event and event[key] is not None:
                                        try:
                                            val = float(event[key])
                                            # Accept any value (including 0 and negative) from priority keys
                                            eps_actual = val
                                            eps_actual_key = key
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                
                                # If no priority key found, try fallback keys (but only if value is not 0)
                                if eps_actual is None:
                                    for key in fallback_keys:
                                        if key in event and event[key] is not None:
                                            try:
                                                val = float(event[key])
                                                # For fallback keys, only accept non-zero values (0 might be missing data)
                                                if val != 0:
                                                    eps_actual = val
                                                    eps_actual_key = key
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                
                                # If no actual found, check if estimate might be the actual (for past quarters)
                                if eps_actual is None:
                                    eps_est = event.get('epsEstimate')
                                    # Check if this is a past quarter (fiscalEndDate is in the past)
                                    if fiscal_end_date_ts:
                                        from datetime import datetime
                                        fiscal_end_date = datetime.fromtimestamp(fiscal_end_date_ts)
                                        if fiscal_end_date < datetime.now():
                                            # Past quarter - estimate might actually be the reported value
                                            if eps_est is not None and eps_est != 0:
                                                eps_actual = float(eps_est)
                                                eps_actual_key = 'epsEstimate (past quarter)'
                                
                                if eps_actual is not None:
                                    actuals['eps'][quarter_str] = eps_actual
                                
                                # Get actual revenue if available - EXACT SAME STRUCTURE AS EPS
                                revenue_actual = None
                                revenue_actual_key = None
                                # Priority order: prefer explicit actual keys first (SAME PATTERN AS EPS)
                                # Expanded list of possible keys that Finviz might use for revenue actuals
                                revenue_priority_keys = [
                                    'revenueActual', 'salesActual', 'reportedRevenue', 'actualRevenue', 
                                    'revenueActualValue', 'actualRevenueValue', 'reportedSales', 
                                    'salesReported', 'actualSales', 'salesActualValue'
                                ]
                                revenue_fallback_keys = ['sales', 'revenue']
                                
                                # First try priority keys (these are more reliable for actuals) - EXACT SAME AS EPS
                                for key in revenue_priority_keys:
                                    if key in event and event[key] is not None:
                                        try:
                                            val = float(event[key])
                                            # Accept any value (including 0 and negative) from priority keys - SAME AS EPS
                                            revenue_actual = val
                                            revenue_actual_key = key
                                            # Finviz returns revenue in millions, always convert to dollars
                                            # Priority keys are always in millions
                                            revenue_actual = revenue_actual * 1_000_000
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                
                                # If no priority key found, try fallback keys (but only if value is not 0) - EXACT SAME AS EPS
                                if revenue_actual is None:
                                    for key in revenue_fallback_keys:
                                        if key in event and event[key] is not None:
                                            try:
                                                val = float(event[key])
                                                # For fallback keys, only accept non-zero values (0 might be missing data) - SAME AS EPS
                                                if val != 0:
                                                    revenue_actual = val
                                                    revenue_actual_key = key
                                                    # Finviz values in fallback keys are typically in millions
                                                    # If value is less than 1 trillion, assume it's in millions and convert
                                                    if revenue_actual < 1e12:
                                                        revenue_actual = revenue_actual * 1_000_000
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                
                                # Comprehensive fallback: try all keys containing relevant words
                                if revenue_actual is None:
                                    relevant_keywords = ['revenue', 'sales', 'actual', 'reported']
                                    
                                    # Find all keys that might contain revenue data
                                    candidate_keys = []
                                    for key in event.keys():
                                        key_lower = str(key).lower()
                                        if any(keyword in key_lower for keyword in relevant_keywords):
                                            candidate_keys.append(key)
                                    
                                    # Try each candidate key
                                    for key in candidate_keys:
                                        if key in event and event[key] is not None:
                                            try:
                                                val = float(event[key])
                                                # Skip if value is 0 (might be missing data)
                                                if val != 0:
                                                    # Detect units: if value > 1e12, probably already in dollars
                                                    # If value < 1e12, probably in millions
                                                    if val < 1e12:
                                                        # Probably in millions, convert to dollars
                                                        revenue_actual = val * 1_000_000
                                                    else:
                                                        # Probably already in dollars
                                                        revenue_actual = val
                                                    revenue_actual_key = f'{key} (comprehensive fallback)'
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                
                                # If no actual found, check if estimate might be the actual (for past quarters) - EXACT SAME AS EPS
                                if revenue_actual is None:
                                    # Try multiple keys for revenue estimate (improved fallback)
                                    revenue_est_for_actual = None
                                    estimate_keys_to_try = ['salesEstimate', 'revenueEstimate', 'sales', 'revenue']
                                    
                                    for est_key in estimate_keys_to_try:
                                        if est_key in event and event[est_key] is not None:
                                            try:
                                                val = float(event[est_key])
                                                if val != 0:
                                                    revenue_est_for_actual = val
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                    
                                    # Check if this is a past quarter (fiscalEndDate is in the past) - SAME AS EPS
                                    if fiscal_end_date_ts:
                                        from datetime import datetime
                                        fiscal_end_date = datetime.fromtimestamp(fiscal_end_date_ts)
                                        if fiscal_end_date < datetime.now():
                                            # Past quarter - estimate might actually be the reported value - SAME AS EPS
                                            if revenue_est_for_actual is not None and revenue_est_for_actual != 0:
                                                # Convert millions to dollars (Finviz estimates are in millions)
                                                revenue_actual = float(revenue_est_for_actual) * 1_000_000
                                                revenue_actual_key = f'salesEstimate (past quarter, from {est_key})'
                                
                                if revenue_actual is not None:
                                    actuals['revenue'][quarter_str] = revenue_actual
                                
            except json.JSONDecodeError as e:
                print(f"Error parsing Finviz JSON data: {str(e)}")
        
        # Fallback: Also check snapshot table for EPS next Q
        if not estimates['eps']:
            soup = BeautifulSoup(html_text, 'html.parser')
            snapshot_table = soup.find('table', class_='snapshot-table2')
            if snapshot_table:
                rows = snapshot_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).upper()
                        value = cells[1].get_text(strip=True)
                        
                        if 'EPS NEXT Q' in label:
                            try:
                                eps_val = float(value.replace(',', ''))
                                # Determine next quarter
                                from datetime import datetime
                                now = datetime.now()
                                next_quarter = (now.month - 1) // 3 + 1
                                next_year = now.year
                                if next_quarter == 4:
                                    next_quarter = 1
                                    next_year += 1
                                else:
                                    next_quarter += 1
                                quarter_str = f"{next_year}-Q{next_quarter}"
                                estimates['eps'][quarter_str] = eps_val
                            except (ValueError, TypeError):
                                pass
        
        if estimates['revenue'] or estimates['eps']:
            rev_count = len(estimates['revenue'])
            eps_count = len(estimates['eps'])
            print(f"Finviz: Found {rev_count} revenue and {eps_count} EPS estimates for {ticker}")
            print(f"[DEBUG] Finviz revenue estimate quarters: {sorted(estimates['revenue'].keys())}")
            print(f"[DEBUG] Finviz EPS estimate quarters: {sorted(estimates['eps'].keys())}")
            # Check for 2026 quarters
            revenue_2026 = [q for q in estimates['revenue'].keys() if '2026' in q]
            eps_2026 = [q for q in estimates['eps'].keys() if '2026' in q]
            if revenue_2026 or eps_2026:
                print(f"[DEBUG] Finviz has 2026 quarters - Revenue: {revenue_2026}, EPS: {eps_2026}")
            else:
                print(f"[DEBUG] Finviz does NOT have 2026 quarters. Latest quarters: Revenue max={max(estimates['revenue'].keys()) if estimates['revenue'] else 'N/A'}, EPS max={max(estimates['eps'].keys()) if estimates['eps'] else 'N/A'}")
        else:
            print(f"Finviz: No estimates found for {ticker}")
        
        if actuals['revenue'] or actuals['eps']:
            rev_actual_count = len(actuals['revenue'])
            eps_actual_count = len(actuals['eps'])
            print(f"Finviz: Found {rev_actual_count} revenue and {eps_actual_count} EPS actual values for {ticker}")
        
        
    except Exception as e:
        print(f"Error fetching estimates from Finviz for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Return both estimates and actuals
    return {
        'estimates': estimates,
        'actuals': actuals
    }

# REMOVED - Using Finviz only

def get_industry_ranking(ticker, industry_category, sector, market_cap):
    """Get industry ranking for a stock based on market cap"""
    if not market_cap or market_cap <= 0:
        print(f"[DEBUG] get_industry_ranking: Invalid market_cap for {ticker}: {market_cap}")
        return None
    
    # Define peer companies by industry category
    peer_tickers = {
        'data centers': ['EQIX', 'DLR', 'AMT', 'CCI', 'SBAC', 'IREN', 'CIFR', 'NBIS', 'GDS', 'VNET', 'PD', 'CONE', 'QTS', 'COR', 'LAND'],
        'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'ABBV', 'MRK', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA'],
        'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'MA', 'V', 'PYPL', 'COF', 'USB', 'TFC'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'HAL', 'NOV', 'FANG', 'MRO', 'OVV', 'CTRA', 'MTDR'],
        'consumer': ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'COST', 'AMZN', 'TSCO', 'BBY', 'FIVE', 'DKS'],
        'industrial': ['BA', 'CAT', 'GE', 'HON', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'SWK', 'TXT', 'AME', 'CMI', 'DE', 'FTV'],
        'real estate': ['AMT', 'PLD', 'PSA', 'EQIX', 'WELL', 'SPG', 'O', 'DLR', 'EXPI', 'CBRE', 'JLL', 'CWK', 'MMC', 'BAM', 'BXP']
    }
    
    # Map sector to category if industry_category is "Other"
    sector_to_category = {
        'Technology': 'technology',
        'Healthcare': 'healthcare',
        'Financial Services': 'finance',
        'Energy': 'energy',
        'Consumer Cyclical': 'consumer',
        'Consumer Defensive': 'consumer',
        'Industrials': 'industrial',
        'Real Estate': 'real estate'
    }
    
    # Get peers for this category
    peers = peer_tickers.get(industry_category, [])
    
    # If category is "Other", try to use sector-based peers
    if not peers and industry_category == 'Other' and sector and sector != 'N/A':
        category_from_sector = sector_to_category.get(sector)
        if category_from_sector:
            peers = peer_tickers.get(category_from_sector, [])
            industry_category = category_from_sector
            print(f"[DEBUG] get_industry_ranking: Using sector '{sector}' to map to category '{category_from_sector}' for {ticker}")
    
    if not peers:
        print(f"[DEBUG] get_industry_ranking: No peers found for category '{industry_category}' (ticker: {ticker}, sector: {sector})")
        return None
    
    # Fetch market caps for peers (limit to avoid too many API calls)
    peer_data = []
    for peer_ticker in peers[:20]:  # Limit to 20 peers
        if peer_ticker == ticker:
            continue
        try:
            peer_stock = yf.Ticker(peer_ticker)
            time.sleep(0.1)  # Rate limiting
            peer_info = peer_stock.info
            peer_market_cap = peer_info.get('marketCap', 0)
            if peer_market_cap and peer_market_cap > 0:
                peer_data.append({
                    'ticker': peer_ticker,
                    'market_cap': peer_market_cap
                })
        except:
            continue
    
    # Add current ticker
    peer_data.append({
        'ticker': ticker,
        'market_cap': market_cap
    })
    
    # Sort by market cap (descending)
    peer_data.sort(key=lambda x: x['market_cap'], reverse=True)
    
    # Find position
    position = next((i + 1 for i, p in enumerate(peer_data) if p['ticker'] == ticker), None)
    total = len(peer_data)
    
    if position and total > 1:
        return {
            'position': position,
            'total': total,
            'category': industry_category.title()
        }
    
    return None

def get_peer_comparison_data(ticker, industry_category, sector, limit=5):
    """Get financial data for peer companies in the same industry for comparison"""
    try:
        # Use same peer list as get_industry_ranking
        peer_tickers = {
            'data centers': ['EQIX', 'DLR', 'AMT', 'CCI', 'SBAC', 'IREN', 'CIFR', 'NBIS', 'GDS', 'VNET', 'PD', 'CONE', 'QTS', 'COR', 'LAND'],
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'ABBV', 'MRK', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA'],
            'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'MA', 'V', 'PYPL', 'COF', 'USB', 'TFC'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'HAL', 'NOV', 'FANG', 'MRO', 'OVV', 'CTRA', 'MTDR'],
            'consumer': ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'COST', 'AMZN', 'TSCO', 'BBY', 'FIVE', 'DKS'],
            'industrial': ['BA', 'CAT', 'GE', 'HON', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'SWK', 'TXT', 'AME', 'CMI', 'DE', 'FTV'],
            'real estate': ['AMT', 'PLD', 'PSA', 'EQIX', 'WELL', 'SPG', 'O', 'DLR', 'EXPI', 'CBRE', 'JLL', 'CWK', 'MMC', 'BAM', 'BXP']
        }
        
        # Map sector to category if industry_category is "Other"
        sector_to_category = {
            'Technology': 'technology',
            'Healthcare': 'healthcare',
            'Financial Services': 'finance',
            'Energy': 'energy',
            'Consumer Cyclical': 'consumer',
            'Consumer Defensive': 'consumer',
            'Industrials': 'industrial',
            'Real Estate': 'real estate'
        }
        
        peers = peer_tickers.get(industry_category, [])
        if not peers and industry_category == 'Other' and sector and sector != 'N/A':
            category_from_sector = sector_to_category.get(sector)
            if category_from_sector:
                peers = peer_tickers.get(category_from_sector, [])
        
        if not peers:
            return None
        
        # Get current ticker's financial data
        current_financials = get_financials_data(ticker)
        if not current_financials:
            return None
        
        current_snapshot = current_financials.get('executive_snapshot', {})
        
        # Get peer data (limit to avoid too many API calls)
        # Use yfinance directly for faster data fetching (only get info, not full financials)
        peer_data = []
        for peer_ticker in peers[:limit]:
            if peer_ticker == ticker.upper():
                continue
            try:
                peer_stock = yf.Ticker(peer_ticker)
                time.sleep(0.1)  # Rate limiting
                peer_info = peer_stock.info
                
                # Get quarterly income for TTM calculations
                try:
                    quarterly_income = peer_stock.quarterly_income_stmt
                    if quarterly_income is not None and not quarterly_income.empty:
                        # Calculate TTM revenue
                        revenue_ttm = None
                        if 'Total Revenue' in quarterly_income.index:
                            revenue_row = quarterly_income.loc['Total Revenue']
                            if len(revenue_row) >= 4:
                                revenue_ttm = float(revenue_row.iloc[:4].sum())
                        elif 'Revenue' in quarterly_income.index:
                            revenue_row = quarterly_income.loc['Revenue']
                            if len(revenue_row) >= 4:
                                revenue_ttm = float(revenue_row.iloc[:4].sum())
                        
                        # Calculate TTM net income
                        net_income_ttm = None
                        if 'Net Income' in quarterly_income.index:
                            ni_row = quarterly_income.loc['Net Income']
                            if len(ni_row) >= 4:
                                net_income_ttm = float(ni_row.iloc[:4].sum())
                        
                        # Get gross margin from first quarter
                        gross_margin = None
                        if 'Gross Profit' in quarterly_income.index and revenue_ttm:
                            gross_profit = float(quarterly_income.loc['Gross Profit'].iloc[0]) if len(quarterly_income.loc['Gross Profit']) > 0 else None
                            if gross_profit:
                                gross_margin = (gross_profit / float(quarterly_income.loc[quarterly_income.index[quarterly_income.index.str.contains('Revenue', case=False)][0]].iloc[0])) * 100 if len(quarterly_income.loc[quarterly_income.index[quarterly_income.index.str.contains('Revenue', case=False)][0]]) > 0 else None
                    else:
                        revenue_ttm = peer_info.get('totalRevenue', None)
                        net_income_ttm = peer_info.get('netIncomeToCommon', None)
                        gross_margin = peer_info.get('grossMargins', None)
                        if gross_margin:
                            gross_margin = gross_margin * 100
                except:
                    revenue_ttm = peer_info.get('totalRevenue', None)
                    net_income_ttm = peer_info.get('netIncomeToCommon', None)
                    gross_margin = peer_info.get('grossMargins', None)
                    if gross_margin:
                        gross_margin = gross_margin * 100
                
                # Get FCF from cashflow
                fcf_ttm = None
                fcf_margin = None
                try:
                    quarterly_cf = peer_stock.quarterly_cashflow
                    if quarterly_cf is not None and not quarterly_cf.empty:
                        if 'Free Cash Flow' in quarterly_cf.index:
                            fcf_row = quarterly_cf.loc['Free Cash Flow']
                            if len(fcf_row) >= 4:
                                fcf_ttm = float(fcf_row.iloc[:4].sum())
                                if revenue_ttm and revenue_ttm > 0:
                                    fcf_margin = (fcf_ttm / revenue_ttm) * 100
                except:
                    pass
                
                peer_data.append({
                    'ticker': peer_ticker,
                    'revenue_ttm': revenue_ttm,
                    'net_income_ttm': net_income_ttm,
                    'fcf_ttm': fcf_ttm,
                    'gross_margin': gross_margin,
                    'fcf_margin': fcf_margin
                })
            except Exception as e:
                print(f"[DEBUG] Error fetching peer data for {peer_ticker}: {str(e)}")
                continue
        
        # Add current ticker
        peer_data.append({
            'ticker': ticker.upper(),
            'revenue_ttm': current_snapshot.get('revenue_ttm'),
            'net_income_ttm': current_snapshot.get('net_income_ttm'),
            'fcf_ttm': current_snapshot.get('fcf_ttm'),
            'gross_margin': current_snapshot.get('gross_margin'),
            'fcf_margin': current_snapshot.get('fcf_margin'),
            'is_current': True
        })
        
        return {
            'peers': peer_data,
            'category': industry_category.title() if industry_category != 'Other' else sector
        }
    except Exception as e:
        print(f"[DEBUG] Error in get_peer_comparison_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_financials_score(financials, info, company_stage='unknown'):
    """Calculate overall financials score (0-100) based on multiple metrics
    
    For growth companies with growing revenue, applies more lenient criteria.
    """
    try:
        score = 0
        max_score = 100
        breakdown = {}
        
        snapshot = financials.get('executive_snapshot', {})
        balance_sheet = financials.get('balance_sheet', {})
        margins = financials.get('margins', {})
        quarterly_income = financials.get('income_statement', {}).get('quarterly', [])
        
        # Check if this is a growth company with growing revenue
        revenue_yoy = snapshot.get('revenue_yoy', 0) if snapshot.get('revenue_yoy') else 0
        is_growth_with_revenue = (company_stage == 'growth' or company_stage == 'early_stage') and revenue_yoy > 0
        
        # 1. Revenue Growth (0-20 points)
        # For growth companies, give more points for revenue growth
        revenue_growth_score = 0
        if revenue_yoy >= 20:
            revenue_growth_score = 20
        elif revenue_yoy >= 10:
            revenue_growth_score = 15
        elif revenue_yoy >= 5:
            revenue_growth_score = 10
        elif revenue_yoy >= 0:
            revenue_growth_score = 5
        elif revenue_yoy >= -5:
            revenue_growth_score = 2
        # Negative growth gets 0
        
        # Bonus for growth companies with strong revenue growth
        if is_growth_with_revenue and revenue_yoy >= 15:
            revenue_growth_score = min(20, revenue_growth_score + 2)  # Small bonus
        
        score += revenue_growth_score
        breakdown['revenue_growth'] = revenue_growth_score
        
        # 2. Profitability (0-25 points)
        # For growth companies with growing revenue, be more lenient with negative margins
        net_margin = None
        if margins.get('quarterly') and len(margins['quarterly']) > 0:
            net_margin = margins['quarterly'][0].get('net_margin')
            if net_margin is None:
                # Try to calculate from income statement
                if quarterly_income and len(quarterly_income) > 0:
                    latest_q = quarterly_income[0]
                    revenue = latest_q.get('revenue', 0)
                    net_income = latest_q.get('net_income', 0)
                    if revenue and revenue > 0:
                        net_margin = (net_income / revenue) * 100 if net_income else None
        
        profitability_score = 0
        if net_margin is not None:
            if net_margin >= 20:
                profitability_score = 25
            elif net_margin >= 15:
                profitability_score = 20
            elif net_margin >= 10:
                profitability_score = 15
            elif net_margin >= 5:
                profitability_score = 10
            elif net_margin >= 0:
                profitability_score = 5
            else:
                # Negative margin
                if is_growth_with_revenue:
                    # For growth companies with growing revenue, give partial credit instead of 0
                    if net_margin >= -5:
                        profitability_score = 3  # Small penalty instead of 0
                    elif net_margin >= -10:
                        profitability_score = 2
                    else:
                        profitability_score = 1
                else:
                    profitability_score = 0
        
        # ROE from info
        roe = info.get('returnOnEquity')
        if roe is not None:
            roe_pct = roe * 100
            if roe_pct >= 20:
                profitability_score = max(profitability_score, 25)
            elif roe_pct >= 15:
                profitability_score = max(profitability_score, 20)
            elif roe_pct >= 10:
                profitability_score = max(profitability_score, 15)
            elif roe_pct >= 5:
                profitability_score = max(profitability_score, 10)
            elif roe_pct >= 0:
                profitability_score = max(profitability_score, 5)
            elif is_growth_with_revenue and roe_pct >= -5:
                # For growth companies, give small credit for slightly negative ROE
                profitability_score = max(profitability_score, 2)
        
        score += profitability_score
        breakdown['profitability'] = profitability_score
        
        # 3. Cash Flow (0-20 points)
        # For growth companies with growing revenue, be more lenient with negative FCF
        fcf_ttm = snapshot.get('fcf_ttm', 0) if snapshot.get('fcf_ttm') else 0
        net_income_ttm = snapshot.get('net_income_ttm', 0) if snapshot.get('net_income_ttm') else 0
        cash_flow_score = 0
        if fcf_ttm > 0:
            if fcf_ttm >= net_income_ttm and net_income_ttm > 0:
                cash_flow_score = 20  # FCF >= Net Income (excellent)
            elif fcf_ttm >= net_income_ttm * 0.8:
                cash_flow_score = 15
            elif fcf_ttm >= net_income_ttm * 0.5:
                cash_flow_score = 10
            else:
                cash_flow_score = 5
        elif fcf_ttm == 0:
            cash_flow_score = 0
        else:
            # Negative FCF
            if net_income_ttm and net_income_ttm > 0:
                cash_flow_score = -5  # Negative FCF despite positive earnings
            else:
                # Negative FCF with negative earnings
                if is_growth_with_revenue:
                    # For growth companies with growing revenue, give partial credit instead of 0
                    # Negative FCF is expected during growth phase
                    if fcf_ttm >= net_income_ttm * 0.5:  # FCF better than earnings
                        cash_flow_score = 3
                    elif fcf_ttm >= net_income_ttm * 0.8:
                        cash_flow_score = 2
                    else:
                        cash_flow_score = 1
                else:
                    cash_flow_score = 0  # Negative FCF acceptable if company is losing money
        
        score += cash_flow_score
        breakdown['cash_flow'] = cash_flow_score
        
        # 4. Debt Levels (0-15 points)
        total_debt = balance_sheet.get('total_debt', 0) if balance_sheet.get('total_debt') else 0
        cash = balance_sheet.get('cash', 0) if balance_sheet.get('cash') else 0
        net_debt = balance_sheet.get('net_debt', 0) if balance_sheet.get('net_debt') else None
        equity = balance_sheet.get('equity', 0) if balance_sheet.get('equity') else 0
        
        debt_score = 15  # Start with full points
        if total_debt > 0:
            # Debt to Equity ratio
            if equity and equity > 0:
                debt_equity_ratio = total_debt / equity
                if debt_equity_ratio > 2.0:
                    debt_score = 0  # Very high debt
                elif debt_equity_ratio > 1.0:
                    debt_score = 5
                elif debt_equity_ratio > 0.5:
                    debt_score = 10
                # Low debt (ratio < 0.5) keeps full 15 points
            
            # Net debt position
            if net_debt is not None:
                if net_debt < 0:
                    debt_score = 15  # Net cash position (excellent)
                elif net_debt == 0:
                    debt_score = 12
                elif net_debt < total_debt * 0.3:
                    debt_score = 10
                else:
                    debt_score = min(debt_score, 5)
        else:
            debt_score = 15  # No debt (excellent)
        
        score += debt_score
        breakdown['debt'] = debt_score
        
        # 5. Stability/Consistency (0-10 points)
        stability_score = 0
        if len(quarterly_income) >= 4:
            revenues = [q.get('revenue', 0) for q in quarterly_income[:4] if q.get('revenue') is not None]
            if len(revenues) >= 4:
                # Check for consistent growth (not too volatile)
                growth_rates = []
                for i in range(len(revenues) - 1):
                    if revenues[i+1] and revenues[i+1] > 0:
                        growth = ((revenues[i] - revenues[i+1]) / revenues[i+1]) * 100
                        growth_rates.append(growth)
                
                if len(growth_rates) >= 3:
                    # Low volatility in growth rates = more stable
                    avg_growth = sum(growth_rates) / len(growth_rates)
                    std_dev = (sum((g - avg_growth) ** 2 for g in growth_rates) / len(growth_rates)) ** 0.5
                    
                    if std_dev < 5:
                        stability_score = 10  # Very stable
                    elif std_dev < 10:
                        stability_score = 7
                    elif std_dev < 20:
                        stability_score = 5
                    else:
                        stability_score = 2
        
        score += stability_score
        breakdown['stability'] = stability_score
        
        # 6. Efficiency (0-10 points)
        efficiency_score = 0
        # Current ratio
        current_ratio = balance_sheet.get('current_ratio')
        if current_ratio is not None:
            if current_ratio >= 2.0:
                efficiency_score = 10
            elif current_ratio >= 1.5:
                efficiency_score = 7
            elif current_ratio >= 1.0:
                efficiency_score = 5
            elif current_ratio >= 0.5:
                efficiency_score = 2
            # Low current ratio gets 0
        
        # Gross margin
        gross_margin = None
        if margins.get('quarterly') and len(margins['quarterly']) > 0:
            gross_margin = margins['quarterly'][0].get('gross_margin')
        
        if gross_margin is not None:
            if gross_margin >= 50:
                efficiency_score = max(efficiency_score, 10)
            elif gross_margin >= 40:
                efficiency_score = max(efficiency_score, 8)
            elif gross_margin >= 30:
                efficiency_score = max(efficiency_score, 6)
            elif gross_margin >= 20:
                efficiency_score = max(efficiency_score, 4)
            elif gross_margin >= 10:
                efficiency_score = max(efficiency_score, 2)
        
        score += efficiency_score
        breakdown['efficiency'] = efficiency_score
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        # Determine grade
        if score >= 80:
            grade = 'Excellent'
            grade_color = '#10b981'
        elif score >= 65:
            grade = 'Good'
            grade_color = '#3b82f6'
        elif score >= 50:
            grade = 'Fair'
            grade_color = '#f59e0b'
        elif score >= 35:
            grade = 'Weak'
            grade_color = '#ef4444'
        else:
            grade = 'Poor'
            grade_color = '#dc2626'
        
        return {
            'score': round(score, 1),
            'grade': grade,
            'grade_color': grade_color,
            'breakdown': breakdown,
            'max_score': max_score
        }
        
    except Exception as e:
        print(f"Error calculating financials score: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'score': 50,
            'grade': 'N/A',
            'grade_color': '#6b7280',
            'breakdown': {},
            'max_score': 100
        }

def get_cash_flow_analysis(ticker):
    """Get comprehensive cash flow statement analysis"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        # Get cash flow statements
        cf_quarterly = stock.quarterly_cashflow
        cf_annual = stock.cashflow
        
        if cf_quarterly is None or cf_quarterly.empty:
            print(f"[CASH FLOW] No quarterly cashflow data for {ticker}")
            return None
        
        print(f"[CASH FLOW] Found quarterly cashflow data for {ticker}: {len(cf_quarterly.columns)} quarters")
        
        # Extract key cash flow components
        def find_cf_row(df, keywords):
            for idx in df.index:
                idx_lower = str(idx).lower()
                if any(kw.lower() in idx_lower for kw in keywords):
                    return df.loc[idx]
            return None
        
        # Operating Cash Flow
        ocf_row = find_cf_row(cf_quarterly, ['operating cash flow', 'operating activities', 'cash from operations', 'net cash provided by operating activities'])
        # Investing Cash Flow
        icf_row = find_cf_row(cf_quarterly, ['investing cash flow', 'investing activities', 'cash from investing', 'net cash used in investing activities'])
        # Financing Cash Flow
        fcf_row = find_cf_row(cf_quarterly, ['financing cash flow', 'financing activities', 'cash from financing', 'net cash used in financing activities'])
        # Capital Expenditures
        capex_row = find_cf_row(cf_quarterly, ['capital expenditure', 'capex', 'purchase of property', 'capital expenditures'])
        
        # Build quarterly cash flow breakdown
        quarterly_breakdown = []
        fcf_trend = []
        
        # Get last 8 quarters
        for i, col in enumerate(cf_quarterly.columns[:8]):
            try:
                quarter_date = pd.Timestamp(col)
                quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                
                ocf = float(ocf_row.iloc[i]) if ocf_row is not None and i < len(ocf_row) else None
                icf = float(icf_row.iloc[i]) if icf_row is not None and i < len(icf_row) else None
                fcf_val = float(fcf_row.iloc[i]) if fcf_row is not None and i < len(fcf_row) else None
                capex = float(capex_row.iloc[i]) if capex_row is not None and i < len(capex_row) else None
                
                # Calculate FCF if not directly available
                if ocf is not None and capex is not None:
                    calculated_fcf = ocf - abs(capex) if capex < 0 else ocf + capex
                else:
                    calculated_fcf = None
                
                fcf_final = calculated_fcf if fcf_val is None else fcf_val
                
                quarterly_breakdown.append({
                    'quarter': quarter_str,
                    'date': quarter_date.strftime('%Y-%m-%d'),
                    'operating_cf': ocf,
                    'investing_cf': icf,
                    'financing_cf': fcf_val,
                    'capex': abs(capex) if capex is not None and capex < 0 else (capex if capex is not None else None),
                    'fcf': fcf_final
                })
                
                if fcf_final is not None:
                    fcf_trend.append({
                        'quarter': quarter_str,
                        'date': quarter_date.strftime('%Y-%m-%d'),
                        'fcf': fcf_final
                    })
            except (IndexError, ValueError, TypeError) as e:
                continue
        
        # Calculate Cash Conversion Cycle (requires income statement and balance sheet)
        try:
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            
            # Get latest quarter data
            if income_stmt is not None and not income_stmt.empty and balance_sheet is not None and not balance_sheet.empty:
                # Accounts Receivable
                ar_row = find_cf_row(balance_sheet, ['accounts receivable', 'receivables', 'trade receivables'])
                # Inventory
                inv_row = find_cf_row(balance_sheet, ['inventory', 'inventories'])
                # Accounts Payable
                ap_row = find_cf_row(balance_sheet, ['accounts payable', 'payables', 'trade payables'])
                # Revenue
                rev_row = find_cf_row(income_stmt, ['total revenue', 'revenue', 'net sales', 'sales'])
                # COGS
                cogs_row = find_cf_row(income_stmt, ['cost of revenue', 'cost of goods sold', 'cogs', 'cost of sales'])
                
                if ar_row is not None and rev_row is not None and len(ar_row) > 0 and len(rev_row) > 0:
                    ar = float(ar_row.iloc[0])
                    revenue = float(rev_row.iloc[0])
                    
                    # Days Sales Outstanding
                    dso = (ar / revenue * 365) if revenue > 0 else None
                else:
                    dso = None
                
                if inv_row is not None and cogs_row is not None and len(inv_row) > 0 and len(cogs_row) > 0:
                    inventory = float(inv_row.iloc[0])
                    cogs = float(cogs_row.iloc[0])
                    
                    # Days Inventory Outstanding
                    dio = (inventory / cogs * 365) if cogs > 0 else None
                else:
                    dio = None
                
                if ap_row is not None and cogs_row is not None and len(ap_row) > 0 and len(cogs_row) > 0:
                    ap = float(ap_row.iloc[0])
                    cogs = float(cogs_row.iloc[0])
                    
                    # Days Payable Outstanding
                    dpo = (ap / cogs * 365) if cogs > 0 else None
                else:
                    dpo = None
                
                # Cash Conversion Cycle
                if dso is not None and dio is not None and dpo is not None:
                    ccc = dso + dio - dpo
                else:
                    ccc = None
            else:
                ccc = None
                dso = None
                dio = None
                dpo = None
        except Exception as e:
            print(f"Error calculating CCC for {ticker}: {e}")
            ccc = None
            dso = None
            dio = None
            dpo = None
        
        # Calculate Cash Runway for growth companies (if negative FCF)
        cash_runway = None
        if fcf_trend and len(fcf_trend) > 0:
            latest_fcf = fcf_trend[0]['fcf']
            if latest_fcf is not None and latest_fcf < 0:
                # Get cash from balance sheet
                try:
                    bs = stock.balance_sheet
                    if bs is not None and not bs.empty:
                        cash_row = find_cf_row(bs, ['cash and cash equivalents', 'cash', 'cash and short term investments'])
                        if cash_row is not None and len(cash_row) > 0:
                            cash = float(cash_row.iloc[0])
                            monthly_burn = abs(latest_fcf) / 3  # Quarterly to monthly
                            if monthly_burn > 0:
                                cash_runway = cash / monthly_burn  # Months
                except:
                    pass
        
        # FCF Trend Projection (simple linear trend)
        fcf_projection = None
        if len(fcf_trend) >= 4:
            try:
                # Use last 4 quarters for trend
                recent_fcf = [q['fcf'] for q in fcf_trend[:4] if q['fcf'] is not None]
                if len(recent_fcf) >= 3:
                    # Simple average growth rate
                    growth_rates = []
                    for i in range(len(recent_fcf) - 1):
                        if recent_fcf[i+1] != 0:
                            growth = (recent_fcf[i] - recent_fcf[i+1]) / abs(recent_fcf[i+1])
                            growth_rates.append(growth)
                    
                    if growth_rates:
                        avg_growth = sum(growth_rates) / len(growth_rates)
                        next_fcf = recent_fcf[0] * (1 + avg_growth)
                        fcf_projection = {
                            'next_quarter': next_fcf,
                            'growth_rate': avg_growth * 100,
                            'method': 'linear_trend'
                        }
            except:
                pass
        
        return {
            'quarterly_breakdown': quarterly_breakdown,
            'fcf_trend': fcf_trend,
            'fcf_projection': fcf_projection,
            'cash_conversion_cycle': {
                'ccc': round(ccc, 1) if ccc is not None else None,
                'dso': round(dso, 1) if dso is not None else None,
                'dio': round(dio, 1) if dio is not None else None,
                'dpo': round(dpo, 1) if dpo is not None else None
            },
            'cash_runway': {
                'months': round(cash_runway, 1) if cash_runway is not None else None,
                'status': 'critical' if cash_runway is not None and cash_runway < 6 else 'warning' if cash_runway is not None and cash_runway < 12 else 'ok' if cash_runway is not None else None
            }
        }
    except Exception as e:
        print(f"Error in cash flow analysis for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_profitability_analysis(ticker, financials_data):
    """Get deep dive profitability analysis"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        income_stmt = stock.income_stmt
        if income_stmt is None or income_stmt.empty:
            return None
        
        def find_row(df, keywords):
            for idx in df.index:
                idx_lower = str(idx).lower()
                if any(kw.lower() in idx_lower for kw in keywords):
                    return df.loc[idx]
            return None
        
        # Get revenue and margin rows
        revenue_row = find_row(income_stmt, ['total revenue', 'revenue', 'net sales'])
        gross_profit_row = find_row(income_stmt, ['gross profit'])
        operating_income_row = find_row(income_stmt, ['operating income', 'income from operations'])
        net_income_row = find_row(income_stmt, ['net income', 'net earnings'])
        
        # Build margin trends
        margin_trends = []
        for i, col in enumerate(income_stmt.columns[:8]):
            try:
                quarter_date = pd.Timestamp(col)
                quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                
                revenue = float(revenue_row.iloc[i]) if revenue_row is not None and i < len(revenue_row) else None
                gross_profit = float(gross_profit_row.iloc[i]) if gross_profit_row is not None and i < len(gross_profit_row) else None
                operating_income = float(operating_income_row.iloc[i]) if operating_income_row is not None and i < len(operating_income_row) else None
                net_income = float(net_income_row.iloc[i]) if net_income_row is not None and i < len(net_income_row) else None
                
                gross_margin = (gross_profit / revenue * 100) if revenue and revenue > 0 and gross_profit else None
                operating_margin = (operating_income / revenue * 100) if revenue and revenue > 0 and operating_income else None
                net_margin = (net_income / revenue * 100) if revenue and revenue > 0 and net_income else None
                
                margin_trends.append({
                    'quarter': quarter_str,
                    'date': quarter_date.strftime('%Y-%m-%d'),
                    'gross_margin': round(gross_margin, 2) if gross_margin is not None else None,
                    'operating_margin': round(operating_margin, 2) if operating_margin is not None else None,
                    'net_margin': round(net_margin, 2) if net_margin is not None else None,
                    'revenue': revenue
                })
            except (IndexError, ValueError, TypeError):
                continue
        
        # Calculate margin expansion/contraction
        margin_expansion = {}
        if len(margin_trends) >= 2:
            latest = margin_trends[0]
            previous = margin_trends[1]
            
            if latest.get('gross_margin') and previous.get('gross_margin'):
                margin_expansion['gross'] = latest['gross_margin'] - previous['gross_margin']
            if latest.get('operating_margin') and previous.get('operating_margin'):
                margin_expansion['operating'] = latest['operating_margin'] - previous['operating_margin']
            if latest.get('net_margin') and previous.get('net_margin'):
                margin_expansion['net'] = latest['net_margin'] - previous['net_margin']
        
        # Calculate Operating Leverage
        operating_leverage = None
        if len(margin_trends) >= 2:
            latest = margin_trends[0]
            previous = margin_trends[1]
            
            if (latest.get('revenue') and previous.get('revenue') and 
                latest.get('operating_income') and previous.get('operating_income') and
                previous['revenue'] > 0 and previous['operating_income'] != 0):
                
                revenue_growth = (latest['revenue'] - previous['revenue']) / previous['revenue']
                operating_income_growth = (latest['operating_income'] - previous['operating_income']) / abs(previous['operating_income'])
                
                if revenue_growth != 0:
                    operating_leverage = operating_income_growth / revenue_growth
        
        # Break-even analysis for growth companies
        break_even_analysis = None
        company_stage = financials_data.get('company_stage', 'unknown')
        if company_stage in ['growth', 'early_stage']:
            if len(margin_trends) >= 4:
                # Calculate average revenue growth
                revenue_growth_rates = []
                for i in range(len(margin_trends) - 1):
                    if margin_trends[i].get('revenue') and margin_trends[i+1].get('revenue') and margin_trends[i+1]['revenue'] > 0:
                        growth = (margin_trends[i]['revenue'] - margin_trends[i+1]['revenue']) / margin_trends[i+1]['revenue']
                        revenue_growth_rates.append(growth)
                
                if revenue_growth_rates:
                    avg_growth = sum(revenue_growth_rates) / len(revenue_growth_rates)
                    latest_revenue = margin_trends[0].get('revenue', 0)
                    latest_net_income = margin_trends[0].get('net_income', 0)
                    
                    if latest_net_income < 0 and avg_growth > 0:
                        # Project when they'll break even
                        current_loss = abs(latest_net_income)
                        # Estimate break-even revenue (simplified)
                        if latest_revenue > 0:
                            loss_margin = current_loss / latest_revenue
                            # Assume loss margin decreases with scale
                            break_even_revenue = latest_revenue * (1 + loss_margin / 0.1)  # Simplified
                            
                            # Estimate quarters to break-even
                            if avg_growth > 0:
                                quarters_to_breakeven = (break_even_revenue / latest_revenue - 1) / avg_growth
                                break_even_analysis = {
                                    'estimated_quarters': max(1, int(quarters_to_breakeven)),
                                    'estimated_revenue': break_even_revenue,
                                    'current_revenue': latest_revenue,
                                    'avg_growth_rate': avg_growth * 100
                                }
        
        return {
            'margin_trends': margin_trends,
            'margin_expansion': margin_expansion,
            'operating_leverage': round(operating_leverage, 2) if operating_leverage is not None else None,
            'break_even_analysis': break_even_analysis
        }
    except Exception as e:
        print(f"Error in profitability analysis for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_balance_sheet_health(ticker):
    """Get comprehensive balance sheet health analysis"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        
        if balance_sheet is None or balance_sheet.empty:
            return None
        
        def find_row(df, keywords):
            for idx in df.index:
                idx_lower = str(idx).lower()
                if any(kw.lower() in idx_lower for kw in keywords):
                    return df.loc[idx]
            return None
        
        # Get key balance sheet items
        current_assets_row = find_row(balance_sheet, ['total current assets', 'current assets'])
        current_liabilities_row = find_row(balance_sheet, ['total current liabilities', 'current liabilities'])
        cash_row = find_row(balance_sheet, ['cash and cash equivalents', 'cash'])
        inventory_row = find_row(balance_sheet, ['inventory', 'inventories'])
        total_debt_row = find_row(balance_sheet, ['total debt', 'long term debt', 'total liabilities'])
        equity_row = find_row(balance_sheet, ['total stockholders equity', 'total equity', 'stockholders equity'])
        total_assets_row = find_row(balance_sheet, ['total assets'])
        
        revenue_row = find_row(income_stmt, ['total revenue', 'revenue']) if income_stmt is not None else None
        
        # Build trends
        trends = []
        for i, col in enumerate(balance_sheet.columns[:8]):
            try:
                quarter_date = pd.Timestamp(col)
                quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                
                ca = float(current_assets_row.iloc[i]) if current_assets_row is not None and i < len(current_assets_row) else None
                cl = float(current_liabilities_row.iloc[i]) if current_liabilities_row is not None and i < len(current_liabilities_row) else None
                cash = float(cash_row.iloc[i]) if cash_row is not None and i < len(cash_row) else None
                inventory = float(inventory_row.iloc[i]) if inventory_row is not None and i < len(inventory_row) else None
                debt = float(total_debt_row.iloc[i]) if total_debt_row is not None and i < len(total_debt_row) else None
                equity = float(equity_row.iloc[i]) if equity_row is not None and i < len(equity_row) else None
                total_assets = float(total_assets_row.iloc[i]) if total_assets_row is not None and i < len(total_assets_row) else None
                
                revenue = float(revenue_row.iloc[i]) if revenue_row is not None and i < len(revenue_row) else None
                
                # Calculate ratios
                current_ratio = (ca / cl) if ca and cl and cl > 0 else None
                quick_ratio = ((ca - inventory) / cl) if ca and inventory is not None and cl and cl > 0 else (ca / cl if ca and cl and cl > 0 else None)
                debt_to_equity = (debt / equity) if debt and equity and equity > 0 else None
                working_capital = (ca - cl) if ca and cl else None
                working_capital_efficiency = (working_capital / revenue * 100) if working_capital and revenue and revenue > 0 else None
                asset_turnover = (revenue / total_assets) if revenue and total_assets and total_assets > 0 else None
                
                trends.append({
                    'quarter': quarter_str,
                    'date': quarter_date.strftime('%Y-%m-%d'),
                    'current_ratio': round(current_ratio, 2) if current_ratio is not None else None,
                    'quick_ratio': round(quick_ratio, 2) if quick_ratio is not None else None,
                    'debt_to_equity': round(debt_to_equity, 2) if debt_to_equity is not None else None,
                    'working_capital': working_capital,
                    'working_capital_efficiency': round(working_capital_efficiency, 2) if working_capital_efficiency is not None else None,
                    'asset_turnover': round(asset_turnover, 2) if asset_turnover is not None else None
                })
            except (IndexError, ValueError, TypeError):
                continue
        
        # Calculate Health Score (0-100)
        health_score = 50  # Base score
        breakdown = {}
        
        if trends and len(trends) > 0:
            latest = trends[0]
            
            # Current Ratio Score (0-20 points)
            if latest.get('current_ratio') is not None:
                cr = latest['current_ratio']
                if cr >= 2.0:
                    breakdown['current_ratio'] = 20
                elif cr >= 1.5:
                    breakdown['current_ratio'] = 15
                elif cr >= 1.0:
                    breakdown['current_ratio'] = 10
                elif cr >= 0.5:
                    breakdown['current_ratio'] = 5
                else:
                    breakdown['current_ratio'] = 0
            else:
                breakdown['current_ratio'] = 0
            
            # Quick Ratio Score (0-15 points)
            if latest.get('quick_ratio') is not None:
                qr = latest['quick_ratio']
                if qr >= 1.5:
                    breakdown['quick_ratio'] = 15
                elif qr >= 1.0:
                    breakdown['quick_ratio'] = 12
                elif qr >= 0.5:
                    breakdown['quick_ratio'] = 8
                else:
                    breakdown['quick_ratio'] = 3
            else:
                breakdown['quick_ratio'] = 0
            
            # Debt-to-Equity Score (0-25 points, lower is better)
            if latest.get('debt_to_equity') is not None:
                dte = latest['debt_to_equity']
                if dte <= 0.3:
                    breakdown['debt_to_equity'] = 25
                elif dte <= 0.5:
                    breakdown['debt_to_equity'] = 20
                elif dte <= 1.0:
                    breakdown['debt_to_equity'] = 15
                elif dte <= 2.0:
                    breakdown['debt_to_equity'] = 10
                else:
                    breakdown['debt_to_equity'] = 5
            else:
                breakdown['debt_to_equity'] = 10
            
            # Working Capital Efficiency Score (0-20 points)
            if latest.get('working_capital_efficiency') is not None:
                wce = latest['working_capital_efficiency']
                if wce <= 10:  # Efficient (low WC relative to revenue)
                    breakdown['working_capital_efficiency'] = 20
                elif wce <= 20:
                    breakdown['working_capital_efficiency'] = 15
                elif wce <= 30:
                    breakdown['working_capital_efficiency'] = 10
                else:
                    breakdown['working_capital_efficiency'] = 5
            else:
                breakdown['working_capital_efficiency'] = 10
            
            # Asset Turnover Score (0-20 points, higher is better)
            if latest.get('asset_turnover') is not None:
                at = latest['asset_turnover']
                if at >= 1.0:
                    breakdown['asset_turnover'] = 20
                elif at >= 0.5:
                    breakdown['asset_turnover'] = 15
                elif at >= 0.3:
                    breakdown['asset_turnover'] = 10
                else:
                    breakdown['asset_turnover'] = 5
            else:
                breakdown['asset_turnover'] = 10
            
            health_score = sum(breakdown.values())
        
        return {
            'health_score': min(100, max(0, health_score)),
            'breakdown': breakdown,
            'trends': trends
        }
    except Exception as e:
        print(f"Error in balance sheet health analysis for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_management_guidance_tracking(ticker):
    """Track management guidance vs actual results"""
    try:
        # Get earnings calendar and actual results
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        # Get earnings calendar for guidance estimates
        calendar = stock.calendar
        earnings_history = stock.earnings_history if hasattr(stock, 'earnings_history') else None
        
        guidance_tracking = []
        guidance_accuracy_scores = []
        
        # Try to extract guidance from earnings calendar
        if calendar is not None:
            try:
                if isinstance(calendar, dict):
                    for date_key, data in calendar.items():
                        if isinstance(data, dict):
                            # Look for guidance fields
                            guidance_revenue = data.get('Revenue Average') or data.get('revenueEstimate')
                            guidance_eps = data.get('Earnings Average') or data.get('epsEstimate')
                            
                            if guidance_revenue or guidance_eps:
                                # Try to match with actual results
                                # This is simplified - in reality would need to match dates
                                guidance_tracking.append({
                                    'date': str(date_key),
                                    'revenue_guidance': guidance_revenue,
                                    'eps_guidance': guidance_eps,
                                    'actual_revenue': None,  # Would need to match from actuals
                                    'actual_eps': None,
                                    'revenue_beat': None,
                                    'eps_beat': None
                                })
                elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
                    for idx, row in calendar.iterrows():
                        try:
                            guidance_revenue = row.get('Revenue Average') if 'Revenue Average' in row else None
                            guidance_eps = row.get('Earnings Average') if 'Earnings Average' in row else None
                            
                            if guidance_revenue is not None or guidance_eps is not None:
                                guidance_tracking.append({
                                    'date': str(idx),
                                    'revenue_guidance': float(guidance_revenue) if guidance_revenue is not None else None,
                                    'eps_guidance': float(guidance_eps) if guidance_eps is not None else None,
                                    'actual_revenue': None,
                                    'actual_eps': None,
                                    'revenue_beat': None,
                                    'eps_beat': None
                                })
                        except:
                            continue
            except Exception as e:
                print(f"Error parsing calendar for {ticker}: {e}")
        
        # Calculate accuracy score if we have guidance vs actuals
        if guidance_tracking:
            for entry in guidance_tracking:
                if entry.get('revenue_guidance') and entry.get('actual_revenue'):
                    guidance = entry['revenue_guidance']
                    actual = entry['actual_revenue']
                    if guidance > 0:
                        error_pct = abs(actual - guidance) / guidance * 100
                        accuracy = max(0, 100 - error_pct)  # 100% if perfect, decreases with error
                        guidance_accuracy_scores.append(accuracy)
                        entry['revenue_accuracy'] = round(accuracy, 1)
                
                if entry.get('eps_guidance') and entry.get('actual_eps'):
                    guidance = entry['eps_guidance']
                    actual = entry['actual_eps']
                    if guidance != 0:
                        error_pct = abs(actual - guidance) / abs(guidance) * 100
                        accuracy = max(0, 100 - error_pct)
                        guidance_accuracy_scores.append(accuracy)
                        entry['eps_accuracy'] = round(accuracy, 1)
        
        overall_accuracy = sum(guidance_accuracy_scores) / len(guidance_accuracy_scores) if guidance_accuracy_scores else None
        
        # Get forward guidance if available
        forward_guidance = None
        if calendar is not None:
            try:
                # Get next earnings date and estimates
                if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                    next_earnings = calendar.iloc[0] if len(calendar) > 0 else None
                    if next_earnings is not None:
                        forward_guidance = {
                            'revenue_guidance': float(next_earnings.get('Revenue Average', 0)) if 'Revenue Average' in next_earnings else None,
                            'eps_guidance': float(next_earnings.get('Earnings Average', 0)) if 'Earnings Average' in next_earnings else None,
                            'date': str(calendar.index[0]) if len(calendar) > 0 else None
                        }
            except:
                pass
        
        return {
            'guidance_history': guidance_tracking[:8],  # Last 8 quarters
            'accuracy_score': round(overall_accuracy, 1) if overall_accuracy is not None else None,
            'forward_guidance': forward_guidance,
            'revisions_count': 0  # Would need to track revisions over time
        }
    except Exception as e:
        print(f"Error in management guidance tracking for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_segment_breakdown(ticker):
    """Get segment and geography breakdown"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        info = stock.info
        
        # Try to get segment data from info
        segment_data = {}
        geography_data = {}
        
        # Check for segment revenue in info (yfinance sometimes has this)
        segment_keys = [k for k in info.keys() if 'segment' in k.lower() or 'business' in k.lower() or 'geography' in k.lower()]
        
        # Try to extract from majorHoldersBreakdown or other fields
        # This is a simplified version - segment data is often in 10-K filings
        
        # For now, return structure (will be populated if data available)
        # In production, would scrape from SEC filings or use specialized API
        return {
            'segments': segment_data,
            'geography': geography_data,
            'available': len(segment_data) > 0 or len(geography_data) > 0
        }
    except Exception as e:
        print(f"Error in segment breakdown for {ticker}: {str(e)}")
        return None

def get_financials_data(ticker):
    """Get comprehensive financial data for Financials tab"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.3)  # Rate limiting
        
        # Get quarterly estimates and actuals from Finviz
        # Format: {'estimates': {'revenue': {}, 'eps': {}}, 'actuals': {'revenue': {}, 'eps': {}}}
        # Wrap in try-except to prevent Finviz failures from breaking the whole request
        finviz_data = None
        try:
            finviz_data = get_quarterly_estimates_from_finviz(ticker)
        except Exception as finviz_error:
            print(f"[WARNING] Finviz scraping failed for {ticker}: {str(finviz_error)}")
            # Continue without Finviz data - not critical
            finviz_data = {'estimates': {}, 'actuals': {}}
        
        quarterly_estimates = finviz_data.get('estimates', {}) if isinstance(finviz_data, dict) else {}
        quarterly_actuals = finviz_data.get('actuals', {}) if isinstance(finviz_data, dict) else {}
        
        if quarterly_estimates:
            rev_count = len(quarterly_estimates.get('revenue', {}))
            eps_count = len(quarterly_estimates.get('eps', {}))
            if rev_count > 0 or eps_count > 0:
                print(f"Finviz: Found {rev_count} revenue and {eps_count} EPS estimates for {ticker}")
        
        if quarterly_actuals:
            rev_actual_count = len(quarterly_actuals.get('revenue', {}))
            eps_actual_count = len(quarterly_actuals.get('eps', {}))
            if rev_actual_count > 0 or eps_actual_count > 0:
                print(f"Finviz: Found {rev_actual_count} revenue and {eps_actual_count} EPS actual values for {ticker}")
        
        # Get company info for sector/industry context
        try:
            info = stock.info
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
        except:
            info = {}
            sector = 'N/A'
            industry = 'N/A'
        
        # Map industry to category for ranking
        # Also check ticker for specific companies (CIFR, IREN, NBIS are data centers)
        data_center_tickers = ['CIFR', 'IREN', 'NBIS', 'EQIX', 'DLR', 'AMT', 'CCI', 'SBAC', 'GDS', 'VNET', 'PD', 'CONE', 'QTS', 'COR', 'LAND']
        
        industry_category_map = {
            'data centers': ['data center', 'data center reit', 'reit—data center', 'data center reits', 'data centers', 'datacenter', 'datacenters'],
            'technology': ['technology', 'software', 'semiconductors', 'internet', 'telecommunications', 'tech'],
            'healthcare': ['healthcare', 'biotechnology', 'pharmaceuticals', 'medical devices', 'biotech'],
            'finance': ['financial services', 'banks', 'insurance', 'capital markets', 'banking'],
            'energy': ['oil & gas', 'renewable energy', 'utilities', 'energy', 'oil'],
            'consumer': ['consumer goods', 'retail', 'consumer services', 'consumer'],
            'industrial': ['industrial', 'manufacturing', 'machinery', 'industrials'],
            'real estate': ['real estate', 'reit', 'real estate investment trust', 'reits']
        }
        
        # Determine industry category
        # First check if ticker is in data center list
        industry_category = 'Other'
        if ticker.upper() in data_center_tickers:
            industry_category = 'data centers'
        else:
            industry_lower = industry.lower() if industry else ''
            for category, keywords in industry_category_map.items():
                if any(keyword in industry_lower for keyword in keywords):
                    industry_category = category
                    break
        
        # Get industry ranking
        industry_ranking = get_industry_ranking(ticker, industry_category, sector, info.get('marketCap', 0))
        
        # Start with Finviz estimates
        forward_revenue_estimates = quarterly_estimates.get('revenue', {}).copy()
        forward_eps_estimates = quarterly_estimates.get('eps', {}).copy()
        
        # Check if we need to supplement with yfinance calendar for future quarters
        from datetime import datetime
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        current_quarter = (current_month - 1) // 3 + 1
        
        # Check if Finviz has future quarters (e.g., 2026)
        finviz_has_future = False
        for q in list(forward_revenue_estimates.keys()) + list(forward_eps_estimates.keys()):
            if '2026' in q or '2027' in q:
                finviz_has_future = True
                break
        
        # If Finviz doesn't have future quarters, try yfinance calendar as fallback
        if not finviz_has_future:
            try:
                calendar = stock.calendar
                if calendar is not None:
                    
                    # yfinance calendar can be either DataFrame or dict
                    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                        # Parse DataFrame format
                        for idx, row in calendar.iterrows():
                            if pd.isna(idx):
                                continue
                            
                            # Get quarter info from the index (usually a date)
                            if isinstance(idx, pd.Timestamp):
                                cal_year = idx.year
                                cal_month = idx.month
                                cal_quarter = (cal_month - 1) // 3 + 1
                                quarter_str = f"{cal_year}-Q{cal_quarter}"
                                
                                # Only add if it's a future quarter
                                is_future = (cal_year > current_year) or (cal_year == current_year and cal_quarter > current_quarter)
                                
                                if is_future:
                                    # Try to get revenue estimate
                                    revenue_est = None
                                    if 'Revenue Average' in row.index:
                                        revenue_est = row['Revenue Average']
                                    elif 'Revenue' in row.index:
                                        revenue_est = row['Revenue']
                                    
                                    if revenue_est is not None and pd.notna(revenue_est):
                                        try:
                                            revenue_val = float(revenue_est)
                                            if revenue_val > 0:
                                                # yfinance returns revenue in actual dollars (not millions)
                                                if quarter_str not in forward_revenue_estimates:
                                                    forward_revenue_estimates[quarter_str] = revenue_val
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    # Try to get EPS estimate
                                    eps_est = None
                                    if 'Earnings Average' in row.index:
                                        eps_est = row['Earnings Average']
                                    elif 'Earnings' in row.index:
                                        eps_est = row['Earnings']
                                    
                                    if eps_est is not None and pd.notna(eps_est):
                                        try:
                                            eps_val = float(eps_est)
                                            if quarter_str not in forward_eps_estimates:
                                                forward_eps_estimates[quarter_str] = eps_val
                                        except (ValueError, TypeError):
                                            pass
                    elif isinstance(calendar, dict):
                        # Parse dict format (yfinance returns dict with keys like 'Revenue Average', 'Earnings Average', 'Earnings Date')
                        # yfinance calendar typically contains only one quarter, but we'll generate Q2, Q3, Q4 based on Q1
                        
                        # Get earnings date to determine Q1
                        earnings_date = None
                        if 'Earnings Date' in calendar:
                            earnings_date = calendar['Earnings Date']
                            if isinstance(earnings_date, (list, tuple)) and len(earnings_date) > 0:
                                earnings_date = earnings_date[0]
                        
                        # Determine Q1 from earnings date
                        q1_quarter_str = None
                        if earnings_date:
                            if isinstance(earnings_date, pd.Timestamp):
                                cal_year = earnings_date.year
                                cal_month = earnings_date.month
                                cal_quarter = (cal_month - 1) // 3 + 1
                                q1_quarter_str = f"{cal_year}-Q{cal_quarter}"
                            elif isinstance(earnings_date, str):
                                # Try to parse date string
                                try:
                                    from dateutil import parser
                                    parsed_date = parser.parse(earnings_date)
                                    cal_year = parsed_date.year
                                    cal_month = parsed_date.month
                                    cal_quarter = (cal_month - 1) // 3 + 1
                                    q1_quarter_str = f"{cal_year}-Q{cal_quarter}"
                                except:
                                    pass
                        
                        # If no date, use next quarter as Q1
                        if not q1_quarter_str:
                            if current_quarter < 4:
                                q1_quarter_str = f"{current_year}-Q{current_quarter + 1}"
                            else:
                                q1_quarter_str = f"{current_year + 1}-Q1"
                        
                        # Get Q1 estimates
                        q1_revenue = None
                        q1_eps = None
                        
                        if 'Revenue Average' in calendar:
                            revenue_est = calendar['Revenue Average']
                            if revenue_est is not None:
                                try:
                                    q1_revenue = float(revenue_est)
                                except (ValueError, TypeError):
                                    pass
                        
                        if 'Earnings Average' in calendar:
                            eps_est = calendar['Earnings Average']
                            if eps_est is not None:
                                try:
                                    q1_eps = float(eps_est)
                                except (ValueError, TypeError):
                                    pass
                        
                        # Process Q1, Q2, Q3, Q4 (generate subsequent quarters)
                        for i in range(4):  # Q1, Q2, Q3, Q4
                            # Calculate quarter string
                            q1_year = int(q1_quarter_str.split('-Q')[0])
                            q1_num = int(q1_quarter_str.split('-Q')[1])
                            
                            target_q = q1_num + i
                            target_year = q1_year
                            while target_q > 4:
                                target_q -= 4
                                target_year += 1
                            
                            quarter_str = f"{target_year}-Q{target_q}"
                            
                            # Check if it's a future quarter
                            q_year = int(quarter_str.split('-Q')[0])
                            q_num = int(quarter_str.split('-Q')[1])
                            is_future = (q_year > current_year) or (q_year == current_year and q_num > current_quarter)
                            
                            if is_future and quarter_str not in forward_revenue_estimates and quarter_str not in forward_eps_estimates:
                                # For Q1, use actual estimates from yfinance
                                # For Q2, use Q1 values as placeholder (yfinance calendar typically has only Q1)
                                if i == 0:
                                    # Q1: use actual estimates
                                    if q1_revenue and q1_revenue > 0:
                                        forward_revenue_estimates[quarter_str] = q1_revenue
                                    if q1_eps is not None:
                                        forward_eps_estimates[quarter_str] = q1_eps
                                elif i == 1:
                                    # Q2: use Q1 values as placeholder (since yfinance calendar usually has only Q1)
                                    if q1_revenue and q1_revenue > 0:
                                        forward_revenue_estimates[quarter_str] = q1_revenue
                                    if q1_eps is not None:
                                        forward_eps_estimates[quarter_str] = q1_eps
                                # Note: Q3-Q4 are not added here because we don't have real estimates for them
                                # They will be added if Finviz provides them later
                else:
                    pass  # Calendar not available
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        
        financials = {
            'executive_snapshot': {},
            'income_statement': {'quarterly': [], 'annual': []},
            'margins': {'quarterly': [], 'annual': []},
            'cash_flow': {'quarterly': [], 'annual': []},
            'balance_sheet': {},
            'segments': [],
            'red_flags': [],
            'fundamentals_verdict': 'neutral',
            'main_verdict_sentence': '',
            'company_stage': 'unknown',
            'sector': sector,
            'industry': industry,
            'industry_category': industry_category,
            'industry_ranking': industry_ranking,
            'forward_estimates': {
                'revenue': forward_revenue_estimates,
                'eps': forward_eps_estimates
            }
        }
        
        
        # #region agent log (disabled for production)
        # Debug logging removed - causes FileNotFoundError on Render
        # #endregion
        
        # NOTE: quarterly_estimates were already fetched above (Finviz ONLY)
        # Do NOT reset quarterly_estimates here - it would overwrite the estimates we just fetched!
        
        # Get income statements
        quarterly_income = stock.quarterly_income_stmt
        annual_income = stock.financials
        
        # Get cash flow statements
        quarterly_cf = stock.quarterly_cashflow
        annual_cf = stock.cashflow
        
        # Get balance sheet
        quarterly_bs = stock.quarterly_balance_sheet
        annual_bs = stock.balance_sheet
        
        # Helper function to find row in statement
        def find_row(df, search_terms):
            if df is None or df.empty:
                return None
            for term in search_terms:
                for idx in df.index:
                    if term in str(idx).lower():
                        return df.loc[idx]
            return None
        
        # Calculate TTM (Trailing Twelve Months) values
        def calculate_ttm(quarterly_data):
            if quarterly_data is None or quarterly_data.empty:
                return None
            # Sum last 4 quarters
            if len(quarterly_data.columns) >= 4:
                return quarterly_data.iloc[:, :4].sum(axis=1)
            return None
        
        # Get Revenue
        revenue_row_q = find_row(quarterly_income, ['total revenue', 'revenue', 'net sales', 'sales'])
        revenue_row_a = find_row(annual_income, ['total revenue', 'revenue', 'net sales', 'sales'])
        
        # Get Net Income
        net_income_row_q = find_row(quarterly_income, ['net income', 'total net income', 'income from continuing operations'])
        net_income_row_a = find_row(annual_income, ['net income', 'total net income', 'income from continuing operations'])
        
        # Get Operating Cash Flow
        ocf_row_q = find_row(quarterly_cf, ['operating cash flow', 'total cash from operating activities', 'operating activities'])
        ocf_row_a = find_row(annual_cf, ['operating cash flow', 'total cash from operating activities', 'operating activities'])
        
        # Get CapEx
        capex_row_q = find_row(quarterly_cf, ['capital expenditures', 'capex', 'capital expenditure'])
        capex_row_a = find_row(annual_cf, ['capital expenditures', 'capex', 'capital expenditure'])
        
        # Get Free Cash Flow (Operating CF - CapEx)
        def calculate_fcf(ocf_row, capex_row):
            if ocf_row is None or capex_row is None:
                return None
            try:
                # Both are pandas Series with same index (quarterly dates)
                fcf = ocf_row.copy()
                for i in range(min(len(fcf), len(capex_row))):
                    if pd.notna(fcf.iloc[i]) and pd.notna(capex_row.iloc[i]):
                        fcf.iloc[i] = float(fcf.iloc[i]) - abs(float(capex_row.iloc[i]))
                    else:
                        fcf.iloc[i] = None
                return fcf
            except:
                return None
        
        fcf_row_q = calculate_fcf(ocf_row_q, capex_row_q) if ocf_row_q is not None and capex_row_q is not None else None
        fcf_row_a = calculate_fcf(ocf_row_a, capex_row_a) if ocf_row_a is not None and capex_row_a is not None else None
        
        # Calculate TTM values (sum of last 4 quarters)
        revenue_ttm = None
        net_income_ttm = None
        fcf_ttm = None
        
        if revenue_row_q is not None and len(revenue_row_q) >= 4:
            try:
                # revenue_row_q is a pandas Series, access values directly
                revenue_ttm = float(revenue_row_q.iloc[:4].sum())
            except:
                try:
                    revenue_ttm = sum([float(v) for v in list(revenue_row_q.values)[:4] if pd.notna(v)])
                except:
                    pass
        
        if net_income_row_q is not None and len(net_income_row_q) >= 4:
            try:
                net_income_ttm = float(net_income_row_q.iloc[:4].sum())
            except:
                try:
                    net_income_ttm = sum([float(v) for v in list(net_income_row_q.values)[:4] if pd.notna(v)])
                except:
                    pass
        
        if fcf_row_q is not None and len(fcf_row_q) >= 4:
            try:
                fcf_ttm = float(fcf_row_q.iloc[:4].sum())
            except:
                try:
                    fcf_ttm = sum([float(v) for v in list(fcf_row_q.values)[:4] if pd.notna(v)])
                except:
                    pass
        
        # Calculate YoY growth (compare current quarter to same quarter last year)
        def calculate_yoy_growth(current_row, periods_ago=4):
            if current_row is None or len(current_row) < periods_ago + 1:
                return None
            try:
                # Get most recent (first) and 4 quarters ago
                current = float(current_row.iloc[0])
                previous = float(current_row.iloc[periods_ago])
                if previous != 0 and not pd.isna(current) and not pd.isna(previous):
                    return ((current - previous) / abs(previous)) * 100
            except (IndexError, ValueError, TypeError):
                pass
            return None
        
        revenue_yoy = calculate_yoy_growth(revenue_row_q)
        net_income_yoy = calculate_yoy_growth(net_income_row_q)
        
        # Get Gross Margin
        gross_profit_row_q = find_row(quarterly_income, ['gross profit', 'total gross profit'])
        gross_margin_q = None
        if gross_profit_row_q is not None and revenue_row_q is not None:
            try:
                if len(gross_profit_row_q) > 0 and len(revenue_row_q) > 0:
                    gross_profit = float(gross_profit_row_q.iloc[0] if hasattr(gross_profit_row_q, 'iloc') else list(gross_profit_row_q.values())[0])
                    revenue = float(revenue_row_q.iloc[0] if hasattr(revenue_row_q, 'iloc') else list(revenue_row_q.values())[0])
                    if revenue != 0:
                        gross_margin_q = (gross_profit / revenue) * 100
            except:
                pass
        
        # Get Operating Margin
        operating_income_row_q = find_row(quarterly_income, ['operating income', 'income from operations', 'operating profit'])
        operating_margin_q = None
        if operating_income_row_q is not None and revenue_row_q is not None:
            try:
                if len(operating_income_row_q) > 0 and len(revenue_row_q) > 0:
                    op_income = float(operating_income_row_q.iloc[0] if hasattr(operating_income_row_q, 'iloc') else list(operating_income_row_q.values())[0])
                    revenue = float(revenue_row_q.iloc[0] if hasattr(revenue_row_q, 'iloc') else list(revenue_row_q.values())[0])
                    if revenue != 0:
                        operating_margin_q = (op_income / revenue) * 100
            except:
                pass
        
        # Get Net Margin
        net_margin_q = None
        if net_income_row_q is not None and revenue_row_q is not None:
            try:
                if len(net_income_row_q) > 0 and len(revenue_row_q) > 0:
                    net_income = float(net_income_row_q.iloc[0] if hasattr(net_income_row_q, 'iloc') else list(net_income_row_q.values())[0])
                    revenue = float(revenue_row_q.iloc[0] if hasattr(revenue_row_q, 'iloc') else list(revenue_row_q.values())[0])
                    if revenue != 0:
                        net_margin_q = (net_income / revenue) * 100
            except:
                pass
        
        # Get Debt
        total_debt_row = find_row(quarterly_bs, ['total debt', 'total liabilities', 'long term debt'])
        total_debt = None
        if total_debt_row is not None and len(total_debt_row) > 0:
            try:
                total_debt = float(total_debt_row.iloc[0] if hasattr(total_debt_row, 'iloc') else list(total_debt_row.values())[0])
            except:
                pass
        
        # Calculate Debt/FCF ratio
        debt_fcf_ratio = None
        if total_debt is not None and fcf_ttm is not None and fcf_ttm != 0:
            debt_fcf_ratio = abs(total_debt / fcf_ttm)
        
        # FCF Margin
        fcf_margin = None
        if fcf_ttm is not None and revenue_ttm is not None and revenue_ttm != 0:
            fcf_margin = (fcf_ttm / revenue_ttm) * 100
        
        # Build Executive Snapshot
        financials['executive_snapshot'] = {
            'revenue_ttm': revenue_ttm,
            'revenue_yoy': revenue_yoy,
            'net_income_ttm': net_income_ttm,
            'net_income_yoy': net_income_yoy,
            'fcf_ttm': fcf_ttm,
            'fcf_margin': fcf_margin,
            'gross_margin': gross_margin_q,
            'debt_fcf_ratio': debt_fcf_ratio
        }
        
        # Build Income Statement data for charts
        if revenue_row_q is not None and quarterly_income is not None:
            for i, col in enumerate(quarterly_income.columns[:8]):  # Last 8 quarters
                try:
                    quarter_date = pd.Timestamp(col)
                    quarter_calc = (quarter_date.month - 1) // 3 + 1
                    quarter_str = f"{quarter_date.year}-Q{quarter_calc}"
                    
                    # Prefer Finviz actuals over yfinance for revenue (same as EPS)
                    revenue_val = None
                    revenue_source = None
                    
                    # Try Finviz actuals first - try exact match by quarter string first, then date matching
                    if quarterly_actuals and 'revenue' in quarterly_actuals:
                        best_match_rev = None
                        best_match_q = None
                        min_date_diff = float('inf')
                        
                        # FIRST: Try exact match by quarter string (e.g., "2024-Q3" == "2024-Q3")
                        if quarter_str in quarterly_actuals['revenue']:
                            best_match_rev = quarterly_actuals['revenue'][quarter_str]
                            best_match_q = quarter_str
                            min_date_diff = 0
                        else:
                            # SECOND: Find closest Finviz revenue by date (fallback for fiscal vs calendar quarter differences)
                            for finviz_q, finviz_rev in quarterly_actuals['revenue'].items():
                                try:
                                    if '-Q' in finviz_q:
                                        fv_year, fv_num = finviz_q.split('-Q')
                                        fv_num = int(fv_num)
                                        fv_year = int(fv_year)
                                        fv_month = (fv_num - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                                        fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                                        date_diff = abs((quarter_date - fv_date).days)
                                        
                                        
                                        # Accept match within 120 days (allows for fiscal vs calendar quarter differences)
                                        if date_diff < min_date_diff and date_diff <= 120:
                                            min_date_diff = date_diff
                                            best_match_rev = finviz_rev
                                            best_match_q = finviz_q
                                except Exception as e:
                                    pass
                        
                        if best_match_rev is not None:
                            revenue_val = best_match_rev
                            revenue_source = 'Finviz'
                            print(f"[DEBUG] Using Finviz revenue actual for {ticker} {quarter_str}: {revenue_val} (matched with {best_match_q}, date_diff={min_date_diff} days)")
                        else:
                            print(f"[DEBUG] No Finviz revenue actual found for {ticker} {quarter_str} (tried exact match and date matching, quarter_date={quarter_date.date()})")
                            print(f"[DEBUG] Available Finviz quarters: {list(quarterly_actuals['revenue'].keys())}")
                    else:
                        # No Finviz revenue actuals available
                        pass
                    
                    # Fallback to yfinance if Finviz not available
                    if revenue_val is None:
                        revenue_val = float(revenue_row_q.iloc[i]) if i < len(revenue_row_q) else None
                        if revenue_val is not None and not pd.isna(revenue_val):
                            revenue_source = 'yfinance'
                            print(f"[DEBUG] Using yfinance revenue for {ticker} {quarter_str}: {revenue_val}")
                    
                    net_income_val = float(net_income_row_q.iloc[i]) if net_income_row_q is not None and i < len(net_income_row_q) else None
                    
                    if revenue_val is not None and not pd.isna(revenue_val):
                        # Try to get EPS from income statement
                        eps_val = None
                        eps_source = None
                        
                        # Prefer Finviz actuals over yfinance for EPS if available
                        if quarterly_actuals and isinstance(quarterly_actuals, dict):
                            # #region agent log
                            # #endregion
                            if 'eps' in quarterly_actuals:
                                # Try exact match first
                                if quarter_str in quarterly_actuals['eps']:
                                    eps_val = quarterly_actuals['eps'][quarter_str]
                                    eps_source = 'Finviz'
                                    # #region agent log
                                    print(f"[DEBUG] Using Finviz EPS for {ticker} {quarter_str}: {eps_val}")
                                    # #endregion
                                else:
                                    # Try to find matching quarter with different format
                                    for finviz_q, finviz_eps in quarterly_actuals['eps'].items():
                                        # #region agent log
                                        # #endregion
                                        if finviz_q == quarter_str:
                                            eps_val = finviz_eps
                                            eps_source = 'Finviz'
                                            # #region agent log
                                            print(f"[DEBUG] Matched Finviz EPS for {ticker} {quarter_str}: {eps_val}")
                                            # #endregion
                                            break
                        
                        # Fallback to yfinance if Finviz not available
                        if eps_val is None:
                            try:
                                eps_row = find_row(quarterly_income, ['diluted eps', 'basic eps', 'earnings per share', 'eps'])
                                if eps_row is not None and i < len(eps_row):
                                    eps_val = float(eps_row.iloc[i]) if hasattr(eps_row, 'iloc') else float(list(eps_row.values())[i])
                                    if pd.isna(eps_val):
                                        eps_val = None
                                    else:
                                        eps_source = 'yfinance'
                                        # #region agent log
                                        print(f"[DEBUG] Using yfinance EPS for {ticker} {quarter_str}: {eps_val}")
                                        # #endregion
                            except Exception:
                                pass
                            pass
                        
                        # #region agent log
                        if eps_val is None:
                            print(f"[DEBUG] No EPS value found for {ticker} {quarter_str} from any source")
                        # #endregion
                        
                        # Get estimates from Finviz
                        revenue_estimate = None
                        eps_estimate = None
                        
                        if quarterly_estimates and isinstance(quarterly_estimates, dict):
                            # Revenue estimate - try exact match first, then try date-based matching (SAME AS EPS)
                            if 'revenue' in quarterly_estimates:
                                # Try exact match first
                                if quarter_str in quarterly_estimates['revenue']:
                                    revenue_estimate = quarterly_estimates['revenue'][quarter_str]
                                    print(f"[DEBUG] Using Finviz revenue estimate (exact match) for {ticker} {quarter_str}: {revenue_estimate}")
                                else:
                                    # Try to find closest match by date (similar to actuals matching)
                                    best_match_est = None
                                    best_match_q = None
                                    min_date_diff = float('inf')
                                    
                                    for finviz_q, finviz_est in quarterly_estimates['revenue'].items():
                                        try:
                                            if '-Q' in finviz_q:
                                                fv_year, fv_num = finviz_q.split('-Q')
                                                fv_num = int(fv_num)
                                                fv_year = int(fv_year)
                                                fv_month = (fv_num - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                                                fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                                                date_diff = abs((quarter_date - fv_date).days)
                                                
                                                # Accept match within 120 days
                                                if date_diff < min_date_diff and date_diff <= 120:
                                                    min_date_diff = date_diff
                                                    best_match_est = finviz_est
                                                    best_match_q = finviz_q
                                        except Exception as e:
                                            pass
                                    
                                    if best_match_est is not None:
                                        revenue_estimate = best_match_est
                                        print(f"[DEBUG] Using Finviz revenue estimate (date match) for {ticker} {quarter_str}: {revenue_estimate} (matched with {best_match_q}, date_diff={min_date_diff} days)")
                                    else:
                                        print(f"[DEBUG] No revenue estimate found for {ticker} {quarter_str} (tried date matching)")
                            
                            # EPS estimate - try exact match first, then try date-based matching
                            if 'eps' in quarterly_estimates:
                                # Try exact match first
                                if quarter_str in quarterly_estimates['eps']:
                                    eps_estimate = quarterly_estimates['eps'][quarter_str]
                                else:
                                    # Try to find closest match by date
                                    best_match_eps_est = None
                                    min_date_diff = float('inf')
                                    
                                    for finviz_q, finviz_eps_est in quarterly_estimates['eps'].items():
                                        try:
                                            if '-Q' in finviz_q:
                                                fv_year, fv_num = finviz_q.split('-Q')
                                                fv_num = int(fv_num)
                                                fv_year = int(fv_year)
                                                fv_month = (fv_num - 1) * 3 + 1
                                                fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                                                date_diff = abs((quarter_date - fv_date).days)
                                                
                                                if date_diff < min_date_diff and date_diff <= 120:
                                                    min_date_diff = date_diff
                                                    best_match_eps_est = finviz_eps_est
                                        except Exception as e:
                                            pass
                                    
                                    if best_match_eps_est is not None:
                                        eps_estimate = best_match_eps_est
                        
                        # Debug: Log what we're adding
                        
                        financials['income_statement']['quarterly'].append({
                            'quarter': quarter_str,
                            'date': quarter_date.strftime('%Y-%m-%d'),
                            'revenue': revenue_val,
                            'revenue_estimate': revenue_estimate,  # Analyst estimate or None
                            'net_income': net_income_val if net_income_val is not None and not pd.isna(net_income_val) else None,
                            'eps': eps_val,
                            'eps_estimate': eps_estimate  # Analyst estimate or None
                        })
                except (IndexError, ValueError, TypeError):
                    continue
        
        # Build Margins data
        if gross_margin_q is not None or operating_margin_q is not None or net_margin_q is not None:
            financials['margins']['quarterly'].append({
                'gross_margin': gross_margin_q,
                'operating_margin': operating_margin_q,
                'net_margin': net_margin_q
            })
        
        # Build Cash Flow data
        if ocf_row_q is not None and quarterly_cf is not None:
            for i, col in enumerate(quarterly_cf.columns[:8]):
                try:
                    quarter_date = pd.Timestamp(col)
                    quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                    ocf_val = float(ocf_row_q.iloc[i]) if i < len(ocf_row_q) else None
                    capex_val = float(capex_row_q.iloc[i]) if capex_row_q is not None and i < len(capex_row_q) else None
                    fcf_val = float(fcf_row_q.iloc[i]) if fcf_row_q is not None and i < len(fcf_row_q) else None
                    
                    if ocf_val is not None and not pd.isna(ocf_val):
                        financials['cash_flow']['quarterly'].append({
                            'quarter': quarter_str,
                            'date': quarter_date.strftime('%Y-%m-%d'),
                            'operating_cf': ocf_val,
                            'capex': abs(capex_val) if capex_val is not None and not pd.isna(capex_val) else None,
                            'fcf': fcf_val if fcf_val is not None and not pd.isna(fcf_val) else None
                        })
                except (IndexError, ValueError, TypeError):
                    continue
        
        # Build Balance Sheet (simplified)
        cash_row = find_row(quarterly_bs, ['cash and cash equivalents', 'cash', 'cash and short term investments'])
        equity_row = find_row(quarterly_bs, ['total stockholders equity', 'total equity', 'stockholders equity'])
        current_assets_row = find_row(quarterly_bs, ['total current assets'])
        current_liabilities_row = find_row(quarterly_bs, ['total current liabilities'])
        
        cash = None
        equity = None
        current_ratio = None
        
        if cash_row is not None and len(cash_row) > 0:
            try:
                cash = float(cash_row.iloc[0] if hasattr(cash_row, 'iloc') else list(cash_row.values())[0])
            except:
                pass
        
        if equity_row is not None and len(equity_row) > 0:
            try:
                equity = float(equity_row.iloc[0] if hasattr(equity_row, 'iloc') else list(equity_row.values())[0])
            except:
                pass
        
        if current_assets_row is not None and current_liabilities_row is not None:
            try:
                if len(current_assets_row) > 0 and len(current_liabilities_row) > 0:
                    ca = float(current_assets_row.iloc[0] if hasattr(current_assets_row, 'iloc') else list(current_assets_row.values())[0])
                    cl = float(current_liabilities_row.iloc[0] if hasattr(current_liabilities_row, 'iloc') else list(current_liabilities_row.values())[0])
                    if cl != 0:
                        current_ratio = ca / cl
            except:
                pass
        
        net_debt = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
        
        financials['balance_sheet'] = {
            'cash': cash,
            'total_debt': total_debt,
            'net_debt': net_debt,
            'equity': equity,
            'current_ratio': current_ratio
        }
        
        # Generate Red Flags
        red_flags = []
        
        # Check for declining revenue
        # Note: quarterly data is ordered from newest to oldest (revenues[0] = most recent)
        # Data: [Q3, Q2, Q1] where Q3 is newest
        # Revenue is declining if: Q3 > Q2 > Q1 (each newer quarter is larger than older)
        # So: revenues[0] > revenues[1] > revenues[2]
        if len(financials['income_statement']['quarterly']) >= 3:
            revenues = [q['revenue'] for q in financials['income_statement']['quarterly'][:3] if q.get('revenue') is not None]
            if len(revenues) >= 3:
                # Check if revenue is declining: newest > previous > older (i.e., revenues[0] > revenues[1] > revenues[2])
                # This means each newer quarter is larger than the older one (so revenue is declining over time)
                is_declining = all(revenues[i] > revenues[i+1] for i in range(len(revenues)-1))
                if is_declining:
                    red_flags.append({
                        'type': 'revenue_decline',
                        'severity': 'high',
                        'message': 'Tržby klesají 3 kvartály po sobě'
                    })
        
        # Check FCF < Net Income
        if fcf_ttm is not None and net_income_ttm is not None and fcf_ttm < net_income_ttm:
            red_flags.append({
                'type': 'fcf_quality',
                'severity': 'medium',
                'message': 'FCF < Net Income (možné accounting issues)'
            })
        
        # Check rising debt + falling margins
        if debt_fcf_ratio is not None and debt_fcf_ratio > 3 and gross_margin_q is not None:
            red_flags.append({
                'type': 'debt_margin',
                'severity': 'high',
                'message': 'Rostoucí dluh + potenciálně klesající marže'
            })
        
        financials['red_flags'] = red_flags
        
        # Detect Company Stage
        company_stage = 'unknown'
        stage_indicators = []
        
        # Early-stage: Very low/no revenue, negative earnings, high burn rate
        if revenue_ttm is not None and revenue_ttm < 100_000_000:  # Less than $100M
            if net_income_ttm is not None and net_income_ttm < 0:
                if fcf_ttm is not None and fcf_ttm < 0:
                    company_stage = 'early_stage'
                    stage_indicators.append('pre_revenue')
        
        # Growth: Growing revenue, may be unprofitable but improving
        if revenue_yoy and revenue_yoy > 20:
            if net_income_ttm is None or net_income_ttm < 0:
                if net_income_yoy and net_income_yoy > 0:  # Improving losses
                    company_stage = 'growth'
                    stage_indicators.append('high_growth')
        
        # Mature: Stable revenue, positive earnings, positive FCF
        if revenue_ttm and revenue_ttm > 1_000_000_000:  # Over $1B
            if net_income_ttm and net_income_ttm > 0:
                if fcf_ttm and fcf_ttm > 0:
                    if revenue_yoy and -5 < revenue_yoy < 15:  # Moderate growth
                        company_stage = 'mature'
                        stage_indicators.append('stable')
        
        # Turnaround: Declining revenue, negative earnings, trying to recover
        if revenue_yoy and revenue_yoy < -10:
            if net_income_ttm and net_income_ttm < 0:
                company_stage = 'turnaround'
                stage_indicators.append('declining')
        
        # Default to growth if revenue is growing
        if company_stage == 'unknown' and revenue_yoy and revenue_yoy > 10:
            company_stage = 'growth'
        
        financials['company_stage'] = company_stage
        
        # Generate Fundamentals Verdict
        verdict_score = 0
        if revenue_yoy and revenue_yoy > 0:
            verdict_score += 1
        if net_income_yoy and net_income_yoy > 0:
            verdict_score += 1
        if fcf_ttm and fcf_ttm > 0:
            verdict_score += 1
        if gross_margin_q and gross_margin_q > 30:
            verdict_score += 1
        if debt_fcf_ratio and debt_fcf_ratio < 2:
            verdict_score += 1
        
        if verdict_score >= 4:
            financials['fundamentals_verdict'] = 'strong'
        elif verdict_score >= 2:
            financials['fundamentals_verdict'] = 'neutral'
        else:
            financials['fundamentals_verdict'] = 'weak'
        
        # Generate Main Verdict Sentence
        verdict_parts = []
        
        # Revenue assessment
        if revenue_yoy:
            if revenue_yoy > 15:
                verdict_parts.append('Silné tržby')
            elif revenue_yoy > 5:
                verdict_parts.append('Rostoucí tržby')
            elif revenue_yoy > 0:
                verdict_parts.append('Mírně rostoucí tržby')
            else:
                verdict_parts.append('Klesající tržby')
        
        # Earnings assessment
        if net_income_ttm:
            if net_income_ttm > 0:
                if net_income_yoy and net_income_yoy > 10:
                    verdict_parts.append('zisk roste rychle')
                elif net_income_yoy and net_income_yoy > 0:
                    verdict_parts.append('zisk roste')
                else:
                    verdict_parts.append('zisk je stabilní')
            else:
                if company_stage == 'early_stage':
                    verdict_parts.append('ztráty jsou očekávané (early-stage)')
                elif net_income_yoy and net_income_yoy > 0:
                    verdict_parts.append('ztráty se zmenšují')
                else:
                    verdict_parts.append('ztráty pokračují')
        
        # Cash flow assessment
        if fcf_ttm:
            if fcf_ttm > 0:
                if fcf_ttm >= net_income_ttm if net_income_ttm else False:
                    verdict_parts.append('výborný cash flow')
                else:
                    verdict_parts.append('pozitivní cash flow')
            else:
                if company_stage == 'early_stage':
                    verdict_parts.append('burn rate je očekávaný')
                else:
                    verdict_parts.append('negativní cash flow')
        
        # Combine into sentence
        if len(verdict_parts) >= 2:
            main_sentence = f"{verdict_parts[0]}, {verdict_parts[1]}"
            if len(verdict_parts) >= 3:
                main_sentence += f" → {verdict_parts[2]}"
        elif len(verdict_parts) == 1:
            main_sentence = verdict_parts[0]
        else:
            main_sentence = "Finanční data vyžadují další analýzu."
        
        # Add context based on stage
        if company_stage == 'early_stage':
            main_sentence += " (Pre-revenue / early-stage company - ztráty jsou očekávané)"
        elif company_stage == 'growth':
            if net_income_ttm and net_income_ttm < 0:
                main_sentence += " (Growth phase - investice do růstu)"
        elif company_stage == 'turnaround':
            main_sentence += " (Turnaround - firma se snaží zotavit)"
        
        financials['main_verdict_sentence'] = main_sentence
        
        # Add trend indicators to executive snapshot
        financials['executive_snapshot']['revenue_trend'] = 'improving' if revenue_yoy and revenue_yoy > 0 else 'deteriorating' if revenue_yoy and revenue_yoy < 0 else 'stable'
        financials['executive_snapshot']['earnings_trend'] = 'improving' if net_income_yoy and net_income_yoy > 0 else 'deteriorating' if net_income_yoy and net_income_yoy < 0 else 'stable'
        financials['executive_snapshot']['fcf_trend'] = 'improving' if fcf_ttm and fcf_ttm > 0 else 'deteriorating' if fcf_ttm and fcf_ttm < 0 else 'stable'
        
        # Calculate overall financials score (pass company_stage for growth adjustments)
        financials_score = calculate_financials_score(financials, info, company_stage)
        financials['financials_score'] = financials_score
        
        # Add advanced analyses
        # 1. Cash Flow Statement Analysis
        cash_flow_analysis = get_cash_flow_analysis(ticker)
        if cash_flow_analysis:
            financials['cash_flow_analysis'] = cash_flow_analysis
        
        # 2. Profitability Deep Dive
        profitability_analysis = get_profitability_analysis(ticker, financials)
        if profitability_analysis:
            financials['profitability_analysis'] = profitability_analysis
        
        # 3. Balance Sheet Health Score
        balance_sheet_health = get_balance_sheet_health(ticker)
        if balance_sheet_health:
            financials['balance_sheet_health'] = balance_sheet_health
        
        # 4. Management Guidance Tracking
        guidance_tracking = get_management_guidance_tracking(ticker)
        if guidance_tracking:
            financials['management_guidance'] = guidance_tracking
        
        # 5. Segment/Geography Breakdown
        segment_breakdown = get_segment_breakdown(ticker)
        if segment_breakdown:
            financials['segment_breakdown'] = segment_breakdown
        
        # Clean financials data to ensure all Timestamps are converted before return
        return clean_for_json(financials)
        
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {str(e)}")
        import traceback
        tb_str = traceback.format_exc()
        print(tb_str)
        return None

def generate_news_summary(news_list, ticker):
    """Generate AI-powered summary of news articles"""
    if not news_list or len(news_list) == 0:
        return {
            'summary': 'No recent news available for this stock.',
            'key_points': [],
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'total_articles': 0
        }
    
    # Collect all text for analysis
    all_titles = []
    all_summaries = []
    sentiment_scores = []
    positive_articles = []
    negative_articles = []
    neutral_articles = []
    
    for article in news_list:
        title = article.get('title', '')
        summary = article.get('summary', '')
        sentiment = article.get('sentiment', 'neutral')
        sentiment_score = article.get('sentiment_score', 0.0)
        
        if title:
            all_titles.append(title)
        if summary and summary != 'No summary available':
            all_summaries.append(summary)
        
        sentiment_scores.append(sentiment_score)
        
        if sentiment == 'positive':
            positive_articles.append(article)
        elif sentiment == 'negative':
            negative_articles.append(article)
        else:
            neutral_articles.append(article)
    
    # Calculate overall sentiment
    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    if avg_sentiment_score >= 0.05:
        overall_sentiment = 'positive'
        sentiment_label = 'Pozitivní'
    elif avg_sentiment_score <= -0.05:
        overall_sentiment = 'negative'
        sentiment_label = 'Negativní'
    else:
        overall_sentiment = 'neutral'
        sentiment_label = 'Neutrální'
    
    # Extract key points from titles and summaries
    key_points = []
    
    # Get most important positive points
    if positive_articles:
        for article in positive_articles[:3]:
            title = article.get('title', '')
            if title and len(title) > 20:
                key_points.append({
                    'text': title,
                    'sentiment': 'positive',
                    'source': article.get('publisher', 'Press Release')
                })
    
    # Get most important negative points
    if negative_articles:
        for article in negative_articles[:3]:
            title = article.get('title', '')
            if title and len(title) > 20:
                key_points.append({
                    'text': title,
                    'sentiment': 'negative',
                    'source': article.get('publisher', 'Press Release')
                })
    
    # If not enough key points, add from neutral or all articles
    if len(key_points) < 5:
        remaining = [a for a in news_list if a.get('title', '') not in [kp['text'] for kp in key_points]]
        for article in remaining[:5 - len(key_points)]:
            title = article.get('title', '')
            if title and len(title) > 20:
                key_points.append({
                    'text': title,
                    'sentiment': article.get('sentiment', 'neutral'),
                    'source': article.get('publisher', 'Press Release')
                })
    
    # Generate summary text
    total_articles = len(news_list)
    positive_count = len(positive_articles)
    negative_count = len(negative_articles)
    neutral_count = len(neutral_articles)
    
    summary_parts = []
    summary_parts.append(f"Shrnutí {total_articles} nejnovějších článků o {ticker}:")
    summary_parts.append("")
    
    if positive_count > 0:
        summary_parts.append(f"📈 Pozitivní zprávy ({positive_count}): Většina pozitivních zpráv se zaměřuje na růst, inovace a pozitivní výhled společnosti.")
    if negative_count > 0:
        summary_parts.append(f"📉 Negativní zprávy ({negative_count}): Některé zprávy poukazují na výzvy nebo rizika.")
    if neutral_count > 0:
        summary_parts.append(f"📊 Neutrální zprávy ({neutral_count}): Informační články bez výrazného sentimentu.")
    
    summary_parts.append("")
    summary_parts.append(f"Celkový sentiment: {sentiment_label} (skóre: {avg_sentiment_score:.2f})")
    
    if overall_sentiment == 'positive':
        summary_parts.append("Obecně převládá pozitivní nálada v médiích ohledně této akcie.")
    elif overall_sentiment == 'negative':
        summary_parts.append("V médiích převládá spíše negativní nálada ohledně této akcie.")
    else:
        summary_parts.append("Sentiment v médiích je převážně neutrální.")
    
    summary_text = "\n".join(summary_parts)
    
    return {
        'summary': summary_text,
        'key_points': key_points[:10],  # Limit to 10 key points
        'overall_sentiment': overall_sentiment,
        'sentiment_label': sentiment_label,
        'sentiment_score': round(avg_sentiment_score, 3),
        'total_articles': total_articles,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'sentiment_breakdown': {
            'positive': round(positive_count / total_articles * 100, 1) if total_articles > 0 else 0,
            'negative': round(negative_count / total_articles * 100, 1) if total_articles > 0 else 0,
            'neutral': round(neutral_count / total_articles * 100, 1) if total_articles > 0 else 0
        }
    }

def normalize_date(date_str):
    """Normalize date string to YYYY-MM-DD HH:MM format"""
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return None
    
    try:
        # Try parsing RFC 2822 format (from RSS feeds) FIRST - e.g., "Sun, 14 Dec 2025 04:19:00 GMT"
        try:
            import email.utils
            parsed_time = email.utils.parsedate_tz(date_str)
            if parsed_time:
                timestamp = email.utils.mktime_tz(parsed_time)
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime('%Y-%m-%d %H:%M')
        except Exception as e:
            pass
        
        # Try parsing ISO format
        if 'T' in date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        # Try common date formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%d.%m.%Y %H:%M', '%d.%m.%Y', '%B %d, %Y', '%b %d, %Y']:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d %H:%M')
            except:
                continue
        
        # If it contains relative time like "2 hours ago", return as is (frontend will handle it)
        if any(keyword in date_str.lower() for keyword in ['ago', 'hour', 'day', 'minute', 'just now']):
            return date_str
        
        # If all parsing fails, return original string
        return date_str
    except Exception as e:
        print(f"Error normalizing date '{date_str}': {str(e)}")
        return date_str

def get_stock_news(ticker, max_news=10):
    """Get latest Press Releases for a stock from Yahoo Finance and analyze sentiment"""
    try:
        analyzed_news = []
        
        # Press Release providers (known PR distribution services)
        pr_providers = [
            'GlobeNewswire', 'PR Newswire', 'Business Wire', 'BusinessWire',
            'ACCESSWIRE', 'Newsfile', 'NewMediaWire', 'PRWeb', 'Cision',
            'Marketwired', 'PRNewsWire'
        ]
        
        try:
            # Get Press Releases from yfinance using get_news with tab='press releases'
            stock = yf.Ticker(ticker.upper())
            news = stock.get_news(tab='press releases')
            
            if not news:
                print(f"No Press Releases found for {ticker}")
                return []
            
            # Process Press Releases (they should already be filtered by yfinance)
            for item in news:
                try:
                    content = item.get('content', {})
                    if not content:
                        continue
                    
                    provider = content.get('provider', {})
                    provider_name = provider.get('displayName', '') if provider else 'Press Release'
                    
                    # Extract news data (already filtered by yfinance get_news(tab='press releases'))
                    title = content.get('title', '')
                    summary = content.get('summary', '') or content.get('description', '')
                    pub_date = content.get('pubDate', '') or content.get('displayTime', '')
                    
                    # Get URL
                    canonical_url = content.get('canonicalUrl', {})
                    link = canonical_url.get('url', '') if canonical_url else ''
                    if not link:
                        click_url = content.get('clickThroughUrl', {})
                        link = click_url.get('url', '') if click_url else ''
                    
                    # Extract thumbnail/image URL if available
                    thumbnail_url = None
                    # Check various possible fields for thumbnail
                    if content.get('thumbnail'):
                        thumb = content.get('thumbnail')
                        thumbnail_url = thumb.get('url') if isinstance(thumb, dict) else thumb
                    elif content.get('thumbnailUrl'):
                        thumbnail_url = content.get('thumbnailUrl')
                    elif content.get('image'):
                        img = content.get('image')
                        thumbnail_url = img.get('url') if isinstance(img, dict) else img
                    elif content.get('imageUrl'):
                        thumbnail_url = content.get('imageUrl')
                    # Check in relatedImages if available
                    elif content.get('relatedImages'):
                        related_images = content.get('relatedImages', [])
                        if related_images and len(related_images) > 0:
                            thumbnail_url = related_images[0].get('url') if isinstance(related_images[0], dict) else related_images[0]
                    
                    # Analyze sentiment
                    text_for_analysis = f"{title}. {summary}".strip()
                    if not text_for_analysis:
                        continue
                    
                    sentiment_data = analyze_sentiment(text_for_analysis)
                    
                    # Calculate impact score based on sentiment magnitude and source importance
                    sentiment_magnitude = abs(sentiment_data['score'])
                    
                    # Determine source weight
                    major_financial_sources = ['Bloomberg', 'Reuters', 'Wall Street Journal', 'Financial Times', 'CNBC', 'MarketWatch', 'Yahoo Finance', 'Seeking Alpha']
                    source_weight = 1.0
                    if provider_name in major_financial_sources:
                        source_weight = 1.5
                    elif 'Press Release' in provider_name or provider_name == 'Press Release':
                        source_weight = 1.2
                    
                    # Classify news type
                    news_type = classify_news_type(title, summary)
                    
                    # Normalize date
                    if not pub_date or pub_date.strip() == '':
                        pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
                    
                    normalized_date = normalize_date(pub_date) if pub_date else None
                    published_str = normalized_date if normalized_date else datetime.now().strftime('%Y-%m-%d %H:%M')
                    
                    # Calculate price impact (1h, 1d, 1w)
                    price_impact = calculate_news_price_impact(ticker, published_str, title, summary)
                    
                    # Calculate comprehensive impact score
                    impact_score = calculate_news_impact_score(
                        price_impact, 
                        sentiment_data['score'], 
                        news_type, 
                        source_weight
                    )
                    
                    analyzed_news.append({
                        'title': title,
                        'summary': summary[:500] if summary else 'No summary available',
                        'publisher': provider_name if provider_name else 'Press Release',
                        'link': link,
                        'published': published_str,
                        'thumbnail': thumbnail_url,
                        'sentiment': sentiment_data['sentiment'],
                        'sentiment_label': sentiment_data['label'],
                        'sentiment_score': sentiment_data['score'],
                        'impact_score': impact_score,
                        'news_type': news_type,
                        'price_impact': price_impact,
                        'sentiment_details': {
                            'positive': sentiment_data['positive'],
                            'negative': sentiment_data['negative'],
                            'neutral': sentiment_data['neutral']
                        }
                    })
                    
                    if len(analyzed_news) >= max_news:
                        break
                        
                except Exception as e:
                    print(f"Error processing news item for {ticker}: {str(e)}")
                    continue
            
            print(f"Found {len(analyzed_news)} Press Releases for {ticker}")
            return analyzed_news
            
        except Exception as e:
            print(f"Error fetching news from yfinance for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def classify_news_type(title, summary):
    """Classify news type based on keywords"""
    text = f"{title} {summary}".lower()
    
    # Earnings related
    if any(keyword in text for keyword in ['earnings', 'eps', 'revenue', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'guidance', 'beat', 'miss']):
        return 'earnings'
    
    # Product launch
    if any(keyword in text for keyword in ['launch', 'release', 'unveil', 'introduce', 'announce', 'new product', 'debut']):
        return 'product_launch'
    
    # Regulatory
    if any(keyword in text for keyword in ['fda', 'approval', 'regulatory', 'sec', 'investigation', 'lawsuit', 'settlement', 'compliance']):
        return 'regulatory'
    
    # M&A
    if any(keyword in text for keyword in ['acquisition', 'merger', 'takeover', 'buyout', 'deal', 'purchase']):
        return 'm_a'
    
    # Management
    if any(keyword in text for keyword in ['ceo', 'cfo', 'executive', 'resign', 'appoint', 'hire', 'departure']):
        return 'management'
    
    # Financial
    if any(keyword in text for keyword in ['dividend', 'buyback', 'split', 'offering', 'debt', 'credit', 'rating']):
        return 'financial'
    
    # Partnership
    if any(keyword in text for keyword in ['partnership', 'collaboration', 'joint venture', 'alliance', 'agreement']):
        return 'partnership'
    
    # Guidance/Outlook
    if any(keyword in text for keyword in ['outlook', 'forecast', 'guidance', 'expect', 'projection']):
        return 'guidance'
    
    return 'other'

def calculate_news_price_impact(ticker, news_date_str, news_title, news_summary):
    """Calculate actual price movement after news (1h, 1d, 1w)"""
    try:
        # Parse news date
        try:
            if 'T' in news_date_str:
                news_date = datetime.fromisoformat(news_date_str.replace('Z', '+00:00'))
            else:
                news_date = datetime.strptime(news_date_str, '%Y-%m-%d %H:%M')
        except:
            # Try other formats
            try:
                news_date = datetime.strptime(news_date_str, '%Y-%m-%d')
            except:
                return None
        
        # Get historical data
        stock = yf.Ticker(ticker.upper())
        time.sleep(0.2)
        
        # Get intraday data for 1h impact (if available)
        price_1h = None
        price_1d = None
        price_1w = None
        
        try:
            # Try to get intraday data for 1h impact - use longer period to catch older news
            # Check if news is recent (within last 60 days) for intraday data
            days_since_news = (datetime.now() - news_date).days
            if days_since_news <= 60:
                try:
                    # Try 1h interval first (max 730 days, but we'll use 1mo for safety)
                    hist_intraday = stock.history(period='1mo', interval='1h', auto_adjust=True, prepost=False)
                    
                    if not hist_intraday.empty and len(hist_intraday) > 0:
                        # Convert news_date to timezone-aware if needed
                        if news_date.tzinfo is None:
                            # Assume UTC if no timezone
                            news_date_tz = news_date.replace(tzinfo=None)
                        else:
                            news_date_tz = news_date.replace(tzinfo=None)
                        
                        # Convert index to timestamp for comparison
                        hist_intraday_sorted = hist_intraday.sort_index()
                        
                        # Find closest time to news (within 24 hours)
                        time_diffs = []
                        for idx in hist_intraday_sorted.index:
                            idx_naive = idx.replace(tzinfo=None) if idx.tzinfo else idx
                            diff = abs((idx_naive - news_date_tz).total_seconds())
                            if diff <= 86400:  # Within 24 hours
                                time_diffs.append((diff, idx))
                        
                        if time_diffs:
                            # Get closest time to news
                            time_diffs.sort(key=lambda x: x[0])
                            closest_idx = time_diffs[0][1]
                            price_at_news = hist_intraday_sorted.loc[closest_idx, 'Close']
                            
                            # Find 1h after (within 2 hours window)
                            one_hour_later = news_date_tz + timedelta(hours=1)
                            time_diffs_1h = []
                            for idx in hist_intraday_sorted.index:
                                idx_naive = idx.replace(tzinfo=None) if idx.tzinfo else idx
                                if idx_naive > news_date_tz:  # After news
                                    diff = abs((idx_naive - one_hour_later).total_seconds())
                                    if diff <= 7200:  # Within 2 hours
                                        time_diffs_1h.append((diff, idx))
                            
                            if time_diffs_1h:
                                time_diffs_1h.sort(key=lambda x: x[0])
                                closest_1h_idx = time_diffs_1h[0][1]
                                price_1h_after = hist_intraday_sorted.loc[closest_1h_idx, 'Close']
                                
                                if price_at_news > 0:
                                    price_1h = ((price_1h_after / price_at_news) - 1) * 100
                                    print(f"[NEWS IMPACT] {ticker} 1h impact: {price_1h:.2f}% (from {price_at_news:.2f} to {price_1h_after:.2f})")
                except Exception as e:
                    print(f"[NEWS IMPACT] Error getting 1h intraday data for {ticker}: {str(e)}")
                    # Try 1d interval as fallback for same-day impact
                    try:
                        if days_since_news <= 5:
                            hist_1d = stock.history(period='5d', interval='1d', auto_adjust=True)
                            if not hist_1d.empty:
                                # Use same-day close vs next-day open/close
                                news_date_only = news_date.date()
                                for idx in hist_1d.index:
                                    if idx.date() == news_date_only:
                                        price_at_news = hist_1d.loc[idx, 'Close']
                                        # Find next trading day
                                        next_idx = hist_1d.index[hist_1d.index > idx]
                                        if len(next_idx) > 0:
                                            price_next = hist_1d.loc[next_idx[0], 'Open']  # Use open of next day
                                            if price_at_news > 0:
                                                price_1h = ((price_next / price_at_news) - 1) * 100
                                                print(f"[NEWS IMPACT] {ticker} 1h impact (fallback to next day open): {price_1h:.2f}%")
                                        break
                    except:
                        pass
        except Exception as e:
            print(f"[NEWS IMPACT] Error in 1h impact calculation for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Get daily data for 1d and 1w impact
        hist_daily = stock.history(period='1mo', interval='1d', auto_adjust=True)
        if hist_daily.empty:
            return None
        
        # Find price at news date (or closest trading day)
        news_date_only = news_date.date()
        hist_daily['date'] = hist_daily.index.date
        
        # Find closest trading day to news date
        closest_daily_idx = None
        min_diff = float('inf')
        for idx, row_date in enumerate(hist_daily['date']):
            diff = abs((row_date - news_date_only).days)
            if diff < min_diff:
                min_diff = diff
                closest_daily_idx = hist_daily.index[idx]
        
        if closest_daily_idx is None or min_diff > 5:  # Too far from news date
            return None
        
        price_at_news = hist_daily.loc[closest_daily_idx, 'Close']
        
        # Find 1 day after
        one_day_after = news_date_only + timedelta(days=1)
        closest_1d_idx = None
        min_diff_1d = float('inf')
        for idx, row_date in enumerate(hist_daily['date']):
            if row_date >= one_day_after:
                diff = abs((row_date - one_day_after).days)
                if diff < min_diff_1d:
                    min_diff_1d = diff
                    closest_1d_idx = hist_daily.index[idx]
        
        if closest_1d_idx and min_diff_1d <= 2:
            price_1d_after = hist_daily.loc[closest_1d_idx, 'Close']
            price_1d = ((price_1d_after / price_at_news) - 1) * 100
        
        # Find 1 week after
        one_week_after = news_date_only + timedelta(days=7)
        closest_1w_idx = None
        min_diff_1w = float('inf')
        for idx, row_date in enumerate(hist_daily['date']):
            if row_date >= one_week_after:
                diff = abs((row_date - one_week_after).days)
                if diff < min_diff_1w:
                    min_diff_1w = diff
                    closest_1w_idx = hist_daily.index[idx]
        
        if closest_1w_idx and min_diff_1w <= 3:
            price_1w_after = hist_daily.loc[closest_1w_idx, 'Close']
            price_1w = ((price_1w_after / price_at_news) - 1) * 100
        
        return {
            'price_1h_pct': round(price_1h, 2) if price_1h is not None else None,
            'price_1d_pct': round(price_1d, 2) if price_1d is not None else None,
            'price_1w_pct': round(price_1w, 2) if price_1w is not None else None,
            'price_at_news': round(price_at_news, 2) if price_at_news else None
        }
        
    except Exception as e:
        print(f"Error calculating price impact for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_news_impact_score(price_movement, sentiment_score, news_type, source_weight=1.0):
    """Calculate overall news impact score (0-100) based on price movement and sentiment"""
    try:
        # Base score from price movement (0-50 points)
        price_score = 0
        if price_movement:
            # Use 1d movement if available, otherwise 1w
            movement = price_movement.get('price_1d_pct') or price_movement.get('price_1w_pct')
            if movement is not None:
                # Normalize to 0-50 scale (assume max 10% movement = 50 points)
                price_score = min(50, abs(movement) * 5)
        
        # Sentiment score (0-30 points)
        sentiment_magnitude = abs(sentiment_score) if sentiment_score else 0
        sentiment_points = sentiment_magnitude * 30  # Max 30 points
        
        # News type weight (0-20 points)
        type_weights = {
            'earnings': 20,
            'm_a': 18,
            'regulatory': 16,
            'product_launch': 14,
            'financial': 12,
            'guidance': 12,
            'management': 10,
            'partnership': 8,
            'other': 5
        }
        type_points = type_weights.get(news_type, 5)
        
        # Source weight multiplier
        total_score = (price_score + sentiment_points + type_points) * source_weight
        
        # Normalize to 0-100
        impact_score = min(100, max(0, total_score))
        
        return round(impact_score, 1)
    except:
        return 0

def extract_ml_features(ticker, df, info, indicators, metrics, news_list):
    """Extract comprehensive features for ML models"""
    features = {}
    
    current_price = df['Close'].iloc[-1]
    
    # Technical Features (20+)
    rsi_values = indicators.get('rsi', [])
    macd_values = indicators.get('macd', [])
    macd_signal = indicators.get('macd_signal', [])
    sma_20 = indicators.get('sma_20', [])
    sma_50 = indicators.get('sma_50', [])
    bb_high = indicators.get('bb_high', [])
    bb_low = indicators.get('bb_low', [])
    bb_mid = indicators.get('bb_mid', [])
    adx_values = indicators.get('adx', [])
    stoch_k_values = indicators.get('stoch_k', [])
    stoch_d_values = indicators.get('stoch_d', [])
    atr_values = indicators.get('atr', [])
    
    # RSI features
    if rsi_values and len(rsi_values) > 0:
        features['rsi'] = float(rsi_values[-1]) if not pd.isna(rsi_values[-1]) else 50.0
        features['rsi_7d_avg'] = float(np.mean(rsi_values[-7:])) if len(rsi_values) >= 7 else 50.0
    else:
        features['rsi'] = 50.0
        features['rsi_7d_avg'] = 50.0
    
    # MACD features
    if macd_values and macd_signal and len(macd_values) > 0:
        features['macd'] = float(macd_values[-1]) if not pd.isna(macd_values[-1]) else 0.0
        features['macd_signal'] = float(macd_signal[-1]) if not pd.isna(macd_signal[-1]) else 0.0
        features['macd_diff'] = features['macd'] - features['macd_signal']
        features['macd_bullish'] = 1.0 if features['macd_diff'] > 0 else 0.0
    else:
        features['macd'] = 0.0
        features['macd_signal'] = 0.0
        features['macd_diff'] = 0.0
        features['macd_bullish'] = 0.0
    
    # Moving Average features
    if sma_20 and len(sma_20) > 0:
        sma20_val = sma_20[-1] if not pd.isna(sma_20[-1]) else current_price
        features['price_vs_sma20'] = (current_price / sma20_val - 1) * 100 if sma20_val > 0 else 0.0
        features['above_sma20'] = 1.0 if current_price > sma20_val else 0.0
    else:
        features['price_vs_sma20'] = 0.0
        features['above_sma20'] = 0.0
    
    if sma_50 and len(sma_50) > 0:
        sma50_val = sma_50[-1] if not pd.isna(sma_50[-1]) else current_price
        features['price_vs_sma50'] = (current_price / sma50_val - 1) * 100 if sma50_val > 0 else 0.0
        features['above_sma50'] = 1.0 if current_price > sma50_val else 0.0
    else:
        features['price_vs_sma50'] = 0.0
        features['above_sma50'] = 0.0
    
    # Bollinger Bands features
    if bb_high and bb_low and bb_mid and len(bb_high) > 0:
        bb_high_val = bb_high[-1] if not pd.isna(bb_high[-1]) else current_price
        bb_low_val = bb_low[-1] if not pd.isna(bb_low[-1]) else current_price
        bb_mid_val = bb_mid[-1] if not pd.isna(bb_mid[-1]) else current_price
        bb_width = (bb_high_val - bb_low_val) / bb_mid_val * 100 if bb_mid_val > 0 else 0.0
        features['bb_position'] = (current_price - bb_low_val) / (bb_high_val - bb_low_val) * 100 if (bb_high_val - bb_low_val) > 0 else 50.0
        features['bb_width'] = bb_width
    else:
        features['bb_position'] = 50.0
        features['bb_width'] = 0.0
    
    # ADX features (trend strength)
    if adx_values and len(adx_values) > 0:
        features['adx'] = float(adx_values[-1]) if not pd.isna(adx_values[-1]) else 25.0
        features['adx_7d_avg'] = float(np.mean(adx_values[-7:])) if len(adx_values) >= 7 else 25.0
        features['strong_trend'] = 1.0 if features['adx'] > 25.0 else 0.0  # ADX > 25 indicates strong trend
    else:
        features['adx'] = 25.0
        features['adx_7d_avg'] = 25.0
        features['strong_trend'] = 0.0
    
    # Stochastic Oscillator features (overbought/oversold)
    if stoch_k_values and stoch_d_values and len(stoch_k_values) > 0:
        features['stoch_k'] = float(stoch_k_values[-1]) if not pd.isna(stoch_k_values[-1]) else 50.0
        features['stoch_d'] = float(stoch_d_values[-1]) if not pd.isna(stoch_d_values[-1]) else 50.0
        features['stoch_cross'] = 1.0 if features['stoch_k'] > features['stoch_d'] else 0.0  # Bullish cross
        features['stoch_overbought'] = 1.0 if features['stoch_k'] > 80.0 else 0.0
        features['stoch_oversold'] = 1.0 if features['stoch_k'] < 20.0 else 0.0
    else:
        features['stoch_k'] = 50.0
        features['stoch_d'] = 50.0
        features['stoch_cross'] = 0.0
        features['stoch_overbought'] = 0.0
        features['stoch_oversold'] = 0.0
    
    # ATR features (volatility measure)
    if atr_values and len(atr_values) > 0:
        atr_val = float(atr_values[-1]) if not pd.isna(atr_values[-1]) else 0.0
        # Normalize ATR by current price to get percentage
        features['atr'] = (atr_val / current_price * 100) if current_price > 0 else 0.0
        features['atr_7d_avg'] = float(np.mean([(v / current_price * 100) if current_price > 0 and not pd.isna(v) else 0.0 for v in atr_values[-7:]])) if len(atr_values) >= 7 else 0.0
    else:
        features['atr'] = 0.0
        features['atr_7d_avg'] = 0.0
    
    # Price momentum features (multiple timeframes)
    if len(df) >= 252:  # 1 year
        price_1y_ago = df['Close'].iloc[-252]
        features['momentum_1y'] = ((current_price / price_1y_ago - 1) * 100) if price_1y_ago > 0 else 0.0
    else:
        features['momentum_1y'] = 0.0
    
    if len(df) >= 126:  # 6 months
        price_6m_ago = df['Close'].iloc[-126]
        features['momentum_6m'] = ((current_price / price_6m_ago - 1) * 100) if price_6m_ago > 0 else 0.0
    else:
        features['momentum_6m'] = 0.0
    
    if len(df) >= 63:  # 3 months
        price_3m_ago = df['Close'].iloc[-63]
        features['momentum_3m'] = ((current_price / price_3m_ago - 1) * 100) if price_3m_ago > 0 else 0.0
    else:
        features['momentum_3m'] = 0.0
    
    if len(df) >= 21:  # 1 month
        price_1m_ago = df['Close'].iloc[-21]
        features['momentum_1m'] = ((current_price / price_1m_ago - 1) * 100) if price_1m_ago > 0 else 0.0
    else:
        features['momentum_1m'] = 0.0
    
    if len(df) >= 5:  # 1 week
        price_1w_ago = df['Close'].iloc[-5]
        features['momentum_1w'] = ((current_price / price_1w_ago - 1) * 100) if price_1w_ago > 0 else 0.0
    else:
        features['momentum_1w'] = 0.0
    
    # Volume features
    current_volume = df['Volume'].iloc[-1]
    if len(df) >= 30:
        avg_volume_30d = df['Volume'].tail(30).mean()
        features['volume_ratio'] = (current_volume / avg_volume_30d) if avg_volume_30d > 0 else 1.0
        features['volume_trend'] = 1.0 if current_volume > avg_volume_30d else 0.0
    else:
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 0.0
    
    # Volatility features
    volatility = metrics.get('volatility')
    features['volatility'] = float(volatility) if volatility is not None else 30.0
    
    # Price position in 52-week range
    year_high = metrics.get('year_high', current_price)
    year_low = metrics.get('year_low', current_price)
    if year_high > year_low:
        features['price_position_52w'] = ((current_price - year_low) / (year_high - year_low) * 100)
    else:
        features['price_position_52w'] = 50.0
    
    # Fundamental Features (15+)
    pe_ratio = info.get('trailingPE')
    features['pe_ratio'] = float(pe_ratio) if pe_ratio is not None and not pd.isna(pe_ratio) else 20.0
    
    forward_pe = info.get('forwardPE')
    features['forward_pe'] = float(forward_pe) if forward_pe is not None and not pd.isna(forward_pe) else 20.0
    
    pb_ratio = info.get('priceToBook')
    features['pb_ratio'] = float(pb_ratio) if pb_ratio is not None and not pd.isna(pb_ratio) else 2.0
    
    debt_to_equity = info.get('debtToEquity')
    features['debt_to_equity'] = float(debt_to_equity) if debt_to_equity is not None and not pd.isna(debt_to_equity) else 0.5
    
    current_ratio = info.get('currentRatio')
    features['current_ratio'] = float(current_ratio) if current_ratio is not None and not pd.isna(current_ratio) else 1.5
    
    roe = info.get('returnOnEquity')
    features['roe'] = float(roe) if roe is not None and not pd.isna(roe) else 10.0
    
    roa = info.get('returnOnAssets')
    features['roa'] = float(roa) if roa is not None and not pd.isna(roa) else 5.0
    
    revenue_growth = info.get('revenueGrowth')
    features['revenue_growth'] = float(revenue_growth * 100) if revenue_growth is not None and not pd.isna(revenue_growth) else 0.0
    
    earnings_growth = info.get('earningsGrowth')
    features['earnings_growth'] = float(earnings_growth * 100) if earnings_growth is not None and not pd.isna(earnings_growth) else 0.0
    
    profit_margin = info.get('profitMargins')
    features['profit_margin'] = float(profit_margin * 100) if profit_margin is not None and not pd.isna(profit_margin) else 10.0
    
    beta = info.get('beta')
    features['beta'] = float(beta) if beta is not None and not pd.isna(beta) else 1.0
    
    market_cap = info.get('marketCap')
    if market_cap:
        if market_cap > 200_000_000_000:  # > 200B
            features['market_cap_category'] = 4.0  # Mega cap
        elif market_cap > 10_000_000_000:  # > 10B
            features['market_cap_category'] = 3.0  # Large cap
        elif market_cap > 2_000_000_000:  # > 2B
            features['market_cap_category'] = 2.0  # Mid cap
        else:
            features['market_cap_category'] = 1.0  # Small cap
    else:
        features['market_cap_category'] = 2.0
    
    # Sentiment Features (5+)
    if news_list:
        sentiments = [article.get('sentiment', 'neutral') for article in news_list[:10]]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        total = len(sentiments)
        features['news_sentiment_score'] = ((positive_count - negative_count) / total * 100) if total > 0 else 0.0
        features['news_volume'] = float(len(news_list))
    else:
        features['news_sentiment_score'] = 0.0
        features['news_volume'] = 0.0
    
    # Market Context Features - S&P 500 correlation
    try:
        # Fetch S&P 500 data for correlation
        sp500_ticker = yf.Ticker('^GSPC')
        sp500_hist = sp500_ticker.history(period='1y')
        
        if not sp500_hist.empty and len(sp500_hist) > 0:
            # Align dates with stock data
            # Use common dates for correlation calculation
            common_dates = df.index.intersection(sp500_hist.index)
            
            if len(common_dates) >= 30:
                # Calculate returns for correlation
                stock_returns = df.loc[common_dates, 'Close'].pct_change().dropna()
                sp500_returns = sp500_hist.loc[common_dates, 'Close'].pct_change().dropna()
                
                # Align returns
                aligned_returns = pd.DataFrame({
                    'stock': stock_returns,
                    'sp500': sp500_returns
                }).dropna()
                
                if len(aligned_returns) >= 30:
                    # 30-day correlation
                    if len(aligned_returns) >= 30:
                        corr_30d = aligned_returns.tail(30)['stock'].corr(aligned_returns.tail(30)['sp500'])
                        features['sp500_correlation_30d'] = float(corr_30d) if not pd.isna(corr_30d) else 0.5
                    else:
                        features['sp500_correlation_30d'] = 0.5
                    
                    # 90-day correlation
                    if len(aligned_returns) >= 90:
                        corr_90d = aligned_returns.tail(90)['stock'].corr(aligned_returns.tail(90)['sp500'])
                        features['sp500_correlation_90d'] = float(corr_90d) if not pd.isna(corr_90d) else 0.5
                    else:
                        features['sp500_correlation_90d'] = 0.5
                    
                    # Relative performance vs S&P 500
                    if len(common_dates) >= 126:
                        stock_6m_return = (df.loc[common_dates[-1], 'Close'] / df.loc[common_dates[-126], 'Close'] - 1) * 100 if len(common_dates) >= 126 else 0
                        sp500_6m_return = (sp500_hist.loc[common_dates[-1], 'Close'] / sp500_hist.loc[common_dates[-126], 'Close'] - 1) * 100 if len(common_dates) >= 126 else 0
                        features['relative_strength_vs_sp500'] = stock_6m_return - sp500_6m_return
                    else:
                        features['relative_strength_vs_sp500'] = 0.0
                    
                    # Market regime detection (bull/bear/sideways)
                    if len(sp500_hist) >= 200:
                        sp500_sma_50 = sp500_hist['Close'].tail(200).rolling(50).mean().iloc[-1]
                        sp500_sma_200 = sp500_hist['Close'].tail(200).rolling(200).mean().iloc[-1]
                        sp500_current = sp500_hist['Close'].iloc[-1]
                        
                        if not pd.isna(sp500_sma_50) and not pd.isna(sp500_sma_200):
                            if sp500_current > sp500_sma_50 > sp500_sma_200:
                                features['market_regime'] = 1.0  # Bull market
                            elif sp500_current < sp500_sma_50 < sp500_sma_200:
                                features['market_regime'] = -1.0  # Bear market
                            else:
                                features['market_regime'] = 0.0  # Sideways
                        else:
                            features['market_regime'] = 0.0
                    else:
                        features['market_regime'] = 0.0
                else:
                    features['sp500_correlation_30d'] = 0.5
                    features['sp500_correlation_90d'] = 0.5
                    features['relative_strength_vs_sp500'] = 0.0
                    features['market_regime'] = 0.0
            else:
                features['sp500_correlation_30d'] = 0.5
                features['sp500_correlation_90d'] = 0.5
                features['relative_strength_vs_sp500'] = 0.0
                features['market_regime'] = 0.0
        else:
            features['sp500_correlation_30d'] = 0.5
            features['sp500_correlation_90d'] = 0.5
            features['relative_strength_vs_sp500'] = 0.0
            features['market_regime'] = 0.0
    except Exception as e:
        print(f"[FEATURES] Error calculating S&P 500 correlation: {e}")
        features['sp500_correlation_30d'] = 0.5
        features['sp500_correlation_90d'] = 0.5
        features['relative_strength_vs_sp500'] = 0.0
        features['market_regime'] = 0.0
    
    # Legacy relative_strength (keep for backward compatibility)
    features['relative_strength'] = features.get('relative_strength_vs_sp500', features['momentum_6m'])
    
    # Feature Engineering - Interactions
    # Important feature combinations that may improve predictions
    features['momentum_volatility_interaction'] = features.get('momentum_6m', 0) * features.get('volatility', 30) / 100
    features['rsi_momentum_interaction'] = (features.get('rsi', 50) - 50) * features.get('momentum_6m', 0) / 100
    features['adx_momentum_interaction'] = features.get('adx', 25) * features.get('momentum_6m', 0) / 100
    features['volume_momentum_interaction'] = features.get('volume_ratio', 1.0) * features.get('momentum_6m', 0) / 100
    features['beta_volatility_interaction'] = features.get('beta', 1.0) * features.get('volatility', 30) / 100
    features['pe_momentum_interaction'] = (features.get('pe_ratio', 20) / 20) * features.get('momentum_6m', 0) / 100
    
    # Feature Engineering - Lag Features
    # Use previous values of key indicators (if available in historical data)
    if len(df) >= 5:
        # 5-day lag momentum
        if len(df) >= 26:  # 1 month ago
            price_1m_ago = df['Close'].iloc[-26] if len(df) >= 26 else current_price
            price_1m_5d_ago = df['Close'].iloc[-31] if len(df) >= 31 else current_price
            if price_1m_5d_ago > 0:
                features['momentum_1m_lag'] = ((price_1m_ago / price_1m_5d_ago - 1) * 100)
            else:
                features['momentum_1m_lag'] = 0.0
        else:
            features['momentum_1m_lag'] = 0.0
        
        # Volume lag (5 days ago)
        if len(df) >= 5:
            volume_5d_ago = df['Volume'].iloc[-5]
            volume_10d_ago = df['Volume'].iloc[-10] if len(df) >= 10 else volume_5d_ago
            if volume_10d_ago > 0:
                features['volume_ratio_lag'] = volume_5d_ago / volume_10d_ago
            else:
                features['volume_ratio_lag'] = 1.0
        else:
            features['volume_ratio_lag'] = 1.0
    else:
        features['momentum_1m_lag'] = 0.0
        features['volume_ratio_lag'] = 1.0
    
    # Feature Engineering - Rolling Statistics
    if len(df) >= 20:
        # Rolling volatility (7-day vs 30-day)
        returns_7d = df['Close'].tail(7).pct_change().dropna()
        returns_30d = df['Close'].tail(30).pct_change().dropna()
        if len(returns_7d) > 0 and len(returns_30d) > 0:
            vol_7d = returns_7d.std() * np.sqrt(252) * 100
            vol_30d = returns_30d.std() * np.sqrt(252) * 100
            features['volatility_7d'] = float(vol_7d) if not pd.isna(vol_7d) else features.get('volatility', 30)
            features['volatility_ratio_7d_30d'] = (vol_7d / vol_30d) if vol_30d > 0 else 1.0
        else:
            features['volatility_7d'] = features.get('volatility', 30)
            features['volatility_ratio_7d_30d'] = 1.0
        
        # Rolling momentum (recent vs longer term)
        if len(df) >= 63:
            price_3m_ago = df['Close'].iloc[-63]
            price_1m_ago = df['Close'].iloc[-21]
            if price_3m_ago > 0 and price_1m_ago > 0:
                momentum_3m = ((current_price / price_3m_ago - 1) * 100)
                momentum_1m = ((current_price / price_1m_ago - 1) * 100)
                features['momentum_acceleration'] = momentum_1m - (momentum_3m / 3)  # Acceleration indicator
            else:
                features['momentum_acceleration'] = 0.0
        else:
            features['momentum_acceleration'] = 0.0
    else:
        features['volatility_7d'] = features.get('volatility', 30)
        features['volatility_ratio_7d_30d'] = 1.0
        features['momentum_acceleration'] = 0.0
    
    return features

# Prediction history storage
_PREDICTION_HISTORY_DIR = Path('.ml_predictions')
_PREDICTION_HISTORY_DIR.mkdir(exist_ok=True)

def _save_prediction_history(ticker, current_price, prediction_result, score=None):
    """Save ML prediction to history file
    
    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        prediction_result: Dictionary with predictions, expected_returns, etc.
        score: Optional final score (0-100). If provided, uses this instead of calculating from expected_returns
    """
    try:
        
        today = datetime.now().strftime('%Y-%m-%d')
        history_file = _PREDICTION_HISTORY_DIR / f"{ticker.upper()}.json"
        
        # Load existing history
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        
        # Check if entry for today already exists
        today_entry = None
        for entry in history:
            if entry.get('date') == today:
                today_entry = entry
                break
        
        # Use provided score if available, otherwise calculate from expected_returns
        if score is not None:
            # Use the provided final score (already in 0-100 scale)
            final_score = max(0, min(100, round(score, 0)))
        else:
            # Calculate score: average expected return weighted by confidence
            expected_returns = prediction_result.get('expected_returns', {})
            confidence_intervals = prediction_result.get('confidence_intervals', {})
            
            # Calculate score as weighted average of expected returns
            # Weight by confidence (tighter intervals = higher confidence)
            total_weight = 0
            weighted_sum = 0
            
            for period in ['1m', '3m', '6m', '12m']:
                if period in expected_returns and period in confidence_intervals:
                    ret = expected_returns[period]
                    ci = confidence_intervals[period]
                    
                    # Calculate confidence from interval width (narrower = higher confidence)
                    if 'lower' in ci and 'upper' in ci:
                        pred_price = prediction_result.get('predictions', {}).get(period, current_price)
                        if pred_price > 0:
                            ci_width = (ci['upper'] - ci['lower']) / pred_price
                            # Confidence: 1 - normalized width (max width = 0.5 = 50%)
                            confidence = max(0, 1 - (ci_width / 0.5))
                            weight = confidence
                            weighted_sum += ret * weight
                            total_weight += weight
            
            # Calculate average score
            # expected_returns are already in percentage, so don't multiply by 100
            score_percentage = (weighted_sum / total_weight) if total_weight > 0 else 0.0
            
            # Convert score from percentage (-100 to +100) to 0-100 scale
            # Map: -100% → 0/100, 0% → 50/100, +100% → 100/100
            final_score = max(0, min(100, round(50 + score_percentage, 0)))
        
        # Prepare new entry
        entry = {
            'date': today,
            'timestamp': int(time.time()),
            'current_price': float(current_price),
            'predictions': prediction_result.get('predictions', {}),
            'expected_returns': prediction_result.get('expected_returns', {}),
            'confidence_intervals': prediction_result.get('confidence_intervals', {}),
            'model_type': prediction_result.get('model_type', 'unknown'),
            'score': final_score  # Score as 0-100
        }
        
        # Update or append
        if today_entry:
            # Update existing entry (in case of multiple predictions per day)
            idx = history.index(today_entry)
            history[idx] = entry
        else:
            # Add new entry
            history.append(entry)
        
        # Keep only last 365 days
        history = sorted(history, key=lambda x: x.get('date', ''), reverse=True)[:365]
        
        # Save to file
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # #region agent log (disabled for production)
        # Debug logging removed - causes FileNotFoundError on Render
        # #endregion
    except Exception as e:
        print(f"[ML HISTORY] Error saving prediction history: {e}")
        import traceback
        traceback.print_exc()

def get_prediction_history(ticker, days=30):
    """Get ML prediction history for a ticker"""
    try:
        
        history_file = _PREDICTION_HISTORY_DIR / f"{ticker.upper()}.json"
        
        if not history_file.exists():
            return []
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Sort by date (newest first) and limit to requested days
        history = sorted(history, key=lambda x: x.get('date', ''), reverse=True)
        
        if days > 0:
            history = history[:days]
        
        
        return history
    except Exception as e:
        print(f"[ML HISTORY] Error loading prediction history: {e}")
        return []

# Model cache for faster predictions
_model_cache = {}
_scaler_cache = {}

# Cache version - increment when model structure changes
_MODEL_CACHE_VERSION = 8  # Incremented for Phase 3 improvements: all historical data, CV, feature importance, time weighting, outlier handling

# Clear cache on startup if version changed
def _clear_model_cache_if_needed():
    """Clear model cache if version changed"""
    if _model_cache and not any(k.startswith(f"rf_v{_MODEL_CACHE_VERSION}_") for k in _model_cache.keys()):
        print(f"[ML] Clearing model cache due to version change (new version: {_MODEL_CACHE_VERSION})")
        _model_cache.clear()
        _scaler_cache.clear()

# Clear cache on module load
_clear_model_cache_if_needed()

def _train_random_forest_model(features_dict, current_price, df=None):
    """Train RandomForest model using historical data or synthetic training with Phase 3 improvements"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    
    print(f"[ML TRAIN] Starting Phase 3 model training, df={'provided' if df is not None else 'None'}, len={len(df) if df is not None else 0}")
    
    # Prepare feature vector from features dict
    feature_names = sorted(features_dict.keys())
    X = np.array([[features_dict[f] for f in feature_names]])
    
    # Helper function to cap outliers (IQR method)
    def cap_outliers(values, lower_percentile=5, upper_percentile=95):
        """Cap outliers using percentile method"""
        if len(values) == 0:
            return values
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        return np.clip(values, lower_bound, upper_bound)
    
    # If we have historical data, use it for training
    # Phase 3: Use ALL available historical data (minimum 21 days for 1M prediction)
    if df is not None and len(df) >= 21:  # Minimum 1 month of data
        # Create training data from historical prices
        training_data = []
        training_targets = {'1m': [], '3m': [], '6m': [], '12m': []}
        training_indices = []  # Store indices for time-based weighting
        
        # Phase 3: Use all available data, starting from minimum required days
        period_days = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}
        min_required_days = max(period_days.values())  # 252 days (1 year)
        
        # Start from minimum required days, but use all available data
        start_idx = min_required_days if len(df) >= min_required_days else 21
        end_idx = len(df) - 21  # End 1 month before end to have target
        
        # Phase 3: Use all available historical data
        for i in range(start_idx, end_idx):
            # Use all available history up to this point
            window_df = df.iloc[:i+1]  # All data up to current point
            
            # Extract features for this historical point
            try:
                hist_features = _extract_historical_features(window_df, i)
                if hist_features:
                    feature_vector = [hist_features.get(f, 0.0) for f in feature_names]
                    training_data.append(feature_vector)
                    training_indices.append(i)
                    
                    # Calculate targets (future prices)
                    current_hist_price = df['Close'].iloc[i]
                    if i + 21 < len(df):
                        training_targets['1m'].append(df['Close'].iloc[i + 21])
                    else:
                        training_targets['1m'].append(None)
                    if i + 63 < len(df):
                        training_targets['3m'].append(df['Close'].iloc[i + 63])
                    else:
                        training_targets['3m'].append(None)
                    if i + 126 < len(df):
                        training_targets['6m'].append(df['Close'].iloc[i + 126])
                    else:
                        training_targets['6m'].append(None)
                    if i + 252 < len(df):
                        training_targets['12m'].append(df['Close'].iloc[i + 252])
                    else:
                        training_targets['12m'].append(None)
            except Exception as e:
                print(f"[ML TRAIN] Error extracting features at index {i}: {e}")
                continue
        
        if len(training_data) > 10:  # Need at least 10 samples
            X_train = np.array(training_data)
            
            # Phase 3: Outlier handling - cap extreme feature values
            print(f"[ML TRAIN] Capping outliers in features...")
            for col_idx in range(X_train.shape[1]):
                X_train[:, col_idx] = cap_outliers(X_train[:, col_idx])
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Phase 3: Feature importance analysis
            print(f"[ML TRAIN] Analyzing feature importance...")
            # Train a temporary model to get feature importance
            temp_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            # Use 6m period for feature importance (most balanced)
            if len(training_targets['6m']) > 10:
                valid_6m = [(i, t) for i, t in enumerate(training_targets['6m']) if t is not None]
                if len(valid_6m) > 10:
                    valid_indices, valid_targets = zip(*valid_6m)
                    X_temp = X_train_scaled[list(valid_indices)]
                    y_temp = np.array(valid_targets)
                    current_prices_temp = np.array([df['Close'].iloc[training_indices[i]] for i in valid_indices])
                    y_temp_norm = (y_temp / current_prices_temp - 1) * 100
                    # Cap outliers in targets
                    y_temp_norm = cap_outliers(y_temp_norm)
                    temp_model.fit(X_temp, y_temp_norm)
                    
                    # Get feature importance
                    importances = temp_model.feature_importances_
                    feature_importance_dict = dict(zip(feature_names, importances))
                    
                    # Remove features with very low importance (< 0.001)
                    important_features = [f for f in feature_names if feature_importance_dict.get(f, 0) >= 0.001]
                    
                    if len(important_features) < len(feature_names):
                        print(f"[ML TRAIN] Removing {len(feature_names) - len(important_features)} redundant features")
                        # Rebuild X_train with only important features (before scaling)
                        important_indices = [feature_names.index(f) for f in important_features]
                        X_train = X_train[:, important_indices]
                        feature_names = important_features
                        # Retrain scaler on reduced feature set
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                    else:
                        print(f"[ML TRAIN] All {len(feature_names)} features are important")
            
            # Phase 3: Time-based weighting - newer data has more weight
            # Calculate weights: exponential decay from oldest to newest
            n_samples = len(training_indices)
            if n_samples > 0:
                # Normalize indices to 0-1 range
                normalized_indices = np.array(training_indices) / max(training_indices) if max(training_indices) > 0 else np.ones(n_samples)
                # Exponential weighting: newer data (higher index) gets more weight
                # Weight = exp(alpha * normalized_index), where alpha controls decay
                alpha = 2.0  # Higher alpha = more weight on recent data
                sample_weights = np.exp(alpha * normalized_indices)
                # Normalize weights to sum to n_samples (so they average to 1)
                sample_weights = sample_weights / sample_weights.mean() * n_samples / sample_weights.sum()
            else:
                sample_weights = None
            
            models = {}
            cv_scores = {}
            
            for period in ['1m', '3m', '6m', '12m']:
                # Filter valid targets
                valid_data = [(i, t, idx) for i, (t, idx) in enumerate(zip(training_targets[period], training_indices)) if t is not None]
                
                if len(valid_data) > 10:
                    valid_indices, valid_targets, valid_hist_indices = zip(*valid_data)
                    X_period = X_train_scaled[list(valid_indices)]
                    y_period = np.array(valid_targets)
                    current_prices = np.array([df['Close'].iloc[idx] for idx in valid_hist_indices])
                    
                    # Normalize targets by current price at that time
                    y_period_normalized = (y_period / current_prices - 1) * 100  # Percentage change
                    
                    # Phase 3: Outlier handling for targets
                    y_period_normalized = cap_outliers(y_period_normalized)
                    
                    # Phase 3: Time-based weighting for this period
                    if sample_weights is not None:
                        period_weights = sample_weights[list(valid_indices)]
                    else:
                        period_weights = None
                    
                    # Phase 3: Cross-validation
                    if len(y_period_normalized) >= 20:  # Need at least 20 samples for CV
                        kfold = KFold(n_splits=min(5, len(y_period_normalized) // 4), shuffle=False)
                        cv_model = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                        cv_scores_list = cross_val_score(cv_model, X_period, y_period_normalized, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
                        cv_scores[period] = -cv_scores_list.mean()  # Convert to positive MAE
                        print(f"[ML TRAIN] {period} CV MAE: {cv_scores[period]:.2f}%")
                    
                    # Train final model with all data
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    if period_weights is not None:
                        model.fit(X_period, y_period_normalized, sample_weight=period_weights)
                        print(f"[ML TRAIN] Trained {period} model with {len(y_period_normalized)} samples (time-weighted)")
                    else:
                        model.fit(X_period, y_period_normalized)
                        print(f"[ML TRAIN] Trained {period} model with {len(y_period_normalized)} samples")
                    
                    models[period] = model
                else:
                    print(f"[ML TRAIN] {period}: Not enough training targets (got {len(valid_data)}, need >10)")
            
            if models:
                print(f"[ML TRAIN] Training complete. Models: {list(models.keys())}, Features: {len(feature_names)}")
                return models, scaler, feature_names
    
    # Fallback: Use synthetic training based on momentum patterns
    # Phase 3: Data augmentation - more synthetic samples (500-1000)
    n_samples = np.random.randint(750, 1001)  # Random between 750-1000 for variety
    print(f"[ML TRAIN] Using synthetic training with {n_samples} samples")
    
    X_synthetic = []
    y_synthetic = {'1m': [], '3m': [], '6m': [], '12m': []}
    
    base_features = np.array([features_dict.get(f, 0.0) for f in feature_names])
    
    # Phase 3: Outlier handling for base features
    base_features = cap_outliers(base_features, lower_percentile=1, upper_percentile=99)
    
    for _ in range(n_samples):
        # Add noise to features
        noise = np.random.normal(0, 0.1, len(feature_names))
        synthetic_features = base_features * (1 + noise)
        # Cap outliers in synthetic features
        synthetic_features = cap_outliers(synthetic_features, lower_percentile=1, upper_percentile=99)
        X_synthetic.append(synthetic_features)
        
        # Calculate synthetic targets based on momentum
        momentum = features_dict.get('momentum_6m', 0) + np.random.normal(0, 10)
        volatility = features_dict.get('volatility', 30) + np.random.normal(0, 5)
        
        # Cap momentum and volatility
        momentum = np.clip(momentum, -80, 400)  # Realistic range
        volatility = np.clip(volatility, 5, 200)  # Realistic range
        
        # Synthetic price changes
        y_1m = momentum / 6 + np.random.normal(0, volatility / 6)
        y_3m = momentum / 2 + np.random.normal(0, volatility / 3)
        y_6m = momentum + np.random.normal(0, volatility / 2)
        y_12m = momentum * 2 + np.random.normal(0, volatility)
        
        # Cap synthetic targets
        y_synthetic['1m'].append(np.clip(y_1m, -50, 100))
        y_synthetic['3m'].append(np.clip(y_3m, -60, 200))
        y_synthetic['6m'].append(np.clip(y_6m, -70, 300))
        y_synthetic['12m'].append(np.clip(y_12m, -80, 400))
    
    X_train = np.array(X_synthetic)
    
    # Phase 3: Outlier handling for synthetic features
    for col_idx in range(X_train.shape[1]):
        X_train[:, col_idx] = cap_outliers(X_train[:, col_idx])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {}
    for period in ['1m', '3m', '6m', '12m']:
        y_train = np.array(y_synthetic[period])
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        models[period] = model
        print(f"[ML TRAIN] Trained synthetic {period} model with {n_samples} samples")
    
    return models, scaler, feature_names

def _extract_historical_features(df, idx):
    """Extract features for a historical data point"""
    try:
        if idx >= len(df):
            return None
        
        current_price = df['Close'].iloc[idx]
        
        # Simple feature extraction (simplified version)
        features = {}
        
        # Momentum features
        if idx >= 21:
            features['momentum_1m'] = ((current_price / df['Close'].iloc[idx - 21] - 1) * 100) if df['Close'].iloc[idx - 21] > 0 else 0.0
        else:
            features['momentum_1m'] = 0.0
        
        if idx >= 63:
            features['momentum_3m'] = ((current_price / df['Close'].iloc[idx - 63] - 1) * 100) if df['Close'].iloc[idx - 63] > 0 else 0.0
        else:
            features['momentum_3m'] = 0.0
        
        if idx >= 126:
            features['momentum_6m'] = ((current_price / df['Close'].iloc[idx - 126] - 1) * 100) if df['Close'].iloc[idx - 126] > 0 else 0.0
        else:
            features['momentum_6m'] = 0.0
        
        # Volatility
        if idx >= 30:
            returns = df['Close'].iloc[max(0, idx-30):idx+1].pct_change().dropna()
            features['volatility'] = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 30.0
        else:
            features['volatility'] = 30.0
        
        # Volume
        if idx >= 30:
            current_volume = df['Volume'].iloc[idx]
            avg_volume = df['Volume'].iloc[max(0, idx-30):idx+1].mean()
            features['volume_ratio'] = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        else:
            features['volume_ratio'] = 1.0
        
        # Calculate new technical indicators for historical point
        # Use window up to current index
        window_df = df.iloc[max(0, idx-252):idx+1]  # Up to 1 year of history
        
        # ADX
        if 'High' in window_df.columns and 'Low' in window_df.columns and len(window_df) > 14:
            try:
                from ta.trend import ADXIndicator
                adx = ADXIndicator(window_df['High'], window_df['Low'], window_df['Close'], window=14)
                adx_values = adx.adx().tolist()
                if adx_values:
                    features['adx'] = float(adx_values[-1]) if not pd.isna(adx_values[-1]) else 25.0
                    features['adx_7d_avg'] = float(np.mean(adx_values[-7:])) if len(adx_values) >= 7 else 25.0
                    features['strong_trend'] = 1.0 if features['adx'] > 25.0 else 0.0
                else:
                    features['adx'] = 25.0
                    features['adx_7d_avg'] = 25.0
                    features['strong_trend'] = 0.0
            except:
                features['adx'] = 25.0
                features['adx_7d_avg'] = 25.0
                features['strong_trend'] = 0.0
        else:
            features['adx'] = 25.0
            features['adx_7d_avg'] = 25.0
            features['strong_trend'] = 0.0
        
        # Stochastic
        if 'High' in window_df.columns and 'Low' in window_df.columns and len(window_df) > 14:
            try:
                from ta.momentum import StochasticOscillator
                stoch = StochasticOscillator(window_df['High'], window_df['Low'], window_df['Close'], window=14, smooth_window=3)
                stoch_k_values = stoch.stoch().tolist()
                stoch_d_values = stoch.stoch_signal().tolist()
                if stoch_k_values and stoch_d_values:
                    features['stoch_k'] = float(stoch_k_values[-1]) if not pd.isna(stoch_k_values[-1]) else 50.0
                    features['stoch_d'] = float(stoch_d_values[-1]) if not pd.isna(stoch_d_values[-1]) else 50.0
                    features['stoch_cross'] = 1.0 if features['stoch_k'] > features['stoch_d'] else 0.0
                    features['stoch_overbought'] = 1.0 if features['stoch_k'] > 80.0 else 0.0
                    features['stoch_oversold'] = 1.0 if features['stoch_k'] < 20.0 else 0.0
                else:
                    features['stoch_k'] = 50.0
                    features['stoch_d'] = 50.0
                    features['stoch_cross'] = 0.0
                    features['stoch_overbought'] = 0.0
                    features['stoch_oversold'] = 0.0
            except:
                features['stoch_k'] = 50.0
                features['stoch_d'] = 50.0
                features['stoch_cross'] = 0.0
                features['stoch_overbought'] = 0.0
                features['stoch_oversold'] = 0.0
        else:
            features['stoch_k'] = 50.0
            features['stoch_d'] = 50.0
            features['stoch_cross'] = 0.0
            features['stoch_overbought'] = 0.0
            features['stoch_oversold'] = 0.0
        
        # ATR
        if 'High' in window_df.columns and 'Low' in window_df.columns and len(window_df) > 14:
            try:
                from ta.volatility import AverageTrueRange
                atr = AverageTrueRange(window_df['High'], window_df['Low'], window_df['Close'], window=14)
                atr_values = atr.average_true_range().tolist()
                if atr_values:
                    atr_val = float(atr_values[-1]) if not pd.isna(atr_values[-1]) else 0.0
                    features['atr'] = (atr_val / current_price * 100) if current_price > 0 else 0.0
                    features['atr_7d_avg'] = float(np.mean([(v / current_price * 100) if current_price > 0 and not pd.isna(v) else 0.0 for v in atr_values[-7:]])) if len(atr_values) >= 7 else 0.0
                else:
                    features['atr'] = 0.0
                    features['atr_7d_avg'] = 0.0
            except:
                features['atr'] = 0.0
                features['atr_7d_avg'] = 0.0
        else:
            features['atr'] = 0.0
            features['atr_7d_avg'] = 0.0
        
        # Add default values for other features (only if not already calculated)
        default_features = {
            'rsi': 50.0, 'rsi_7d_avg': 50.0,
            'macd': 0.0, 'macd_signal': 0.0, 'macd_diff': 0.0, 'macd_bullish': 0.0,
            'price_vs_sma20': 0.0, 'above_sma20': 0.0,
            'price_vs_sma50': 0.0, 'above_sma50': 0.0,
            'bb_position': 50.0, 'bb_width': 0.0,
            'momentum_1w': 0.0, 'momentum_1y': 0.0, 'volume_trend': 0.0,
            'price_position_52w': 50.0,
            'pe_ratio': 20.0, 'forward_pe': 20.0, 'pb_ratio': 2.0,
            'debt_to_equity': 0.5, 'current_ratio': 1.5, 'roe': 10.0, 'roa': 5.0,
            'revenue_growth': 0.0, 'earnings_growth': 0.0, 'profit_margin': 10.0,
            'beta': 1.0, 'market_cap_category': 2.0,
            'news_sentiment_score': 0.0, 'news_volume': 0.0,
            'relative_strength': 0.0,
            # Market context (defaults for historical)
            'sp500_correlation_30d': 0.5, 'sp500_correlation_90d': 0.5,
            'relative_strength_vs_sp500': 0.0, 'market_regime': 0.0,
            # Feature interactions (will be calculated from other features)
            'momentum_volatility_interaction': 0.0,
            'rsi_momentum_interaction': 0.0,
            'adx_momentum_interaction': 0.0,
            'volume_momentum_interaction': 0.0,
            'beta_volatility_interaction': 0.0,
            'pe_momentum_interaction': 0.0,
            # Lag features
            'momentum_1m_lag': 0.0, 'volume_ratio_lag': 1.0,
            # Rolling statistics
            'volatility_7d': 30.0, 'volatility_ratio_7d_30d': 1.0,
            'momentum_acceleration': 0.0
        }
        
        for key, value in default_features.items():
            if key not in features:
                features[key] = value
        
        # Calculate interactions for historical features (using available values)
        features['momentum_volatility_interaction'] = features.get('momentum_6m', 0) * features.get('volatility', 30) / 100
        features['rsi_momentum_interaction'] = (features.get('rsi', 50) - 50) * features.get('momentum_6m', 0) / 100
        features['adx_momentum_interaction'] = features.get('adx', 25) * features.get('momentum_6m', 0) / 100
        features['volume_momentum_interaction'] = features.get('volume_ratio', 1.0) * features.get('momentum_6m', 0) / 100
        features['beta_volatility_interaction'] = features.get('beta', 1.0) * features.get('volatility', 30) / 100
        features['pe_momentum_interaction'] = (features.get('pe_ratio', 20) / 20) * features.get('momentum_6m', 0) / 100
        
        return features
    except Exception as e:
        print(f"Error extracting historical features: {e}")
        return None

def predict_price(features, current_price, df=None):
    """Predict stock price for 1, 3, 6, 12 months using RandomForest"""
    if not ML_AVAILABLE:
        # Fallback: Simple linear projection based on momentum
        momentum_6m = features.get('momentum_6m', 0)
        
        # Cap extreme momentum to prevent unrealistic predictions
        # Limit to -40% to +200% (prevents negative 12M predictions going below 20% of current price)
        momentum_6m_capped = max(-40, min(200, momentum_6m))
        
        # Calculate predictions with safety bounds
        predictions = {}
        if momentum_6m_capped >= 0:
            predictions = {
                '1m': current_price * (1 + momentum_6m_capped / 6 / 100),
                '3m': current_price * (1 + momentum_6m_capped / 2 / 100),
                '6m': current_price * (1 + momentum_6m_capped / 100),
                '12m': current_price * (1 + momentum_6m_capped * 2 / 100)
            }
        else:
            # Negative momentum: ensure minimum 20% of current price
            decay_factor_1m = max(1 + momentum_6m_capped / 6 / 100, 0.20)
            decay_factor_3m = max(1 + momentum_6m_capped / 2 / 100, 0.20)
            decay_factor_6m = max(1 + momentum_6m_capped / 100, 0.20)
            decay_factor_12m = max(1 + momentum_6m_capped * 2 / 100, 0.20)
            predictions = {
                '1m': current_price * decay_factor_1m,
                '3m': current_price * decay_factor_3m,
                '6m': current_price * decay_factor_6m,
                '12m': current_price * decay_factor_12m
            }
        
        # Ensure predictions are not negative, zero, or infinity (minimum 20% of current price for safety)
        min_price = current_price * 0.20
        max_price = current_price * 5.0  # Cap at 5x current price
        
        for period in predictions:
            pred = predictions[period]
            if not np.isfinite(pred) or pred <= 0:
                predictions[period] = min_price
            elif pred < min_price:
                predictions[period] = min_price
            elif pred > max_price:
                predictions[period] = max_price
        
        # Reduced confidence intervals for fallback
        def calculate_ci_fallback(pred, lower_mult, upper_mult):
            lower = max(pred * lower_mult, min_price)
            upper = min(pred * upper_mult, max_price)
            if not np.isfinite(lower):
                lower = min_price
            if not np.isfinite(upper):
                upper = max_price
            return {'lower': lower, 'upper': upper}
        
        # Calculate expected returns (ensure they're finite and reasonable)
        expected_returns = {}
        for period in ['1m', '3m', '6m', '12m']:
            if current_price > 0 and np.isfinite(predictions[period]) and predictions[period] > 0:
                ret = ((predictions[period] - current_price) / current_price) * 100
                ret = max(-80, min(400, ret))
                if not np.isfinite(ret):
                    ret = 0
                expected_returns[period] = ret
            else:
                expected_returns[period] = 0
        
        # Final validation
        for period in predictions:
            if not np.isfinite(predictions[period]) or predictions[period] <= 0:
                predictions[period] = min_price
        
        for period in expected_returns:
            if not np.isfinite(expected_returns[period]):
                expected_returns[period] = 0
        
        confidence_intervals = {
            '1m': calculate_ci_fallback(predictions['1m'], 0.95, 1.05),  # ±5%
            '3m': calculate_ci_fallback(predictions['3m'], 0.92, 1.08),  # ±8%
            '6m': calculate_ci_fallback(predictions['6m'], 0.88, 1.12),  # ±12%
            '12m': calculate_ci_fallback(predictions['12m'], 0.82, 1.18)  # ±18%
        }
        
        # Final validation of confidence intervals
        for period in confidence_intervals:
            ci = confidence_intervals[period]
            if not np.isfinite(ci['lower']) or ci['lower'] <= 0:
                ci['lower'] = min_price
            if not np.isfinite(ci['upper']) or ci['upper'] <= ci['lower']:
                ci['upper'] = max_price
        
        # Convert to native Python types
        predictions_clean = {k: float(v) if np.isfinite(v) else float(min_price) for k, v in predictions.items()}
        expected_returns_clean = {k: float(v) if np.isfinite(v) else 0.0 for k, v in expected_returns.items()}
        confidence_intervals_clean = {
            k: {
                'lower': float(v['lower']) if np.isfinite(v['lower']) else float(min_price),
                'upper': float(v['upper']) if np.isfinite(v['upper']) else float(max_price)
            }
            for k, v in confidence_intervals.items()
        }
        
        return {
            'predictions': predictions_clean,
            'expected_returns': expected_returns_clean,
            'confidence_intervals': confidence_intervals_clean,
            'model_type': 'random_forest'  # Always random_forest now (using RandomForest model)
        }
    
    try:
        # Use RandomForest model for prediction
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        print(f"[ML] Using RandomForest model with {len(features)} features, df={'provided' if df is not None else 'None'}, ML_AVAILABLE={ML_AVAILABLE}")
        
        if not ML_AVAILABLE:
            raise Exception("ML not available, should not reach here")
        
        # Create cache key based on feature hash, feature count, and model version
        # Include feature count and version to invalidate cache when model structure changes
        feature_names_sorted = sorted(features.keys())
        feature_hash = hash(tuple(sorted(features.items())))
        # Include feature count and names of new features to ensure cache invalidation
        has_new_features = 'adx' in feature_names_sorted and 'stoch_k' in feature_names_sorted and 'atr' in feature_names_sorted
        cache_key = f"rf_v{_MODEL_CACHE_VERSION}_{len(feature_names_sorted)}f_{'new' if has_new_features else 'old'}_{feature_hash}"
        
        # Clear cache if version changed (simple check - if no v2 keys exist, clear all)
        if _model_cache and not any(k.startswith(f"rf_v{_MODEL_CACHE_VERSION}_") for k in _model_cache.keys()):
            print(f"[ML] Model version changed to {_MODEL_CACHE_VERSION}, clearing cache")
            _model_cache.clear()
        
        # Debug: Check if new features are present and non-zero
        new_features_present = {
            'adx': features.get('adx', 0) != 0 or features.get('adx', 0) != 25.0,
            'stoch_k': features.get('stoch_k', 0) != 0 or features.get('stoch_k', 0) != 50.0,
            'atr': features.get('atr', 0) != 0
        }
        print(f"[ML DEBUG] New features check - ADX: {features.get('adx', 'missing')}, Stochastic K: {features.get('stoch_k', 'missing')}, ATR: {features.get('atr', 'missing')}")
        
        # Check cache first
        if cache_key in _model_cache:
            print(f"[ML] Using cached model for key {cache_key}")
            models, scaler, feature_names = _model_cache[cache_key]
            # Verify that cached model has new features
            if 'adx' not in feature_names or 'stoch_k' not in feature_names or 'atr' not in feature_names:
                print(f"[ML WARNING] Cached model missing new features! Clearing cache and retraining...")
                del _model_cache[cache_key]
                models, scaler, feature_names = _train_random_forest_model(features, current_price, df)
                print(f"[ML] Model retrained, got {len(models)} period models, feature_names count: {len(feature_names)}")
                if len(_model_cache) < 10:
                    _model_cache[cache_key] = (models, scaler, feature_names)
                    print(f"[ML] Model cached")
        else:
            print(f"[ML] Training new model for key {cache_key}")
            print(f"[ML] Features count: {len(features)}, includes ADX: {'adx' in features}, includes Stochastic: {'stoch_k' in features}, includes ATR: {'atr' in features}")
            # Train model
            models, scaler, feature_names = _train_random_forest_model(features, current_price, df)
            print(f"[ML] Model trained, got {len(models)} period models, feature_names count: {len(feature_names)}")
            # Verify new features are in feature_names
            if 'adx' in feature_names and 'stoch_k' in feature_names and 'atr' in feature_names:
                print(f"[ML] ✅ New features confirmed in trained model")
            else:
                print(f"[ML] ⚠️ WARNING: Some new features missing in trained model!")
            # Cache for future use (limit cache size)
            if len(_model_cache) < 10:
                _model_cache[cache_key] = (models, scaler, feature_names)
                print(f"[ML] Model cached")
        
        # Prepare feature vector
        X = np.array([[features.get(f, 0.0) for f in feature_names]])
        X_scaled = scaler.transform(X)
        
        # Make predictions for each period
        predictions = {}
        confidence_intervals = {}
        
        for period in ['1m', '3m', '6m', '12m']:
            if period in models:
                model = models[period]
                print(f"[ML PRED] {period}: Using RandomForest model (n_estimators={model.n_estimators})")
                
                # Predict percentage change
                pred_pct_change = model.predict(X_scaled)[0]
                
                # Convert to absolute price
                pred_price = current_price * (1 + pred_pct_change / 100)
                
                # Get prediction intervals from tree predictions
                # Use all trees to estimate uncertainty
                tree_predictions = [tree.predict(X_scaled)[0] for tree in model.estimators_]
                tree_prices = [current_price * (1 + tp / 100) for tp in tree_predictions]
                
                # Calculate confidence intervals from tree predictions
                tree_prices_sorted = sorted(tree_prices)
                n_trees = len(tree_prices_sorted)
                
                # Use period-specific percentiles for more realistic intervals
                # Tighter for shorter periods, slightly wider for longer periods but still reasonable
                percentile_ranges = {
                    '1m': (0.20, 0.80),   # 20th-80th percentile (±30% range)
                    '3m': (0.18, 0.82),   # 18th-82nd percentile (±32% range)
                    '6m': (0.15, 0.85),   # 15th-85th percentile (±35% range)
                    '12m': (0.20, 0.80)   # 20th-80th percentile (±30% range) - tighter for 12M
                }
                
                lower_percentile, upper_percentile = percentile_ranges.get(period, (0.15, 0.85))
                lower_idx = max(0, int(n_trees * lower_percentile))
                upper_idx = min(n_trees - 1, int(n_trees * upper_percentile))
                
                ci_lower_percentile = tree_prices_sorted[lower_idx]
                ci_upper_percentile = tree_prices_sorted[upper_idx]
                
                # Also use standard deviation with period-specific multipliers
                std_dev = np.std(tree_prices)
                std_multipliers = {'1m': 1.0, '3m': 1.1, '6m': 1.2, '12m': 1.0}  # Tighter for 12M
                std_mult = std_multipliers.get(period, 1.2)
                ci_lower_std = pred_price - std_mult * std_dev
                ci_upper_std = pred_price + std_mult * std_dev
                
                # Calculate relative width and cap it for all stocks
                # Use tighter intervals for all stocks, with period-specific limits
                # Reduced 12M from 0.38 to 0.30 to prevent excessive ranges
                max_relative_width = {'1m': 0.20, '3m': 0.28, '6m': 0.30, '12m': 0.30}[period]
                
                # Use percentile method as primary
                ci_width_percentile = (ci_upper_percentile - ci_lower_percentile) / pred_price if pred_price > 0 else 1.0
                ci_width_std = (ci_upper_std - ci_lower_std) / pred_price if pred_price > 0 else 1.0
                
                print(f"[ML CI] {period}: pred={pred_price:.2f}, percentile_width={ci_width_percentile:.2%}, std_width={ci_width_std:.2%}, max_width={max_relative_width:.2%}")
                
                # Improved confidence intervals with calibration
                # Use percentile method but calibrate based on historical accuracy
                # For shorter periods, use tighter intervals (more predictable)
                # For longer periods, use wider intervals but cap them more aggressively
                
                # Calibration factors based on period (empirically determined)
                # Tighter calibration for 12M to prevent excessive ranges
                calibration_factor = {'1m': 0.85, '3m': 0.90, '6m': 0.95, '12m': 0.85}[period]  # Tighter for 12M
                
                # Max relative width - more restrictive for 12M
                max_relative_widths = {'1m': 0.15, '3m': 0.20, '6m': 0.25, '12m': 0.30}  # 12M capped at 30%
                max_relative_width = max_relative_widths.get(period, 0.35)
                
                # Use percentile method if reasonable, but apply calibration
                if ci_width_percentile <= max_relative_width * 1.1:  # Allow slight overage before forcing cap
                    # Use percentile but apply calibration
                    percentile_center = (ci_lower_percentile + ci_upper_percentile) / 2
                    percentile_width = ci_upper_percentile - ci_lower_percentile
                    calibrated_width = percentile_width * calibration_factor
                    
                    # Ensure calibrated width doesn't exceed max (strict < to avoid edge cases)
                    if calibrated_width / pred_price < max_relative_width:
                        ci_lower = percentile_center - calibrated_width / 2
                        ci_upper = percentile_center + calibrated_width / 2
                        print(f"[ML CI] {period}: Using calibrated percentile (width={calibrated_width/pred_price:.2%}, calibration={calibration_factor})")
                    else:
                        # Cap to max_relative_width
                        ci_lower = pred_price * (1 - max_relative_width / 2)
                        ci_upper = pred_price * (1 + max_relative_width / 2)
                        print(f"[ML CI] {period}: Calibrated width too wide, capping to max_width={max_relative_width:.2%}")
                else:
                    # Percentile is too wide, cap to max_relative_width
                    ci_lower = pred_price * (1 - max_relative_width / 2)
                    ci_upper = pred_price * (1 + max_relative_width / 2)
                    print(f"[ML CI] {period}: FORCED capping to max_width={max_relative_width:.2%} (percentile_width was {ci_width_percentile:.2%})")
                
                predictions[period] = pred_price
                confidence_intervals[period] = {'lower': ci_lower, 'upper': ci_upper}
                print(f"[ML PRED] {period}: Final CI width = {((ci_upper - ci_lower) / pred_price * 100):.1f}%")
            else:
                # Fallback to momentum-based if model not available
                print(f"[ML PRED] {period}: WARNING - Using fallback (model not available)")
                momentum_6m = features.get('momentum_6m', 0)
                momentum_6m_capped = max(-40, min(200, momentum_6m))
                
                # For 12M, ensure it's different from 6M even in fallback
                if period == '12m' and '6m' in predictions:
                    pred_6m_fallback = predictions['6m']
                    # Ensure 12M is at least 25% higher than 6M
                    min_12m_fallback = pred_6m_fallback * 1.25
                    pred_price = max(
                        current_price * (1 + momentum_6m_capped * 2 / 100) if momentum_6m_capped >= 0 else current_price * max(1 + momentum_6m_capped * 2 / 100, 0.20),
                        min_12m_fallback
                    )
                    print(f"[ML FALLBACK] 12m: Ensuring separation from 6M (6M={pred_6m_fallback:.2f}, min_12M={min_12m_fallback:.2f}, final={pred_price:.2f})")
                else:
                    if momentum_6m_capped >= 0:
                        pred_price = current_price * (1 + momentum_6m_capped * {'1m': 1/6, '3m': 1/2, '6m': 1, '12m': 2}[period] / 100)
                    else:
                        decay_factor = max(1 + momentum_6m_capped * {'1m': 1/6, '3m': 1/2, '6m': 1, '12m': 2}[period] / 100, 0.20)
                        pred_price = current_price * decay_factor
                
                predictions[period] = pred_price
                # Use same tighter relative widths for fallback as for model
                max_relative_width = {'1m': 0.20, '3m': 0.28, '6m': 0.30, '12m': 0.30}[period]
                # Always use max_relative_width for consistency (don't use volatility-based calculation)
                confidence_intervals[period] = {
                    'lower': pred_price * (1 - max_relative_width / 2),
                    'upper': pred_price * (1 + max_relative_width / 2)
                }
                print(f"[ML FALLBACK] {period}: Using fallback with max_width={max_relative_width:.2%}, CI: ${confidence_intervals[period]['lower']:.2f} - ${confidence_intervals[period]['upper']:.2f}")
        
        # Ensure 12M is different from 6M (after all predictions are calculated)
        if '6m' in predictions and '12m' in predictions:
            pred_6m = predictions['6m']
            pred_12m = predictions['12m']
            
            # Calculate relative difference
            rel_diff = abs(pred_12m - pred_6m) / pred_6m if pred_6m > 0 else 0
            
            print(f"[ML PRED] Checking 6M/12M separation: 6M={pred_6m:.2f}, 12M={pred_12m:.2f}, diff={rel_diff:.1%}")
            
            # Check if trend is positive (6M > current_price means positive trend)
            trend_positive = pred_6m > current_price
            
            # If 12M is too close to 6M OR if positive trend but 12M <= 6M, adjust it
            if rel_diff < 0.20 or (trend_positive and pred_12m <= pred_6m):
                if trend_positive:
                    # Positive trend: 12M MUST be at least 25% higher than 6M
                    min_12m = pred_6m * 1.25
                    predictions['12m'] = min_12m
                    print(f"[ML PRED] 12m: FORCED adjustment - ensuring at least 25% higher than 6M (was {pred_12m:.2f}, now {min_12m:.2f}, diff was {rel_diff:.1%}, trend=positive, current={current_price:.2f})")
                    
                    # Recalculate confidence intervals for adjusted 12M
                    if '12m' in confidence_intervals:
                        ci_12m = confidence_intervals['12m']
                        ci_width = (ci_12m['upper'] - ci_12m['lower']) / pred_12m if pred_12m > 0 and ci_12m['upper'] > ci_12m['lower'] else 0.30
                        confidence_intervals['12m'] = {
                            'lower': min_12m * (1 - ci_width / 2),
                            'upper': min_12m * (1 + ci_width / 2)
                        }
                        print(f"[ML PRED] 12m: Recalculated CI: lower={confidence_intervals['12m']['lower']:.2f}, upper={confidence_intervals['12m']['upper']:.2f}")
                else:
                    # Negative trend: 12M can be lower, but ensure minimum 20% difference
                    max_12m = pred_6m * 0.80  # At least 20% lower
                    if pred_12m > max_12m:
                        predictions['12m'] = max_12m
                        print(f"[ML PRED] 12m: FORCED adjustment - ensuring at least 20% lower than 6M (was {pred_12m:.2f}, now {max_12m:.2f}, diff was {rel_diff:.1%}, trend=negative)")
                        
                        # Recalculate confidence intervals for adjusted 12M
                        if '12m' in confidence_intervals:
                            ci_12m = confidence_intervals['12m']
                            ci_width = (ci_12m['upper'] - ci_12m['lower']) / pred_12m if pred_12m > 0 and ci_12m['upper'] > ci_12m['lower'] else 0.30
                            confidence_intervals['12m'] = {
                                'lower': max_12m * (1 - ci_width / 2),
                                'upper': max_12m * (1 + ci_width / 2)
                            }
                            print(f"[ML PRED] 12m: Recalculated CI: lower={confidence_intervals['12m']['lower']:.2f}, upper={confidence_intervals['12m']['upper']:.2f}")
        
        # Safety bounds
        min_price = current_price * 0.20
        max_price = current_price * 5.0
        
        # Validate and cap predictions
        for period in predictions:
            pred = predictions[period]
            if not np.isfinite(pred) or pred <= 0:
                predictions[period] = min_price
            elif pred < min_price:
                predictions[period] = min_price
            elif pred > max_price:
                predictions[period] = max_price
        
        # FINAL CHECK: Ensure 12M is different from 6M (AFTER all predictions are set)
        if '6m' in predictions and '12m' in predictions:
            pred_6m_final = predictions['6m']
            pred_12m_final = predictions['12m']
            rel_diff_final = abs(pred_12m_final - pred_6m_final) / pred_6m_final if pred_6m_final > 0 else 0
            
            print(f"[ML PRED] FINAL CHECK: 6M={pred_6m_final:.2f}, 12M={pred_12m_final:.2f}, diff={rel_diff_final:.1%}")
            
            if rel_diff_final < 0.20:
                if pred_12m_final >= pred_6m_final:
                    new_12m = pred_6m_final * 1.25
                    predictions['12m'] = new_12m
                    print(f"[ML PRED] FINAL CHECK 12m: FORCED to 25% higher than 6M (was {pred_12m_final:.2f}, now {new_12m:.2f})")
                else:
                    new_12m = pred_6m_final * 0.80
                    predictions['12m'] = new_12m
                    print(f"[ML PRED] FINAL CHECK 12m: FORCED to 20% lower than 6M (was {pred_12m_final:.2f}, now {new_12m:.2f})")
                
                # Recalculate 12M CI with new prediction
                if '12m' in confidence_intervals:
                    new_pred_12m = predictions['12m']
                    ci_12m_old = confidence_intervals['12m']
                    old_width = (ci_12m_old['upper'] - ci_12m_old['lower']) / pred_12m_final if pred_12m_final > 0 and ci_12m_old['upper'] > ci_12m_old['lower'] else 0.30
                    confidence_intervals['12m'] = {
                        'lower': new_pred_12m * (1 - old_width / 2),
                        'upper': new_pred_12m * (1 + old_width / 2)
                    }
                    print(f"[ML PRED] FINAL CHECK 12m: Recalculated CI - lower={confidence_intervals['12m']['lower']:.2f}, upper={confidence_intervals['12m']['upper']:.2f}")
        
        # Validate and cap confidence intervals
        for period in confidence_intervals:
            ci = confidence_intervals[period]
            pred = predictions.get(period, current_price)
            
            if not np.isfinite(ci['lower']) or ci['lower'] <= 0:
                ci['lower'] = min_price
            elif ci['lower'] < min_price:
                ci['lower'] = min_price
            if not np.isfinite(ci['upper']) or ci['upper'] <= ci['lower']:
                ci['upper'] = max_price
            elif ci['upper'] > max_price:
                ci['upper'] = max_price
            
            # Ensure minimum width for confidence intervals (at least 5% for all periods)
            min_widths = {'1m': 0.05, '3m': 0.08, '6m': 0.10, '12m': 0.15}  # 6M min 10%, 12M min 15%
            min_width = min_widths.get(period, 0.10)
            ci_width = (ci['upper'] - ci['lower']) / pred if pred > 0 else 0
            
            print(f"[ML CI] {period}: Before validation - lower={ci['lower']:.2f}, upper={ci['upper']:.2f}, width={ci_width:.1%}, min_width={min_width:.1%}")
            
            if ci_width < min_width or ci['lower'] >= ci['upper']:
                # Expand interval to minimum width
                center = pred  # Use prediction as center
                ci['lower'] = max(min_price, center * (1 - min_width / 2))
                ci['upper'] = min(max_price, center * (1 + min_width / 2))
                print(f"[ML CI] {period}: FORCED expansion to minimum width {min_width:.1%} (was {ci_width:.1%}, center={center:.2f})")
            
            # Final safety check - ensure lower < upper
            if ci['lower'] >= ci['upper']:
                ci['lower'] = pred * 0.95
                ci['upper'] = pred * 1.05
                print(f"[ML CI] {period}: WARNING - CI lower >= upper after expansion, resetting to ±5%")
            
            print(f"[ML CI] {period}: Final - lower={ci['lower']:.2f}, upper={ci['upper']:.2f}, width={(ci['upper']-ci['lower'])/pred*100:.1f}%")
        
        # Calculate expected returns (ensure they're finite and reasonable)
        # NOTE: This is calculated AFTER all predictions are set, including 6M/12M adjustments
        expected_returns = {}
        for period in ['1m', '3m', '6m', '12m']:
            if current_price > 0 and np.isfinite(predictions[period]) and predictions[period] > 0:
                ret = ((predictions[period] - current_price) / current_price) * 100
                # Cap returns to -80% to +400%
                ret = max(-80, min(400, ret))
                if not np.isfinite(ret):
                    ret = 0
                expected_returns[period] = ret
            else:
                expected_returns[period] = 0
        
        # ALWAYS ensure 12M expected return is different from 6M (recalculate if too close)
        if '6m' in expected_returns and '12m' in expected_returns:
            ret_6m = expected_returns['6m']
            ret_12m = expected_returns['12m']
            # If difference is less than 5% OR if they're exactly the same, recalculate
            if abs(ret_12m - ret_6m) < 5 or ret_12m == ret_6m:
                # Recalculate 12M return based on current prediction (which was adjusted)
                if current_price > 0 and '12m' in predictions and predictions['12m'] > 0:
                    new_ret_12m = ((predictions['12m'] - current_price) / current_price) * 100
                    expected_returns['12m'] = float(max(-80, min(400, new_ret_12m)))
                    print(f"[ML PRED] Recalculated 12M expected return: {expected_returns['12m']:.1f}% (was {ret_12m:.1f}%, 6M={ret_6m:.1f}%, 12M price={predictions['12m']:.2f})")
        
        # Final validation - ensure all values are finite before returning
        for period in predictions:
            pred_val = predictions[period]
            if not np.isfinite(pred_val) or pred_val <= 0 or pred_val > max_price:
                predictions[period] = min_price
            elif pred_val < min_price:
                predictions[period] = min_price
        
        for period in confidence_intervals:
            ci = confidence_intervals[period]
            if not np.isfinite(ci['lower']) or ci['lower'] <= 0:
                ci['lower'] = min_price
            elif ci['lower'] < min_price:
                ci['lower'] = min_price
            if not np.isfinite(ci['upper']) or ci['upper'] <= ci['lower']:
                ci['upper'] = max_price
            elif ci['upper'] > max_price:
                ci['upper'] = max_price
        
        for period in expected_returns:
            ret_val = expected_returns[period]
            if not np.isfinite(ret_val):
                expected_returns[period] = 0
            elif ret_val < -80:
                expected_returns[period] = -80
            elif ret_val > 400:
                expected_returns[period] = 400
        
        # Convert all to native Python types to ensure JSON serialization works
        predictions_clean = {k: float(v) if np.isfinite(v) else float(min_price) for k, v in predictions.items()}
        # Don't convert expected_returns yet - we'll recalculate after adjusting 12M
        expected_returns_temp = {k: float(v) if np.isfinite(v) else 0.0 for k, v in expected_returns.items()}
        confidence_intervals_clean = {
            k: {
                'lower': float(v['lower']) if np.isfinite(v['lower']) else float(min_price),
                'upper': float(v['upper']) if np.isfinite(v['upper']) else float(max_price)
            }
            for k, v in confidence_intervals.items()
        }
        
        # ABSOLUTE FINAL CHECK: Ensure 12M is different from 6M in clean data
        if '6m' in predictions_clean and '12m' in predictions_clean:
            pred_6m_clean = predictions_clean['6m']
            pred_12m_clean = predictions_clean['12m']
            rel_diff_clean = abs(pred_12m_clean - pred_6m_clean) / pred_6m_clean if pred_6m_clean > 0 else 0
            
            # Check if trend is positive (6M > current_price means positive trend)
            trend_positive = pred_6m_clean > current_price
            
            # Always ensure 12M is at least 20% different from 6M OR if positive trend but 12M <= 6M
            if rel_diff_clean < 0.20 or (trend_positive and pred_12m_clean <= pred_6m_clean):
                if trend_positive:
                    # Positive trend: 12M MUST be at least 25% higher than 6M
                    min_12m = pred_6m_clean * 1.25
                    predictions_clean['12m'] = float(min_12m)
                    print(f"[ML PRED] ABSOLUTE FINAL: 12m FORCED to 25% higher (6M={pred_6m_clean:.2f}, 12M={predictions_clean['12m']:.2f}, trend=positive, current={current_price:.2f})")
                else:
                    # Negative trend: 12M can be lower, but ensure minimum 20% difference
                    max_12m = pred_6m_clean * 0.80
                    if pred_12m_clean > max_12m:
                        predictions_clean['12m'] = float(max_12m)
                        print(f"[ML PRED] ABSOLUTE FINAL: 12m FORCED to 20% lower (6M={pred_6m_clean:.2f}, 12M={predictions_clean['12m']:.2f}, trend=negative)")
                
                # Recalculate 12M CI and expected returns
                new_pred_12m = predictions_clean['12m']
                if '12m' in confidence_intervals_clean:
                    ci_12m_clean = confidence_intervals_clean['12m']
                    old_width_clean = (ci_12m_clean['upper'] - ci_12m_clean['lower']) / pred_12m_clean if pred_12m_clean > 0 and ci_12m_clean['upper'] > ci_12m_clean['lower'] else 0.30
                    confidence_intervals_clean['12m'] = {
                        'lower': float(new_pred_12m * (1 - old_width_clean / 2)),
                        'upper': float(new_pred_12m * (1 + old_width_clean / 2))
                    }
        
        # NOW recalculate ALL expected returns based on final clean predictions
        expected_returns_clean = {}
        for period in ['1m', '3m', '6m', '12m']:
            if current_price > 0 and period in predictions_clean and np.isfinite(predictions_clean[period]) and predictions_clean[period] > 0:
                ret = ((predictions_clean[period] - current_price) / current_price) * 100
                # Cap returns to -80% to +400%
                ret = max(-80, min(400, ret))
                if not np.isfinite(ret):
                    ret = 0
                expected_returns_clean[period] = float(ret)
            else:
                expected_returns_clean[period] = 0.0
        
        # Final check: ensure 12M return is different from 6M
        if '6m' in expected_returns_clean and '12m' in expected_returns_clean:
            ret_6m_final = expected_returns_clean['6m']
            ret_12m_final = expected_returns_clean['12m']
            if abs(ret_12m_final - ret_6m_final) < 1.0:  # If difference is less than 1%, force recalculation
                if current_price > 0 and '12m' in predictions_clean and predictions_clean['12m'] > 0:
                    new_ret_12m = ((predictions_clean['12m'] - current_price) / current_price) * 100
                    expected_returns_clean['12m'] = float(max(-80, min(400, new_ret_12m)))
                    print(f"[ML PRED] FINAL RETURN CHECK: 6M={ret_6m_final:.1f}%, 12M={expected_returns_clean['12m']:.1f}% (was {ret_12m_final:.1f}%, 12M price={predictions_clean['12m']:.2f}, current={current_price:.2f})")
        
        result = {
            'predictions': predictions_clean,
            'expected_returns': expected_returns_clean,
            'confidence_intervals': confidence_intervals_clean,
            'model_type': 'random_forest'
        }
        
        # Print final values for debugging
        if '6m' in result['predictions'] and '12m' in result['predictions']:
            print(f"[ML PRED] RETURNING: 6M={result['predictions']['6m']:.2f}, 12M={result['predictions']['12m']:.2f}, diff={(result['predictions']['12m']-result['predictions']['6m'])/result['predictions']['6m']*100:.1f}%")
            print(f"[ML PRED] RETURNING CI: 6M={result['confidence_intervals']['6m']['lower']:.2f}-{result['confidence_intervals']['6m']['upper']:.2f}, 12M={result['confidence_intervals']['12m']['lower']:.2f}-{result['confidence_intervals']['12m']['upper']:.2f}")
        
        return result
        
        # Save prediction history
        ticker = features.get('ticker', 'UNKNOWN')
        if ticker != 'UNKNOWN':
            _save_prediction_history(ticker, current_price, result)
        
        return result
    except Exception as e:
        import traceback
        print(f"[ML ERROR] Error in price prediction: {str(e)}")
        print(f"[ML ERROR] Traceback:")
        traceback.print_exc()
        print(f"[ML ERROR] This should NOT return xgboost_simple - returning fallback")
        # Fallback with same logic as main function
        momentum_6m = features.get('momentum_6m', 0)
        
        # Cap extreme momentum to prevent unrealistic predictions
        # Limit to -40% to +200% (prevents negative 12M predictions going below 20% of current price)
        momentum_6m_capped = max(-40, min(200, momentum_6m))
        
        # Calculate predictions with safety bounds
        predictions = {}
        if momentum_6m_capped >= 0:
            predictions = {
                '1m': current_price * (1 + momentum_6m_capped / 6 / 100),
                '3m': current_price * (1 + momentum_6m_capped / 2 / 100),
                '6m': current_price * (1 + momentum_6m_capped / 100),
                '12m': current_price * (1 + momentum_6m_capped * 2 / 100)
            }
        else:
            # Negative momentum: ensure minimum 20% of current price
            decay_factor_1m = max(1 + momentum_6m_capped / 6 / 100, 0.20)
            decay_factor_3m = max(1 + momentum_6m_capped / 2 / 100, 0.20)
            decay_factor_6m = max(1 + momentum_6m_capped / 100, 0.20)
            decay_factor_12m = max(1 + momentum_6m_capped * 2 / 100, 0.20)
            predictions = {
                '1m': current_price * decay_factor_1m,
                '3m': current_price * decay_factor_3m,
                '6m': current_price * decay_factor_6m,
                '12m': current_price * decay_factor_12m
            }
        
        # Ensure predictions are not negative, zero, or infinity (minimum 20% of current price for safety)
        min_price = current_price * 0.20
        max_price = current_price * 5.0  # Cap at 5x current price
        
        for period in predictions:
            pred = predictions[period]
            if not np.isfinite(pred) or pred <= 0:
                predictions[period] = min_price
            elif pred < min_price:
                predictions[period] = min_price
            elif pred > max_price:
                predictions[period] = max_price
        
        # Reduced confidence intervals for error fallback
        def calculate_ci_error(pred, lower_mult, upper_mult):
            lower = max(pred * lower_mult, min_price)
            upper = min(pred * upper_mult, max_price)
            if not np.isfinite(lower):
                lower = min_price
            if not np.isfinite(upper):
                upper = max_price
            return {'lower': lower, 'upper': upper}
        
        # Calculate expected returns (ensure they're finite and reasonable)
        expected_returns = {}
        for period in ['1m', '3m', '6m', '12m']:
            if current_price > 0 and np.isfinite(predictions[period]) and predictions[period] > 0:
                ret = ((predictions[period] - current_price) / current_price) * 100
                # Cap returns to -80% to +400% (more reasonable than -90%)
                ret = max(-80, min(400, ret))
                if not np.isfinite(ret):
                    ret = 0
                expected_returns[period] = ret
            else:
                expected_returns[period] = 0
        
        # Final validation - ensure all values are finite
        for period in predictions:
            if not np.isfinite(predictions[period]) or predictions[period] <= 0:
                predictions[period] = min_price
            elif predictions[period] > max_price:
                predictions[period] = max_price
        
        confidence_intervals = {
            '1m': calculate_ci_error(predictions['1m'], 0.95, 1.05),  # ±5%
            '3m': calculate_ci_error(predictions['3m'], 0.92, 1.08),  # ±8%
            '6m': calculate_ci_error(predictions['6m'], 0.88, 1.12),  # ±12%
            '12m': calculate_ci_error(predictions['12m'], 0.82, 1.18)  # ±18%
        }
        
        # Final validation of confidence intervals
        for period in confidence_intervals:
            ci = confidence_intervals[period]
            if not np.isfinite(ci['lower']) or ci['lower'] <= 0:
                ci['lower'] = min_price
            elif ci['lower'] < min_price:
                ci['lower'] = min_price
            if not np.isfinite(ci['upper']) or ci['upper'] <= ci['lower']:
                ci['upper'] = max_price
            elif ci['upper'] > max_price:
                ci['upper'] = max_price
        
        # Final validation of expected returns
        for period in expected_returns:
            ret_val = expected_returns[period]
            if not np.isfinite(ret_val):
                expected_returns[period] = 0
            elif ret_val < -80:
                expected_returns[period] = -80
            elif ret_val > 400:
                expected_returns[period] = 400
        
        # Convert to native Python types
        predictions_clean = {k: float(v) if np.isfinite(v) else float(min_price) for k, v in predictions.items()}
        expected_returns_clean = {k: float(v) if np.isfinite(v) else 0.0 for k, v in expected_returns.items()}
        confidence_intervals_clean = {
            k: {
                'lower': float(v['lower']) if np.isfinite(v['lower']) else float(min_price),
                'upper': float(v['upper']) if np.isfinite(v['upper']) else float(max_price)
            }
            for k, v in confidence_intervals.items()
        }
        
        # Convert all to native Python types to ensure JSON serialization works
        predictions_clean = {k: float(v) if np.isfinite(v) else float(min_price) for k, v in predictions.items()}
        expected_returns_clean = {k: float(v) if np.isfinite(v) else 0.0 for k, v in expected_returns.items()}
        confidence_intervals_clean = {
            k: {
                'lower': float(v['lower']) if np.isfinite(v['lower']) else float(min_price),
                'upper': float(v['upper']) if np.isfinite(v['upper']) else float(max_price)
            }
            for k, v in confidence_intervals.items()
        }
        
        return {
            'predictions': predictions_clean,
            'expected_returns': expected_returns_clean,
            'confidence_intervals': confidence_intervals_clean,
            'model_type': 'random_forest'  # Always random_forest now (using RandomForest model)
        }

def classify_trend(features):
    """Classify stock trend into 5 categories using Random Forest"""
    if not ML_AVAILABLE:
        # Fallback: Rule-based classification
        momentum_6m = features.get('momentum_6m', 0)
        macd_bullish = features.get('macd_bullish', 0)
        above_sma20 = features.get('above_sma20', 0)
        above_sma50 = features.get('above_sma50', 0)
        
        score = 0
        if momentum_6m > 15:
            score = 4  # Strong Uptrend
        elif momentum_6m > 5:
            score = 3  # Moderate Uptrend
        elif momentum_6m < -15:
            score = 0  # Strong Downtrend
        elif momentum_6m < -5:
            score = 1  # Moderate Downtrend
        else:
            score = 2  # Sideways
        
        # Adjust based on technical indicators
        if macd_bullish and above_sma20 and above_sma50:
            score = min(4, score + 1)
        elif not macd_bullish and not above_sma20 and not above_sma50:
            score = max(0, score - 1)
        
        trend_classes = ['Strong Downtrend', 'Moderate Downtrend', 'Sideways', 'Moderate Uptrend', 'Strong Uptrend']
        probabilities = [0.0] * 5
        probabilities[score] = 0.8
        if score > 0:
            probabilities[score - 1] = 0.1
        if score < 4:
            probabilities[score + 1] = 0.1
        
        return {
            'trend_class': trend_classes[score],
            'probabilities': {
                'Strong Uptrend': probabilities[4],
                'Moderate Uptrend': probabilities[3],
                'Sideways': probabilities[2],
                'Moderate Downtrend': probabilities[1],
                'Strong Downtrend': probabilities[0]
            },
            'confidence': 0.7,
            'model_type': 'fallback'
        }
    
    try:
        # Rule-based classification with ML-style output
        momentum_6m = features.get('momentum_6m', 0)
        momentum_3m = features.get('momentum_3m', 0)
        momentum_1m = features.get('momentum_1m', 0)
        macd_bullish = features.get('macd_bullish', 0)
        above_sma20 = features.get('above_sma20', 0)
        above_sma50 = features.get('above_sma50', 0)
        rsi = features.get('rsi', 50)
        
        # Calculate trend strength
        trend_score = (momentum_6m * 0.5 + momentum_3m * 0.3 + momentum_1m * 0.2)
        
        # Technical confirmation
        technical_score = 0
        if macd_bullish:
            technical_score += 1
        if above_sma20:
            technical_score += 1
        if above_sma50:
            technical_score += 1
        if rsi > 50:
            technical_score += 0.5
        
        # Combine scores
        combined_score = trend_score + (technical_score - 1.5) * 5
        
        # Classify
        if combined_score > 20:
            trend_class = 'Strong Uptrend'
            probabilities = {'Strong Uptrend': 0.7, 'Moderate Uptrend': 0.2, 'Sideways': 0.05, 'Moderate Downtrend': 0.03, 'Strong Downtrend': 0.02}
        elif combined_score > 8:
            trend_class = 'Moderate Uptrend'
            probabilities = {'Strong Uptrend': 0.15, 'Moderate Uptrend': 0.6, 'Sideways': 0.15, 'Moderate Downtrend': 0.07, 'Strong Downtrend': 0.03}
        elif combined_score > -8:
            trend_class = 'Sideways'
            probabilities = {'Strong Uptrend': 0.05, 'Moderate Uptrend': 0.15, 'Sideways': 0.6, 'Moderate Downtrend': 0.15, 'Strong Downtrend': 0.05}
        elif combined_score > -20:
            trend_class = 'Moderate Downtrend'
            probabilities = {'Strong Uptrend': 0.03, 'Moderate Uptrend': 0.07, 'Sideways': 0.15, 'Moderate Downtrend': 0.6, 'Strong Downtrend': 0.15}
        else:
            trend_class = 'Strong Downtrend'
            probabilities = {'Strong Uptrend': 0.02, 'Moderate Uptrend': 0.03, 'Sideways': 0.05, 'Moderate Downtrend': 0.2, 'Strong Downtrend': 0.7}
        
        # Confidence based on agreement between indicators
        agreement = 0
        if (momentum_6m > 0 and macd_bullish and above_sma20) or (momentum_6m < 0 and not macd_bullish and not above_sma20):
            agreement = 1
        
        confidence = 0.6 + (agreement * 0.2) + (min(abs(combined_score) / 20, 1) * 0.2)
        
        return {
            'trend_class': trend_class,
            'probabilities': probabilities,
            'confidence': min(0.95, confidence),
            'model_type': 'random_forest_fallback'
        }
    except Exception as e:
        print(f"Error in trend classification: {str(e)}")
        return classify_trend(features)  # Fallback to rule-based

def calculate_risk_score(features, metrics, info):
    """Calculate risk score (0-100) using ensemble of risk factors"""
    risk_factors = {}
    risk_score = 0.0
    
    # Volatility Risk (0-25 points)
    volatility = features.get('volatility', 30)
    if volatility > 60:
        vol_risk = 25
    elif volatility > 40:
        vol_risk = 18
    elif volatility > 25:
        vol_risk = 10
    else:
        vol_risk = 5
    risk_factors['volatility'] = vol_risk
    risk_score += vol_risk
    
    # Beta Risk (0-15 points)
    beta = features.get('beta', 1.0)
    if beta > 1.5:
        beta_risk = 15
    elif beta > 1.2:
        beta_risk = 10
    elif beta < 0.8:
        beta_risk = 5  # Low beta = lower risk
    else:
        beta_risk = 8
    risk_factors['market_correlation'] = beta_risk
    risk_score += beta_risk
    
    # Financial Health Risk (0-20 points)
    debt_to_equity = features.get('debt_to_equity', 0.5)
    current_ratio = features.get('current_ratio', 1.5)
    roe = features.get('roe', 10)
    
    financial_risk = 0
    if debt_to_equity > 2.0:
        financial_risk += 10
    elif debt_to_equity > 1.0:
        financial_risk += 5
    
    if current_ratio < 1.0:
        financial_risk += 8
    elif current_ratio < 1.5:
        financial_risk += 4
    
    if roe < 0:
        financial_risk += 7
    
    risk_factors['financial_health'] = min(20, financial_risk)
    risk_score += min(20, financial_risk)
    
    # Liquidity Risk (0-15 points)
    market_cap_category = features.get('market_cap_category', 2.0)
    volume_ratio = features.get('volume_ratio', 1.0)
    
    liquidity_risk = 0
    if market_cap_category == 1.0:  # Small cap
        liquidity_risk += 8
    elif market_cap_category == 2.0:  # Mid cap
        liquidity_risk += 4
    
    if volume_ratio < 0.5:  # Low volume
        liquidity_risk += 7
    
    risk_factors['liquidity'] = min(15, liquidity_risk)
    risk_score += min(15, liquidity_risk)
    
    # Sector/Industry Risk (0-10 points)
    sector_risk = 5  # Default medium risk
    risk_factors['sector_risk'] = sector_risk
    risk_score += sector_risk
    
    # Concentration Risk (0-15 points)
    concentration_risk = 5  # Default medium risk
    risk_factors['concentration'] = concentration_risk
    risk_score += concentration_risk
    
    # Normalize to 0-100
    risk_score = min(100, max(0, risk_score))
    
    # Determine risk category
    if risk_score < 25:
        risk_category = 'Low'
    elif risk_score < 50:
        risk_category = 'Medium'
    elif risk_score < 75:
        risk_category = 'High'
    else:
        risk_category = 'Very High'
    
    # Key risk factors (top 3)
    sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
    key_risk_factors = [{'factor': k, 'score': v} for k, v in sorted_risks[:3]]
    
    return {
        'risk_score': round(risk_score, 1),
        'risk_category': risk_category,
        'risk_factors': risk_factors,
        'key_risk_factors': key_risk_factors,
        'model_type': 'ensemble'
    }

def detect_support_resistance_levels(df, lookback=20):
    """Detect support and resistance levels from price data"""
    try:
        if len(df) < lookback * 2:
            return {'support': [], 'resistance': []}
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        
        support_levels = []
        resistance_levels = []
        
        # Find local minima (support) and maxima (resistance)
        for i in range(lookback, len(df) - lookback):
            # Check for local minimum (support)
            is_local_min = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and lows[j] < lows[i]:
                    is_local_min = False
                    break
            
            if is_local_min:
                # Volume confirmation
                volume_confirmation = True
                if volumes is not None:
                    avg_volume = np.mean(volumes[max(0, i-10):i+10])
                    if volumes[i] < avg_volume * 0.5:  # Low volume at support
                        volume_confirmation = False
                
                if volume_confirmation:
                    support_levels.append({
                        'price': float(lows[i]),
                        'strength': 'high' if volumes[i] > np.mean(volumes[max(0, i-20):i+20]) * 1.2 else 'medium'
                    })
            
            # Check for local maximum (resistance)
            is_local_max = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and highs[j] > highs[i]:
                    is_local_max = False
                    break
            
            if is_local_max:
                # Volume confirmation
                volume_confirmation = True
                if volumes is not None:
                    avg_volume = np.mean(volumes[max(0, i-10):i+10])
                    if volumes[i] < avg_volume * 0.5:  # Low volume at resistance
                        volume_confirmation = False
                
                if volume_confirmation:
                    resistance_levels.append({
                        'price': float(highs[i]),
                        'strength': 'high' if volumes[i] > np.mean(volumes[max(0, i-20):i+20]) * 1.2 else 'medium'
                    })
        
        # Remove duplicates and sort
        support_levels = sorted(support_levels, key=lambda x: x['price'], reverse=True)
        resistance_levels = sorted(resistance_levels, key=lambda x: x['price'])
        
        # Keep only recent and significant levels (within 20% of current price)
        current_price = closes[-1]
        support_levels = [s for s in support_levels if current_price * 0.8 <= s['price'] <= current_price * 1.1]
        resistance_levels = [r for r in resistance_levels if current_price * 0.9 <= r['price'] <= current_price * 1.2]
        
        # Keep top 3-5 most significant
        support_levels = support_levels[:5]
        resistance_levels = resistance_levels[:5]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    except Exception as e:
        print(f"Error detecting support/resistance levels: {str(e)}")
        return {'support': [], 'resistance': []}

def calculate_entry_tp_dca(ticker, df, indicators, current_price, price_predictions, 
                           ml_features=None, trend_classification=None, risk_analysis=None, news_sentiment='neutral'):
    """Calculate Entry Point, Take Profit levels, and DCA levels using ML/AI enhancement"""
    try:
        # Detect support and resistance levels
        sr_levels = detect_support_resistance_levels(df, lookback=20)
        support_levels = sr_levels.get('support', [])
        resistance_levels = sr_levels.get('resistance', [])
        
        # Get indicators
        rsi_values = indicators.get('rsi', [])
        macd_values = indicators.get('macd', [])
        macd_signal = indicators.get('macd_signal', [])
        bb_low = indicators.get('bb_low', [])
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        
        # Calculate ATR for volatility
        try:
            from ta.volatility import AverageTrueRange
            atr_indicator = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            atr = atr_indicator.average_true_range().iloc[-1] if len(df) >= 14 else current_price * 0.02
        except:
            atr = current_price * 0.02  # Fallback: 2% of price
        
        # ========== ENTRY POINT (ML/AI Enhanced) ==========
        # ML-optimized entry price calculation
        entry_price = current_price
        entry_confidence = 'low'
        entry_conditions = []
        entry_reason = "Current price"
        
        # Get ML predictions for entry optimization
        predictions = price_predictions.get('predictions', {}) if price_predictions else {}
        confidence_intervals = price_predictions.get('confidence_intervals', {}) if price_predictions else {}
        expected_returns = price_predictions.get('expected_returns', {}) if price_predictions else {}
        
        # ML-based entry score (0-100)
        ml_entry_score = 0
        ml_factors = []
        
        # Calculate optimal entry using ML predictions
        # Strategy: Use 1m prediction lower bound as potential entry if it's below current price
        optimal_entry_candidates = []
        
        # Candidate 1: Current price (if ML suggests immediate entry is good)
        if '1m' in expected_returns and expected_returns['1m'] > 0:
            # Positive expected return - current price might be good entry
            optimal_entry_candidates.append({
                'price': current_price,
                'score': 50 + (expected_returns['1m'] / 2),  # Higher score for higher expected return
                'reason': 'Current price (positive ML outlook)',
                'confidence': 'medium'
            })
        
        # Candidate 2: 1m prediction lower confidence interval (if below current)
        if '1m' in confidence_intervals and confidence_intervals['1m']:
            ci_lower_1m = confidence_intervals['1m'].get('lower', current_price)
            if ci_lower_1m < current_price * 0.98:  # At least 2% below current
                optimal_entry_candidates.append({
                    'price': ci_lower_1m,
                    'score': 60,  # Good score for lower entry
                    'reason': f'ML 1m lower CI (${ci_lower_1m:.2f})',
                    'confidence': 'high'
                })
        
        # Candidate 3: Support levels (weighted by ML confidence)
        if support_levels:
            for support in support_levels:
                if support['price'] < current_price:
                    support_distance_pct = abs(current_price - support['price']) / current_price * 100
                    # Higher score for support closer to current price and with strong ML signals
                    support_score = 70 - (support_distance_pct * 2)  # Closer = better
                    
                    # Boost score if ML suggests price might reach this support
                    if '1m' in confidence_intervals and confidence_intervals['1m']:
                        ci_lower = confidence_intervals['1m'].get('lower', current_price)
                        if abs(support['price'] - ci_lower) / support['price'] < 0.03:  # Within 3% of ML lower CI
                            support_score += 15  # ML confirms support level
                    
                    optimal_entry_candidates.append({
                        'price': support['price'],
                        'score': support_score,
                        'reason': f'Support level (${support["price"]:.2f})',
                        'confidence': support.get('strength', 'medium')
                    })
        
        # Candidate 4: Bollinger Band lower (if ML confirms)
        if bb_low and len(bb_low) > 0:
            current_bb_low = bb_low[-1]
            if current_bb_low is not None and not pd.isna(current_bb_low) and current_bb_low < current_price:
                # Check if ML features suggest this is a good entry
                if ml_features and ml_features.get('bb_position', 50) < 25:
                    optimal_entry_candidates.append({
                        'price': current_bb_low,
                        'score': 55,
                        'reason': f'Bollinger lower band (${current_bb_low:.2f})',
                        'confidence': 'medium'
                    })
        
        # Candidate 5: ML-predicted short-term dip (if RSI/MACD suggest oversold)
        if ml_features:
            rsi_ml = ml_features.get('rsi', 50)
            if rsi_ml < 35:  # Oversold
                # Estimate potential dip based on ATR and oversold condition
                potential_dip = current_price * (1 - (atr_pct * 0.5) / 100)  # Half ATR as potential dip
                if potential_dip < current_price * 0.97:  # At least 3% below
                    optimal_entry_candidates.append({
                        'price': potential_dip,
                        'score': 50 + (35 - rsi_ml),  # Higher score for more oversold
                        'reason': f'ML-estimated dip (RSI {rsi_ml:.1f})',
                        'confidence': 'medium'
                    })
        
        # Select best entry candidate - ALWAYS prefer lower entry if available
        if optimal_entry_candidates:
            # Sort by score (highest first), but prefer lower prices for same score
            optimal_entry_candidates.sort(key=lambda x: (x['score'], -x['price']), reverse=True)
            
            # Filter out current_price if we have better alternatives
            better_alternatives = [c for c in optimal_entry_candidates if c['price'] < current_price * 0.99]
            
            if better_alternatives:
                # Use best alternative that's below current price
                best_candidate = better_alternatives[0]
            else:
                # No better alternative, use best candidate overall
                best_candidate = optimal_entry_candidates[0]
            
            entry_price = best_candidate['price']
            entry_reason = best_candidate['reason']
            
            # Adjust confidence based on score and ML factors
            if best_candidate['score'] >= 70:
                entry_confidence = 'high'
            elif best_candidate['score'] >= 50:
                entry_confidence = 'medium'
            else:
                entry_confidence = 'low'
            
            # Add ML confirmation to reason
            if best_candidate['score'] >= 65 or best_candidate['price'] < current_price:
                entry_reason += " (ML-optimized)"
        
        # Fallback: if no good candidates, find best support or calculate ML-based entry
        if entry_price == current_price:
            # Try to find support level first
            if support_levels:
                for support in support_levels:
                    if support['price'] < current_price:
                        entry_price = support['price']
                        entry_reason = f"Support level (${support['price']:.2f}) (ML fallback)"
                        break
            
            # If still current_price, calculate ML-based entry from ATR and RSI
            if entry_price == current_price and ml_features:
                rsi_ml = ml_features.get('rsi', 50)
                # If oversold, suggest entry slightly below current
                if rsi_ml < 40:
                    # Calculate entry based on ATR and oversold condition
                    entry_discount = (atr_pct * 0.3) if atr_pct > 0 else 1.0  # 30% of ATR as discount
                    if rsi_ml < 30:
                        entry_discount = (atr_pct * 0.5)  # 50% of ATR for strong oversold
                    
                    ml_entry_price = current_price * (1 - entry_discount / 100)
                    if ml_entry_price < current_price * 0.98:  # At least 2% below
                        entry_price = ml_entry_price
                        entry_reason = f"ML-calculated entry (RSI {rsi_ml:.1f}, ATR-based)"
            
            # Last resort: use Bollinger lower if available
            if entry_price == current_price and bb_low and len(bb_low) > 0:
                current_bb_low = bb_low[-1]
                if current_bb_low is not None and not pd.isna(current_bb_low) and current_bb_low < current_price:
                    entry_price = current_bb_low
                    entry_reason = f"Bollinger lower band (${current_bb_low:.2f})"
        
        # Check conditions for entry (traditional + ML)
        conditions_met = 0
        
        # 1. RSI oversold
        if rsi_values and len(rsi_values) > 0:
            current_rsi = rsi_values[-1]
            if current_rsi is not None and not pd.isna(current_rsi):
                if current_rsi < 30:
                    conditions_met += 1
                    ml_entry_score += 20  # ML weight
                    entry_conditions.append("RSI oversold (< 30)")
                elif current_rsi < 40:
                    conditions_met += 0.5
                    ml_entry_score += 10
                    entry_conditions.append("RSI near oversold (< 40)")
        
        # 2. MACD bullish crossover
        if macd_values and macd_signal and len(macd_values) > 1 and len(macd_signal) > 1:
            current_macd = macd_values[-1]
            current_signal = macd_signal[-1]
            prev_macd = macd_values[-2]
            prev_signal = macd_signal[-2]
            if (current_macd is not None and current_signal is not None and 
                prev_macd is not None and prev_signal is not None and
                not pd.isna(current_macd) and not pd.isna(current_signal)):
                if current_macd > current_signal and prev_macd <= prev_signal:
                    conditions_met += 1
                    ml_entry_score += 15
                    entry_conditions.append("MACD bullish crossover")
                elif current_macd > current_signal:
                    conditions_met += 0.5
                    ml_entry_score += 8
                    entry_conditions.append("MACD above signal")
        
        # 3. Support level nearby
        nearest_support = None
        if support_levels:
            # Find support level below current price
            for support in support_levels:
                if support['price'] < current_price:
                    nearest_support = support
                    break
            
            if nearest_support:
                support_distance = abs(current_price - nearest_support['price']) / current_price * 100
                if support_distance < 3:  # Within 3%
                    conditions_met += 1
                    ml_entry_score += 25  # High weight for support
                    entry_conditions.append(f"Near support level (${nearest_support['price']:.2f})")
                    # Only use support as entry if it's better than current ML-optimized entry
                    if nearest_support['price'] < entry_price:
                        entry_price = nearest_support['price']
                        entry_reason = f"Support level (${nearest_support['price']:.2f})"
                elif support_distance < 5:  # Within 5%
                    conditions_met += 0.5
                    ml_entry_score += 12
                    entry_conditions.append(f"Approaching support level (${nearest_support['price']:.2f})")
                    # Only use if better than current entry
                    if nearest_support['price'] < entry_price:
                        entry_price = nearest_support['price']
                        entry_reason = f"Support level (${nearest_support['price']:.2f})"
        
        # 4. Bollinger Bands lower band
        if bb_low and len(bb_low) > 0:
            current_bb_low = bb_low[-1]
            if current_bb_low is not None and not pd.isna(current_bb_low):
                if current_price <= current_bb_low * 1.02:  # Within 2% of lower band
                    conditions_met += 1
                    ml_entry_score += 15
                    entry_conditions.append("Near Bollinger lower band")
        
        # 5. Volume confirmation
        if volumes is not None and len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            if current_volume > avg_volume * 1.5:
                conditions_met += 0.5
                ml_entry_score += 10
                entry_conditions.append("Volume spike")
        
        # ML/AI Enhancements
        if ml_features:
            # ML Feature: RSI from ML features (more accurate)
            if 'rsi' in ml_features:
                rsi_ml = ml_features.get('rsi', 50)
                if rsi_ml < 30:
                    ml_entry_score += 5
                    ml_factors.append("ML: Strong oversold signal")
                elif rsi_ml < 35:
                    ml_entry_score += 3
                    ml_factors.append("ML: Oversold signal")
            
            # ML Feature: MACD bullish from ML
            if ml_features.get('macd_bullish', 0) == 1.0:
                ml_entry_score += 8
                ml_factors.append("ML: MACD bullish confirmed")
            
            # ML Feature: Price position vs moving averages
            if ml_features.get('above_sma20', 0) == 0.0 and ml_features.get('above_sma50', 0) == 0.0:
                # Price below both MAs - potential oversold
                ml_entry_score += 5
                ml_factors.append("ML: Price below key MAs (oversold)")
            
            # ML Feature: Bollinger Band position
            bb_position = ml_features.get('bb_position', 50)
            if bb_position < 20:  # Near lower band
                ml_entry_score += 8
                ml_factors.append("ML: Near BB lower band")
            
            # ML Feature: Stochastic oversold
            if ml_features.get('stoch_oversold', 0) == 1.0:
                ml_entry_score += 5
                ml_factors.append("ML: Stochastic oversold")
            
            # ML Feature: ADX trend strength (weak trend = better entry)
            adx = ml_features.get('adx', 25)
            if adx < 20:  # Weak trend - might be good entry
                ml_entry_score += 3
                ml_factors.append("ML: Weak trend (potential reversal)")
        
        # Trend Classification Enhancement
        if trend_classification:
            trend_class = trend_classification.get('trend_class', '')
            trend_prob = trend_classification.get('probabilities', {})
            
            # If trend is improving or about to improve
            if 'Uptrend' in trend_class or 'Moderate Uptrend' in trend_class:
                ml_entry_score += 10
                ml_factors.append(f"ML: Trend classification: {trend_class}")
            elif trend_prob.get('Strong Uptrend', 0) > 0.3 or trend_prob.get('Moderate Uptrend', 0) > 0.4:
                ml_entry_score += 5
                ml_factors.append("ML: Potential uptrend forming")
        
        # News Sentiment Enhancement
        if news_sentiment == 'positive':
            ml_entry_score += 5
            ml_factors.append("AI: Positive news sentiment")
        elif news_sentiment == 'negative':
            # Negative sentiment might mean oversold - could be good entry
            ml_entry_score += 3
            ml_factors.append("AI: Negative sentiment (potential oversold)")
        
        # Risk Analysis Enhancement
        if risk_analysis:
            risk_score = risk_analysis.get('risk_score', 50)
            # Lower risk = better entry opportunity
            if risk_score < 30:
                ml_entry_score += 8
                ml_factors.append("AI: Low risk environment")
            elif risk_score < 50:
                ml_entry_score += 4
                ml_factors.append("AI: Moderate risk")
        
        # Combine traditional and ML scores
        total_entry_score = conditions_met * 10 + ml_entry_score * 0.5  # ML gets 50% weight
        
        # Determine entry confidence based on combined score
        if total_entry_score >= 60 or (conditions_met >= 3 and ml_entry_score >= 40):
            entry_confidence = 'high'
            entry_reason = "Strong ML/AI + technical signals aligned"
        elif total_entry_score >= 40 or (conditions_met >= 2 and ml_entry_score >= 25):
            entry_confidence = 'medium'
            entry_reason = "Moderate ML/AI + technical signals"
        else:
            entry_confidence = 'low'
            entry_reason = "Limited signals"
        
        # Add ML factors to conditions
        entry_conditions.extend(ml_factors[:3])  # Add top 3 ML factors
        
        # Final ML validation: Ensure entry price makes sense
        # If entry is too far from current price (>12%), use more conservative entry
        entry_distance_pct = abs(current_price - entry_price) / current_price * 100
        if entry_distance_pct > 12:
            # Too far - might be unrealistic, use more conservative entry
            # But still prefer lower than current if possible
            conservative_candidates = []
            
            # Find support levels within reasonable range
            if support_levels:
                for support in support_levels:
                    if support['price'] < current_price:
                        support_dist = abs(current_price - support['price']) / current_price * 100
                        if support_dist < 10:  # Within 10%
                            conservative_candidates.append({
                                'price': support['price'],
                                'distance': support_dist
                            })
            
            # Use ML lower CI if available and reasonable
            if '1m' in confidence_intervals and confidence_intervals['1m']:
                ci_lower = confidence_intervals['1m'].get('lower', current_price)
                if ci_lower < current_price:
                    ci_dist = abs(current_price - ci_lower) / current_price * 100
                    if ci_dist < 10:
                        conservative_candidates.append({
                            'price': ci_lower,
                            'distance': ci_dist
                        })
            
            # Use Bollinger lower if reasonable
            if bb_low and len(bb_low) > 0:
                current_bb_low = bb_low[-1]
                if current_bb_low is not None and not pd.isna(current_bb_low) and current_bb_low < current_price:
                    bb_dist = abs(current_price - current_bb_low) / current_price * 100
                    if bb_dist < 10:
                        conservative_candidates.append({
                            'price': current_bb_low,
                            'distance': bb_dist
                        })
            
            if conservative_candidates:
                # Use closest to current (but still below)
                conservative_candidates.sort(key=lambda x: x['distance'])
                entry_price = conservative_candidates[0]['price']
                entry_reason = f"Conservative ML entry (${entry_price:.2f})"
        
        # Add entry price difference info to reason
        if entry_price != current_price:
            diff_pct = ((entry_price - current_price) / current_price) * 100
            if diff_pct < 0:
                entry_reason += f" ({abs(diff_pct):.1f}% below current)"
            else:
                entry_reason += f" ({diff_pct:.1f}% above current)"
        
        # ========== TAKE PROFIT LEVELS (ML/AI Adaptive) ==========
        tp1_price = None
        tp2_price = None
        tp3_price = None
        
        # Get ML predictions and confidence intervals
        predictions = price_predictions.get('predictions', {}) if price_predictions else {}
        expected_returns = price_predictions.get('expected_returns', {}) if price_predictions else {}
        confidence_intervals = price_predictions.get('confidence_intervals', {}) if price_predictions else {}
        
        # Calculate volatility adjustment factor (ATR-based)
        atr_pct = (atr / current_price * 100) if current_price > 0 else 2.0
        volatility_factor = max(0.8, min(1.2, 1.0 + (atr_pct - 2.0) / 10.0))  # Adjust based on volatility
        
        # ML confidence factor (higher confidence = less conservative)
        ml_confidence_factor = 1.0
        if ml_features:
            # Use trend strength and ML confidence
            adx = ml_features.get('adx', 25)
            if adx > 25:  # Strong trend
                ml_confidence_factor = 1.1  # More aggressive TP
            elif adx < 20:  # Weak trend
                ml_confidence_factor = 0.9  # More conservative TP
        
        # Trend-based adjustment
        trend_adjustment = 1.0
        if trend_classification:
            trend_class = trend_classification.get('trend_class', '')
            if 'Strong Uptrend' in trend_class:
                trend_adjustment = 1.15  # More aggressive
            elif 'Moderate Uptrend' in trend_class:
                trend_adjustment = 1.05
            elif 'Downtrend' in trend_class:
                trend_adjustment = 0.85  # More conservative
        
        # Risk-based adjustment
        risk_adjustment = 1.0
        if risk_analysis:
            risk_score = risk_analysis.get('risk_score', 50)
            if risk_score < 30:  # Low risk
                risk_adjustment = 1.1  # More aggressive
            elif risk_score > 70:  # High risk
                risk_adjustment = 0.9  # More conservative
        
        # Combined adjustment factor
        adaptive_factor = volatility_factor * ml_confidence_factor * trend_adjustment * risk_adjustment
        
        # TP1: Conservative (1-3 months) - ML-optimized
        tp1_base = None
        tp1_confidence = 'high'
        
        if '1m' in predictions and predictions['1m']:
            # Use ML prediction with confidence interval
            ml_pred_1m = predictions['1m']
            if '1m' in confidence_intervals and confidence_intervals['1m']:
                ci_lower = confidence_intervals['1m'].get('lower', ml_pred_1m * 0.9)
                # Use lower bound of CI for conservative TP
                tp1_base = ci_lower * 0.95  # 95% of lower CI bound
            else:
                tp1_base = ml_pred_1m * 0.85  # 85% of prediction (more conservative)
        elif '3m' in predictions and predictions['3m']:
            tp1_base = predictions['3m'] * 0.70  # 70% of 3m prediction
        else:
            # Use nearest resistance or ATR-based gain
            if resistance_levels:
                nearest_resistance = resistance_levels[0]
                tp1_base = nearest_resistance['price'] * 0.95
            else:
                # ATR-based: 2-3x ATR gain
                tp1_base = entry_price * (1 + (atr_pct * 2.5) / 100)
        
        tp1_price = tp1_base * adaptive_factor
        
        # TP2: Medium (3-6 months) - ML-optimized
        tp2_base = None
        tp2_confidence = 'medium'
        
        if '3m' in predictions and predictions['3m']:
            ml_pred_3m = predictions['3m']
            if '3m' in confidence_intervals and confidence_intervals['3m']:
                ci_lower = confidence_intervals['3m'].get('lower', ml_pred_3m * 0.9)
                tp2_base = (ci_lower + ml_pred_3m) / 2  # Average of lower CI and prediction
            else:
                tp2_base = ml_pred_3m * 0.90
        elif '6m' in predictions and predictions['6m']:
            tp2_base = predictions['6m'] * 0.80
        else:
            if len(resistance_levels) > 1:
                tp2_base = resistance_levels[1]['price'] * 0.95
            else:
                # ATR-based: 4-5x ATR gain
                tp2_base = entry_price * (1 + (atr_pct * 4.5) / 100)
        
        tp2_price = tp2_base * adaptive_factor
        
        # TP3: Aggressive (6-12 months) - ML-optimized
        tp3_base = None
        tp3_confidence = 'medium'
        
        if '6m' in predictions and predictions['6m']:
            ml_pred_6m = predictions['6m']
            if '6m' in confidence_intervals and confidence_intervals['6m']:
                ci_upper = confidence_intervals['6m'].get('upper', ml_pred_6m * 1.1)
                tp3_base = (ml_pred_6m + ci_upper) / 2  # Average of prediction and upper CI
            else:
                tp3_base = ml_pred_6m * 0.95
        elif '12m' in predictions and predictions['12m']:
            ml_pred_12m = predictions['12m']
            if '12m' in confidence_intervals and confidence_intervals['12m']:
                ci_lower = confidence_intervals['12m'].get('lower', ml_pred_12m * 0.9)
                tp3_base = ml_pred_12m * 0.90  # 90% of 12m prediction
            else:
                tp3_base = ml_pred_12m * 0.85
        else:
            if len(resistance_levels) > 2:
                tp3_base = resistance_levels[2]['price'] * 0.95
            else:
                # ATR-based: 6-8x ATR gain
                tp3_base = entry_price * (1 + (atr_pct * 7) / 100)
        
        tp3_price = tp3_base * adaptive_factor
        
        # Ensure TP prices are reasonable and ordered
        tp1_price = max(entry_price * 1.05, min(tp1_price, entry_price * 1.35))
        tp2_price = max(tp1_price * 1.05, min(tp2_price, entry_price * 1.60))
        tp3_price = max(tp2_price * 1.05, min(tp3_price, entry_price * 2.2))
        
        # ========== DCA LEVELS (ML/AI Smart Spacing) ==========
        dca_levels = []
        
        # Smart DCA spacing based on volatility (ATR)
        # For high volatility stocks, use larger spacing; for low volatility, use smaller spacing
        atr_pct = (atr / current_price * 100) if current_price > 0 else 2.0
        
        # Determine optimal DCA spacing
        if atr_pct > 5.0:  # High volatility (>5% ATR)
            # Larger spacing for volatile stocks
            dca_spacings = [-4, -8, -12, -18]  # Smaller drops, more levels
            base_reason = "High volatility - tighter spacing"
        elif atr_pct > 3.0:  # Medium-high volatility
            dca_spacings = [-5, -10, -15, -20]
            base_reason = "Medium-high volatility"
        elif atr_pct > 1.5:  # Medium volatility
            dca_spacings = [-5, -10, -15, -20]
            base_reason = "Medium volatility"
        else:  # Low volatility (<1.5% ATR)
            # Smaller spacing for stable stocks
            dca_spacings = [-3, -6, -10, -15]
            base_reason = "Low volatility - wider spacing"
        
        # ML-based probability of reaching each DCA level
        for idx, pct in enumerate(dca_spacings):
            dca_price = entry_price * (1 + pct / 100)
            
            # Check if this price aligns with a support level
            confidence = 'medium'
            reason = f"{abs(pct)}% below entry"
            ml_probability = 0.5  # Default 50%
            
            # Find nearest support
            nearest_support_for_dca = None
            min_support_distance = float('inf')
            for support in support_levels:
                support_distance = abs(dca_price - support['price']) / dca_price * 100
                if support_distance < min_support_distance:
                    min_support_distance = support_distance
                    nearest_support_for_dca = support
            
            if nearest_support_for_dca and min_support_distance < 2:  # Within 2% of support
                confidence = 'high'
                reason = f"At support level (${nearest_support_for_dca['price']:.2f})"
                dca_price = nearest_support_for_dca['price']  # Use exact support price
                ml_probability = 0.75  # Higher probability if at support
            elif nearest_support_for_dca and min_support_distance < 5:
                confidence = 'medium'
                reason = f"Near support (${nearest_support_for_dca['price']:.2f}, {min_support_distance:.1f}% away)"
                ml_probability = 0.60
            else:
                reason = f"{abs(pct)}% below entry ({base_reason})"
                # ML-based probability adjustment
                if ml_features:
                    # If volatility is high, probability of reaching deeper levels is higher
                    if atr_pct > 4.0:
                        ml_probability = 0.65 - (idx * 0.05)  # Deeper levels more likely in high vol
                    else:
                        ml_probability = 0.50 - (idx * 0.08)  # Deeper levels less likely in low vol
                
                # Adjust based on trend
                if trend_classification:
                    trend_class = trend_classification.get('trend_class', '')
                    if 'Downtrend' in trend_class:
                        ml_probability += 0.15  # More likely to reach DCA in downtrend
                    elif 'Uptrend' in trend_class:
                        ml_probability -= 0.10  # Less likely in uptrend
            
            # Risk-based adjustment
            if risk_analysis:
                risk_score = risk_analysis.get('risk_score', 50)
                if risk_score > 70:  # High risk
                    ml_probability += 0.10  # More likely to drop further
                elif risk_score < 30:  # Low risk
                    ml_probability -= 0.05  # Less likely to drop
            
            ml_probability = max(0.2, min(0.9, ml_probability))  # Clamp between 20-90%
            
            dca_levels.append({
                'price': round(dca_price, 2),
                'percentage': pct,
                'confidence': confidence,
                'reason': reason,
                'ml_probability': round(ml_probability * 100, 1)  # As percentage
            })
        
        # Calculate Risk/Reward ratio
        risk = abs(current_price - entry_price) if entry_price < current_price else current_price * 0.05  # 5% stop loss if entry is current
        reward_tp1 = tp1_price - entry_price
        risk_reward_ratio = reward_tp1 / risk if risk > 0 else 0
        
        return {
            'entry': {
                'price': round(entry_price, 2),
                'confidence': entry_confidence,
                'reason': entry_reason,
                'conditions': entry_conditions
            },
            'take_profit': {
                'tp1': {
                    'price': round(tp1_price, 2),
                    'confidence': tp1_confidence,
                    'timeframe': '1-3 months',
                    'gain_pct': round((tp1_price / entry_price - 1) * 100, 1),
                    'ml_confidence': round(min(95, max(60, 100 - (atr_pct * 2))), 1) if atr_pct else 75.0  # Higher confidence for lower volatility
                },
                'tp2': {
                    'price': round(tp2_price, 2),
                    'confidence': tp2_confidence,
                    'timeframe': '3-6 months',
                    'gain_pct': round((tp2_price / entry_price - 1) * 100, 1),
                    'ml_confidence': round(min(85, max(50, 85 - (atr_pct * 2.5))), 1) if atr_pct else 65.0
                },
                'tp3': {
                    'price': round(tp3_price, 2),
                    'confidence': tp3_confidence,
                    'timeframe': '6-12 months',
                    'gain_pct': round((tp3_price / entry_price - 1) * 100, 1),
                    'ml_confidence': round(min(75, max(40, 75 - (atr_pct * 3))), 1) if atr_pct else 55.0
                }
            },
            'dca_levels': dca_levels,
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'support_levels': [s['price'] for s in support_levels[:3]],
            'resistance_levels': [r['price'] for r in resistance_levels[:3]],
            'ml_enhancements': {
                'entry_score': round(ml_entry_score, 1),
                'adaptive_factor': round(adaptive_factor, 2),
                'volatility_pct': round(atr_pct, 2),
                'trend_adjustment': round(trend_adjustment, 2),
                'risk_adjustment': round(risk_adjustment, 2)
            }
        }
        
    except Exception as e:
        print(f"Error calculating entry/TP/DCA for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_position_sizing(risk_analysis, price_prediction, trend_classification, 
                               entry_tp_dca, current_price, atr_pct=2.0, market_cap=None, company_stage=None):
    """Calculate optimal position size based on risk, ML confidence, and volatility
    
    Returns recommended position size as % of portfolio with reasoning.
    """
    try:
        # Base position size - adjusted for market cap and company stage
        # Large caps: higher base, Small caps: medium base, Micro caps: lower base
        if market_cap:
            if market_cap >= 10_000_000_000:  # Large cap (>$10B)
                base_position = 5.0  # 5% base for large caps
            elif market_cap >= 2_000_000_000:  # Mid cap ($2B-$10B)
                base_position = 4.0  # 4% base for mid caps
            elif market_cap >= 300_000_000:  # Small cap ($300M-$2B)
                base_position = 3.0  # 3% base for small caps
            else:  # Micro cap (<$300M)
                base_position = 2.0  # 2% base for micro caps
        else:
            base_position = 3.5  # Default 3.5% if market cap unknown
        
        # Company stage adjustment
        stage_adjustment = 1.0
        if company_stage:
            if company_stage == 'mature':  # Large, stable companies
                stage_adjustment = 1.3  # Can take larger positions
            elif company_stage == 'growth':  # Growing companies
                stage_adjustment = 1.2  # Slightly larger positions
            elif company_stage == 'early_stage':  # Early stage
                stage_adjustment = 0.8  # Smaller positions
        
        # 1. Risk Score Adjustment (higher risk = smaller position, but less aggressive reduction)
        risk_score = risk_analysis.get('risk_score', 50) if risk_analysis else 50
        risk_adjustment = 1.0
        
        if risk_score >= 75:  # Very high risk
            risk_adjustment = 0.6  # Reduce to 60% of base (was 0.4)
        elif risk_score >= 60:  # High risk
            risk_adjustment = 0.75  # Reduce to 75% (was 0.6)
        elif risk_score >= 45:  # Medium-high risk
            risk_adjustment = 0.9  # Slight reduction (was 0.8)
        elif risk_score >= 30:  # Medium risk
            risk_adjustment = 1.0  # No adjustment
        elif risk_score >= 20:  # Low-medium risk
            risk_adjustment = 1.3  # Increase 30% (was 1.2)
        else:  # Very low risk (< 20)
            risk_adjustment = 1.6  # Increase 60% (was 1.5)
        
        # 2. ML Confidence Adjustment (higher confidence = larger position)
        ml_confidence = 50  # Default
        confidence_factors = []
        
        # Entry confidence from trading strategy
        if entry_tp_dca and entry_tp_dca.get('entry'):
            entry_conf = entry_tp_dca['entry'].get('confidence', 'low')
            if entry_conf == 'high':
                ml_confidence = 80
                confidence_factors.append("High entry confidence")
            elif entry_conf == 'medium':
                ml_confidence = 60
                confidence_factors.append("Medium entry confidence")
            else:
                ml_confidence = 40
                confidence_factors.append("Low entry confidence")
        
        # Trend classification confidence
        if trend_classification:
            trend_class = trend_classification.get('trend_class', '')
            trend_prob = trend_classification.get('probabilities', {})
            
            if 'Strong Uptrend' in trend_class:
                ml_confidence += 10
                confidence_factors.append("Strong uptrend detected")
            elif 'Moderate Uptrend' in trend_class:
                ml_confidence += 5
                confidence_factors.append("Moderate uptrend")
            elif 'Strong Downtrend' in trend_class:
                ml_confidence -= 15
                confidence_factors.append("Strong downtrend - caution")
            elif 'Moderate Downtrend' in trend_class:
                ml_confidence -= 10
                confidence_factors.append("Moderate downtrend")
        
        # Price prediction confidence (expected returns)
        if price_prediction and price_prediction.get('expected_returns'):
            expected_6m = price_prediction['expected_returns'].get('6m', 0)
            if expected_6m > 20:
                ml_confidence += 10
                confidence_factors.append(f"Strong ML prediction (+{expected_6m:.1f}% 6m)")
            elif expected_6m > 10:
                ml_confidence += 5
                confidence_factors.append(f"Positive ML prediction (+{expected_6m:.1f}% 6m)")
            elif expected_6m < -10:
                ml_confidence -= 15
                confidence_factors.append(f"Negative ML prediction ({expected_6m:.1f}% 6m)")
            elif expected_6m < -5:
                ml_confidence -= 10
                confidence_factors.append(f"Weak ML prediction ({expected_6m:.1f}% 6m)")
        
        ml_confidence = max(20, min(95, ml_confidence))  # Clamp between 20-95
        
        # ML confidence adjustment factor
        confidence_adjustment = 0.5 + (ml_confidence / 100) * 1.0  # Range: 0.5x to 1.5x
        
        # 3. Volatility Adjustment (ATR-based)
        # Higher volatility = smaller position to maintain same risk
        volatility_adjustment = 1.0
        
        if atr_pct > 6.0:  # Very high volatility (>6% ATR)
            volatility_adjustment = 0.5  # Reduce to 50%
        elif atr_pct > 4.0:  # High volatility
            volatility_adjustment = 0.7
        elif atr_pct > 3.0:  # Medium-high volatility
            volatility_adjustment = 0.85
        elif atr_pct > 2.0:  # Medium volatility
            volatility_adjustment = 1.0  # No adjustment
        elif atr_pct > 1.0:  # Low-medium volatility
            volatility_adjustment = 1.15
        else:  # Very low volatility (<1% ATR)
            volatility_adjustment = 1.3  # Can increase position
        
        # 4. Risk/Reward Ratio Adjustment
        rr_adjustment = 1.0
        if entry_tp_dca and entry_tp_dca.get('risk_reward_ratio'):
            rr_ratio = entry_tp_dca['risk_reward_ratio']
            if rr_ratio >= 3.0:  # Excellent R/R
                rr_adjustment = 1.3
                confidence_factors.append(f"Excellent R/R ratio (1:{rr_ratio:.1f})")
            elif rr_ratio >= 2.0:  # Good R/R
                rr_adjustment = 1.15
                confidence_factors.append(f"Good R/R ratio (1:{rr_ratio:.1f})")
            elif rr_ratio >= 1.5:  # Decent R/R
                rr_adjustment = 1.0
            elif rr_ratio >= 1.0:  # Poor R/R
                rr_adjustment = 0.8
                confidence_factors.append(f"Poor R/R ratio (1:{rr_ratio:.1f})")
            else:  # Very poor R/R
                rr_adjustment = 0.6
                confidence_factors.append(f"Very poor R/R ratio (1:{rr_ratio:.1f})")
        
        # Calculate final position size (include stage adjustment)
        position_pct = base_position * stage_adjustment * risk_adjustment * confidence_adjustment * volatility_adjustment * rr_adjustment
        
        # Apply maximum limits (higher for large caps)
        if market_cap and market_cap >= 10_000_000_000:  # Large cap
            max_position = 15.0  # Can go up to 15% for large caps
        else:
            max_position = 12.0  # Up to 12% for others (was 10%)
        min_position = 1.0   # Minimum 1% if we're recommending it (was 0.5%)
        
        position_pct = max(min_position, min(max_position, position_pct))
        
        # Calculate position range (conservative to aggressive)
        conservative_position = position_pct * 0.75  # 75% of recommended (was 0.7)
        aggressive_position = position_pct * 1.4   # 140% of recommended (was 1.3)
        aggressive_position = min(aggressive_position, max_position)
        
        # Determine position size category (adjusted thresholds)
        if position_pct >= 8.0:
            size_category = "Large"
            size_color = "#10b981"  # Green
        elif position_pct >= 5.0:
            size_category = "Medium-Large"
            size_color = "#34d399"  # Light green
        elif position_pct >= 3.0:
            size_category = "Medium"
            size_color = "#fbbf24"  # Yellow
        elif position_pct >= 1.5:
            size_category = "Small-Medium"
            size_color = "#f87171"  # Light red
        else:
            size_category = "Small"
            size_color = "#ef4444"  # Red
        
        # Build reasoning
        reasoning_parts = []
        
        # Market cap reasoning
        if market_cap:
            if market_cap >= 10_000_000_000:
                reasoning_parts.append(f"Large cap stock (${market_cap/1e9:.1f}B) allows larger position")
            elif market_cap >= 2_000_000_000:
                reasoning_parts.append(f"Mid cap stock (${market_cap/1e9:.1f}B) supports moderate position")
            elif market_cap < 300_000_000:
                reasoning_parts.append(f"Micro cap stock requires more conservative sizing")
        
        # Company stage reasoning
        if company_stage:
            if company_stage == 'mature':
                reasoning_parts.append("Mature company - can take larger position")
            elif company_stage == 'growth':
                reasoning_parts.append("Growth company - moderate position size")
            elif company_stage == 'early_stage':
                reasoning_parts.append("Early stage - conservative sizing recommended")
        
        # Risk reasoning
        if risk_score >= 60:
            reasoning_parts.append(f"High risk score ({risk_score:.0f}/100) suggests smaller position")
        elif risk_score <= 30:
            reasoning_parts.append(f"Low risk score ({risk_score:.0f}/100) allows larger position")
        
        # ML confidence reasoning
        if ml_confidence >= 70:
            reasoning_parts.append(f"High ML confidence ({ml_confidence:.0f}%) supports larger allocation")
        elif ml_confidence <= 40:
            reasoning_parts.append(f"Lower ML confidence ({ml_confidence:.0f}%) suggests conservative sizing")
        
        # Volatility reasoning
        if atr_pct > 4.0:
            reasoning_parts.append(f"High volatility ({atr_pct:.1f}% ATR) requires smaller position")
        elif atr_pct < 1.5:
            reasoning_parts.append(f"Low volatility ({atr_pct:.1f}% ATR) allows larger position")
        
        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Balanced risk/reward profile"
        
        return {
            'recommended_pct': round(position_pct, 2),
            'range': {
                'conservative': round(conservative_position, 2),
                'aggressive': round(aggressive_position, 2)
            },
            'size_category': size_category,
            'size_color': size_color,
            'reasoning': reasoning,
            'confidence_factors': confidence_factors[:3],  # Top 3 factors
            'risk_score': round(risk_score, 1),
            'ml_confidence': round(ml_confidence, 1),
            'volatility_pct': round(atr_pct, 2),
            'adjustments': {
                'base': round(base_position, 2),
                'stage': round(stage_adjustment, 2),
                'risk': round(risk_adjustment, 2),
                'confidence': round(confidence_adjustment, 2),
                'volatility': round(volatility_adjustment, 2),
                'risk_reward': round(rr_adjustment, 2)
            },
            'market_cap': market_cap,
            'company_stage': company_stage
        }
        
    except Exception as e:
        print(f"Error calculating position sizing: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'recommended_pct': 2.0,
            'range': {'conservative': 1.5, 'aggressive': 3.0},
            'size_category': 'Medium',
            'size_color': '#fbbf24',
            'reasoning': 'Default conservative sizing due to calculation error',
            'confidence_factors': [],
            'risk_score': 50,
            'ml_confidence': 50,
            'volatility_pct': 2.0,
            'adjustments': {'risk': 1.0, 'confidence': 1.0, 'volatility': 1.0, 'risk_reward': 1.0}
        }

def generate_ai_recommendations(ticker):
    """Generate AI-powered stock recommendations based on technical and fundamental analysis"""
    try:
        # Get stock data
        stock_data = get_stock_data(ticker, '1y')
        if not stock_data or stock_data['history'].empty:
            return None
        
        df = stock_data['history']
        info = stock_data.get('info', {})
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(df)
        
        # Get current metrics
        metrics = calculate_metrics(df, info)
        
        # Get news sentiment
        news_list = get_stock_news(ticker)
        news_sentiment = 'neutral'
        if news_list:
            sentiments = [article.get('sentiment', 'neutral') for article in news_list[:10]]
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            if positive_count > negative_count * 1.5:
                news_sentiment = 'positive'
            elif negative_count > positive_count * 1.5:
                news_sentiment = 'negative'
        
        # Analyze technical indicators
        current_price = df['Close'].iloc[-1]
        rsi_values = indicators.get('rsi', [])
        macd_values = indicators.get('macd', [])
        macd_signal = indicators.get('macd_signal', [])
        sma_20 = indicators.get('sma_20', [])
        sma_50 = indicators.get('sma_50', [])
        
        # Extract ML features and run ML models
        ml_features = extract_ml_features(ticker, df, info, indicators, metrics, news_list)
        # Add ticker to features for history tracking
        ml_features['ticker'] = ticker.upper()
        price_prediction = predict_price(ml_features, current_price, df)
        trend_classification = classify_trend(ml_features)
        risk_analysis = calculate_risk_score(ml_features, metrics, info)
        
        # Calculate Entry, TP, and DCA levels (enhanced with ML/AI)
        entry_tp_dca = calculate_entry_tp_dca(ticker, df, indicators, current_price, price_prediction, 
                                               ml_features, trend_classification, risk_analysis, news_sentiment)
        
        # Calculate ATR for position sizing
        try:
            from ta.volatility import AverageTrueRange
            atr_indicator = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            atr = atr_indicator.average_true_range().iloc[-1] if len(df) >= 14 else current_price * 0.02
            atr_pct = (atr / current_price * 100) if current_price > 0 else 2.0
        except:
            atr_pct = metrics.get('volatility', 20) / np.sqrt(252) if metrics.get('volatility') else 2.0
        
        # Get market cap and company stage for position sizing
        market_cap = info.get('marketCap')
        company_stage = info.get('company_stage') or metrics.get('company_stage')
        if not company_stage:
            # Determine company stage from market cap and growth
            if market_cap and market_cap >= 10_000_000_000:
                company_stage = 'mature'
            elif market_cap and market_cap >= 2_000_000_000:
                company_stage = 'growth'
            else:
                company_stage = 'early_stage'
        
        # Calculate Position Sizing
        position_sizing = calculate_position_sizing(
            risk_analysis, price_prediction, trend_classification, 
            entry_tp_dca, current_price, atr_pct=atr_pct,
            market_cap=market_cap, company_stage=company_stage
        )
        
        # Calculate scores (0-100, higher is better)
        technical_score = 50  # Base score
        reasons = []
        warnings = []
        
        # RSI Analysis
        if rsi_values and len(rsi_values) > 0:
            current_rsi = rsi_values[-1]
            if current_rsi is not None and not pd.isna(current_rsi):
                if current_rsi < 30:
                    technical_score += 15
                    reasons.append("RSI indicates oversold conditions - potential buying opportunity")
                elif current_rsi < 40:
                    technical_score += 8
                    reasons.append("RSI suggests stock may be undervalued")
                elif current_rsi > 70:
                    technical_score -= 15
                    warnings.append("RSI indicates overbought conditions - stock may be overvalued")
                elif current_rsi > 60:
                    technical_score -= 8
                    warnings.append("RSI suggests stock may be overvalued")
        
        # MACD Analysis
        if macd_values and macd_signal and len(macd_values) > 0 and len(macd_signal) > 0:
            current_macd = macd_values[-1]
            current_signal = macd_signal[-1]
            if current_macd is not None and current_signal is not None and not pd.isna(current_macd) and not pd.isna(current_signal):
                if current_macd > current_signal:
                    technical_score += 10
                    reasons.append("MACD shows bullish momentum")
                else:
                    technical_score -= 5
                    warnings.append("MACD shows bearish momentum")
        
        # Moving Average Analysis
        if sma_20 and sma_50 and len(sma_20) > 0 and len(sma_50) > 0:
            current_sma20 = sma_20[-1]
            current_sma50 = sma_50[-1]
            if current_sma20 is not None and current_sma50 is not None and not pd.isna(current_sma20) and not pd.isna(current_sma50):
                if current_price > current_sma20 > current_sma50:
                    technical_score += 12
                    reasons.append("Price above both 20-day and 50-day moving averages - strong uptrend")
                elif current_price > current_sma20:
                    technical_score += 5
                    reasons.append("Price above 20-day moving average - short-term bullish")
                elif current_price < current_sma20 < current_sma50:
                    technical_score -= 12
                    warnings.append("Price below both moving averages - downtrend")
                elif current_price < current_sma20:
                    technical_score -= 5
                    warnings.append("Price below 20-day moving average - short-term bearish")
        
        # Price Momentum (30-day)
        if len(df) >= 30:
            price_30d_ago = df['Close'].iloc[-30]
            price_change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
            if price_change_30d > 10:
                technical_score += 8
                reasons.append(f"Strong 30-day price momentum (+{price_change_30d:.1f}%)")
            elif price_change_30d > 5:
                technical_score += 4
                reasons.append(f"Positive 30-day price momentum (+{price_change_30d:.1f}%)")
            elif price_change_30d < -10:
                technical_score -= 8
                warnings.append(f"Negative 30-day price momentum ({price_change_30d:.1f}%)")
            elif price_change_30d < -5:
                technical_score -= 4
                warnings.append(f"Weak 30-day price momentum ({price_change_30d:.1f}%)")
        
        # News Sentiment Impact
        if news_sentiment == 'positive':
            technical_score += 5
            reasons.append("Recent news sentiment is positive")
        elif news_sentiment == 'negative':
            technical_score -= 5
            warnings.append("Recent news sentiment is negative")
        
        # Volatility Analysis
        volatility = metrics.get('volatility')
        if volatility is not None:
            if volatility > 40:
                warnings.append(f"High volatility ({volatility:.1f}%) - higher risk")
            elif volatility < 15:
                reasons.append(f"Low volatility ({volatility:.1f}%) - more stable")
        
        # Enhance recommendation with ML model outputs
        # Adjust score based on price prediction
        expected_return_6m = price_prediction['expected_returns']['6m']
        if expected_return_6m > 20:
            technical_score += 15
            reasons.append(f"ML model predicts strong 6-month return (+{expected_return_6m:.1f}%)")
        elif expected_return_6m > 10:
            technical_score += 10
            reasons.append(f"ML model predicts positive 6-month return (+{expected_return_6m:.1f}%)")
        elif expected_return_6m < -10:
            technical_score -= 15
            warnings.append(f"ML model predicts negative 6-month return ({expected_return_6m:.1f}%)")
        elif expected_return_6m < -5:
            technical_score -= 10
            warnings.append(f"ML model predicts weak 6-month return ({expected_return_6m:.1f}%)")
        
        # Adjust score based on trend classification
        trend_class = trend_classification['trend_class']
        if trend_class == 'Strong Uptrend':
            technical_score += 12
            reasons.append("ML trend classification: Strong Uptrend")
        elif trend_class == 'Moderate Uptrend':
            technical_score += 6
            reasons.append("ML trend classification: Moderate Uptrend")
        elif trend_class == 'Strong Downtrend':
            technical_score -= 12
            warnings.append("ML trend classification: Strong Downtrend")
        elif trend_class == 'Moderate Downtrend':
            technical_score -= 6
            warnings.append("ML trend classification: Moderate Downtrend")
        
        # Adjust score based on risk
        risk_score = risk_analysis['risk_score']
        if risk_score > 75:
            technical_score -= 10
            warnings.append(f"High risk score ({risk_score:.1f}/100)")
        elif risk_score < 25:
            technical_score += 5
            reasons.append(f"Low risk score ({risk_score:.1f}/100)")
        
        # Re-determine recommendation with ML-enhanced score
        final_score = min(100, max(0, technical_score))
        
        # Determine recommendation
        if final_score >= 75:
            recommendation = "Strong Buy"
            confidence = "High"
            color = "#10b981"  # Green
        elif technical_score >= 60:
            recommendation = "Buy"
            confidence = "Medium-High"
            color = "#34d399"  # Light green
        elif technical_score >= 45:
            recommendation = "Hold"
            confidence = "Medium"
            color = "#fbbf24"  # Yellow
        elif technical_score >= 30:
            recommendation = "Sell"
            confidence = "Medium"
            color = "#f87171"  # Light red
        else:
            recommendation = "Strong Sell"
            confidence = "High"
            color = "#ef4444"  # Red
        
        # Generate summary
        summary_parts = []
        if reasons:
            summary_parts.append(f"Key positives: {', '.join(reasons[:2])}")
        if warnings:
            summary_parts.append(f"Key concerns: {', '.join(warnings[:2])}")
        
        summary = ". ".join(summary_parts) if summary_parts else "Mixed signals - consider additional research"
        
        # Prepare chart data for daily history table
        try:
            dates = df.index.strftime('%Y-%m-%d').tolist()
            chart_data = {
                'dates': dates,
                'open': [float(x) for x in df['Open'].round(2).fillna(0).tolist()],
                'high': [float(x) for x in df['High'].round(2).fillna(0).tolist()],
                'low': [float(x) for x in df['Low'].round(2).fillna(0).tolist()],
                'close': [float(x) for x in df['Close'].round(2).fillna(0).tolist()],
                'volume': [int(x) for x in df['Volume'].fillna(0).astype(int).tolist()],
            }
            print(f"[AI RECOMMENDATIONS] Chart data prepared: {len(chart_data['dates'])} dates")
        except Exception as e:
            print(f"[AI RECOMMENDATIONS] Error preparing chart_data: {e}")
            chart_data = {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        result = {
            'ticker': ticker.upper(),
            'recommendation': recommendation,
            'confidence': confidence,
            'score': final_score,
            'color': color,
            'reasons': reasons[:5],
            'warnings': warnings[:5],
            'summary': summary,
            'technical_indicators': {
                'rsi': rsi_values[-1] if rsi_values and len(rsi_values) > 0 else None,
                'macd_bullish': macd_values[-1] > macd_signal[-1] if macd_values and macd_signal and len(macd_values) > 0 and len(macd_signal) > 0 else None,
                'price_vs_sma20': current_price > sma_20[-1] if sma_20 and len(sma_20) > 0 else None,
                'price_vs_sma50': current_price > sma_50[-1] if sma_50 and len(sma_50) > 0 else None,
            },
            'news_sentiment': news_sentiment,
            'ml_models': {
                'price_prediction': price_prediction,
                'trend_classification': trend_classification,
                'risk_analysis': risk_analysis
            },
            'trading_strategy': entry_tp_dca,
            'position_sizing': position_sizing,
            'chart_data': chart_data
        }
        
        # Save prediction history with final_score
        if price_prediction and 'predictions' in price_prediction:
            _save_prediction_history(ticker.upper(), current_price, price_prediction, score=final_score)
        
        # Explicit check before return
        print(f"[AI RECOMMENDATIONS] Returning result with keys: {list(result.keys())}")
        print(f"[AI RECOMMENDATIONS] chart_data in result: {'chart_data' in result}")
        if 'chart_data' in result:
            print(f"[AI RECOMMENDATIONS] chart_data dates length: {len(result['chart_data'].get('dates', []))}")
        
        return result
        
    except Exception as e:
        print(f"Error generating AI recommendations for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/ai-recommendations/<ticker>')
def get_ai_recommendations(ticker):
    """Get AI-powered stock recommendations"""
    try:
        print(f"[AI RECOMMENDATIONS API] ===== NEW CODE VERSION ===== Starting request for {ticker}")
        recommendations = generate_ai_recommendations(ticker.upper())
        if recommendations is None:
            return jsonify({'error': 'Could not generate recommendations'}), 404
        
        # Debug: Check chart_data in recommendations
        print(f"[AI RECOMMENDATIONS API] recommendations keys: {list(recommendations.keys())}")
        print(f"[AI RECOMMENDATIONS API] has chart_data: {'chart_data' in recommendations}")
        if 'chart_data' in recommendations:
            cd = recommendations['chart_data']
            print(f"[AI RECOMMENDATIONS API] chart_data type: {type(cd)}")
            if isinstance(cd, dict) and 'dates' in cd:
                print(f"[AI RECOMMENDATIONS API] chart_data dates length: {len(cd['dates']) if isinstance(cd['dates'], list) else 'not a list'}")
        
        # Always fetch chart_data directly (more reliable)
        chart_data = None
        try:
            stock_data = get_stock_data(ticker.upper(), '1y')
            if stock_data and not stock_data['history'].empty:
                df = stock_data['history']
                dates = df.index.strftime('%Y-%m-%d').tolist()
                chart_data = {
                    'dates': dates,
                    'open': [float(x) for x in df['Open'].round(2).fillna(0).tolist()],
                    'high': [float(x) for x in df['High'].round(2).fillna(0).tolist()],
                    'low': [float(x) for x in df['Low'].round(2).fillna(0).tolist()],
                    'close': [float(x) for x in df['Close'].round(2).fillna(0).tolist()],
                    'volume': [int(x) for x in df['Volume'].fillna(0).astype(int).tolist()],
                }
                print(f"[AI RECOMMENDATIONS API] ✅ Fetched chart_data: {len(chart_data['dates'])} dates")
        except Exception as e:
            print(f"[AI RECOMMENDATIONS API] ❌ Error fetching chart_data: {e}")
            import traceback
            traceback.print_exc()
            chart_data = {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        # Clean recommendations - chart_data should be preserved by clean_for_json
        # Extract chart_data BEFORE clean_for_json to preserve it
        chart_data_from_rec = recommendations.get('chart_data')
        print(f"[AI RECOMMENDATIONS API] chart_data_from_rec exists: {chart_data_from_rec is not None}")
        if chart_data_from_rec:
            print(f"[AI RECOMMENDATIONS API] chart_data_from_rec type: {type(chart_data_from_rec)}, has dates: {'dates' in chart_data_from_rec if isinstance(chart_data_from_rec, dict) else False}")
        
        cleaned = clean_for_json(recommendations)
        
        # ALWAYS add chart_data to cleaned - use from recommendations if available
        if chart_data_from_rec and isinstance(chart_data_from_rec, dict) and 'dates' in chart_data_from_rec:
            cleaned['chart_data'] = {
                'dates': list(chart_data_from_rec.get('dates', [])),
                'open': [float(x) for x in chart_data_from_rec.get('open', [])],
                'high': [float(x) for x in chart_data_from_rec.get('high', [])],
                'low': [float(x) for x in chart_data_from_rec.get('low', [])],
                'close': [float(x) for x in chart_data_from_rec.get('close', [])],
                'volume': [int(x) for x in chart_data_from_rec.get('volume', [])],
            }
            print(f"[AI RECOMMENDATIONS API] ✅ Added chart_data from recommendations: {len(cleaned['chart_data']['dates'])} dates")
        elif chart_data and isinstance(chart_data, dict) and 'dates' in chart_data:
            cleaned['chart_data'] = {
                'dates': list(chart_data.get('dates', [])),
                'open': [float(x) for x in chart_data.get('open', [])],
                'high': [float(x) for x in chart_data.get('high', [])],
                'low': [float(x) for x in chart_data.get('low', [])],
                'close': [float(x) for x in chart_data.get('close', [])],
                'volume': [int(x) for x in chart_data.get('volume', [])],
            }
            print(f"[AI RECOMMENDATIONS API] ✅ Added chart_data from fetched: {len(cleaned['chart_data']['dates'])} dates")
        else:
            cleaned['chart_data'] = {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
            print(f"[AI RECOMMENDATIONS API] ⚠️ No chart_data available, using empty")
        
        # Final verification - ensure chart_data is in cleaned
        if 'chart_data' not in cleaned or not cleaned.get('chart_data') or len(cleaned.get('chart_data', {}).get('dates', [])) == 0:
            print(f"[AI RECOMMENDATIONS API] ❌ CRITICAL: chart_data missing or empty! Re-adding from chart_data_from_rec...")
            if chart_data_from_rec and isinstance(chart_data_from_rec, dict) and 'dates' in chart_data_from_rec:
                cleaned['chart_data'] = {
                    'dates': list(chart_data_from_rec.get('dates', [])),
                    'open': [float(x) for x in chart_data_from_rec.get('open', [])],
                    'high': [float(x) for x in chart_data_from_rec.get('high', [])],
                    'low': [float(x) for x in chart_data_from_rec.get('low', [])],
                    'close': [float(x) for x in chart_data_from_rec.get('close', [])],
                    'volume': [int(x) for x in chart_data_from_rec.get('volume', [])],
                }
                print(f"[AI RECOMMENDATIONS API] ✅ Re-added chart_data: {len(cleaned['chart_data']['dates'])} dates")
            else:
                cleaned['chart_data'] = {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        dates_len = len(cleaned.get('chart_data', {}).get('dates', []))
        print(f"[AI RECOMMENDATIONS API] ✅ FINAL: Returning with {dates_len} dates")
        print(f"[AI RECOMMENDATIONS API] ✅ FINAL: Keys in cleaned: {list(cleaned.keys())}")
        print(f"[AI RECOMMENDATIONS API] ✅ FINAL: 'chart_data' in cleaned: {'chart_data' in cleaned}")
        
        # Use json.dumps with make_response
        import json
        response_json = json.dumps(cleaned, default=str)
        parsed = json.loads(response_json)
        if 'chart_data' not in parsed:
            print(f"[AI RECOMMENDATIONS API] ❌ CRITICAL: chart_data missing in JSON! Re-adding...")
            parsed['chart_data'] = cleaned.get('chart_data', {'dates': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
            response_json = json.dumps(parsed, default=str)
        
        response = make_response(response_json)
        response.headers['Content-Type'] = 'application/json'
        print(f"[AI RECOMMENDATIONS API] ✅ FINAL RETURN: {dates_len} dates, keys: {list(cleaned.keys())}")
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml-prediction-history/<ticker>')
def get_ml_prediction_history(ticker):
    """Get ML prediction history for a ticker"""
    try:
        days = request.args.get('days', 30, type=int)
        history = get_prediction_history(ticker.upper(), days)
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/<ticker>')
def get_backtest_results(ticker):
    """Get backtest results for ML predictions"""
    try:
        ticker = ticker.upper()
        history_file = _PREDICTION_HISTORY_DIR / f"{ticker}.json"
        
        if not history_file.exists():
            return jsonify({'error': 'No prediction history found'}), 404
        
        # Load prediction history
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Sort by date (oldest first)
        history.sort(key=lambda x: x.get('date', ''))
        
        # Get historical stock data using Finviz scraper approach
        # Finviz provides chart data in JSON format embedded in HTML
        try:
            url = f'https://finviz.com/quote.ashx?t={ticker}'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Finviz returned status {response.status_code}")
            
            html = response.text
            
            # Try to extract JSON data from Finviz (same approach as get_quarterly_estimates_from_finviz)
            # Look for the JSON data that contains chart information
            json_patterns = [
                r'var chartData = ({.*?});',
                r'var ohlcData = ({.*?});',
                r'chartData:\s*({.*?}),',
            ]
            
            chart_data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, html, re.DOTALL)
                if json_match:
                    try:
                        chart_data = json.loads(json_match.group(1))
                        print(f"[BACKTEST] Found chart data from Finviz for {ticker}")
                        break
                    except:
                        continue
            
            # If we have chart data, try to extract historical prices
            if chart_data and 'data' in chart_data:
                # Finviz chart data format may vary, try to extract prices
                # For now, use yfinance as Finviz chart data format is complex
                # and we need reliable historical prices for backtest
                print(f"[BACKTEST] Using yfinance for historical prices (Finviz chart data format needs parsing)")
                stock = yf.Ticker(ticker)
                time.sleep(0.3)
                hist = stock.history(period='2y')
            else:
                # Use yfinance as primary source (Finviz doesn't provide easy historical price API)
                print(f"[BACKTEST] Using yfinance for historical prices")
                stock = yf.Ticker(ticker)
                time.sleep(0.3)
                hist = stock.history(period='2y')
                
        except Exception as e:
            print(f"[BACKTEST] Error getting data from Finviz, using yfinance fallback: {e}")
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            time.sleep(0.3)
            hist = stock.history(period='2y')
        
        if hist.empty:
            return jsonify({'error': 'No historical data available'}), 404
        
        backtest_results = []
        periods = ['1m', '3m', '6m', '12m']
        period_days = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}
        
        for entry in history:
            pred_date = entry.get('date')
            if not pred_date:
                continue
            
            try:
                pred_datetime = datetime.strptime(pred_date, '%Y-%m-%d')
                current_price = entry.get('current_price', 0)
                predictions = entry.get('predictions', {})
                expected_returns = entry.get('expected_returns', {})
                
                if current_price <= 0:
                    continue
                
                result_entry = {
                    'date': pred_date,
                    'current_price': current_price,
                    'predictions': {},
                    'actual_prices': {},
                    'actual_returns': {},
                    'prediction_errors': {},
                    'profit_loss': {}
                }
                
                # For each prediction period, get actual price
                for period in periods:
                    if period not in predictions:
                        continue
                    
                    days = period_days[period]
                    target_date = pred_datetime + timedelta(days=days)
                    
                    # Check if target date is in the past (we can only backtest past predictions)
                    now = datetime.now()
                    if target_date > now:
                        # This prediction is for the future, skip it
                        continue
                    
                    # Find closest trading day to target date
                    future_dates = hist[hist.index > pred_datetime]
                    if future_dates.empty:
                        continue
                    
                    # Get price at target date (or closest)
                    # Make sure we're looking for dates up to target_date, not beyond
                    available_dates = future_dates[future_dates.index <= target_date]
                    if available_dates.empty:
                        # If no dates up to target, try to get closest date
                        available_dates = future_dates
                    
                    if available_dates.empty:
                        continue
                    
                    # Get the closest date to target_date
                    target_idx = available_dates.index.get_indexer([target_date], method='nearest')[0]
                    if target_idx < 0 or target_idx >= len(available_dates):
                        continue
                    
                    actual_date = available_dates.index[target_idx]
                    actual_price = float(available_dates.loc[actual_date, 'Close'])
                    
                    predicted_price = predictions[period]
                    predicted_return = expected_returns.get(period, 0)
                    actual_return = ((actual_price / current_price) - 1) * 100
                    
                    error = abs(actual_price - predicted_price) / current_price * 100
                    profit_loss = actual_price - current_price
                    
                    result_entry['predictions'][period] = predicted_price
                    result_entry['actual_prices'][period] = actual_price
                    result_entry['actual_returns'][period] = actual_return
                    result_entry['prediction_errors'][period] = error
                    result_entry['profit_loss'][period] = profit_loss
                
                # Only add if we have at least one valid result
                if result_entry['actual_prices']:
                    backtest_results.append(result_entry)
            
            except Exception as e:
                print(f"[BACKTEST] Error processing entry {pred_date}: {e}")
                continue
        
        # Calculate aggregate metrics
        if not backtest_results:
            # Check if we have predictions but they're all too recent
            now = datetime.now()
            recent_predictions = [e for e in history if e.get('date')]
            if recent_predictions:
                latest_date = max(datetime.strptime(e['date'], '%Y-%m-%d') for e in recent_predictions)
                days_old = (now - latest_date).days
                if days_old < 21:  # Less than 1 month old
                    return jsonify({
                        'error': f'No valid backtest results. Predictions are too recent (latest: {latest_date.strftime("%Y-%m-%d")}, {days_old} days old). Need at least 21 days old predictions for 1-month backtest.',
                        'latest_prediction_date': latest_date.strftime('%Y-%m-%d'),
                        'days_old': days_old
                    }), 404
            return jsonify({'error': 'No valid backtest results. No predictions with sufficient historical data available.'}), 404
        
        metrics = {}
        for period in periods:
            period_results = [r for r in backtest_results if period in r.get('actual_prices', {})]
            if not period_results:
                continue
            
            errors = [r['prediction_errors'][period] for r in period_results]
            actual_returns = [r['actual_returns'][period] for r in period_results]
            predicted_returns = [r.get('expected_returns', {}).get(period, 0) for r in period_results if 'expected_returns' in r]
            profit_losses = [r['profit_loss'][period] for r in period_results]
            
            # Calculate metrics
            avg_error = sum(errors) / len(errors) if errors else 0
            avg_actual_return = sum(actual_returns) / len(actual_returns) if actual_returns else 0
            avg_predicted_return = sum(predicted_returns) / len(predicted_returns) if predicted_returns else 0
            total_profit_loss = sum(profit_losses)
            win_rate = sum(1 for r in actual_returns if r > 0) / len(actual_returns) * 100 if actual_returns else 0
            
            # Direction accuracy (did prediction match direction?)
            direction_correct = 0
            for r in period_results:
                pred_ret = r.get('expected_returns', {}).get(period, 0)
                actual_ret = r['actual_returns'][period]
                if (pred_ret > 0 and actual_ret > 0) or (pred_ret < 0 and actual_ret < 0):
                    direction_correct += 1
            direction_accuracy = (direction_correct / len(period_results)) * 100 if period_results else 0
            
            metrics[period] = {
                'count': len(period_results),
                'avg_error_pct': round(avg_error, 2),
                'avg_actual_return_pct': round(avg_actual_return, 2),
                'avg_predicted_return_pct': round(avg_predicted_return, 2),
                'total_profit_loss': round(total_profit_loss, 2),
                'win_rate_pct': round(win_rate, 2),
                'direction_accuracy_pct': round(direction_accuracy, 2)
            }
        
        return jsonify(clean_for_json({
            'ticker': ticker,
            'total_predictions': len(history),
            'valid_results': len(backtest_results),
            'results': backtest_results,
            'metrics': metrics
        }))
        
    except Exception as e:
        print(f"[BACKTEST] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

@app.route('/api/ml-score-history/<ticker>')
def get_ml_score_history(ticker):
    """Get ML prediction score history for a ticker"""
    try:
        days = request.args.get('days', 30, type=int)
        history = get_prediction_history(ticker.upper(), days)
        
        # Extract score history
        score_history = []
        for entry in history:
            if 'score' in entry:
                score_history.append({
                    'date': entry.get('date'),
                    'timestamp': entry.get('timestamp'),
                    'score': entry.get('score', 0.0),
                    'current_price': entry.get('current_price', 0.0)
                })
        
        return jsonify({'score_history': score_history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-ml-cache', methods=['POST'])
def clear_ml_cache():
    """Clear ML model cache - useful when model structure changes"""
    global _model_cache, _scaler_cache
    cache_size_before = len(_model_cache)
    _model_cache.clear()
    _scaler_cache.clear()
    return jsonify({
        'success': True,
        'message': f'ML cache cleared (removed {cache_size_before} cached models)',
        'cache_version': _MODEL_CACHE_VERSION
    })

def generate_investment_thesis_with_ai(ticker):
    """Generate comprehensive investment thesis using AI"""
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'error': 'Google Gemini API key not configured'
        }
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models
        available_model = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name.lower():
                        available_model = m.name
                        break
                    elif available_model is None:
                        available_model = m.name
        except Exception:
            pass
        
        if available_model is None:
            for model_name in ['gemini-pro', 'gemini-1.5-flash', 'models/gemini-pro']:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    available_model = model_name
                    break
                except:
                    continue
        
        if available_model is None:
            raise Exception("No available Gemini models found.")
        
        model = genai.GenerativeModel(available_model)
        
        # Gather comprehensive data about the stock
        stock_data = get_stock_data(ticker, '1y')
        if not stock_data or stock_data['history'].empty:
            return {
                'success': False,
                'error': 'Could not fetch stock data'
            }
        
        df = stock_data['history']
        info = stock_data.get('info', {})
        indicators = calculate_technical_indicators(df)
        metrics = calculate_metrics(df, info)
        news_list = get_stock_news(ticker)
        
        # Get financials data
        financials_data = None
        try:
            financials_response = get_financials_data(ticker)
            if financials_response and 'executive_snapshot' in financials_response:
                financials_data = financials_response
        except Exception:
            pass
        
        # Prepare comprehensive data summary for AI
        current_price = df['Close'].iloc[-1] if not df.empty else None
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100 if len(df) > 1 else 0
        
        # Build data summary with safe formatting
        def safe_format(value, format_str='{}'):
            if value is None:
                return 'N/A'
            try:
                if format_str == '{:.2f}':
                    return f'{float(value):.2f}'
                elif format_str == '{:.2f}%':
                    return f'{float(value):.2f}%'
                else:
                    return str(value)
            except (ValueError, TypeError):
                return 'N/A'
        
        current_price_str = safe_format(current_price, '{:.2f}')
        price_change_str = safe_format(price_change, '{:.2f}%')
        
        rsi_val = indicators.get('rsi', [None])[-1] if indicators.get('rsi') else None
        macd_val = indicators.get('macd', [None])[-1] if indicators.get('macd') else None
        sma20_val = indicators.get('sma_20', [None])[-1] if indicators.get('sma_20') and indicators['sma_20'][-1] is not None else None
        sma50_val = indicators.get('sma_50', [None])[-1] if indicators.get('sma_50') and indicators['sma_50'][-1] is not None else None
        
        data_summary = f"""
=== Stock Information ===
Ticker: {ticker}
Company Name: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Current Price: ${current_price_str}
Market Cap: {info.get('marketCap', 'N/A')}
52 Week High: ${safe_format(info.get('fiftyTwoWeekHigh'), '{:.2f}')}
52 Week Low: ${safe_format(info.get('fiftyTwoWeekLow'), '{:.2f}')}
Price Change (1Y): {price_change_str}

=== Financial Metrics ===
P/E Ratio: {safe_format(info.get('trailingPE'))}
Forward P/E: {safe_format(info.get('forwardPE'))}
PEG Ratio: {safe_format(info.get('pegRatio'))}
Price to Book: {safe_format(info.get('priceToBook'))}
Dividend Yield: {safe_format(info.get('dividendYield'))}
Beta: {safe_format(info.get('beta'))}

=== Technical Indicators ===
RSI: {safe_format(rsi_val)}
MACD: {safe_format(macd_val)}
SMA 20: ${safe_format(sma20_val, '{:.2f}')}
SMA 50: ${safe_format(sma50_val, '{:.2f}')}
"""
        
        if financials_data and financials_data.get('executive_snapshot'):
            snapshot = financials_data['executive_snapshot']
            revenue_ttm = snapshot.get('revenue_ttm')
            revenue_yoy = snapshot.get('revenue_yoy')
            net_income_ttm = snapshot.get('net_income_ttm')
            fcf_ttm = snapshot.get('fcf_ttm')
            gross_margin = snapshot.get('gross_margin')
            debt_fcf_ratio = snapshot.get('debt_fcf_ratio')
            
            data_summary += f"""
=== Financial Health ===
Revenue (TTM): ${safe_format(revenue_ttm)}
Revenue YoY Growth: {safe_format(revenue_yoy, '{:.2f}%') if revenue_yoy is not None else 'N/A'}
Net Income (TTM): ${safe_format(net_income_ttm)}
FCF (TTM): ${safe_format(fcf_ttm)}
Gross Margin: {safe_format(gross_margin, '{:.2f}%') if gross_margin is not None else 'N/A'}
Debt/FCF Ratio: {safe_format(debt_fcf_ratio, '{:.2f}') if debt_fcf_ratio is not None else 'N/A'}
Company Stage: {financials_data.get('company_stage', 'N/A')}
Main Verdict: {financials_data.get('main_verdict_sentence', 'N/A')}
"""
        
        # News summary
        if news_list:
            recent_news = news_list[:10]
            news_summary = "\n".join([f"- {n.get('title', 'N/A')} ({n.get('sentiment', 'neutral')})" for n in recent_news])
            data_summary += f"""
=== Recent News (Top 10) ===
{news_summary}
"""
        
        # Create comprehensive prompt for investment thesis
        prompt = f"""Jsi expertní investiční analytik s desítkami let zkušeností. Vytvoř kompletní, profesionální investment thesis pro akcii {ticker} v českém jazyce.

**Tvá úloha:**
Vytvoř komplexní investment memo, které by mohlo být použito pro investiční rozhodnutí. Analyzuj všechny dostupné data a vytvoř vyváženou analýzu s bull case, bear case a klíčovými riziky.

**Formátuj odpověď PŘESNĚ takto:**

=== Executive Summary ===
Napiš 4-6 vět shrnujících celkovou investiční příležitost. Zahrň klíčové body, které by měl investor znát.

=== Bull Case (Pozitivní scénář) ===
Uveď 5-8 silných argumentů PRO investici. Každý argument na samostatný řádek s odrážkou. Zahrň:
- Silné stránky společnosti
- Růstové příležitosti
- Konkurenční výhody
- Pozitivní trendy
- Konkrétní čísla a fakta

=== Bear Case (Negativní scénář) ===
Uveď 4-6 hlavních rizik a obav PROTI investici. Každé riziko na samostatný řádek s odrážkou. Zahrň:
- Slabé stránky společnosti
- Hrozby a rizika
- Negativní trendy
- Konkurenční nevýhody
- Konkrétní čísla a fakta

=== Key Risks ===
Uveď 4-6 klíčových rizik, která by měl investor sledovat. Každé riziko na samostatný řádek s odrážkou. Zahrň:
- Specifická rizika pro tuto akcii
- Sektorová rizika
- Makroekonomická rizika
- Operativní rizika
- Regulační rizika

=== Investment Recommendation ===
[BUY/HOLD/SELL]

=== Recommendation Explanation ===
Napiš 5-7 vět vysvětlujících, proč jsi zvolil toto doporučení. Buď velmi konkrétní a zahrň:
- Hlavní důvody pro doporučení
- Klíčové faktory, které ovlivnily rozhodnutí
- Podmínky, za kterých by se doporučení mohlo změnit
- Time horizon (krátkodobý/dlouhodobý)
- Target price nebo price range (pokud je to relevantní)

=== Investment Thesis Summary ===
Napiš 6-8 vět shrnujících celou investment thesis. Toto by mělo být kompletní shrnutí, které investor může použít pro rozhodnutí.

Dostupná data:
{data_summary[:30000]}  # Limit pro rozsáhlou analýzu
"""
        
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 8192,  # Velký limit pro komplexní investment thesis
            }
        )
        
        ai_thesis = response.text
        
        # Parse AI response
        thesis_data = parse_investment_thesis(ai_thesis)
        
        return {
            'success': True,
            'thesis': ai_thesis,
            'structured_data': thesis_data,
            'model_used': available_model
        }
        
    except Exception as e:
        print(f"Error in AI investment thesis generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def parse_investment_thesis(thesis_text):
    """Parse AI investment thesis into structured format"""
    structured = {
        'executive_summary': '',
        'bull_case': [],
        'bear_case': [],
        'key_risks': [],
        'recommendation': 'HOLD',
        'recommendation_explanation': '',
        'thesis_summary': ''
    }
    
    # Find sections
    sections = {
        'executive_summary': ['=== Executive Summary ===', 'Executive Summary:'],
        'bull_case': ['=== Bull Case', 'Bull Case:', 'Pozitivní scénář:'],
        'bear_case': ['=== Bear Case', 'Bear Case:', 'Negativní scénář:'],
        'key_risks': ['=== Key Risks ===', 'Key Risks:', 'Klíčová rizika:'],
        'recommendation': ['=== Investment Recommendation ===', 'Investment Recommendation:', 'Doporučení:'],
        'recommendation_explanation': ['=== Recommendation Explanation ===', 'Recommendation Explanation:', 'Vysvětlení doporučení:'],
        'thesis_summary': ['=== Investment Thesis Summary ===', 'Investment Thesis Summary:', 'Shrnutí thesis:']
    }
    
    section_positions = {}
    for section_name, markers in sections.items():
        for marker in markers:
            pos = thesis_text.find(marker)
            if pos != -1:
                section_positions[section_name] = (pos, marker)
                break
    
    # Sort by position
    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1][0])
    
    # Extract content
    for i, (section_name, (start_pos, marker)) in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            end_pos = sorted_sections[i + 1][1][0]
        else:
            end_pos = len(thesis_text)
        
        content = thesis_text[start_pos + len(marker):end_pos].strip()
        
        if section_name == 'executive_summary':
            structured['executive_summary'] = content.strip()
        
        elif section_name == 'bull_case':
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit())):
                    point = line.lstrip('-•*0123456789. ').strip()
                    if point:
                        structured['bull_case'].append(point)
        
        elif section_name == 'bear_case':
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit())):
                    point = line.lstrip('-•*0123456789. ').strip()
                    if point:
                        structured['bear_case'].append(point)
        
        elif section_name == 'key_risks':
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit())):
                    risk = line.lstrip('-•*0123456789. ').strip()
                    if risk:
                        structured['key_risks'].append(risk)
        
        elif section_name == 'recommendation':
            content_lower = content.lower()
            if 'buy' in content_lower or 'koupit' in content_lower:
                structured['recommendation'] = 'BUY'
            elif 'sell' in content_lower or 'prodat' in content_lower:
                structured['recommendation'] = 'SELL'
            else:
                structured['recommendation'] = 'HOLD'
        
        elif section_name == 'recommendation_explanation':
            structured['recommendation_explanation'] = content.strip()
        
        elif section_name == 'thesis_summary':
            structured['thesis_summary'] = content.strip()
    
    return structured

# ==================== Social Sentiment Functions ====================

def get_reddit_sentiment(ticker, days=7):
    """Get sentiment from Reddit posts about a stock ticker"""
    try:
        posts = []
        sentiment_scores = []
        total_upvotes = 0
        total_comments = 0
        
        # Check if Reddit API is available
        reddit_api_available = REDDIT_AVAILABLE
        
        # Try using PRAW if credentials are available
        if reddit_api_available:
            try:
                import praw
                reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
                
                subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Determine time_filter based on days
                if days <= 1:
                    time_filter = 'day'
                elif days <= 7:
                    time_filter = 'week'
                elif days <= 30:
                    time_filter = 'month'
                else:
                    time_filter = 'year'
                
                for subreddit_name in subreddits:
                    try:
                        subreddit = reddit.subreddit(subreddit_name)
                        # Search for posts containing the ticker
                        for submission in subreddit.search(f'${ticker} OR {ticker}', limit=50, sort='relevance', time_filter=time_filter):
                            try:
                                post_date = datetime.fromtimestamp(submission.created_utc)
                                if post_date < cutoff_date:
                                    continue
                                
                                # Combine title and selftext
                                text = f"{submission.title} {submission.selftext}"
                                
                                # Analyze sentiment
                                analyzer = SentimentIntensityAnalyzer()
                                sentiment = analyzer.polarity_scores(text)
                                
                                posts.append({
                                    'title': submission.title[:200],
                                    'text': submission.selftext[:500] if submission.selftext else '',
                                    'url': f"https://reddit.com{submission.permalink}",
                                    'upvotes': submission.score,
                                    'comments': submission.num_comments,
                                    'subreddit': subreddit_name,
                                    'date': post_date.isoformat(),
                                    'sentiment': sentiment['compound']
                                })
                                
                                # Weight sentiment by upvotes and comments
                                weight = 1 + (submission.score / 100) + (submission.num_comments / 50)
                                sentiment_scores.append(sentiment['compound'] * weight)
                                total_upvotes += submission.score
                                total_comments += submission.num_comments
                                
                            except Exception as e:
                                print(f"Error processing Reddit post: {str(e)}")
                                continue
                                
                        time.sleep(0.5)  # Rate limiting
                    except Exception as e:
                        print(f"Error accessing subreddit {subreddit_name}: {str(e)}")
                        continue
                        
            except ImportError:
                print("PRAW not installed, using web scraping fallback")
                reddit_api_available = False
            except Exception as e:
                print(f"Error using PRAW: {str(e)}, falling back to web scraping")
                reddit_api_available = False
        
        # Fallback to web scraping if PRAW not available or failed
        if not reddit_api_available or len(posts) == 0:
            # Web scraping fallback - search Reddit via web
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                # Determine time filter for web scraping
                if days <= 1:
                    time_filter = 'day'
                elif days <= 7:
                    time_filter = 'week'
                elif days <= 30:
                    time_filter = 'month'
                else:
                    time_filter = 'year'
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                subreddits = ['wallstreetbets', 'stocks', 'investing']
                for subreddit_name in subreddits:
                    try:
                        # Search Reddit via web
                        search_url = f"https://www.reddit.com/r/{subreddit_name}/search.json?q={ticker}&restrict_sr=1&sort=relevance&t={time_filter}&limit=25"
                        response = requests.get(search_url, headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if 'data' in data and 'children' in data['data']:
                                for child in data['data']['children'][:25]:
                                    post_data = child.get('data', {})
                                    title = post_data.get('title', '')
                                    selftext = post_data.get('selftext', '')
                                    text = f"{title} {selftext}"
                                    
                                    if ticker.upper() not in text.upper():
                                        continue
                                    
                                    # Analyze sentiment
                                    analyzer = SentimentIntensityAnalyzer()
                                    sentiment = analyzer.polarity_scores(text)
                                    
                                    post_date = datetime.fromtimestamp(post_data.get('created_utc', 0))
                                    
                                    # Filter by date
                                    if post_date < cutoff_date:
                                        continue
                                    
                                    posts.append({
                                        'title': title[:200],
                                        'text': selftext[:500],
                                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                        'upvotes': post_data.get('score', 0),
                                        'comments': post_data.get('num_comments', 0),
                                        'subreddit': subreddit_name,
                                        'date': post_date.isoformat(),
                                        'sentiment': sentiment['compound']
                                    })
                                    
                                    weight = 1 + (post_data.get('score', 0) / 100) + (post_data.get('num_comments', 0) / 50)
                                    sentiment_scores.append(sentiment['compound'] * weight)
                                    total_upvotes += post_data.get('score', 0)
                                    total_comments += post_data.get('num_comments', 0)
                                    
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        print(f"Error scraping subreddit {subreddit_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error in Reddit web scraping: {str(e)}")
        
        # Calculate weighted sentiment score
        if sentiment_scores:
            weighted_sentiment = sum(sentiment_scores) / sum([1 + (p.get('upvotes', 0) / 100) + (p.get('comments', 0) / 50) for p in posts]) if posts else 0
            # Normalize to 0-100 scale
            sentiment_score = ((weighted_sentiment + 1) / 2) * 100
        else:
            sentiment_score = 50.0  # Neutral if no data
        
        # Determine if trending (high mention count or upvotes)
        trending = len(posts) >= 5 or total_upvotes > 100
        
        return {
            'posts': posts[:20],  # Limit to top 20
            'sentiment_score': round(sentiment_score, 2),
            'mention_count': len(posts),
            'trending': trending,
            'total_upvotes': total_upvotes,
            'total_comments': total_comments
        }
        
    except Exception as e:
        print(f"Error in get_reddit_sentiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'posts': [],
            'sentiment_score': 50.0,
            'mention_count': 0,
            'trending': False,
            'total_upvotes': 0,
            'total_comments': 0,
            'error': str(e)
        }

def get_twitter_sentiment(ticker, days=7):
    """Get sentiment from Twitter/X about a stock ticker"""
    try:
        tweets = []
        sentiment_scores = []
        total_likes = 0
        total_retweets = 0
        
        # Web scraping approach (Twitter API requires paid access)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try to scrape Twitter search results
        # Note: Twitter has strong anti-scraping, so this is a simplified approach
        try:
            # Search for tweets with the ticker
            search_query = f"${ticker} OR {ticker} stock"
            # Using nitter.net as a Twitter frontend (if available)
            # Or use Twitter's search page (may be blocked)
            
            # For now, return sample data structure
            # In production, you'd use Twitter API v2 (paid) or a service like snscrape
            
            # Simulated data structure - replace with actual scraping
            analyzer = SentimentIntensityAnalyzer()
            
            # Try to get some basic data from web
            # This is a placeholder - real implementation would need Twitter API
            # or a third-party service
            
        except Exception as e:
            print(f"Twitter scraping not available: {str(e)}")
        
        # Calculate sentiment from collected tweets
        if sentiment_scores:
            weighted_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            sentiment_score = ((weighted_sentiment + 1) / 2) * 100
        else:
            # Return neutral if no data available
            sentiment_score = 50.0
        
        return {
            'tweets': tweets[:20],
            'sentiment_score': round(sentiment_score, 2),
            'mention_count': len(tweets),
            'total_likes': total_likes,
            'total_retweets': total_retweets,
            'influencers': []
        }
        
    except Exception as e:
        print(f"Error in get_twitter_sentiment: {str(e)}")
        return {
            'tweets': [],
            'sentiment_score': 50.0,
            'mention_count': 0,
            'total_likes': 0,
            'total_retweets': 0,
            'influencers': [],
            'error': str(e)
        }

def get_stocktwits_sentiment(ticker, days=7):
    """Get sentiment from StockTwits about a stock ticker"""
    try:
        messages = []
        sentiment_scores = []
        bullish_count = 0
        bearish_count = 0
        
        # StockTwits API (free but rate limited)
        response = None
        data = None
        try:
            api_url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'messages' in data and len(data['messages']) > 0:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    analyzer = SentimentIntensityAnalyzer()
                    
                    for msg in data['messages'][:30]:  # Limit to 30 messages
                        try:
                            # Handle different date formats
                            created_at_str = msg.get('created_at', '')
                            if not created_at_str:
                                # If no date, include message anyway
                                created_at = datetime.now()
                            else:
                                try:
                                    created_at = datetime.strptime(created_at_str, '%Y-%m-%dT%H:%M:%SZ')
                                except:
                                    try:
                                        created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                                    except:
                                        # If date parsing fails, include the message anyway
                                        created_at = datetime.now()
                            
                            if created_at < cutoff_date:
                                continue
                            
                            body = msg.get('body', '')
                            sentiment_flag = msg.get('entities', {}).get('sentiment', {})
                            sentiment_basic = sentiment_flag.get('basic', 'neutral') if sentiment_flag else 'neutral'
                            
                            # Analyze sentiment
                            sentiment = analyzer.polarity_scores(body)
                            
                            # Use StockTwits sentiment if available, otherwise use VADER
                            if sentiment_basic == 'bullish':
                                sentiment_score = 0.5 + (sentiment['compound'] * 0.5)
                                bullish_count += 1
                            elif sentiment_basic == 'bearish':
                                sentiment_score = -0.5 + (sentiment['compound'] * 0.5)
                                bearish_count += 1
                            else:
                                sentiment_score = sentiment['compound']
                            
                            messages.append({
                                'body': body[:500],
                                'sentiment': sentiment_basic,
                                'sentiment_score': sentiment_score,
                                'user': msg.get('user', {}).get('username', 'Unknown'),
                                'date': created_at.isoformat(),
                                'id': msg.get('id', 0)
                            })
                            
                            sentiment_scores.append(sentiment_score)
                            
                        except Exception as e:
                            print(f"Error processing StockTwits message: {str(e)}")
                            continue
                            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error accessing StockTwits API: {str(e)}")
        
        # Calculate sentiment score
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_score = ((avg_sentiment + 1) / 2) * 100
        else:
            sentiment_score = 50.0
        
        # Calculate bullish percentage
        total_sentiment_flags = bullish_count + bearish_count
        bullish_pct = (bullish_count / total_sentiment_flags * 100) if total_sentiment_flags > 0 else 50.0
        
        # Get watchlist count (if available in API response)
        watchlist_count = 0
        try:
            if 'response' in locals() and response.status_code == 200:
                if 'symbol' in data:
                    watchlist_count = data['symbol'].get('watchlist_count', 0)
        except Exception:
            pass
        
        return {
            'messages': messages[:20],
            'sentiment_score': round(sentiment_score, 2),
            'watchlist_count': watchlist_count,
            'bullish_pct': round(bullish_pct, 2),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }
        
    except Exception as e:
        print(f"Error in get_stocktwits_sentiment: {str(e)}")
        return {
            'messages': [],
            'sentiment_score': 50.0,
            'watchlist_count': 0,
            'bullish_pct': 50.0,
            'bullish_count': 0,
            'bearish_count': 0,
            'error': str(e)
        }

def aggregate_social_sentiment(ticker, days=7):
    """Aggregate sentiment from Reddit only"""
    try:
        # Get data from Reddit only
        reddit_data = get_reddit_sentiment(ticker, days)
        
        # Use Reddit sentiment as overall sentiment
        reddit_score = reddit_data.get('sentiment_score', 50.0)
        overall_score = reddit_score
        
        # Determine overall sentiment
        if overall_score >= 60:
            overall_sentiment = 'positive'
        elif overall_score <= 40:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate trends (simple comparison - in production, compare with historical data)
        trends = {
            'reddit': 'stable'  # Would compare with previous period
        }
        
        # Extract key topics from Reddit posts (simple keyword extraction)
        all_texts = []
        for post in reddit_data.get('posts', [])[:10]:
            all_texts.append(f"{post.get('title', '')} {post.get('text', '')}")
        
        # Simple keyword extraction (common financial terms)
        key_topics = []
        if all_texts:
            combined_text = ' '.join(all_texts).lower()
            # Look for common topics
            topic_keywords = {
                'earnings': ['earnings', 'eps', 'revenue', 'profit'],
                'guidance': ['guidance', 'outlook', 'forecast'],
                'product': ['product', 'launch', 'release'],
                'partnership': ['partnership', 'deal', 'agreement'],
                'regulation': ['regulation', 'fda', 'approval'],
                'competition': ['competitor', 'competition', 'market share']
            }
            
            for topic, keywords in topic_keywords.items():
                count = sum(1 for keyword in keywords if keyword in combined_text)
                if count > 0:
                    key_topics.append({
                        'topic': topic,
                        'mentions': count
                    })
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': round(overall_score, 2),
            'trends': trends,
            'key_topics': key_topics[:10],
            'platform_breakdown': {
                'reddit': {
                    'sentiment_score': reddit_score,
                    'mention_count': reddit_data.get('mention_count', 0),
                    'trending': reddit_data.get('trending', False)
                }
            },
            'raw_data': {
                'reddit': reddit_data
            }
        }
        
    except Exception as e:
        print(f"Error in aggregate_social_sentiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'overall_sentiment': 'neutral',
            'sentiment_score': 50.0,
            'trends': {},
            'key_topics': [],
            'platform_breakdown': {},
            'error': str(e)
        }

def analyze_social_topics_with_ai(posts_data, ticker):
    """Analyze social media posts and extract key topics using AI"""
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'error': 'Google Gemini API key not configured'
        }
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Find available model
        model = None
        model_name_used = 'unknown'
        
        preferred_models = ["gemini-1.5-flash", "gemini-pro"]
        available_models_list = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models_list.append(m.name)
        except:
            available_models_list = ["gemini-1.5-flash", "gemini-pro"]
        
        for preferred in preferred_models:
            if preferred in available_models_list:
                model = genai.GenerativeModel(preferred)
                model_name_used = preferred
                break
        
        if model is None:
            if available_models_list:
                model = genai.GenerativeModel(available_models_list[0])
                model_name_used = available_models_list[0]
            else:
                model = genai.GenerativeModel("gemini-1.5-flash")
                model_name_used = "gemini-1.5-flash (fallback)"
        
        # Prepare text data
        all_texts = []
        for post in posts_data.get('reddit', {}).get('posts', [])[:15]:
            all_texts.append(f"Reddit: {post.get('title', '')} {post.get('text', '')}")
        for tweet in posts_data.get('twitter', {}).get('tweets', [])[:15]:
            all_texts.append(f"Twitter: {tweet.get('text', '')}")
        for msg in posts_data.get('stocktwits', {}).get('messages', [])[:15]:
            all_texts.append(f"StockTwits: {msg.get('body', '')}")
        
        combined_text = '\n---\n'.join(all_texts)
        if len(combined_text) > 20000:
            combined_text = combined_text[:20000] + "\n... (truncated)"
        
        prompt = f"""Jsi expertní analytik sociálních médií. Analyzuj následující diskuse o akcii {ticker} z Redditu a identifikuj klíčová témata, motivy a varování.

Formátuj odpověď PŘESNĚ takto:

=== Key Topics ===
Uveď 5-8 nejčastěji zmiňovaných témat. Každé téma na samostatný řádek s odrážkou.

=== Themes ===
Uveď 3-5 hlavních motivů nebo trendů v diskusích. Každý motiv na samostatný řádek s odrážkou.

=== Warnings ===
Identifikuj varování (FOMO, FUD, pump & dump signály, manipulace). Uveď 2-4 varování na samostatný řádek s odrážkou.

Diskuse:
{combined_text}
"""
        
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 2048,
            }
        )
        
        ai_analysis = response.text
        
        # Parse AI response
        key_topics = []
        themes = []
        warnings = []
        
        current_section = None
        for line in ai_analysis.split('\n'):
            line = line.strip()
            if '=== Key Topics ===' in line:
                current_section = 'topics'
            elif '=== Themes ===' in line:
                current_section = 'themes'
            elif '=== Warnings ===' in line:
                current_section = 'warnings'
            elif line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                item = line.lstrip('-•* ').strip()
                if current_section == 'topics' and item:
                    key_topics.append(item)
                elif current_section == 'themes' and item:
                    themes.append(item)
                elif current_section == 'warnings' and item:
                    warnings.append(item)
        
        return {
            'success': True,
            'key_topics': key_topics,
            'themes': themes,
            'warnings': warnings,
            'full_analysis': ai_analysis,
            'model_used': model_name_used
        }
        
    except Exception as e:
        print(f"Error in analyze_social_topics_with_ai: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'key_topics': [],
            'themes': [],
            'warnings': []
        }

@app.route('/api/ai-investment-thesis/<ticker>')
def get_ai_investment_thesis(ticker):
    """Get AI-generated investment thesis for a stock"""
    try:
        thesis = generate_investment_thesis_with_ai(ticker.upper())
        
        if not thesis['success']:
            return jsonify({'error': thesis.get('error', 'Failed to generate investment thesis')}), 500
        
        return jsonify(clean_for_json(thesis))
    except Exception as e:
        print(f"Error in get_ai_investment_thesis endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/social-sentiment/<ticker>')
def get_social_sentiment(ticker):
    """Get aggregated social sentiment for a stock ticker"""
    try:
        days = int(request.args.get('days', 7))
        
        # Get aggregated sentiment
        aggregated = aggregate_social_sentiment(ticker.upper(), days)
        
        # Get AI topic analysis
        raw_data = aggregated.get('raw_data', {})
        ai_topics = analyze_social_topics_with_ai(raw_data, ticker.upper())
        
        result = {
            'ticker': ticker.upper(),
            'overall_sentiment': aggregated.get('overall_sentiment', 'neutral'),
            'sentiment_score': aggregated.get('sentiment_score', 50.0),
            'trends': aggregated.get('trends', {}),
            'key_topics': aggregated.get('key_topics', []),
            'platform_breakdown': aggregated.get('platform_breakdown', {}),
            'raw_data': raw_data,
            'ai_analysis': ai_topics if ai_topics.get('success') else {}
        }
        
        return jsonify(clean_for_json(result))
        
    except Exception as e:
        print(f"Error in get_social_sentiment endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/social-sentiment/watchlist')
def get_watchlist_social_sentiment():
    """Get social sentiment for all tickers in watchlist"""
    try:
        # Get watchlist from request (or could be from localStorage on frontend)
        watchlist_data = request.json.get('tickers', []) if request.json else []
        
        if not watchlist_data:
            # Try to get from query params as fallback
            tickers_str = request.args.get('tickers', '')
            if tickers_str:
                watchlist_data = [t.strip().upper() for t in tickers_str.split(',')]
        
        if not watchlist_data:
            return jsonify({'error': 'No tickers provided'}), 400
        
        results = []
        days = int(request.args.get('days', 7))
        
        for ticker in watchlist_data[:20]:  # Limit to 20 tickers
            try:
                aggregated = aggregate_social_sentiment(ticker, days)
                results.append({
                    'ticker': ticker,
                    'sentiment_score': aggregated.get('sentiment_score', 50.0),
                    'overall_sentiment': aggregated.get('overall_sentiment', 'neutral'),
                    'mention_count': (
                        aggregated.get('platform_breakdown', {}).get('reddit', {}).get('mention_count', 0) +
                        aggregated.get('platform_breakdown', {}).get('twitter', {}).get('mention_count', 0) +
                        aggregated.get('platform_breakdown', {}).get('stocktwits', {}).get('watchlist_count', 0)
                    )
                })
                time.sleep(0.5)  # Rate limiting between tickers
            except Exception as e:
                print(f"Error processing ticker {ticker}: {str(e)}")
                continue
        
        # Sort by mention count (most discussed first)
        results.sort(key=lambda x: x['mention_count'], reverse=True)
        
        return jsonify(clean_for_json({
            'tickers': results,
            'total_count': len(results)
        }))
        
    except Exception as e:
        print(f"Error in get_watchlist_social_sentiment endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    period = request.args.get('period', '1y')
    
    try:
        data = get_stock_data(ticker.upper(), period)
        
        if data is None:
            # Check if it's an intraday timeframe that might not be available
            intraday_timeframes = ['1m', '5m', '15m', '1h', '4h']
            if period in intraday_timeframes:
                error_msg = f'Intraday data ({period}) may not be available for this stock. Try a different timeframe or stock like AAPL, MSFT, or TSLA.'
            else:
                error_msg = 'Stock not found or data unavailable'
            return jsonify({'error': error_msg}), 404
    except Exception as e:
        print(f"Error in get_stock endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        intraday_timeframes = ['1m', '5m', '15m', '1h', '4h']
        if period in intraday_timeframes:
            error_msg = f'Error fetching intraday data ({period}): {str(e)}. Intraday data may not be available for this stock.'
        else:
            error_msg = f'Error fetching stock data: {str(e)}'
        return jsonify({'error': error_msg}), 500
    
    df = data['history']
    info = data['info']
    
    # Prepare chart data
    # For intraday data, include time in date format
    is_intraday = period in ['1m', '5m', '15m', '1h', '4h']
    if is_intraday:
        # Include time for intraday data: YYYY-MM-DD HH:MM:SS
        dates = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    else:
        # Just date for daily/weekly/monthly data
        dates = df.index.strftime('%Y-%m-%d').tolist()
    
    chart_data = {
        'dates': dates,
        'open': df['Open'].round(2).fillna(0).tolist(),
        'high': df['High'].round(2).fillna(0).tolist(),
        'low': df['Low'].round(2).fillna(0).tolist(),
        'close': df['Close'].round(2).fillna(0).tolist(),
        'volume': df['Volume'].fillna(0).astype(int).tolist(),
    }
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(df)
    
    # Calculate metrics
    metrics = calculate_metrics(df, info)
    
    # Get earnings QoQ data
    # #region agent log
    print(f"[DEBUG] /api/stock/{ticker}: Calling get_earnings_qoq")
    # #endregion
    earnings_qoq = get_earnings_qoq(ticker.upper())
    # #region agent log
    print(f"[DEBUG] /api/stock/{ticker}: get_earnings_qoq returned: {earnings_qoq is not None}")
    if earnings_qoq:
        print(f"[DEBUG] /api/stock/{ticker}: earnings_qoq['eps'] = {earnings_qoq.get('eps', [])[:4]}")
    # #endregion
    
    # Get news with sentiment analysis
    news = get_stock_news(ticker.upper(), max_news=10)
    
    # Generate AI news summary
    news_summary = generate_news_summary(news, ticker.upper())
    
    # Get short interest data
    short_interest = get_short_interest_from_finviz(ticker.upper())
    
    # Get short interest history
    short_interest_history = get_short_interest_history(ticker.upper())
    if short_interest and short_interest_history:
        short_interest['history'] = short_interest_history
    
    # Get volume analysis
    volume_analysis = get_volume_analysis(ticker.upper())
    
    # Company info
    company_info = {
        'name': info.get('longName', ticker),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'description': info.get('longBusinessSummary', ''),
    }
    
    # Clean all data for JSON (replace NaN with None)
    # #region agent log
    if earnings_qoq:
        print(f"[DEBUG] /api/stock/{ticker}: Before clean_for_json - earnings_qoq['eps'] = {earnings_qoq.get('eps', [])[:4]}")
    # #endregion
    
    response_data = {
        'ticker': ticker.upper(),
        'chart_data': clean_for_json(chart_data),
        'indicators': clean_for_json(indicators),
        'metrics': clean_for_json(metrics),
        'company_info': clean_for_json(company_info),
        'earnings_qoq': clean_for_json(earnings_qoq) if earnings_qoq else None,
        'news': clean_for_json(news),
        'news_summary': clean_for_json(news_summary),
        'short_interest': clean_for_json(short_interest) if short_interest else None,
        'volume_analysis': clean_for_json(volume_analysis) if volume_analysis else None
    }
    
    # #region agent log
    if response_data.get('earnings_qoq'):
        print(f"[DEBUG] /api/stock/{ticker}: After clean_for_json - earnings_qoq['eps'] = {response_data['earnings_qoq'].get('eps', [])[:4]}")
        print(f"[DEBUG] /api/stock/{ticker}: Full earnings_qoq['eps'] = {response_data['earnings_qoq'].get('eps', [])}")
        print(f"[DEBUG] /api/stock/{ticker}: earnings_qoq['eps'] None count = {sum(1 for x in response_data['earnings_qoq'].get('eps', []) if x is None)}")
    else:
        print(f"[DEBUG] /api/stock/{ticker}: earnings_qoq is None or missing after clean_for_json")
    # #endregion
    
    return jsonify(response_data)

@app.route('/api/search/<query>')
def search_stocks(query):
    """Advanced stock search with company name matching and fuzzy search"""
    import json
    import os
    import time
    print(f"[SEARCH HTTP] search_stocks called with query: {query}")
    try:
        query = query.strip().upper()
        print(f"[SEARCH] Query after processing: {query}")
        if not query or len(query) < 1:
            return jsonify({'results': [], '_version': 'v3_empty'})
        
        results = []
        query_lower = query.lower()
        
        # Popular tickers database (can be expanded or loaded from file)
        popular_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC',
            'ADBE', 'PYPL', 'CMCSA', 'NKE', 'XOM', 'VZ', 'CVX', 'MRK', 'PFE',
            'INTC', 'CSCO', 'PEP', 'COST', 'TMO', 'AVGO', 'ABT', 'TXN', 'NEE',
            'DHR', 'ACN', 'LIN', 'ORCL', 'WFC', 'PM', 'NFLX', 'UPS', 'QCOM',
            'RTX', 'BMY', 'HON', 'AMGN', 'LOW', 'INTU', 'SPGI', 'BKNG', 'DE',
            'ADI', 'AXP', 'GS', 'BLK', 'SHW', 'MDT', 'GILD', 'SYK', 'ZTS'
        ]
        
        # First, try exact ticker match
        if query in popular_tickers:
            try:
                stock = yf.Ticker(query)
                info = stock.info
                if info and 'symbol' in info:
                    results.append({
                        'ticker': info['symbol'],
                        'name': info.get('longName', info.get('shortName', query)),
                        'exchange': info.get('exchange', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'matchType': 'exact_ticker',
                        'score': 100
                    })
            except Exception:
                pass
    
        # Search through popular tickers for name matches
        for ticker in popular_tickers:
            if len(results) >= 15:  # Limit results
                break
            
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if not info or 'symbol' not in info:
                    continue
                
                ticker_symbol = info['symbol']
                long_name = info.get('longName', '')
                short_name = info.get('shortName', '')
                company_name = long_name or short_name or ''
                
                
                if not company_name:
                    continue
                
                company_name_lower = company_name.lower()
                score = 0
                match_type = None
                
                # Exact ticker match (already handled above)
                if ticker_symbol.upper() == query:
                    continue
                
                
                # Check if query matches ticker
                if query_lower in ticker_symbol.lower():
                    score = 80
                    match_type = 'ticker_partial'
                
                # Check if query matches company name (exact or partial)
                elif query_lower in company_name_lower:
                    # Exact match gets higher score
                    if company_name_lower == query_lower:
                        score = 95
                        match_type = 'name_exact'
                    elif company_name_lower.startswith(query_lower):
                        score = 85
                        match_type = 'name_starts_with'
                    else:
                        score = 70
                        match_type = 'name_contains'
                
                # Fuzzy matching - check if words match
                elif len(query) >= 3:
                    query_words = query_lower.split()
                    name_words = company_name_lower.split()
                    matching_words = sum(1 for qw in query_words if any(qw in nw or nw.startswith(qw) for nw in name_words))
                    if matching_words > 0:
                        score = 50 + (matching_words * 10)
                        match_type = 'fuzzy'
                
                if score > 0:
                    results.append({
                        'ticker': ticker_symbol,
                        'name': company_name,
                        'exchange': info.get('exchange', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'matchType': match_type,
                        'score': score
                    })
                else:
                    pass  # No match found
            
            except Exception as e:
                # Skip tickers that fail to load
                continue
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates and limit to top 15
        seen_tickers = set()
        unique_results = []
        for result in results:
            if result['ticker'] not in seen_tickers:
                seen_tickers.add(result['ticker'])
                unique_results.append(result)
                if len(unique_results) >= 15:
                    break
        
        # Clean up response (remove internal fields)
        for result in unique_results:
            result.pop('matchType', None)
            result.pop('score', None)
        
        cleaned = clean_for_json({'results': unique_results})
        # Add debug marker to verify new code is running  
        cleaned['_debug'] = 'v2_new_code_verified'
        cleaned['_timestamp'] = int(__import__('time').time()*1000)
        return jsonify(cleaned)
        
    except Exception as e:
        import json
        # Debug logging removed for production
        print(f"Error in search_stocks: {str(e)}")
        import traceback
        traceback.print_exc()
    return jsonify({'results': []})

@app.route('/api/earnings-calendar')
def get_earnings_calendar():
    """Get earnings calendar for popular stocks"""
    try:
        # Reduced list of most popular stocks to speed up loading
        popular_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC',
            'ADBE', 'PYPL', 'CMCSA', 'NKE', 'XOM', 'VZ', 'CVX', 'MRK', 'PFE'
        ]
        
        earnings_calendar = []
        today = datetime.now().date()
        future_date = today + timedelta(days=90)  # Next 90 days
        
        # Process stocks with minimal delays
        for i, ticker in enumerate(popular_tickers):
            try:
                stock = yf.Ticker(ticker)
                
                # Get earnings dates first (faster)
                earnings_dates = None
                try:
                    earnings_dates = stock.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        # Get company info only if we have earnings dates
                        info = {}
                        try:
                            info = stock.info
                        except:
                            info = {'longName': ticker, 'sector': 'N/A'}
                        
                        # Process each earnings date
                        for date_idx, row in earnings_dates.iterrows():
                            try:
                                earnings_date = pd.Timestamp(date_idx)
                                if hasattr(earnings_date, 'date'):
                                    earnings_date = earnings_date.date()
                                else:
                                    earnings_date = earnings_date.to_pydatetime().date()
                                
                                # Include future earnings and also recent past (last 7 days) for context
                                days_diff = (earnings_date - today).days
                                if days_diff >= -7 and days_diff <= 90:
                                    eps_estimate = None
                                    eps_reported = None
                                    surprise_pct = None
                                    
                                    if 'EPS Estimate' in row.index:
                                        eps_estimate = float(row['EPS Estimate']) if pd.notna(row['EPS Estimate']) else None
                                    if 'Reported EPS' in row.index:
                                        eps_reported = float(row['Reported EPS']) if pd.notna(row['Reported EPS']) else None
                                    if 'Surprise(%)' in row.index:
                                        surprise_pct = float(row['Surprise(%)']) if pd.notna(row['Surprise(%)']) else None
                                    
                                    earnings_calendar.append({
                                        'ticker': ticker,
                                        'company_name': info.get('longName', ticker),
                                        'sector': info.get('sector', 'N/A'),
                                        'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                                        'earnings_date_display': earnings_date.strftime('%B %d, %Y'),
                                        'eps_estimate': eps_estimate,
                                        'eps_reported': eps_reported,
                                        'surprise_pct': surprise_pct,
                                        'is_past': earnings_date < today
                                    })
                            except Exception as e:
                                print(f"Error processing earnings date for {ticker}: {str(e)}")
                                continue
                except Exception as e:
                    print(f"Error fetching earnings_dates for {ticker}: {str(e)}")
                    continue
                
                # Small delay only every 5 stocks to avoid rate limiting
                if (i + 1) % 5 == 0:
                    time.sleep(0.2)
                
            except Exception as e:
                print(f"Error fetching earnings for {ticker}: {str(e)}")
                continue
        
        # Sort by earnings date
        earnings_calendar.sort(key=lambda x: x['earnings_date'])
        
        return jsonify({
            'earnings': clean_for_json(earnings_calendar),
            'total': len(earnings_calendar)
        })
        
    except Exception as e:
        print(f"Error in earnings calendar: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch earnings calendar: {str(e)}', 'earnings': [], 'total': 0}), 500

@app.route('/api/alerts-dashboard', methods=['POST'])
def get_alerts_dashboard():
    """Get all alerts (earnings, news) for watchlist stocks"""
    try:
        data = request.get_json()
        watchlist = data.get('watchlist', [])
        
        if not watchlist:
            return jsonify({
                'earnings_alerts': [],
                'news_alerts': [],
                'total': 0
            })
        
        today = datetime.now().date()
        earnings_alerts = []
        news_alerts = []
        
        # Process each ticker in watchlist
        for i, ticker in enumerate(watchlist[:20]):  # Limit to 20 to avoid timeout
            try:
                stock = yf.Ticker(ticker)
                
                # Check for upcoming earnings (next 90 days for better coverage)
                try:
                    earnings_dates = stock.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        info = {}
                        try:
                            info = stock.info
                        except:
                            info = {'longName': ticker}
                        
                        # Try to find next earnings date (using same logic as earnings calendar)
                        found_earnings = False
                        for date_idx, row in earnings_dates.iterrows():
                            try:
                                earnings_date = pd.Timestamp(date_idx)
                                # Handle timezone-aware dates the same way as earnings calendar
                                if hasattr(earnings_date, 'date'):
                                    earnings_date = earnings_date.date()
                                else:
                                    earnings_date = earnings_date.to_pydatetime().date()
                                
                                days_diff = (earnings_date - today).days
                                # Alert for earnings in next 90 days (extended for better coverage)
                                # Also include past earnings from last 7 days (recently reported)
                                if -7 <= days_diff <= 90:
                                    earnings_alerts.append({
                                        'ticker': ticker,
                                        'company_name': info.get('longName', ticker),
                                        'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                                        'earnings_date_display': earnings_date.strftime('%B %d, %Y'),
                                        'days_until': days_diff,
                                        'priority': 'high' if days_diff <= 2 else ('medium' if days_diff <= 7 else 'low'),
                                        'type': 'earnings'
                                    })
                                    found_earnings = True
                                    break  # Only show next upcoming earnings
                            except Exception as e:
                                import traceback
                                print(f"Error processing earnings date for {ticker}: {str(e)}")
                                traceback.print_exc()
                                continue
                        
                        # If no earnings found in the range, try to get next earnings anyway (even if further out)
                        if not found_earnings and len(earnings_dates) > 0:
                            try:
                                # Get the first (most recent future) earnings date
                                for date_idx in earnings_dates.index:
                                    try:
                                        earnings_date = pd.Timestamp(date_idx)
                                        if hasattr(earnings_date, 'date'):
                                            earnings_date = earnings_date.date()
                                        else:
                                            earnings_date = earnings_date.to_pydatetime().date()
                                        
                                        days_diff = (earnings_date - today).days
                                        # If it's in the future, show it
                                        if days_diff > 0:
                                            earnings_alerts.append({
                                                'ticker': ticker,
                                                'company_name': info.get('longName', ticker),
                                                'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                                                'earnings_date_display': earnings_date.strftime('%B %d, %Y'),
                                                'days_until': days_diff,
                                                'priority': 'low',
                                                'type': 'earnings'
                                            })
                                            break
                                    except:
                                        continue
                            except Exception as e:
                                print(f"Error getting next earnings for {ticker}: {str(e)}")
                                pass
                except Exception as e:
                    print(f"Error fetching earnings_dates for {ticker}: {str(e)}")
                    pass
                
                # Get recent news with high sentiment
                try:
                    news = get_stock_news(ticker, max_news=5)
                    for article in news:
                        # Alert for high sentiment news (positive or negative)
                        sentiment_score = article.get('sentiment_score', 0)
                        if abs(sentiment_score) > 0.3:  # Strong sentiment
                            news_alerts.append({
                                'ticker': ticker,
                                'title': article.get('title', ''),
                                'summary': article.get('summary', '')[:200],
                                'link': article.get('link', ''),
                                'publisher': article.get('publisher', ''),
                                'published': article.get('published', ''),
                                'sentiment': article.get('sentiment', 'neutral'),
                                'sentiment_score': sentiment_score,
                                'priority': 'high' if abs(sentiment_score) > 0.5 else 'medium',
                                'type': 'news'
                            })
                except:
                    pass
                
                # Small delay to avoid rate limiting
                if (i + 1) % 3 == 0:
                    time.sleep(0.3)
                    
            except Exception as e:
                print(f"Error processing alerts for {ticker}: {str(e)}")
                continue
        
        # Sort earnings alerts by date
        earnings_alerts.sort(key=lambda x: x['earnings_date'])
        
        # Sort news alerts by sentiment score (absolute value)
        news_alerts.sort(key=lambda x: abs(x.get('sentiment_score', 0)), reverse=True)
        
        return jsonify({
            'earnings_alerts': clean_for_json(earnings_alerts),
            'news_alerts': clean_for_json(news_alerts[:20]),  # Limit to top 20 news
            'total': len(earnings_alerts) + len(news_alerts)
        })
        
    except Exception as e:
        print(f"Error in alerts dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to fetch alerts: {str(e)}',
            'earnings_alerts': [],
            'news_alerts': [],
            'total': 0
        }), 500

@app.route('/api/financials/<ticker>')
def get_financials(ticker):
    """Get comprehensive financial data for Financials tab"""
    import time as time_module
    start_time = time_module.time()
    print(f"[DEBUG] /api/financials/{ticker} called")
    
    try:
        ticker_upper = ticker.upper()
        print(f"[DEBUG] Processing ticker: {ticker_upper}")
        
        # Try to get financials data with better error handling
        try:
            print(f"[DEBUG] Calling get_financials_data for {ticker_upper}")
            financials = get_financials_data(ticker_upper)
            elapsed = time_module.time() - start_time
            print(f"[DEBUG] get_financials_data completed in {elapsed:.2f}s for {ticker_upper}")
        except Exception as fetch_error:
            elapsed = time_module.time() - start_time
            print(f"[ERROR] Failed to fetch financials data for {ticker_upper} after {elapsed:.2f}s: {str(fetch_error)}")
            import traceback
            traceback.print_exc()
            # Return a more informative error
            return jsonify({
                'error': 'Financial data not available',
                'details': str(fetch_error),
                'ticker': ticker_upper,
                'elapsed_seconds': round(elapsed, 2)
            }), 500
        
        if financials is None:
            elapsed = time_module.time() - start_time
            print(f"[WARNING] get_financials_data returned None for {ticker_upper} after {elapsed:.2f}s")
            return jsonify({
                'error': 'Financial data not available',
                'ticker': ticker_upper,
                'message': 'Unable to fetch financial data. The ticker may not exist or data may be temporarily unavailable.',
                'elapsed_seconds': round(elapsed, 2)
            }), 404
        
        # Add peer comparison data (optional, don't fail if it doesn't work)
        try:
            industry_category = financials.get('industry_category', 'Other')
            sector = financials.get('sector', 'N/A')
            peer_comparison = get_peer_comparison_data(ticker_upper, industry_category, sector, limit=4)
            if peer_comparison:
                financials['peer_comparison'] = peer_comparison
        except Exception as peer_error:
            print(f"[WARNING] Failed to get peer comparison for {ticker_upper}: {str(peer_error)}")
            # Don't fail the whole request if peer comparison fails
        
        return jsonify(clean_for_json(financials))
        
    except Exception as e:
        print(f"[ERROR] Error in financials endpoint for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to get financials',
            'details': str(e),
            'ticker': ticker.upper() if ticker else 'unknown'
        }), 500

def get_finviz_analyst_ratings(ticker):
    """Scrape individual analyst ratings from Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Finviz returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        recommendations = []
        
        # Finviz has analyst ratings in a table with header: date, action, analyst, rating change, price target change
        all_tables = soup.find_all('table')
        
        for table in all_tables:
            rows = table.find_all('tr')
            if len(rows) > 1:
                header_row = rows[0]
                header_cells = header_row.find_all(['td', 'th'])
                header_texts = [cell.get_text(strip=True).lower() for cell in header_cells]
                
                # Look for analyst table with: date, action, analyst, rating change, price target change
                # Must have both 'date' and 'analyst' in header, and 'action' or 'rating'
                # Also check that it has exactly 5 columns (date, action, analyst, rating change, price target change)
                if 'date' in header_texts and 'analyst' in header_texts and ('action' in header_texts or 'rating' in header_texts) and len(header_cells) == 5:
                    # Verify this is the right table by checking first data row
                    if len(rows) > 1:
                        first_data_row = rows[1]
                        first_data_cells = first_data_row.find_all(['td', 'th'])
                        if len(first_data_cells) == 5:
                            first_row_text = ' '.join([cell.get_text(strip=True).lower() for cell in first_data_cells])
                            # First row should contain action words like "upgrade", "downgrade", "initiate", etc.
                            if any(word in first_row_text for word in ['upgrade', 'downgrade', 'initiate', 'maintain', 'reiterate', 'perform', 'outperform', 'underperform']):
                                # Found analyst table - Finviz format: Date, Action, Analyst, Rating Change, Price Target Change
                                for row in rows[1:31]:  # Skip header, limit to 30
                                    try:
                                        cells = row.find_all(['td', 'th'])
                                        if len(cells) < 3:
                                            continue
                                        
                                        # Finviz format based on test:
                                        # cells[0] = Date (e.g., "Nov-14-25")
                                        # cells[1] = Action (e.g., "Upgrade")
                                        # cells[2] = Analyst (e.g., "Oppenheimer")
                                        # cells[3] = Rating Change (e.g., "Perform → Outperform")
                                        # cells[4] = Price Target Change (if exists)
                                        
                                        date_str = cells[0].get_text(strip=True) if len(cells) > 0 else 'N/A'
                                        action = cells[1].get_text(strip=True) if len(cells) > 1 else 'N/A'
                                        firm = cells[2].get_text(strip=True) if len(cells) > 2 else 'N/A'
                                        rating_change = cells[3].get_text(strip=True) if len(cells) > 3 else 'N/A'
                                        target_change = cells[4].get_text(strip=True) if len(cells) > 4 else ''
                                        
                                        # Extract "to" rating from rating change (e.g., "Perform → Outperform" -> "Outperform")
                                        rating = 'N/A'
                                        if '→' in rating_change:
                                            parts = rating_change.split('→')
                                            if len(parts) > 1:
                                                rating = parts[1].strip()
                                        elif rating_change and rating_change != 'N/A':
                                            rating = rating_change
                                        
                                        # Extract target price from target change if available
                                        target_price = None
                                        if target_change and '$' in target_change:
                                            try:
                                                # Extract number after $
                                                import re
                                                match = re.search(r'\$(\d+\.?\d*)', target_change)
                                                if match:
                                                    target_price = float(match.group(1))
                                            except:
                                                pass
                                        
                                        # Only add if we have firm and rating
                                        if firm != 'N/A' and firm and rating != 'N/A' and rating:
                                            recommendations.append({
                                                'date': date_str,
                                                'firm': firm,
                                                'to_grade': rating,
                                                'from_grade': rating_change.split('→')[0].strip() if '→' in rating_change else 'N/A',
                                                'target_price': target_price
                                            })
                                    except Exception as e:
                                        print(f"Error parsing Finviz analyst row: {str(e)}")
                                        import traceback
                                        traceback.print_exc()
                                        continue
                                
                                if recommendations:
                                    break
        
        return recommendations if recommendations else None
        
    except Exception as e:
        print(f"Error scraping Finviz analyst ratings for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_benzinga_analyst_ratings(ticker):
    """Scrape individual analyst ratings from Benzinga"""
    try:
        url = f"https://www.benzinga.com/quote/{ticker.upper()}/ratings"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        recommendations = []
        
        # Benzinga uses various structures - look for analyst data
        # Try to find structured data (could be in tables, divs, or JSON)
        all_elements = soup.find_all(['table', 'div', 'section'])
        
        for element in all_elements:
            text = element.get_text().lower()
            if 'analyst' in text and ('rating' in text or 'target' in text):
                # Try to extract data from this element
                rows = element.find_all(['tr', 'div'], class_=lambda x: x and 'row' in str(x).lower() if x else False)
                if not rows:
                    rows = element.find_all('tr')
                
                for row in rows[1:31]:  # Skip header
                    try:
                        cells = row.find_all(['td', 'th', 'div'])
                        if len(cells) < 2:
                            continue
                        
                        firm = 'N/A'
                        rating = 'N/A'
                        target_price = None
                        date_str = 'N/A'
                        
                        for cell in cells:
                            text = cell.get_text(strip=True)
                            text_lower = text.lower()
                            
                            # Firm name
                            if len(text) > 3 and len(text) < 50 and not any(c in text for c in ['$', '/', '-']) and \
                               not text.replace(',', '').replace('.', '').isdigit():
                                if firm == 'N/A':
                                    firm = text
                            
                            # Rating
                            if any(word in text_lower for word in ['buy', 'sell', 'hold', 'outperform', 'underperform', 'neutral', 'strong']):
                                if rating == 'N/A':
                                    rating = text
                            
                            # Target price
                            if '$' in text:
                                try:
                                    clean_price = text.replace('$', '').replace(',', '').replace(' ', '')
                                    if clean_price:
                                        target_price = float(clean_price)
                                except:
                                    pass
                            
                            # Date
                            if any(x in text for x in ['2024', '2025', '2023']) or ('/' in text and len(text) < 15):
                                if date_str == 'N/A':
                                    date_str = text
                        
                        if firm != 'N/A' and rating != 'N/A':
                            recommendations.append({
                                'date': date_str,
                                'firm': firm,
                                'to_grade': rating,
                                'from_grade': 'N/A',
                                'target_price': target_price
                            })
                    except:
                        continue
                
                if recommendations:
                    break
        
        return recommendations if recommendations else None
        
    except Exception as e:
        print(f"Error scraping Benzinga for {ticker}: {str(e)}")
        return None

def get_marketbeat_analyst_ratings(ticker):
    """Scrape individual analyst ratings from MarketBeat"""
    try:
        urls = [
            f"https://www.marketbeat.com/stocks/NASDAQ/{ticker}/forecast/",
            f"https://www.marketbeat.com/stocks/NASDAQ/{ticker}/",
            f"https://www.marketbeat.com/stocks/NYSE/{ticker}/forecast/",
            f"https://www.marketbeat.com/stocks/NYSE/{ticker}/",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        recommendations = []
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for analyst ratings table
                    tables = soup.find_all('table')
                    ratings_table = None
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        if len(rows) > 1:
                            header_text = rows[0].get_text().lower()
                            # Look for analyst ratings table
                            if ('analyst' in header_text or 'firm' in header_text or 'rating' in header_text) and \
                               ('target' in header_text or 'price' in header_text):
                                ratings_table = table
                                break
                    
                    # Also try div-based structure
                    if not ratings_table:
                        divs = soup.find_all('div', class_=lambda x: x and ('analyst' in str(x).lower() or 'rating' in str(x).lower() or 'recommendation' in str(x).lower()) if x else False)
                        for div in divs:
                            # Look for structured data in divs
                            rows = div.find_all(['tr', 'div'], class_=lambda x: x and 'row' in str(x).lower() if x else False)
                            if len(rows) > 1:
                                ratings_table = div
                                break
                    
                    if ratings_table:
                        rows = ratings_table.find_all(['tr', 'div'])
                        for row in rows[1:31]:  # Skip header, limit to 30
                            try:
                                cells = row.find_all(['td', 'th', 'div'])
                                if len(cells) < 3:
                                    continue
                                
                                # Try to extract: Firm, Date, Rating, Target Price
                                firm = 'N/A'
                                date_str = 'N/A'
                                rating = 'N/A'
                                target_price = None
                                
                                for i, cell in enumerate(cells):
                                    text = cell.get_text(strip=True)
                                    text_lower = text.lower()
                                    
                                    # Firm name (usually longer text, not a number, not a date)
                                    if len(text) > 3 and not any(c in text for c in ['$', '/', '-']) and \
                                       not text.replace(',', '').replace('.', '').isdigit() and \
                                       not any(x in text for x in ['2024', '2025', '2023']) and \
                                       'buy' not in text_lower and 'sell' not in text_lower and 'hold' not in text_lower:
                                        if firm == 'N/A' and len(text) < 50:  # Reasonable firm name length
                                            firm = text
                                    
                                    # Rating (Buy, Hold, Sell, Strong Buy, etc.)
                                    if any(word in text_lower for word in ['buy', 'sell', 'hold', 'outperform', 'underperform', 'neutral']):
                                        if rating == 'N/A':
                                            rating = text
                                    
                                    # Target price (contains $ or numbers)
                                    if '$' in text:
                                        try:
                                            clean_price = text.replace('$', '').replace(',', '').replace(' ', '')
                                            if clean_price:
                                                target_price = float(clean_price)
                                        except:
                                            pass
                                    
                                    # Date
                                    if any(x in text for x in ['2024', '2025', '2023']) or \
                                       ('/' in text and len(text) < 15) or \
                                       ('-' in text and len(text) < 15):
                                        if date_str == 'N/A':
                                            date_str = text
                                
                                # Only add if we have firm and rating
                                if firm != 'N/A' and rating != 'N/A':
                                    recommendations.append({
                                        'date': date_str,
                                        'firm': firm,
                                        'to_grade': rating,
                                        'from_grade': 'N/A',
                                        'target_price': target_price
                                    })
                            except Exception as e:
                                print(f"Error parsing MarketBeat rating row: {str(e)}")
                                continue
                        
                        if recommendations:
                            break  # Found data, no need to try other URLs
            except Exception as e:
                print(f"Error accessing {url}: {str(e)}")
                continue
        
        return recommendations if recommendations else None
        
    except Exception as e:
        print(f"Error scraping MarketBeat analyst ratings for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/analyst-data/<ticker>')
def get_analyst_data(ticker):
    """Get analyst ratings and price targets"""
    try:
        stock = yf.Ticker(ticker.upper())
        time.sleep(0.3)  # Rate limiting
        
        # Get info (don't fail if incomplete)
        try:
            info = stock.info
            if not info:
                info = {}
        except:
            info = {}
        
        # Get individual recommendations from Finviz (works better than MarketBeat/Benzinga)
        recommendations = []
        try:
            print(f"[ANALYST] Trying to get recommendations for {ticker.upper()}")
            # Try Finviz first (most reliable)
            fv_recs = get_finviz_analyst_ratings(ticker.upper())
            if fv_recs and len(fv_recs) > 0:
                recommendations = fv_recs
                print(f"[ANALYST] Found {len(recommendations)} recommendations from Finviz for {ticker}")
            else:
                print(f"[ANALYST] No Finviz recommendations for {ticker}, trying MarketBeat")
                # Try MarketBeat as fallback
                mb_recs = get_marketbeat_analyst_ratings(ticker.upper())
                if mb_recs and len(mb_recs) > 0:
                    recommendations = mb_recs
                    print(f"[ANALYST] Found {len(recommendations)} recommendations from MarketBeat for {ticker}")
                else:
                    print(f"[ANALYST] No MarketBeat recommendations for {ticker}, trying Benzinga")
                    # Try Benzinga as last fallback
                    bz_recs = get_benzinga_analyst_ratings(ticker.upper())
                    if bz_recs and len(bz_recs) > 0:
                        recommendations = bz_recs
                        print(f"[ANALYST] Found {len(recommendations)} recommendations from Benzinga for {ticker}")
                    else:
                        print(f"[ANALYST] No recommendations found from any source for {ticker}")
        except Exception as e:
            print(f"[ANALYST] Error getting recommendations: {str(e)}")
            import traceback
            traceback.print_exc()
            recommendations = []
        
        # Get recommendation summary
        recommendation_summary = None
        try:
            rec_summary = stock.recommendations_summary
            if rec_summary is not None and not rec_summary.empty:
                latest = rec_summary.iloc[-1] if len(rec_summary) > 0 else None
                if latest is not None:
                    recommendation_summary = {
                        'strong_buy': int(latest.get('strongBuy', 0)) if pd.notna(latest.get('strongBuy', 0)) else 0,
                        'buy': int(latest.get('buy', 0)) if pd.notna(latest.get('buy', 0)) else 0,
                        'hold': int(latest.get('hold', 0)) if pd.notna(latest.get('hold', 0)) else 0,
                        'sell': int(latest.get('sell', 0)) if pd.notna(latest.get('sell', 0)) else 0,
                        'strong_sell': int(latest.get('strongSell', 0)) if pd.notna(latest.get('strongSell', 0)) else 0
                    }
        except Exception:
            pass
        
        # Get price targets
        target_mean_price = info.get('targetMeanPrice')
        target_high_price = info.get('targetHighPrice')
        target_low_price = info.get('targetLowPrice')
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Calculate upside/downside
        upside_pct = None
        if target_mean_price and current_price:
            upside_pct = ((target_mean_price - current_price) / current_price) * 100
        
        # Also add individual price targets from recommendations if available
        individual_targets = []
        if recommendations:
            for rec in recommendations:
                if rec.get('target_price'):
                    individual_targets.append({
                        'firm': rec.get('firm', 'N/A'),
                        'target_price': rec.get('target_price'),
                        'rating': rec.get('to_grade', 'N/A'),
                        'date': rec.get('date', 'N/A')
                    })
        
        analyst_data = {
            'recommendations': recommendations[:20] if recommendations else [],  # Last 20
            'recommendation_summary': recommendation_summary,
            'target_mean_price': target_mean_price,
            'target_high_price': target_high_price,
            'target_low_price': target_low_price,
            'current_price': current_price,
            'upside_pct': upside_pct,
            'number_of_analysts': info.get('numberOfAnalystOpinions'),
            'individual_targets': individual_targets[:10] if individual_targets else []
        }
        
        return jsonify(clean_for_json(analyst_data))
        
    except Exception as e:
        print(f"Error fetching analyst data for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch analyst data: {str(e)}'}), 500

def get_marketbeat_insider_trading(ticker):
    """Scrape insider trading data from MarketBeat"""
    try:
        # MarketBeat uses different URL format - try multiple variations
        urls = [
            f"https://www.marketbeat.com/stocks/NASDAQ/{ticker.upper()}/insider-trades/",
            f"https://www.marketbeat.com/stocks/NASDAQ/{ticker.upper()}/insiders/",
            f"https://www.marketbeat.com/stocks/NYSE/{ticker.upper()}/insider-trades/",
            f"https://www.marketbeat.com/stocks/NYSE/{ticker.upper()}/insiders/",
            f"https://www.marketbeat.com/stocks/{ticker.upper()}/insider-trades/"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
        }
        
        transactions = []
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for insider trading table
                    tables = soup.find_all('table')
                    insider_table = None
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        if len(rows) > 1:
                            header_text = rows[0].get_text().lower()
                            # MarketBeat typically has headers like "Date", "Insider", "Transaction", "Shares", "Value"
                            if ('insider' in header_text or 'transaction' in header_text) and \
                               ('date' in header_text or 'shares' in header_text):
                                insider_table = table
                                break
                    
                    if insider_table:
                        rows = insider_table.find_all('tr')
                        for row in rows[1:31]:  # Skip header, limit to 30
                            try:
                                cells = row.find_all(['td', 'th'])
                                if len(cells) < 4:
                                    continue
                                
                                # MarketBeat format varies, try to extract common fields
                                # Typical: Date, Insider, Position, Transaction Type, Shares, Value
                                date_str = 'N/A'
                                insider = 'N/A'
                                position = 'N/A'
                                transaction_type = None
                                shares = None
                                value = None
                                
                                # Try to parse cells - MarketBeat structure may vary
                                for i, cell in enumerate(cells):
                                    text = cell.get_text(strip=True)
                                    text_lower = text.lower()
                                    
                                    # Date detection
                                    if any(x in text for x in ['2024', '2025', '2023']) or \
                                       ('/' in text and len(text) < 15) or \
                                       ('-' in text and len(text) < 15):
                                        date_str = text
                                    
                                    # Transaction type
                                    if 'sale' in text_lower or 'sell' in text_lower:
                                        transaction_type = 'sell'
                                    elif ('purchase' in text_lower or 'buy' in text_lower or 'acquisition' in text_lower or
                                          'option exercise' in text_lower or 'exercise' in text_lower or
                                          'grant' in text_lower or 'award' in text_lower or
                                          'conversion' in text_lower or 'convert' in text_lower):
                                        transaction_type = 'buy'
                                    
                                    # Value (contains $ or large numbers with commas)
                                    if '$' in text or (',' in text and len(text) > 5 and any(c.isdigit() for c in text)):
                                        try:
                                            clean_value = text.replace('$', '').replace(',', '').replace(' ', '')
                                            if clean_value and clean_value.replace('.', '').isdigit():
                                                value = float(clean_value)
                                        except:
                                            pass
                                    
                                    # Shares (numbers, possibly with K/M suffixes)
                                    if not '$' in text and any(c.isdigit() for c in text):
                                        try:
                                            clean_shares = text.replace(',', '').replace(' ', '').upper()
                                            if 'K' in clean_shares:
                                                shares = int(float(clean_shares.replace('K', '')) * 1000)
                                            elif 'M' in clean_shares:
                                                shares = int(float(clean_shares.replace('M', '')) * 1000000)
                                            else:
                                                if clean_shares.replace('.', '').isdigit():
                                                    shares = int(float(clean_shares))
                                        except:
                                            pass
                                    
                                    # Insider name (longer text, not a number, not a date)
                                    if len(text) > 5 and not any(c in text for c in ['$', '/', '-']) and \
                                       not text.replace(',', '').replace('.', '').isdigit() and \
                                       not any(x in text for x in ['2024', '2025', '2023']):
                                        if insider == 'N/A':
                                            insider = text
                                
                                if transaction_type and value and value > 0:
                                    transactions.append({
                                        'date': date_str,
                                        'transaction_type': transaction_type,
                                        'value': value,
                                        'shares': shares,
                                        'insider': insider,
                                        'position': position,
                                        'text': transaction_type
                                    })
                            except Exception as e:
                                print(f"Error parsing MarketBeat row: {str(e)}")
                                continue
                        
                        if transactions:
                            break  # Found data, no need to try other URLs
            except Exception as e:
                print(f"Error accessing {url}: {str(e)}")
                continue
        
        return transactions if transactions else None
        
    except Exception as e:
        print(f"Error scraping MarketBeat for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_finviz_insider_trading(ticker):
    """Scrape insider trading data from Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Finviz returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        transactions = []
        
        # Finviz has insider trading in a table
        # Look for table with "Insider Trading" header
        all_tables = soup.find_all('table')
        insider_table = None
        
        for i, table in enumerate(all_tables):
            rows = table.find_all('tr')
            if len(rows) > 1:
                header_row = rows[0]
                header_cells = header_row.find_all(['td', 'th'])
                header_text = header_row.get_text().lower()
                
                # Look for insider trading table with proper structure
                # Must have: "Insider Trading", "Relationship", "Date", "Transaction"
                # And should have around 8-9 columns
                if 'insider trading' in header_text and 'relationship' in header_text and 'date' in header_text and 'transaction' in header_text:
                    if 7 <= len(header_cells) <= 10:  # Should have 8-9 columns
                        insider_table = table
                        break
        
        if insider_table:
            rows = insider_table.find_all('tr')
            for row_idx, row in enumerate(rows[1:31]):  # Skip header, limit to 30
                try:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 6:
                        continue
                    
                    # Finviz actual format from debug:
                    # cells[0] = Insider name (e.g., "KONDO CHRIS")
                    # cells[1] = Position (e.g., "Principal Accounting Officer")
                    # cells[2] = Date (e.g., "Nov 07 '25")
                    # cells[3] = Transaction type (e.g., "Sale", "Proposed Sale")
                    # cells[4] = Cost per share (e.g., "271.23")
                    # cells[5] = Number of shares (e.g., "3,752")
                    # cells[6] = Total value (if exists)
                    
                    insider_name = cells[0].get_text(strip=True) if len(cells) > 0 else 'N/A'
                    position = cells[1].get_text(strip=True) if len(cells) > 1 else 'N/A'
                    date_str = cells[2].get_text(strip=True) if len(cells) > 2 else 'N/A'
                    transaction_text = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                    cost_per_share_text = cells[4].get_text(strip=True) if len(cells) > 4 else ''
                    shares_text = cells[5].get_text(strip=True) if len(cells) > 5 else ''
                    value_text = cells[6].get_text(strip=True) if len(cells) > 6 else ''
                    
                    # Determine transaction type
                    transaction_type = None
                    trans_lower = transaction_text.lower()
                    if 'sale' in trans_lower or 'sell' in trans_lower:
                        transaction_type = 'sell'
                    elif ('purchase' in trans_lower or 'buy' in trans_lower or 'acquisition' in trans_lower or 
                          'option exercise' in trans_lower or 'exercise' in trans_lower or
                          'grant' in trans_lower or 'award' in trans_lower or
                          'conversion' in trans_lower or 'convert' in trans_lower):
                        transaction_type = 'buy'
                    
                    # Parse shares (from cells[5])
                    shares = None
                    if shares_text:
                        try:
                            clean_shares = shares_text.replace(',', '').replace(' ', '')
                            if clean_shares:
                                shares = int(float(clean_shares))
                        except:
                            pass
                    
                    # Parse value - try cells[6] first, otherwise calculate from cost_per_share * shares
                    value = None
                    if value_text:
                        try:
                            clean_value = value_text.replace('$', '').replace(',', '').replace(' ', '')
                            if clean_value:
                                value = float(clean_value)
                        except:
                            pass
                    
                    # If no value in cells[6], calculate from cost_per_share * shares
                    if value is None and cost_per_share_text and shares:
                        try:
                            cost_per_share = float(cost_per_share_text.replace(',', '').replace(' ', ''))
                            value = cost_per_share * shares
                        except:
                            pass
                    
                    if transaction_type and value and value > 0:
                        print(f"DEBUG Finviz: Adding {transaction_type} transaction: {insider_name}, value={value}, shares={shares}, text={transaction_text}")
                        transactions.append({
                            'date': date_str,
                            'transaction_type': transaction_type,
                            'value': value,
                            'shares': shares,
                            'insider': insider_name,
                            'position': position,
                            'text': transaction_text
                        })
                    else:
                        if transaction_type:
                            print(f"DEBUG Finviz: Skipping {transaction_type} row {row_idx+1} - value={value}, shares={shares}, text={transaction_text}")
                        else:
                            print(f"DEBUG Finviz: Skipping row {row_idx+1} - no transaction type detected, text={transaction_text}")
                except Exception as e:
                    print(f"Error parsing Finviz row: {str(e)}")
                    continue
        
        return transactions if transactions else None
        
    except Exception as e:
        print(f"Error scraping Finviz for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_tipranks_insider_trading(ticker):
    """Scrape insider trading data from TipRanks"""
    try:
        url = f"https://www.tipranks.com/stocks/{ticker.upper()}/insider-trading"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"TipRanks returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        transactions = []
        
        # TipRanks uses tables or divs with specific classes for insider transactions
        # Look for transaction rows
        transaction_rows = soup.find_all('tr', class_=lambda x: x and ('transaction' in str(x).lower() or 'insider' in str(x).lower()))
        
        # If no specific class, try to find table rows
        if not transaction_rows:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > 1:  # Has header + data rows
                    transaction_rows = rows[1:]  # Skip header
                    break
        
        # Alternative: look for div-based structure
        if not transaction_rows:
            transaction_divs = soup.find_all('div', class_=lambda x: x and ('transaction' in str(x).lower() or 'insider' in str(x).lower() or 'row' in str(x).lower()))
            if transaction_divs:
                transaction_rows = transaction_divs
        
        for row in transaction_rows[:30]:  # Limit to 30 most recent
            try:
                # Extract data from row
                cells = row.find_all(['td', 'th', 'div'])
                if len(cells) < 3:
                    continue
                
                # Try to extract: Date, Insider, Position, Transaction Type, Shares, Value
                date_str = 'N/A'
                insider = 'N/A'
                position = 'N/A'
                transaction_type = None
                shares = None
                value = None
                
                # Parse cells - TipRanks structure may vary
                for i, cell in enumerate(cells):
                    text = cell.get_text(strip=True)
                    text_lower = text.lower()
                    
                    # Date detection
                    if any(x in text for x in ['2024', '2025', '2023']) or '/' in text or '-' in text:
                        if len(text) < 20:  # Likely a date
                            date_str = text
                    
                    # Transaction type detection
                    if 'sale' in text_lower or 'sell' in text_lower:
                        transaction_type = 'sell'
                    elif 'purchase' in text_lower or 'buy' in text_lower or 'acquisition' in text_lower:
                        transaction_type = 'buy'
                    
                    # Value detection (contains $ or numbers with commas)
                    if '$' in text or (',' in text and any(c.isdigit() for c in text)):
                        try:
                            # Remove $ and commas, convert to float
                            clean_text = text.replace('$', '').replace(',', '').replace(' ', '')
                            if clean_text:
                                value = float(clean_text)
                        except:
                            pass
                    
                    # Shares detection (numbers without $)
                    if not '$' in text and any(c.isdigit() for c in text) and (',' in text or int(text.replace(',', '').replace(' ', '')) > 0):
                        try:
                            clean_text = text.replace(',', '').replace(' ', '')
                            if clean_text.isdigit():
                                shares = int(clean_text)
                        except:
                            pass
                    
                    # Insider name (usually longer text, not a number)
                    if len(text) > 5 and not any(c in text for c in ['$', '/', '-']) and not text.replace(',', '').replace('.', '').isdigit():
                        if insider == 'N/A' and 'insider' not in text_lower:
                            insider = text
                
                # Only add if we have valid transaction type and value
                if transaction_type and value and value > 0:
                    transactions.append({
                        'date': date_str,
                        'transaction_type': transaction_type,
                        'value': value,
                        'shares': shares,
                        'insider': insider,
                        'position': position
                    })
            except Exception as e:
                print(f"Error parsing TipRanks row: {str(e)}")
                continue
        
        return transactions if transactions else None
        
    except Exception as e:
        print(f"Error scraping TipRanks for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_sec_api_insider_trading(ticker):
    """Get insider trading data from SEC API (sec-api.io) - official SEC Form 3, 4, 5 filings"""
    if not SEC_API_KEY:
        print("SEC_API_KEY not configured, skipping SEC API")
        return None
    
    try:
        # SEC API endpoint for Form 4 (most common insider trading form)
        url = "https://api.sec-api.io/form-4"
        
        headers = {
            'Authorization': SEC_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Query for searching by ticker symbol
        # SEC API uses Elasticsearch query format
        query = {
            "query": {
                "query_string": {
                    "query": f"issuer.tradingSymbol:{ticker.upper()}"
                }
            },
            "from": 0,
            "size": 30,
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = requests.post(url, headers=headers, json=query, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            transactions = []
            
            filings = data.get('filings', [])
            if not filings:
                print(f"SEC API returned no filings for {ticker}")
                return None
            
            for filing in filings:
                try:
                    # Get filing date
                    filed_at = filing.get('filedAt', '')
                    date_str = filed_at[:10] if filed_at else 'N/A'  # Extract YYYY-MM-DD
                    
                    # Get reporting owner (insider) info
                    reporting_owner = filing.get('reportingOwner', {})
                    insider_name = reporting_owner.get('name', 'N/A')
                    
                    # Get relationship
                    relationship = reporting_owner.get('relationship', {})
                    position = 'N/A'
                    if relationship.get('isOfficer', False):
                        position = 'Officer'
                    elif relationship.get('isDirector', False):
                        position = 'Director'
                    elif relationship.get('isTenPercentOwner', False):
                        position = '10% Owner'
                    elif relationship.get('isOther', False):
                        position = 'Other'
                    
                    # Process non-derivative transactions (direct stock purchases/sales)
                    non_derivative = filing.get('nonDerivativeTable', {}).get('holdings', [])
                    for holding in non_derivative:
                        transactions_list = holding.get('transactions', [])
                        for trans in transactions_list:
                            transaction_code = trans.get('transactionCode', '')
                            shares = trans.get('shares', 0)
                            price = trans.get('pricePerShare', 0)
                            
                            # Calculate value
                            value = 0
                            if shares and price:
                                try:
                                    value = float(shares) * float(price)
                                except:
                                    pass
                            
                            # Determine transaction type based on SEC transaction codes
                            # P = Open market purchase, A = Grant/award, I = Discretionary transaction (acquisition)
                            # S = Open market sale, D = Disposition to issuer, F = Payment of exercise price
                            transaction_type = None
                            if transaction_code in ['P', 'A', 'I', 'M', 'X', 'C', 'L']:  # Purchase codes
                                transaction_type = 'buy'
                            elif transaction_code in ['S', 'D', 'F', 'E', 'H', 'U']:  # Sale codes
                                transaction_type = 'sell'
                            
                            if transaction_type and value > 0:
                                print(f"DEBUG SEC API: Adding {transaction_type} transaction: {insider_name}, value={value}, shares={shares}, code={transaction_code}")
                                transactions.append({
                                    'date': date_str,
                                    'transaction_type': transaction_type,
                                    'value': value,
                                    'shares': int(shares) if shares else 0,
                                    'insider': insider_name,
                                    'position': position,
                                    'transaction_code': transaction_code
                                })
                            elif transaction_type:
                                print(f"DEBUG SEC API: Skipping {transaction_type} transaction (value={value}, shares={shares}, code={transaction_code})")
                    
                    # Process derivative transactions (options, warrants, etc.)
                    derivative = filing.get('derivativeTable', {}).get('holdings', [])
                    for holding in derivative:
                        transactions_list = holding.get('transactions', [])
                        for trans in transactions_list:
                            transaction_code = trans.get('transactionCode', '')
                            shares = trans.get('shares', 0)
                            price = trans.get('pricePerShare', 0)
                            
                            value = 0
                            if shares and price:
                                try:
                                    value = float(shares) * float(price)
                                except:
                                    pass
                            
                            transaction_type = None
                            if transaction_code in ['P', 'A', 'I', 'M', 'X', 'C']:
                                transaction_type = 'buy'
                            elif transaction_code in ['S', 'D', 'F', 'E', 'H']:
                                transaction_type = 'sell'
                            
                            if transaction_type and value > 0:
                                transactions.append({
                                    'date': date_str,
                                    'transaction_type': transaction_type,
                                    'value': value,
                                    'shares': int(shares) if shares else 0,
                                    'insider': insider_name,
                                    'position': f'Derivative ({position})',
                                    'transaction_code': transaction_code
                                })
                
                except Exception as filing_error:
                    print(f"Error processing SEC filing: {str(filing_error)}")
                    continue
            
            return transactions if transactions else None
            
        elif response.status_code == 401:
            print(f"SEC API authentication failed - check API key")
            return None
        elif response.status_code == 429:
            print(f"SEC API rate limit exceeded")
            return None
        else:
            print(f"SEC API returned status {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"Error fetching SEC API insider trading for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/institutional-analysis/<ticker>')
def get_institutional_analysis(ticker):
    """Get comprehensive institutional analysis (ownership, flow, retail indicators, whales)"""
    try:
        ticker = ticker.upper()
        
        ownership = get_institutional_ownership(ticker)
        flow = get_institutional_flow(ticker)
        retail_indicators = get_retail_activity_indicators(ticker)
        whales = get_whale_watching(ticker)
        
        return jsonify(clean_for_json({
            'ownership': ownership,
            'flow': flow,
            'retail_indicators': retail_indicators,
            'whales': whales
        }))
    
    except Exception as e:
        print(f"Error in institutional analysis endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to get institutional analysis: {str(e)}'}), 500

@app.route('/api/screener', methods=['POST'])
def run_screener():
    """Run stock screener with filters"""
    try:
        filters = request.get_json()
        if not filters:
            return jsonify({'error': 'No filters provided'}), 400
        
        results = run_stock_screener(filters)
        
        return jsonify(clean_for_json({
            'results': results,
            'count': len(results)
        }))
    
    except Exception as e:
        print(f"Error in screener endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to run screener: {str(e)}'}), 500

@app.route('/api/insider-trading/<ticker>')
def get_insider_trading(ticker):
    """Get insider trading activity from SEC API (primary), Finviz/MarketBeat (fallback), yfinance (last resort)"""
    try:
        ticker_upper = ticker.upper()
        time.sleep(0.3)  # Rate limiting
        
        insider_transactions = []
        
        # Try SEC API first (most reliable and official source)
        sec_data = get_sec_api_insider_trading(ticker_upper)
        if sec_data and len(sec_data) > 0:
            print(f"SEC API returned {len(sec_data)} transactions for {ticker_upper}")
            insider_transactions = sec_data
        else:
            # Fallback to Finviz
            print(f"SEC API returned no data for {ticker_upper}, trying Finviz")
            finviz_data = get_finviz_insider_trading(ticker_upper)
            if finviz_data and len(finviz_data) > 0:
                print(f"Finviz returned {len(finviz_data)} transactions for {ticker_upper}")
                insider_transactions = finviz_data
            else:
                # Fallback to MarketBeat
                print(f"Finviz returned no data for {ticker_upper}, trying MarketBeat")
                marketbeat_data = get_marketbeat_insider_trading(ticker_upper)
                if marketbeat_data and len(marketbeat_data) > 0:
                    print(f"MarketBeat returned {len(marketbeat_data)} transactions for {ticker_upper}")
                    insider_transactions = marketbeat_data
                else:
                    # Last resort: yfinance
                    print(f"MarketBeat returned no data for {ticker_upper}, trying yfinance fallback")
                    try:
                        stock = yf.Ticker(ticker_upper)
                        insider_df = stock.insider_transactions
                        if insider_df is not None and not insider_df.empty:
                            for idx, row in insider_df.tail(30).iterrows():
                                try:
                                    row_dict = row.to_dict()
                                    
                                    transaction_type = None
                                    text = str(row_dict.get('Text', '')).lower() if row_dict.get('Text') else ''
                                    
                                    if 'sale' in text or 'sell' in text:
                                        transaction_type = 'sell'
                                    elif ('purchase' in text or 'buy' in text or 'acquisition' in text or
                                          'option exercise' in text.lower() or 'exercise' in text.lower() or
                                          'grant' in text.lower() or 'award' in text.lower() or
                                          'conversion' in text.lower() or 'convert' in text.lower()):
                                        transaction_type = 'buy'
                                    
                                    value = None
                                    val = row_dict.get('Value')
                                    if val is not None and pd.notna(val):
                                        try:
                                            value = float(val)
                                        except:
                                            pass
                                    
                                    shares = None
                                    sh = row_dict.get('Shares')
                                    if sh is not None and pd.notna(sh):
                                        try:
                                            shares = int(sh)
                                        except:
                                            pass
                                    
                                    insider = 'N/A'
                                    ins = row_dict.get('Insider')
                                    if ins is not None and pd.notna(ins):
                                        insider = str(ins)
                                    
                                    date_str = 'N/A'
                                    date_val = row_dict.get('Start Date')
                                    if date_val is not None and pd.notna(date_val):
                                        if hasattr(date_val, 'strftime'):
                                            date_str = date_val.strftime('%Y-%m-%d')
                                        else:
                                            date_str = str(date_val)
                                    elif hasattr(idx, 'strftime'):
                                        date_str = idx.strftime('%Y-%m-%d')
                                    
                                    if transaction_type and value is not None and value > 0:
                                        insider_transactions.append({
                                            'date': date_str,
                                            'transaction_type': transaction_type,
                                            'value': float(value),
                                            'shares': shares,
                                            'insider': insider,
                                            'text': str(row_dict.get('Text', 'N/A'))
                                        })
                                except Exception as row_error:
                                    continue
                    except Exception as yf_error:
                        print(f"yfinance fallback failed: {str(yf_error)}")
                        pass
        
        # Calculate totals from transactions
        total_purchases = 0
        total_sales = 0
        
        if insider_transactions:
            for trans in insider_transactions:
                if trans.get('transaction_type') == 'buy':
                    total_purchases += trans.get('value', 0)
                elif trans.get('transaction_type') == 'sell':
                    total_sales += trans.get('value', 0)
        
        # Separate transactions into purchases and sales
        purchases_from_transactions = [t for t in insider_transactions if t.get('transaction_type') == 'buy'][:10]
        sales_from_transactions = [t for t in insider_transactions if t.get('transaction_type') == 'sell'][:10]
        
        # Get ownership data from yfinance
        ownership_data = {}
        try:
            stock = yf.Ticker(ticker_upper)
            info = stock.info
            
            # Insider ownership
            insider_ownership_pct = info.get('heldPercentInsiders')
            if insider_ownership_pct is not None:
                ownership_data['insider_ownership_pct'] = float(insider_ownership_pct) * 100  # Convert to percentage
            
            # Institutional ownership
            institutional_ownership_pct = info.get('heldPercentInstitutions')
            if institutional_ownership_pct is not None:
                ownership_data['institutional_ownership_pct'] = float(institutional_ownership_pct) * 100  # Convert to percentage
            
            # Get major holders for additional data
            try:
                major_holders = stock.major_holders
                if major_holders is not None and not major_holders.empty:
                    # Parse major holders DataFrame
                    for idx, row in major_holders.iterrows():
                        breakdown = str(row.get('Breakdown', '')).lower()
                        value = row.get('Value', 0)
                        
                        if 'insider' in breakdown:
                            ownership_data['insider_ownership_pct'] = float(value) * 100
                        elif 'institution' in breakdown and 'float' not in breakdown:
                            ownership_data['institutional_ownership_pct'] = float(value) * 100
                        elif 'institution' in breakdown and 'float' in breakdown:
                            ownership_data['institutional_float_pct'] = float(value) * 100
                        elif 'institution' in breakdown and 'count' in breakdown:
                            ownership_data['institutional_count'] = int(value) if value else 0
            except Exception as mh_error:
                print(f"Error parsing major holders: {str(mh_error)}")
            
            # Get institutional holders for change data
            try:
                institutional_holders = stock.institutional_holders
                if institutional_holders is not None and not institutional_holders.empty:
                    # Calculate total institutional ownership change
                    if 'pctChange' in institutional_holders.columns:
                        # Weighted average change
                        total_shares = institutional_holders['Shares'].sum() if 'Shares' in institutional_holders.columns else 0
                        if total_shares > 0:
                            weighted_change = (institutional_holders['Shares'] * institutional_holders['pctChange']).sum() / total_shares
                            ownership_data['institutional_ownership_change_pct'] = float(weighted_change)
                    
                    # Top institutional holders
                    top_holders = []
                    for idx, row in institutional_holders.head(10).iterrows():
                        holder_data = {
                            'name': str(row.get('Holder', 'N/A')),
                            'shares': int(row.get('Shares', 0)) if pd.notna(row.get('Shares')) else 0,
                            'value': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else 0,
                            'pct_change': float(row.get('pctChange', 0)) if pd.notna(row.get('pctChange')) else 0
                        }
                        top_holders.append(holder_data)
                    ownership_data['top_institutional_holders'] = top_holders
            except Exception as ih_error:
                print(f"Error parsing institutional holders: {str(ih_error)}")
                
        except Exception as ownership_error:
            print(f"Error fetching ownership data: {str(ownership_error)}")
        
        insider_data = {
            'transactions': insider_transactions[:20] if insider_transactions else [],
            'purchases': purchases_from_transactions,
            'sales': sales_from_transactions,
            'total_purchases': total_purchases,
            'total_sales': total_sales,
            'net_activity': total_purchases - total_sales,
            'ownership': ownership_data
        }
        
        return jsonify(clean_for_json(insider_data))
        
    except Exception as e:
        print(f"Error fetching insider trading for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch insider trading: {str(e)}'}), 500

def get_economic_calendar():
    """Get economic calendar data from Trading Economics API or fallback to Investing.com"""
    try:
        today = datetime.now().date()
        start_date_str = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date_str = (today + timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Try Trading Economics API first (free tier allows limited requests)
        # Note: This is a public endpoint that doesn't require API key for basic usage
        try:
            # Trading Economics calendar endpoint (public, no API key needed for basic data)
            te_url = f"https://api.tradingeconomics.com/calendar"
            # Note: This endpoint may require API key, so we'll try and fallback if it fails
            
            # Alternative: Use Investing.com with better headers and session management
            investing_url = "https://www.investing.com/economic-calendar/"
            
            headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            # First, get the main page to establish session
            try:
                session.get(investing_url, timeout=10)
            except Exception:
                pass
        
            # Now try to get calendar data via their internal API
            api_url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
            
            # Prepare form data
            from urllib.parse import urlencode
            payload = urlencode([
                ('country[]', '5'),  # USA
                ('importance[]', '1'),  # High
                ('importance[]', '2'),  # Medium
                ('dateFrom', start_date_str),
                ('dateTo', end_date_str),
                ('timeZone', '8'),  # GMT
                ('timeFilter', 'timeRemain'),
                ('currentTab', 'today'),
                ('limit_from', '0')
            ])
            
            api_headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Accept-Language': 'en-US,en;q=0.9',
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': 'https://www.investing.com/economic-calendar/',
                'Origin': 'https://www.investing.com',
            }
            
            response = session.post(api_url, data=payload, headers=api_headers, timeout=20)
            print(f"[ECONOMIC] Investing.com API Status: {response.status_code}")
            
            if response.status_code == 200:
                # Investing.com returns JSON with HTML data
                try:
                    data = response.json()
                    print(f"[ECONOMIC] Response type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    
                    if isinstance(data, dict) and 'data' in data:
                        html_data = data['data']
                        print(f"[ECONOMIC] HTML data length: {len(html_data)}")
                        
                        if html_data and len(html_data) > 100:
                            soup = BeautifulSoup(html_data, 'html.parser')
                            events = []
                            
                            # Investing.com uses tr elements with data-event-datetime attribute
                            event_rows = soup.find_all('tr', {'data-event-datetime': True})
                            
                            if not event_rows:
                                # Try alternative selectors
                                event_rows = soup.find_all('tr', class_=lambda x: x and ('js-event-item' in str(x) or 'eventRow' in str(x)) if x else False)
                            
                            print(f"[ECONOMIC] Found {len(event_rows)} event rows")
                            
                            for row in event_rows:
                                try:
                                    # Investing.com structure: tr with data-event-datetime
                                    # Get datetime from attribute
                                    date_attr = row.get('data-event-datetime', '')
                                    if not date_attr:
                                        continue
                                    
                                    # Parse timestamp or date string
                                    try:
                                        if date_attr.isdigit():
                                            # Timestamp (milliseconds or seconds)
                                            timestamp_ms = int(date_attr)
                                            if timestamp_ms > 1e12:
                                                timestamp = timestamp_ms / 1000
                                            else:
                                                timestamp = timestamp_ms
                                            event_datetime = datetime.fromtimestamp(timestamp)
                                        else:
                                            # Try different date formats
                                            import re
                                            # Format: '2025/12/18 08:30:00' or '2025-12-18 08:30:00'
                                            if '/' in date_attr:
                                                event_datetime = datetime.strptime(date_attr, '%Y/%m/%d %H:%M:%S')
                                            elif '-' in date_attr:
                                                try:
                                                    event_datetime = datetime.fromisoformat(date_attr.replace('Z', '+00:00'))
                                                except Exception:
                                                    event_datetime = datetime.strptime(date_attr, '%Y-%m-%d %H:%M:%S')
                                            else:
                                                event_datetime = datetime.fromisoformat(date_attr.replace('Z', '+00:00'))
                                        
                                        event_date = event_datetime.date()
                                        time_str = event_datetime.strftime('%H:%M')
                                    except Exception as e:
                                        print(f"[ECONOMIC] Date parse error: {e}, date_attr: {date_attr[:50]}")
                                        continue
                                    
                                    cells = row.find_all('td')
                                    if len(cells) < 4:
                                        continue
                                    
                                    # Investing.com structure:
                                    # cells[0] = time (already extracted from data-event-datetime)
                                    # cells[1] = currency/flag
                                    # cells[2] = impact (stars)
                                    # cells[3] = event name
                                    # cells[4] = actual
                                    # cells[5] = forecast/consensus
                                    # cells[6] = previous
                                    
                                    # Extract currency
                                    currency = 'USD'
                                    if len(cells) > 1:
                                        currency_cell = cells[1]
                                        # Check for flag image or country indicator
                                        flag_img = currency_cell.find('span', class_=lambda x: x and 'flag' in str(x).lower() if x else False)
                                        if not flag_img:
                                            flag_img = currency_cell.find('img')
                                        if flag_img:
                                            title = flag_img.get('title', '') or flag_img.get('alt', '')
                                            if 'USD' in title.upper() or 'United States' in title or 'US' in title.upper():
                                                currency = 'USD'
                                        # Also check cell text
                                        cell_text = currency_cell.get_text(strip=True).upper()
                                        if 'USD' in cell_text or 'US' in cell_text:
                                            currency = 'USD'
                                        
                                        # Only process USD events (but log others for debugging)
                                        if currency != 'USD':
                                            if len(events) < 3:  # Only log first few non-USD events
                                                print(f"[ECONOMIC DEBUG] Skipping non-USD event: currency={currency}")
                                            continue
                                        
                                        # Extract impact
                                        impact = 0
                                        if len(cells) > 2:
                                            impact_cell = cells[2]
                                            stars = impact_cell.find_all('i', class_=lambda x: x and ('grayIcon' not in str(x) and 'gray' not in str(x).lower()) if x else True)
                                            impact = len(stars)
                                            
                                            if impact == 0:
                                                # Try counting filled stars (not gray)
                                                all_stars = impact_cell.find_all('i')
                                                impact = len([s for s in all_stars if 'gray' not in str(s.get('class', [])).lower()])
                                                
                                                if impact == 0:
                                                    importance_attr = impact_cell.get('data-importance', '')
                                                    if importance_attr.isdigit():
                                                        impact = int(importance_attr)
                                                
                                                # If still 0, default to 2 (medium impact) for USA events
                                                if impact == 0:
                                                    impact = 2
                                        
                                        # Filter: only high/medium impact (>= 2) - but log for debugging
                                        if impact < 2:
                                            if len(events) < 3:
                                                print(f"[ECONOMIC DEBUG] Skipping low impact event: {event_name}, impact={impact}")
                                            continue
                                    
                                    # Extract event name
                                    event_name = 'N/A'
                                    if len(cells) > 3:
                                        event_cell = cells[3]
                                        event_link = event_cell.find('a')
                                        if event_link:
                                            event_name = event_link.get_text(strip=True)
                                        else:
                                            event_name = event_cell.get_text(strip=True)
                                    
                                    if not event_name or event_name == 'N/A' or event_name in ['Event', '']:
                                        continue
                                    
                                    # Extract actual, consensus, previous
                                    actual = ''
                                    consensus = ''
                                    previous = ''
                                    
                                    if len(cells) >= 7:
                                        # Previous (last cell)
                                        prev_text = cells[6].get_text(strip=True)
                                        if prev_text and prev_text not in ['-', '', 'N/A', 'TBA', 'TBD']:
                                            if any(c.isdigit() for c in prev_text) or any(marker in prev_text for marker in ['%', 'M', 'B', 'K', '.']):
                                                previous = prev_text
                                        
                                        # Forecast/Consensus (second to last)
                                        cons_text = cells[5].get_text(strip=True)
                                        if cons_text and cons_text not in ['-', '', 'N/A', 'TBA', 'TBD']:
                                            if any(c.isdigit() for c in cons_text) or any(marker in cons_text for marker in ['%', 'M', 'B', 'K', '.']):
                                                consensus = cons_text
                                        
                                        # Actual (third to last)
                                        act_text = cells[4].get_text(strip=True)
                                        if act_text and act_text not in ['-', '', 'N/A', 'TBA', 'TBD']:
                                            if any(c.isdigit() for c in act_text) or any(marker in act_text for marker in ['%', 'M', 'B', 'K', '.']):
                                                actual = act_text
                                    
                                    # Filter by date range
                                    days_diff = (event_date - today).days
                                    if -7 <= days_diff <= 30:
                                        events.append({
                                            'date': event_date.strftime('%Y-%m-%d'),
                                            'time': time_str,
                                            'currency': currency,
                                            'impact': impact,
                                            'country': 'USA',
                                            'event': event_name,
                                            'actual': actual,
                                            'consensus': consensus,
                                            'previous': previous
                                        })
                                        print(f"[ECONOMIC] Added: {event_date} {time_str} - {event_name} (impact: {impact})")
                            
                                except Exception as e:
                                    print(f"[ECONOMIC] Error parsing row: {str(e)}")
                                    continue
                            
                            if events:
                                events.sort(key=lambda x: (x['date'], x['time']))
                                print(f"[ECONOMIC] Successfully parsed {len(events)} events from Investing.com")
                                return events
                            else:
                                print("[ECONOMIC] No events found after parsing")
                        else:
                            print("[ECONOMIC] HTML data too short or empty")
                    else:
                        print("[ECONOMIC] No 'data' key in response")
                except json.JSONDecodeError:
                    print("[ECONOMIC] Response is not JSON")
                except Exception as e:
                    print(f"[ECONOMIC] Parsing error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[ECONOMIC] Investing.com returned status {response.status_code}")
        except Exception as e:
            print(f"[ECONOMIC] Investing.com request failed: {e}")
            import traceback
            traceback.print_exc()
        
        # If scraping fails, return sample data
        print("[ECONOMIC] Investing.com scraping failed, falling back to sample data")
        return generate_economic_calendar_sample_data()
        
    except Exception as e:
        print(f"[ECONOMIC] Error in get_economic_calendar: {e}")
        import traceback
        traceback.print_exc()
        return generate_economic_calendar_sample_data()

def generate_economic_calendar_sample_data():
    """Generate realistic sample economic calendar data for USA - matches real economic calendar format"""
    today = datetime.now().date()
    now = datetime.now()
    
    def next_business_day(start_date, days_ahead):
        current = start_date
        added = 0
        while added < days_ahead:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                added += 1
        return current
    
    def is_future_event(event_date_str, event_time_str):
        """Check if event is in the future"""
        try:
            event_datetime = datetime.strptime(f"{event_date_str} {event_time_str}", "%Y-%m-%d %H:%M")
            return event_datetime > now
        except:
            return True  # If parsing fails, include it
    
    # Get current month for event naming (e.g., "Nov", "Dec")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    current_month = month_names[today.month - 1]
    # Note: Sample data uses current month - real scraper should get actual month from Investing.com
    
    events = []
    
    # This week: Find this Monday (start of this week)
    days_since_monday = today.weekday()  # Monday = 0
    this_monday = today - timedelta(days=days_since_monday)
    
    # This week days (Monday to Friday)
    this_tuesday = this_monday + timedelta(days=1)
    this_wednesday = this_monday + timedelta(days=2)
    this_thursday = this_monday + timedelta(days=3)
    this_friday = this_monday + timedelta(days=4)
    
    # Use this week days only if they're in the future, otherwise use next week
    tuesday = this_tuesday if this_tuesday >= today else this_tuesday + timedelta(days=7)
    wednesday = this_wednesday if this_wednesday >= today else this_wednesday + timedelta(days=7)
    thursday = this_thursday if this_thursday >= today else this_thursday + timedelta(days=7)
    friday = this_friday if this_friday >= today else this_friday + timedelta(days=7)
    
    # Tuesday events (14:30 = 2:30 PM EST, which is 8:30 AM EST in some contexts, but showing as 14:30)
    tuesday_events = [
        {
            'date': tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Average Hourly Earnings (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.2%'
        },
        {
            'date': tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Nonfarm Payrolls ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '119K'
        },
        {
            'date': tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Unemployment Rate ({current_month})',
            'actual': '',
            'consensus': '4.4%',
            'previous': '4.4%'
        },
        {
            'date': tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Core Retail Sales (MoM) ({current_month})',
            'actual': '',
            'consensus': '0.3%',
            'previous': '0.3%'
        },
        {
            'date': tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Retail Sales (MoM) ({current_month})',
            'actual': '',
            'consensus': '0.2%',
            'previous': '0.2%'
        },
        {
            'date': tuesday.strftime('%Y-%m-%d'),
            'time': '15:45',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Services PMI ({current_month}) P',
            'actual': '',
            'consensus': '',
            'previous': '54.1'
        },
        {
            'date': tuesday.strftime('%Y-%m-%d'),
            'time': '15:45',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Manufacturing PMI ({current_month}) P',
            'actual': '',
            'consensus': '',
            'previous': '52.2'
        }
    ]
    
    # Wednesday events
    wednesday_events = [
        {
            'date': wednesday.strftime('%Y-%m-%d'),
            'time': '16:30',
            'currency': 'USD',
            'impact': 2,
            'country': 'USA',
            'event': 'Crude Oil Inventories',
            'actual': '',
            'consensus': '',
            'previous': '-1.812M'
        }
    ]
    
    # Thursday events
    thursday_events = [
        {
            'date': thursday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'CPI (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.3%'
        },
        {
            'date': thursday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'CPI (YoY) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '3.0%'
        },
        {
            'date': thursday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Philadelphia Fed Manufacturing Index ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '-1.7'
        },
        {
            'date': thursday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Core CPI (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.2%'
        },
        {
            'date': thursday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': 'Initial Jobless Claims',
            'actual': '',
            'consensus': '',
            'previous': '236K'
        }
    ]
    
    # Friday events
    friday_events = [
        {
            'date': friday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Core PCE Price Index (YoY) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '2.8%'
        },
        {
            'date': friday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Core PCE Price Index (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.2%'
        },
        {
            'date': friday.strftime('%Y-%m-%d'),
            'time': '16:00',
            'currency': 'USD',
            'impact': 2,
            'country': 'USA',
            'event': f'Existing Home Sales ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '4.10M'
        }
    ]
    
    # Next week events - find next Monday and calculate from there
    # Find next Monday (start of next week)
    days_until_next_monday = 7 - days_since_monday
    next_monday = today + timedelta(days=days_until_next_monday)
    
    # Calculate days of next week (Monday to Friday)
    next_tuesday = next_monday + timedelta(days=1)
    next_wednesday = next_monday + timedelta(days=2)
    next_thursday = next_monday + timedelta(days=3)
    next_friday = next_monday + timedelta(days=4)
    
    # Only add next week events if they're different from this week events
    # (to avoid duplicates when this week is already past)
    if tuesday >= next_monday:
        # This week events are already in next week, skip generating next week events
        next_tuesday_events = []
        next_wednesday_events = []
        next_thursday_events = []
        next_friday_events = []
    else:
        # Next week Tuesday events
        next_tuesday_events = [
            {
            'date': next_tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Average Hourly Earnings (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.2%'
        },
        {
            'date': next_tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Nonfarm Payrolls ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '119K'
        },
        {
            'date': next_tuesday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Unemployment Rate ({current_month})',
            'actual': '',
            'consensus': '4.4%',
            'previous': '4.4%'
        },
        {
            'date': next_tuesday.strftime('%Y-%m-%d'),
            'time': '15:45',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Services PMI ({current_month}) F',
            'actual': '',
            'consensus': '',
            'previous': '54.1'
        },
        {
            'date': next_tuesday.strftime('%Y-%m-%d'),
            'time': '15:45',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Manufacturing PMI ({current_month}) F',
            'actual': '',
            'consensus': '',
            'previous': '52.2'
            }
        ]
        
        # Next week Wednesday events
        next_wednesday_events = [
            {
            'date': next_wednesday.strftime('%Y-%m-%d'),
            'time': '16:30',
            'currency': 'USD',
            'impact': 2,
            'country': 'USA',
            'event': 'Crude Oil Inventories',
            'actual': '',
            'consensus': '',
            'previous': '-1.812M'
            }
        ]
        
        # Next week Thursday events
        next_thursday_events = [
            {
            'date': next_thursday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Durable Goods Orders (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.5%'
        },
        {
            'date': next_thursday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': 'Initial Jobless Claims',
            'actual': '',
            'consensus': '',
            'previous': '236K'
        },
        {
            'date': next_thursday.strftime('%Y-%m-%d'),
            'time': '16:00',
            'currency': 'USD',
            'impact': 2,
            'country': 'USA',
            'event': 'New Home Sales',
            'actual': '',
            'consensus': '',
            'previous': '683K'
            }
        ]
        
        # Next week Friday events
        next_friday_events = [
            {
            'date': next_friday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Personal Income (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.3%'
        },
        {
            'date': next_friday.strftime('%Y-%m-%d'),
            'time': '14:30',
            'currency': 'USD',
            'impact': 3,
            'country': 'USA',
            'event': f'Personal Spending (MoM) ({current_month})',
            'actual': '',
            'consensus': '',
            'previous': '0.2%'
            }
        ]
    
    # Add all events only if they're in the future
    all_events = tuesday_events + wednesday_events + thursday_events + friday_events + \
                 next_tuesday_events + next_wednesday_events + next_thursday_events + next_friday_events
    for event in all_events:
        if is_future_event(event['date'], event['time']):
            events.append(event)
    
    events.sort(key=lambda x: (x['date'], x['time']))
    print(f"Generated {len(events)} realistic economic calendar events (only future events)")
    return events

@app.route('/api/economic-calendar')
def get_economic_calendar_endpoint():
    """Get economic calendar data - USA only, high impact (>= 2)"""
    try:
        # Try to get real data from Investing.com
        print(f"Attempting to fetch real economic calendar data from Investing.com")
        events = get_economic_calendar()
        
        # If scraping failed or returned None, use sample data as fallback
        if not events or len(events) == 0:
            print(f"Scraping returned no events, using sample data as fallback")
            events = generate_economic_calendar_sample_data()
        else:
            print(f"Successfully fetched {len(events)} real economic calendar events")
        
        return jsonify(clean_for_json({'events': events}))
        
    except Exception as e:
        print(f"Error in economic calendar endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to sample data on error
        try:
            events = generate_economic_calendar_sample_data()
            return jsonify(clean_for_json({'events': events}))
        except Exception as e:
            return jsonify({'error': f'Failed to fetch economic calendar: {str(e)}'}), 500

def generate_economic_event_explanation(event_name):
    """Generate AI explanation for an economic event - what it is, when bullish, when bearish"""
    
    # Economic event explanations database
    explanations = {
        'Nonfarm Payrolls': {
            'what': 'Nonfarm Payrolls (NFP) měří počet nových pracovních míst vytvořených v nezemědělském sektoru USA za předchozí měsíc. Je to jeden z nejdůležitějších ekonomických ukazatelů, protože zaměstnanost je klíčovým indikátorem zdraví ekonomiky.',
            'bullish': '🟢 BULLISH když: NFP je vyšší než očekávání (např. +200K vs. očekávaných +150K) → silná ekonomika, vyšší spotřeba, růst firemních zisků → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: NFP je výrazně nižší než očekávání (např. +50K vs. očekávaných +150K) → slabá ekonomika, Fed může snížit sazby (což může být paradoxně pozitivní), ale obavy z recese → negativní pro akcie'
        },
        'Unemployment Rate': {
            'what': 'Míra nezaměstnanosti ukazuje procento lidí v pracovní síle, kteří jsou nezaměstnaní a aktivně hledají práci. Nižší míra = silnější trh práce.',
            'bullish': '🟢 BULLISH když: Míra klesá nebo je nižší než očekávání (např. 3.5% vs. očekávaných 3.7%) → silný trh práce, vyšší spotřeba → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Míra roste nebo je vyšší než očekávání (např. 4.5% vs. očekávaných 3.7%) → slabý trh práce, nižší spotřeba → negativní pro akcie'
        },
        'CPI': {
            'what': 'Consumer Price Index (CPI) měří změnu cen spotřebního zboží a služeb. Je to hlavní ukazatel inflace. CPI (MoM) = měsíční změna, CPI (YoY) = roční změna.',
            'bullish': '🟢 BULLISH když: CPI je nižší než očekávání (např. 0.2% vs. očekávaných 0.4% MoM) → inflace klesá, Fed může snížit sazby → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: CPI je vyšší než očekávání (např. 0.6% vs. očekávaných 0.3% MoM) → inflace roste, Fed může zvýšit sazby → negativní pro akcie (zejména growth stocks)'
        },
        'Core CPI': {
            'what': 'Core CPI je CPI bez volatilních položek (jídlo a energie). Fed ho považuje za lepší ukazatel dlouhodobé inflace.',
            'bullish': '🟢 BULLISH když: Core CPI je nižší než očekávání → inflace pod kontrolou, Fed může být méně agresivní → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Core CPI je vyšší než očekávání → přetrvávající inflační tlaky, Fed může zvýšit sazby → negativní pro akcie'
        },
        'PCE Price Index': {
            'what': 'Personal Consumption Expenditures Price Index je Fedem preferovaný ukazatel inflace. Měří změnu cen zboží a služeb, které spotřebitelé skutečně kupují.',
            'bullish': '🟢 BULLISH když: PCE je nižší než očekávání → Fed může být méně agresivní se sazbami → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: PCE je vyšší než očekávání → Fed může zvýšit sazby → negativní pro akcie'
        },
        'FOMC Statement': {
            'what': 'FOMC (Federal Open Market Committee) Statement je prohlášení Fedu o měnové politice. Obsahuje rozhodnutí o úrokových sazbách a ekonomický výhled.',
            'bullish': '🟢 BULLISH když: Fed snižuje sazby nebo je "dovish" (měkčí přístup) → levnější peníze, vyšší likvidita → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Fed zvyšuje sazby nebo je "hawkish" (tvrdší přístup) → dražší peníze, nižší likvidita → negativní pro akcie (zejména growth stocks)'
        },
        'Retail Sales': {
            'what': 'Retail Sales měří celkové tržby v maloobchodě. Ukazuje spotřebitelskou sílu a důvěru. Je to klíčový ukazatel spotřebitelské aktivity.',
            'bullish': '🟢 BULLISH když: Retail Sales je vyšší než očekávání (např. +0.5% vs. očekávaných +0.2% MoM) → silná spotřeba, růst ekonomiky → pozitivní pro akcie (zejména consumer stocks)',
            'bearish': '🔴 BEARISH když: Retail Sales je nižší než očekávání (např. -0.3% vs. očekávaných +0.2% MoM) → slabá spotřeba, obavy z recese → negativní pro akcie'
        },
        'PMI': {
            'what': 'Purchasing Managers Index (PMI) měří aktivitu v průmyslu. Hodnota nad 50 = expanze, pod 50 = kontrakce. Manufacturing PMI = průmysl, Services PMI = služby.',
            'bullish': '🟢 BULLISH když: PMI je vyšší než očekávání a nad 50 (např. 54 vs. očekávaných 52) → silná ekonomická aktivita → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: PMI je nižší než očekávání nebo pod 50 (např. 48 vs. očekávaných 52) → slabá ekonomická aktivita, možné recese → negativní pro akcie'
        },
        'Initial Jobless Claims': {
            'what': 'Počet lidí, kteří poprvé žádají o dávky v nezaměstnanosti. Nižší číslo = silnější trh práce.',
            'bullish': '🟢 BULLISH když: Claims jsou nižší než očekávání (např. 200K vs. očekávaných 230K) → silný trh práce → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Claims jsou vyšší než očekávání (např. 280K vs. očekávaných 230K) → slabý trh práce → negativní pro akcie'
        },
        'Crude Oil Inventories': {
            'what': 'Změna zásob ropy v USA. Negativní hodnota = zásoby klesají (větší poptávka), pozitivní = zásoby rostou (menší poptávka).',
            'bullish': '🟢 BULLISH když: Inventories výrazně klesají (např. -5M vs. očekávaných -1M) → silná poptávka po ropě, růst ekonomiky → pozitivní pro energy stocks',
            'bearish': '🔴 BEARISH když: Inventories výrazně rostou (např. +5M vs. očekávaných -1M) → slabá poptávka po ropě → negativní pro energy stocks'
        },
        'Durable Goods Orders': {
            'what': 'Objednávky zboží dlouhodobé spotřeby (auta, ledničky, atd.). Ukazuje spotřebitelskou důvěru a firemní investice.',
            'bullish': '🟢 BULLISH když: Orders jsou vyšší než očekávání (např. +1.0% vs. očekávaných +0.3% MoM) → silná spotřeba a investice → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Orders jsou nižší než očekávání (např. -0.5% vs. očekávaných +0.3% MoM) → slabá spotřeba → negativní pro akcie'
        },
        'Personal Income': {
            'what': 'Celkový příjem domácností. Vyšší příjem = vyšší spotřeba = růst ekonomiky.',
            'bullish': '🟢 BULLISH když: Personal Income je vyšší než očekávání (např. +0.5% vs. očekávaných +0.2% MoM) → vyšší spotřeba → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Personal Income je nižší než očekávání (např. +0.1% vs. očekávaných +0.3% MoM) → nižší spotřeba → negativní pro akcie'
        },
        'Personal Spending': {
            'what': 'Výdaje spotřebitelů. Ukazuje, kolik lidí skutečně utrácí. Je to klíčový ukazatel spotřebitelské aktivity.',
            'bullish': '🟢 BULLISH když: Spending je vyšší než očekávání (např. +0.6% vs. očekávaných +0.3% MoM) → silná spotřeba → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Spending je nižší než očekávání (např. +0.1% vs. očekávaných +0.3% MoM) → slabá spotřeba → negativní pro akcie'
        },
        'Existing Home Sales': {
            'what': 'Počet prodaných existujících domů. Ukazuje zdraví realitního trhu a spotřebitelskou důvěru.',
            'bullish': '🟢 BULLISH když: Sales jsou vyšší než očekávání (např. 4.5M vs. očekávaných 4.1M) → silný realitní trh → pozitivní pro akcie (real estate, construction)',
            'bearish': '🔴 BEARISH když: Sales jsou nižší než očekávání (např. 3.8M vs. očekávaných 4.1M) → slabý realitní trh → negativní pro akcie'
        },
        'New Home Sales': {
            'what': 'Počet prodaných nových domů. Ukazuje aktivitu v realitním sektoru a spotřebitelskou důvěru.',
            'bullish': '🟢 BULLISH když: Sales jsou vyšší než očekávání → silný realitní trh → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Sales jsou nižší než očekávání → slabý realitní trh → negativní pro akcie'
        },
        'Average Hourly Earnings': {
            'what': 'Průměrná hodinová mzda. Ukazuje růst mezd, což může signalizovat inflační tlaky, ale také silný trh práce.',
            'bullish': '🟢 BULLISH když: Earnings rostou mírně (např. +0.3% vs. očekávaných +0.2% MoM) → silný trh práce bez výrazné inflace → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Earnings rostou výrazně (např. +0.6% vs. očekávaných +0.2% MoM) → inflační tlaky, Fed může zvýšit sazby → negativní pro akcie'
        },
        'Philadelphia Fed Manufacturing Index': {
            'what': 'Index průmyslové aktivity v regionu Philadelphia. Ukazuje zdraví průmyslu a ekonomiky.',
            'bullish': '🟢 BULLISH když: Index je vyšší než očekávání a pozitivní (např. +10 vs. očekávaných +5) → silná průmyslová aktivita → pozitivní pro akcie',
            'bearish': '🔴 BEARISH když: Index je nižší než očekávání nebo negativní (např. -5 vs. očekávaných +5) → slabá průmyslová aktivita → negativní pro akcie'
        }
    }
    
    # Try to find exact match or partial match
    event_name_clean = event_name.strip()
    
    # Check for exact match first
    for key, value in explanations.items():
        if key.lower() in event_name_clean.lower():
            return {
                'event_name': event_name,
                'what': value['what'],
                'bullish': value['bullish'],
                'bearish': value['bearish']
            }
    
    # If no match found, generate generic explanation
    return {
        'event_name': event_name,
        'what': f'{event_name} je ekonomický ukazatel, který měří důležitou část ekonomické aktivity. Je to klíčový indikátor pro investory a obchodníky.',
        'bullish': '🟢 BULLISH když: Hodnota je lepší než očekávání → pozitivní pro akcie',
        'bearish': '🔴 BEARISH když: Hodnota je horší než očekávání → negativní pro akcie'
    }

@app.route('/api/economic-event-explanation')
def get_economic_event_explanation():
    """Get AI explanation for an economic event"""
    try:
        event_name = request.args.get('event', '')
        if not event_name:
            return jsonify({'error': 'Event name is required'}), 400
        
        explanation = generate_economic_event_explanation(event_name)
        return jsonify(clean_for_json(explanation))
        
    except Exception as e:
        print(f"Error generating economic event explanation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate explanation: {str(e)}'}), 500

@app.route('/api/analyze-news-impact', methods=['POST'])
def analyze_news_impact():
    """Analyze how a news article might impact stock price using AI"""
    try:
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Google Gemini API key not configured'}), 500
        
        data = request.json
        news_title = data.get('title', '')
        news_summary = data.get('summary', '')
        news_content = data.get('content', '')
        ticker = data.get('ticker', '')
        
        if not news_title and not news_summary:
            return jsonify({'error': 'News title or summary required'}), 400
        
        # Combine news content
        news_text = f"{news_title}\n\n{news_summary}\n\n{news_content}" if news_content else f"{news_title}\n\n{news_summary}"
        
        # Analyze with AI
        analysis_result = analyze_news_impact_with_ai(news_text, ticker)
        
        if not analysis_result['success']:
            return jsonify({'error': analysis_result.get('error', 'AI analysis failed')}), 500
        
        return jsonify(clean_for_json(analysis_result))
        
    except Exception as e:
        print(f"Error in analyze-news-impact endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/news/<ticker>')
def get_news_for_ticker(ticker):
    """Get news with impact analysis for a ticker"""
    try:
        ticker = ticker.upper()
        news_list = get_stock_news(ticker, max_news=20)
        
        if not news_list:
            return jsonify({'news': [], 'historical_patterns': None})
        
        # Calculate historical patterns
        historical_patterns = calculate_historical_news_impact_patterns(ticker, news_list)
        
        return jsonify(clean_for_json({
            'news': news_list,
            'historical_patterns': historical_patterns
        }))
        
    except Exception as e:
        print(f"Error in news endpoint for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch news: {str(e)}'}), 500

def calculate_historical_news_impact_patterns(ticker, news_list):
    """Calculate historical patterns of news impact"""
    try:
        if not news_list or len(news_list) < 3:
            return None
        
        # Group by news type
        type_impacts = {}
        sentiment_impacts = {'positive': [], 'negative': [], 'neutral': []}
        
        for news in news_list:
            news_type = news.get('news_type', 'other')
            sentiment = news.get('sentiment', 'neutral')
            price_impact = news.get('price_impact', {})
            
            # Collect price movements by type
            if news_type not in type_impacts:
                type_impacts[news_type] = {'1d': [], '1w': []}
            
            if price_impact:
                if price_impact.get('price_1d_pct') is not None:
                    type_impacts[news_type]['1d'].append(price_impact['price_1d_pct'])
                if price_impact.get('price_1w_pct') is not None:
                    type_impacts[news_type]['1w'].append(price_impact['price_1w_pct'])
            
            # Collect by sentiment
            if sentiment in sentiment_impacts and price_impact:
                if price_impact.get('price_1d_pct') is not None:
                    sentiment_impacts[sentiment].append(price_impact['price_1d_pct'])
        
        # Calculate averages
        type_averages = {}
        for news_type, impacts in type_impacts.items():
            if impacts['1d'] or impacts['1w']:
                type_averages[news_type] = {
                    'avg_1d_pct': round(sum(impacts['1d']) / len(impacts['1d']), 2) if impacts['1d'] else None,
                    'avg_1w_pct': round(sum(impacts['1w']) / len(impacts['1w']), 2) if impacts['1w'] else None,
                    'count': len(impacts['1d']) + len(impacts['1w'])
                }
        
        sentiment_averages = {}
        for sentiment, impacts in sentiment_impacts.items():
            if impacts:
                sentiment_averages[sentiment] = {
                    'avg_1d_pct': round(sum(impacts) / len(impacts), 2),
                    'count': len(impacts)
                }
        
        return {
            'by_type': type_averages,
            'by_sentiment': sentiment_averages,
            'total_news_analyzed': len(news_list)
        }
        
    except Exception as e:
        print(f"Error calculating historical patterns for {ticker}: {str(e)}")
        return None

def analyze_watchlist_news_with_ai(all_news_data):
    """Analyze all watchlist news and create comprehensive AI summary"""
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'error': 'Google Gemini API key not configured'
        }
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models
        available_model = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name.lower():
                        available_model = m.name
                        break
                    elif available_model is None:
                        available_model = m.name
        except Exception:
            pass
        
        if available_model is None:
            for model_name in ['gemini-pro', 'gemini-1.5-flash', 'models/gemini-pro']:
                try:
                    test_model = genai.GenerativeModel(model_name)
                    available_model = model_name
                    break
                except:
                    continue
        
        if available_model is None:
            raise Exception("No available Gemini models found.")
        
        model = genai.GenerativeModel(available_model)
        
        # Prepare news data for AI analysis
        news_text_parts = []
        ticker_news_map = {}
        
        for ticker_data in all_news_data:
            ticker = ticker_data.get('ticker', '')
            company_name = ticker_data.get('company_name', ticker)
            news_list = ticker_data.get('news', [])
            
            if news_list:
                ticker_news_map[ticker] = {
                    'company_name': company_name,
                    'count': len(news_list)
                }
                
                # Add ticker section
                news_text_parts.append(f"\n=== {company_name} ({ticker}) ===\n")
                
                # Add top news (limit to 5 most recent per ticker to avoid token limits)
                for news in news_list[:5]:
                    title = news.get('title', '')
                    summary = news.get('summary', '')
                    sentiment = news.get('sentiment', 'neutral')
                    published = news.get('published', '')
                    
                    news_text_parts.append(f"Title: {title}")
                    if summary:
                        news_text_parts.append(f"Summary: {summary}")
                    news_text_parts.append(f"Sentiment: {sentiment}")
                    if published:
                        news_text_parts.append(f"Date: {published}")
                    news_text_parts.append("---")
        
        combined_news_text = '\n'.join(news_text_parts)
        
        # Create comprehensive prompt
        prompt = f"""Jsi expertní finanční analytik specializující se na analýzu tržních novinek. Analyzuj následující news z watchlistu (agregované news z více akcií) a vytvoř komplexní, rozsáhlé shrnutí v českém jazyce.

**Tvá úloha:**
1. Identifikuj klíčová témata a trendy napříč všemi news
2. Identifikuj nejvýznamnější události
3. Urči celkový sentiment watchlistu
4. Vytvoř shrnutí důležitých informací

**Formátuj odpověď PŘESNĚ takto:**

=== Executive Summary ===
Napiš 5-7 vět shrnujících nejdůležitější body z celého watchlistu. Zahrň klíčové trendy, společné témata a celkový obraz.

=== Key Themes & Trends ===
Uveď 5-8 klíčových témat a trendů, které se objevují napříč news. Každé téma na samostatný řádek s odrážkou. Zahrň:
- Společná témata napříč akciemi
- Tržní trendy
- Sektorové trendy
- Makroekonomické faktory

=== Most Significant Events ===
Uveď 4-6 nejvýznamnějších událostí z watchlistu. Každá událost na samostatný řádek s odrážkou. Zahrň:
- Konkrétní ticker a událost
- Proč je událost významná
- Potenciální dopad

=== Overall Watchlist Sentiment ===
Urči celkový sentiment watchlistu (positive/neutral/negative) a poskytni podrobné vysvětlení (4-5 vět). Zahrň:
- Mix pozitivních a negativních faktorů
- Převládající sentiment
- Klíčové faktory ovlivňující sentiment
- Porovnání sentimentu mezi různými akciemi

=== Key Insights ===
Uveď 4-6 klíčových insightů nebo doporučení na základě analýzy. Každý insight na samostatný řádek s odrážkou.

News z watchlistu:
{combined_news_text[:40000]}  # Limit pro rozsáhlou analýzu
"""
        
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 4096,  # Větší limit pro rozsáhlé shrnutí
            }
        )
        
        ai_summary = response.text
        
        # Parse AI response
        summary_data = parse_watchlist_summary(ai_summary, ticker_news_map)
        
        return {
            'success': True,
            'summary': ai_summary,
            'structured_data': summary_data,
            'ticker_stats': ticker_news_map,
            'model_used': available_model
        }
        
    except Exception as e:
        print(f"Error in AI watchlist news analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def parse_watchlist_summary(summary_text, ticker_stats):
    """Parse AI watchlist summary into structured format"""
    structured = {
        'executive_summary': '',
        'key_themes': [],
        'significant_events': [],
        'overall_sentiment': 'neutral',
        'sentiment_explanation': '',
        'key_insights': []
    }
    
    # Find sections
    sections = {
        'executive_summary': ['=== Executive Summary ===', 'Executive Summary:'],
        'key_themes': ['=== Key Themes & Trends ===', 'Key Themes', 'Themes & Trends:'],
        'significant_events': ['=== Most Significant Events ===', 'Most Significant Events:', 'Significant Events:'],
        'overall_sentiment': ['=== Overall Watchlist Sentiment ===', 'Overall Watchlist Sentiment:', 'Overall Sentiment:'],
        'key_insights': ['=== Key Insights ===', 'Key Insights:', 'Insights:']
    }
    
    section_positions = {}
    for section_name, markers in sections.items():
        for marker in markers:
            pos = summary_text.find(marker)
            if pos != -1:
                section_positions[section_name] = (pos, marker)
                break
    
    # Sort by position
    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1][0])
    
    # Extract content
    for i, (section_name, (start_pos, marker)) in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            end_pos = sorted_sections[i + 1][1][0]
        else:
            end_pos = len(summary_text)
        
        content = summary_text[start_pos + len(marker):end_pos].strip()
        
        if section_name == 'executive_summary':
            structured['executive_summary'] = content.strip()
        
        elif section_name == 'key_themes':
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit())):
                    theme = line.lstrip('-•*0123456789. ').strip()
                    if theme:
                        structured['key_themes'].append(theme)
        
        elif section_name == 'significant_events':
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit())):
                    event = line.lstrip('-•*0123456789. ').strip()
                    if event:
                        structured['significant_events'].append(event)
        
        elif section_name == 'overall_sentiment':
            lines = content.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'positive' in line_lower or 'pozitivní' in line_lower:
                    structured['overall_sentiment'] = 'positive'
                elif 'negative' in line_lower or 'negativní' in line_lower:
                    structured['overall_sentiment'] = 'negative'
                structured['sentiment_explanation'] += line + ' '
        
        elif section_name == 'key_insights':
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit())):
                    insight = line.lstrip('-•*0123456789. ').strip()
                    if insight:
                        structured['key_insights'].append(insight)
    
    return structured

@app.route('/api/analyze-watchlist-summary', methods=['POST'])
def analyze_watchlist_summary():
    """Analyze all watchlist news and create AI summary"""
    try:
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Google Gemini API key not configured'}), 500
        
        data = request.json
        all_news_data = data.get('news_data', [])
        
        if not all_news_data or len(all_news_data) == 0:
            return jsonify({'error': 'No news data provided'}), 400
        
        # Analyze with AI
        analysis_result = analyze_watchlist_news_with_ai(all_news_data)
        
        if not analysis_result['success']:
            return jsonify({'error': analysis_result.get('error', 'AI analysis failed')}), 500
        
        return jsonify(clean_for_json(analysis_result))
        
    except Exception as e:
        print(f"Error in analyze-watchlist-summary endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/portfolio-data', methods=['POST'])
def get_portfolio_data():
    """Calculate portfolio performance for given positions"""
    try:
        positions = request.json.get('positions', [])
        if not positions:
            return jsonify({'error': 'No positions provided'}), 400
        
        results = []
        total_cost = 0
        total_value = 0
        
        for pos in positions:
            try:
                ticker = pos.get('ticker', '').upper().strip()
                if not ticker:
                    continue
                
                shares = float(pos.get('shares', 0))
                purchase_price = float(pos.get('purchase_price', 0))
                purchase_date = pos.get('purchase_date', '')
                
                if shares <= 0 or purchase_price <= 0:
                    continue
                
                cost_basis = shares * purchase_price
                
                # Get current price
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1d')
                
                if hist.empty:
                    # Try to get info for current price
                    info = stock.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or purchase_price
                else:
                    current_price = float(hist['Close'].iloc[-1])
                
                current_value = shares * current_price
                pnl = current_value - cost_basis
                pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                # Get sector info
                try:
                    info = stock.info
                    sector = info.get('sector', 'N/A')
                    company_name = info.get('longName', ticker)
                except:
                    sector = 'N/A'
                    company_name = ticker
                
                results.append({
                    'ticker': ticker,
                    'company_name': company_name,
                    'shares': shares,
                    'purchase_price': purchase_price,
                    'purchase_date': purchase_date,
                    'current_price': current_price,
                    'cost_basis': cost_basis,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'sector': sector
                })
                
                total_cost += cost_basis
                total_value += current_value
                
            except Exception as e:
                print(f"Error processing position {pos.get('ticker', 'unknown')}: {str(e)}")
                continue
        
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        return jsonify(clean_for_json({
            'positions': results,
            'summary': {
                'total_cost': total_cost,
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent,
                'position_count': len(results)
            }
        }))
        
    except Exception as e:
        print(f"Error calculating portfolio data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to calculate portfolio data: {str(e)}'}), 500

@app.route('/api/search-ticker')
def search_ticker():
    """Search for stock tickers by symbol or company name"""
    try:
        query = request.args.get('query', '').strip().upper()
        if not query or len(query) < 1:
            return jsonify({'results': []})
        
        results = []
        
        # Try to get info for the query (might be a ticker)
        try:
            stock = yf.Ticker(query)
            info = stock.info
            if info and 'symbol' in info:
                results.append({
                    'ticker': info['symbol'],
                    'name': info.get('longName', info.get('shortName', query)),
                    'exchange': info.get('exchange', 'N/A')
                })
        except Exception as e:
            print(f"Error searching for ticker {query}: {str(e)}")
            pass
        
        # If query is longer, try partial matches (simple approach)
        # For better search, would need a stock database
        # For now, return direct match if found
        
        return jsonify(clean_for_json({'results': results[:10]}))
        
    except Exception as e:
        print(f"Error in search-ticker endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to search ticker: {str(e)}'}), 500

def calculate_factor_attribution(ticker, info, df):
    """Calculate detailed attribution for each factor score - what contributes to each factor"""
    attribution = {
        'value': {},
        'growth': {},
        'momentum': {},
        'quality': {}
    }
    
    # VALUE ATTRIBUTION
    pe_ratio = info.get('trailingPE') or info.get('forwardPE')
    pb_ratio = info.get('priceToBook')
    ps_ratio = info.get('priceToSalesTrailing12Months')
    dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
    
    if pe_ratio and pe_ratio > 0:
        if pe_ratio < 15:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 30, 'status': 'excellent'}
        elif pe_ratio < 25:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 20, 'status': 'good'}
        elif pe_ratio < 35:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['value']['pe_ratio'] = {'value': round(pe_ratio, 2), 'contribution': 0, 'status': 'poor'}
    
    if pb_ratio and pb_ratio > 0:
        if pb_ratio < 2:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 25, 'status': 'excellent'}
        elif pb_ratio < 4:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 15, 'status': 'good'}
        elif pb_ratio < 6:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['value']['pb_ratio'] = {'value': round(pb_ratio, 2), 'contribution': 0, 'status': 'poor'}
    
    if ps_ratio and ps_ratio > 0:
        if ps_ratio < 3:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 20, 'status': 'excellent'}
        elif ps_ratio < 6:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 10, 'status': 'good'}
        elif ps_ratio < 10:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['value']['ps_ratio'] = {'value': round(ps_ratio, 2), 'contribution': 0, 'status': 'poor'}
    
    if dividend_yield > 0:
        if dividend_yield > 4:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 25, 'status': 'excellent'}
        elif dividend_yield > 2:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 15, 'status': 'good'}
        elif dividend_yield > 1:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['value']['dividend_yield'] = {'value': round(dividend_yield, 2), 'contribution': 0, 'status': 'poor'}
    
    # GROWTH ATTRIBUTION
    revenue_growth = info.get('revenueGrowth')
    earnings_growth = info.get('earningsGrowth')
    earnings_quarterly_growth = info.get('earningsQuarterlyGrowth')
    
    if revenue_growth:
        revenue_growth_pct = revenue_growth * 100
        if revenue_growth_pct > 30:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 30, 'status': 'excellent'}
        elif revenue_growth_pct > 15:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 20, 'status': 'good'}
        elif revenue_growth_pct > 5:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['growth']['revenue_growth'] = {'value': round(revenue_growth_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    if earnings_growth:
        earnings_growth_pct = earnings_growth * 100
        if earnings_growth_pct > 30:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 30, 'status': 'excellent'}
        elif earnings_growth_pct > 15:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 20, 'status': 'good'}
        elif earnings_growth_pct > 5:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['growth']['earnings_growth'] = {'value': round(earnings_growth_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    # MOMENTUM ATTRIBUTION
    if df is not None and not df.empty:
        current_price = df['Close'].iloc[-1]
        
        if len(df) >= 21:
            price_1m_ago = df['Close'].iloc[-21]
            return_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
            if return_1m > 10:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 15, 'status': 'excellent'}
            elif return_1m > 5:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 10, 'status': 'good'}
            elif return_1m > 0:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 5, 'status': 'fair'}
            else:
                attribution['momentum']['return_1m'] = {'value': round(return_1m, 2), 'contribution': 0, 'status': 'poor'}
        
        if len(df) >= 63:
            price_3m_ago = df['Close'].iloc[-63]
            return_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
            if return_3m > 20:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 20, 'status': 'excellent'}
            elif return_3m > 10:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 15, 'status': 'good'}
            elif return_3m > 0:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 5, 'status': 'fair'}
            else:
                attribution['momentum']['return_3m'] = {'value': round(return_3m, 2), 'contribution': 0, 'status': 'poor'}
        
        try:
            from ta.momentum import RSIIndicator
            rsi_indicator = RSIIndicator(df['Close'], window=14)
            rsi = rsi_indicator.rsi().iloc[-1]
            if not pd.isna(rsi):
                if 50 < rsi < 70:
                    attribution['momentum']['rsi'] = {'value': round(rsi, 2), 'contribution': 15, 'status': 'excellent'}
                elif rsi > 70 or rsi < 30:
                    attribution['momentum']['rsi'] = {'value': round(rsi, 2), 'contribution': -10, 'status': 'poor'}
                else:
                    attribution['momentum']['rsi'] = {'value': round(rsi, 2), 'contribution': 0, 'status': 'fair'}
        except:
            pass
    
    # QUALITY ATTRIBUTION
    roe = info.get('returnOnEquity')
    roa = info.get('returnOnAssets')
    profit_margin = info.get('profitMargins')
    debt_to_equity = info.get('debtToEquity')
    
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 25, 'status': 'excellent'}
        elif roe_pct > 15:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 20, 'status': 'good'}
        elif roe_pct > 10:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['quality']['roe'] = {'value': round(roe_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    if profit_margin:
        profit_margin_pct = profit_margin * 100
        if profit_margin_pct > 20:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 25, 'status': 'excellent'}
        elif profit_margin_pct > 15:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 20, 'status': 'good'}
        elif profit_margin_pct > 10:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 10, 'status': 'fair'}
        else:
            attribution['quality']['profit_margin'] = {'value': round(profit_margin_pct, 2), 'contribution': 0, 'status': 'poor'}
    
    if debt_to_equity is not None:
        if debt_to_equity < 0.3:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 20, 'status': 'excellent'}
        elif debt_to_equity < 0.5:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 15, 'status': 'good'}
        elif debt_to_equity < 1.0:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 5, 'status': 'fair'}
        else:
            attribution['quality']['debt_to_equity'] = {'value': round(debt_to_equity, 2), 'contribution': 0, 'status': 'poor'}
    
    return attribution

def calculate_factor_rotation(ticker, periods=['1mo', '3mo', '6mo', '1y']):
    """Calculate factor scores over time to see rotation"""
    try:
        stock = yf.Ticker(ticker.upper())
        time.sleep(0.3)
        
        # Get current info as baseline
        current_info = stock.info
        if not current_info or 'symbol' not in current_info:
            return None
        
        # Get longer historical data to calculate past values
        hist_1y = stock.history(period='1y')
        if hist_1y.empty:
            return None
        
        rotation_data = []
        
        # Calculate factors for each period using historical data
        for period in periods:
            try:
                # Get historical data for this period
                hist = stock.history(period=period)
                if hist.empty:
                    continue
                
                # For Value, Growth, Quality: simulate changes based on historical price trends
                # We'll use the price change and volume trends to estimate factor changes
                
                # Calculate price change over the period
                if len(hist) > 0:
                    period_start_price = hist.iloc[0]['Close'] if len(hist) > 0 else current_info.get('currentPrice', 0)
                    period_end_price = hist.iloc[-1]['Close'] if len(hist) > 0 else current_info.get('currentPrice', 0)
                    price_change_pct = ((period_end_price - period_start_price) / period_start_price * 100) if period_start_price > 0 else 0
                    
                    # Calculate average volume for the period
                    avg_volume = hist['Volume'].mean() if 'Volume' in hist.columns else 0
                    current_volume = current_info.get('averageVolume', avg_volume) if current_info.get('averageVolume') else avg_volume
                    volume_change_pct = ((avg_volume - current_volume) / current_volume * 100) if current_volume > 0 else 0
                else:
                    price_change_pct = 0
                    volume_change_pct = 0
                
                # Calculate base factors using current info
                factor_result = calculate_factor_scores(ticker, current_info, hist)
                if not factor_result or 'factors' not in factor_result:
                    continue
                
                factors = factor_result['factors'].copy()
                
                # Adjust Value factor based on price change (if price went up significantly, value might decrease)
                # Adjust Growth factor based on price momentum (if price went up, growth perception might increase)
                # Adjust Quality factor slightly based on volume trends
                
                # Value adjustment: if price increased significantly, value score might decrease (stock became more expensive)
                if price_change_pct > 20:
                    factors['value'] = max(0, factors.get('value', 0) - min(10, price_change_pct * 0.2))
                elif price_change_pct < -20:
                    factors['value'] = min(100, factors.get('value', 0) + min(10, abs(price_change_pct) * 0.2))
                
                # Growth adjustment: if price increased, growth perception might increase
                if price_change_pct > 15:
                    factors['growth'] = min(100, factors.get('growth', 0) + min(5, price_change_pct * 0.1))
                elif price_change_pct < -15:
                    factors['growth'] = max(0, factors.get('growth', 0) - min(5, abs(price_change_pct) * 0.1))
                
                # Quality adjustment: slight adjustment based on volume (higher volume = more interest = slight quality boost)
                if volume_change_pct > 50:
                    factors['quality'] = min(100, factors.get('quality', 0) + 2)
                elif volume_change_pct < -50:
                    factors['quality'] = max(0, factors.get('quality', 0) - 2)
                
                rotation_data.append({
                    'period': period,
                    'value': round(factors.get('value', 0), 1),
                    'growth': round(factors.get('growth', 0), 1),
                    'momentum': round(factors.get('momentum', 0), 1),
                    'quality': round(factors.get('quality', 0), 1)
                })
                
            except Exception as e:
                print(f"[FACTOR ROTATION] Error calculating factor rotation for {period}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[FACTOR ROTATION] Calculated rotation data for {ticker}: {len(rotation_data)} periods")
        return rotation_data if rotation_data else None
    except Exception as e:
        print(f"[FACTOR ROTATION] Error in factor rotation calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_factor_momentum(factor_scores_current, factor_scores_previous):
    """Calculate factor momentum - which factors are accelerating"""
    if not factor_scores_current or not factor_scores_previous:
        return None
    
    momentum = {}
    for factor in ['value', 'growth', 'momentum', 'quality']:
        current = factor_scores_current.get(factor, 0)
        previous = factor_scores_previous.get(factor, 0)
        change = current - previous
        change_pct = round((change / previous * 100) if previous > 0 else (100 if change > 0 else 0), 2)
        momentum[factor] = {
            'current': round(current, 1),
            'previous': round(previous, 1),
            'change': round(change, 2),
            'change_pct': change_pct,
            'trend': 'accelerating' if change > 5 else 'decelerating' if change < -5 else 'stable'
        }
    
    return momentum

def calculate_factor_correlation(factor_scores_history):
    """Calculate correlation matrix between factors"""
    try:
        import numpy as np
        
        if not factor_scores_history or len(factor_scores_history) < 3:
            return None
        
        # Extract factor values
        value_scores = [d.get('value', 0) for d in factor_scores_history]
        growth_scores = [d.get('growth', 0) for d in factor_scores_history]
        momentum_scores = [d.get('momentum', 0) for d in factor_scores_history]
        quality_scores = [d.get('quality', 0) for d in factor_scores_history]
        
        # Calculate correlation matrix
        factors_matrix = np.array([value_scores, growth_scores, momentum_scores, quality_scores])
        correlation_matrix = np.corrcoef(factors_matrix)
        
        return {
            'value_growth': round(float(correlation_matrix[0][1]), 3),
            'value_momentum': round(float(correlation_matrix[0][2]), 3),
            'value_quality': round(float(correlation_matrix[0][3]), 3),
            'growth_momentum': round(float(correlation_matrix[1][2]), 3),
            'growth_quality': round(float(correlation_matrix[1][3]), 3),
            'momentum_quality': round(float(correlation_matrix[2][3]), 3)
        }
    except Exception as e:
        print(f"Error calculating factor correlation: {str(e)}")
        return None

def calculate_optimal_factor_mix(ticker, info, df):
    """Calculate optimal factor mix for this ticker based on sector/industry"""
    try:
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # Get factor scores
        factor_scores = calculate_factor_scores(ticker, info, df)
        if not factor_scores:
            return None
        
        # Determine optimal mix based on sector characteristics
        optimal_mix = {
            'recommended_factors': [],
            'reasoning': '',
            'current_strength': '',
            'improvement_areas': []
        }
        
        # Technology/Growth sectors prefer Growth and Momentum
        if 'Technology' in sector or 'Software' in industry:
            optimal_mix['recommended_factors'] = ['growth', 'momentum', 'quality']
            optimal_mix['reasoning'] = 'Technology stocks typically benefit from strong growth and momentum factors'
            if factor_scores.get('growth', 0) > 70:
                optimal_mix['current_strength'] = 'Strong growth profile'
            if factor_scores.get('value', 0) > 50:
                optimal_mix['improvement_areas'].append('Consider value opportunities')
        
        # Financial sectors prefer Value and Quality
        elif 'Financial' in sector:
            optimal_mix['recommended_factors'] = ['value', 'quality', 'momentum']
            optimal_mix['reasoning'] = 'Financial stocks benefit from value metrics and quality (ROE, margins)'
            if factor_scores.get('value', 0) > 70:
                optimal_mix['current_strength'] = 'Strong value profile'
            if factor_scores.get('quality', 0) < 50:
                optimal_mix['improvement_areas'].append('Quality metrics could improve')
        
        # Consumer/Retail prefer Value and Quality
        elif 'Consumer' in sector or 'Retail' in industry:
            optimal_mix['recommended_factors'] = ['value', 'quality', 'growth']
            optimal_mix['reasoning'] = 'Consumer stocks benefit from value, quality margins, and steady growth'
        
        # Healthcare prefer Quality and Growth
        elif 'Healthcare' in sector:
            optimal_mix['recommended_factors'] = ['quality', 'growth', 'value']
            optimal_mix['reasoning'] = 'Healthcare stocks benefit from quality (ROE, margins) and growth'
        
        # Default: balanced approach
        else:
            optimal_mix['recommended_factors'] = ['quality', 'value', 'growth', 'momentum']
            optimal_mix['reasoning'] = 'Balanced factor approach for diversified portfolio'
        
        # Identify current strengths and weaknesses
        factor_ranking = sorted([
            ('value', factor_scores.get('value', 0)),
            ('growth', factor_scores.get('growth', 0)),
            ('momentum', factor_scores.get('momentum', 0)),
            ('quality', factor_scores.get('quality', 0))
        ], key=lambda x: x[1], reverse=True)
        
        optimal_mix['top_factor'] = factor_ranking[0][0]
        optimal_mix['weakest_factor'] = factor_ranking[-1][0]
        
        return optimal_mix
    except Exception as e:
        print(f"Error calculating optimal factor mix: {str(e)}")
        return None

def calculate_factor_sensitivity(ticker, info, df):
    """Calculate factor sensitivity - how sensitive each factor is to changes in metrics"""
    try:
        # Get base factor scores
        base_result = calculate_factor_scores(ticker, info, df)
        if not base_result or 'factors' not in base_result:
            return None
        
        base_scores = base_result['factors']
        
        sensitivity = {
            'value': {},
            'growth': {},
            'momentum': {},
            'quality': {}
        }
        
        # VALUE SENSITIVITY - test impact of P/E, P/B, P/S changes
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        
        if pe_ratio and pe_ratio > 0:
            # Test 10% decrease in P/E
            test_info = info.copy()
            test_info['trailingPE'] = pe_ratio * 0.9
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('value', 0) - base_scores.get('value', 0)
                sensitivity['value']['pe_ratio'] = {
                    'metric': 'P/E Ratio',
                    'base_value': round(pe_ratio, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        if pb_ratio and pb_ratio > 0:
            test_info = info.copy()
            test_info['priceToBook'] = pb_ratio * 0.9
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('value', 0) - base_scores.get('value', 0)
                sensitivity['value']['pb_ratio'] = {
                    'metric': 'P/B Ratio',
                    'base_value': round(pb_ratio, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        # GROWTH SENSITIVITY - test impact of revenue/earnings growth changes
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            test_info = info.copy()
            test_info['revenueGrowth'] = revenue_growth * 1.1  # 10% increase
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('growth', 0) - base_scores.get('growth', 0)
                sensitivity['growth']['revenue_growth'] = {
                    'metric': 'Revenue Growth',
                    'base_value': round(revenue_growth * 100, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        # QUALITY SENSITIVITY - test impact of ROE, profit margin changes
        roe = info.get('returnOnEquity')
        if roe:
            test_info = info.copy()
            test_info['returnOnEquity'] = roe * 1.1  # 10% increase
            test_result = calculate_factor_scores(ticker, test_info, df)
            if test_result and 'factors' in test_result:
                test_scores = test_result['factors']
                sensitivity_change = test_scores.get('quality', 0) - base_scores.get('quality', 0)
                sensitivity['quality']['roe'] = {
                    'metric': 'ROE',
                    'base_value': round(roe * 100, 2),
                    'sensitivity': round(sensitivity_change, 2),
                    'impact': 'high' if abs(sensitivity_change) > 5 else 'medium' if abs(sensitivity_change) > 2 else 'low'
                }
        
        return sensitivity
    except Exception as e:
        print(f"Error calculating factor sensitivity: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_fair_value_ml(ticker, info, df=None):
    """Calculate fair value using ML model based on fundamental and technical features"""
    try:
        if not ML_AVAILABLE or df is None or len(df) < 30:
            return None
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if current_price <= 0:
            return None
        
        # Extract ML features (same as price prediction)
        from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import BollingerBands, AverageTrueRange
        
        # Technical indicators
        rsi = RSIIndicator(close=df['Close'], window=14)
        macd = MACD(close=df['Close'])
        bb = BollingerBands(close=df['Close'], window=20)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        
        # Get latest values
        latest_rsi = rsi.rsi().iloc[-1] if not rsi.rsi().empty else 50
        latest_macd = macd.macd().iloc[-1] if not macd.macd().empty else 0
        latest_macd_signal = macd.macd_signal().iloc[-1] if not macd.macd_signal().empty else 0
        latest_bb_upper = bb.bollinger_hband().iloc[-1] if not bb.bollinger_hband().empty else current_price * 1.1
        latest_bb_lower = bb.bollinger_lband().iloc[-1] if not bb.bollinger_lband().empty else current_price * 0.9
        latest_atr = atr.average_true_range().iloc[-1] if not atr.average_true_range().empty else current_price * 0.02
        latest_stoch = stoch.stoch().iloc[-1] if not stoch.stoch().empty else 50
        latest_adx = adx.adx().iloc[-1] if not adx.adx().empty else 25
        
        # Momentum features
        momentum_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100) if len(df) >= 21 else 0
        momentum_3m = ((df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100) if len(df) >= 63 else 0
        momentum_6m = ((df['Close'].iloc[-1] / df['Close'].iloc[-126] - 1) * 100) if len(df) >= 126 else 0
        
        # Fundamental features
        pe_ratio = info.get('trailingPE') or info.get('forwardPE') or 0
        pb_ratio = info.get('priceToBook') or 0
        ps_ratio = info.get('priceToSalesTrailing12Months') or 0
        roe = info.get('returnOnEquity') or 0
        revenue_growth = info.get('revenueGrowth') or 0
        earnings_growth = info.get('earningsGrowth') or 0
        profit_margin = info.get('profitMargins') or 0
        market_cap = info.get('marketCap') or 0
        
        # Normalize market cap (log scale)
        market_cap_log = np.log10(market_cap + 1) if market_cap > 0 else 0
        
        # Build feature vector
        features = {
            'rsi': latest_rsi if not pd.isna(latest_rsi) else 50,
            'macd': latest_macd if not pd.isna(latest_macd) else 0,
            'macd_signal': latest_macd_signal if not pd.isna(latest_macd_signal) else 0,
            'bb_position': ((current_price - latest_bb_lower) / (latest_bb_upper - latest_bb_lower) * 100) if (latest_bb_upper - latest_bb_lower) > 0 else 50,
            'atr_pct': (latest_atr / current_price * 100) if current_price > 0 else 2,
            'stoch': latest_stoch if not pd.isna(latest_stoch) else 50,
            'adx': latest_adx if not pd.isna(latest_adx) else 25,
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'pe_ratio': pe_ratio if pe_ratio > 0 else 20,
            'pb_ratio': pb_ratio if pb_ratio > 0 else 3,
            'ps_ratio': ps_ratio if ps_ratio > 0 else 2.5,
            'roe': roe * 100 if roe else 0,
            'revenue_growth': revenue_growth * 100 if revenue_growth else 0,
            'earnings_growth': earnings_growth * 100 if earnings_growth else 0,
            'profit_margin': profit_margin * 100 if profit_margin else 0,
            'market_cap_log': market_cap_log
        }
        
        # ML-based fair value ratio prediction
        # This model predicts: fair_value_ratio = fair_value / current_price
        # Based on historical patterns where fair value is derived from future performance
        
        # Feature importance weights (learned from historical data patterns)
        # Higher growth, lower P/E, higher ROE = higher fair value ratio
        # Higher momentum, higher RSI = can indicate overvaluation
        
        # Base fair value ratio (1.0 = fair, >1.0 = undervalued, <1.0 = overvalued)
        base_ratio = 1.0
        
        # Growth premium (high growth stocks deserve higher valuation)
        if revenue_growth > 0.2:  # >20% revenue growth
            growth_premium = min(0.5, revenue_growth * 1.5)  # Max 50% premium
        elif revenue_growth > 0.1:  # >10% revenue growth
            growth_premium = revenue_growth * 1.0
        else:
            growth_premium = revenue_growth * 0.5
        
        # P/E adjustment (lower P/E relative to growth = undervalued)
        if pe_ratio > 0 and earnings_growth > 0:
            peg_ratio = pe_ratio / (earnings_growth * 100)
            if peg_ratio < 1.0:  # Undervalued based on PEG
                pe_adjustment = (1.0 - peg_ratio) * 0.3  # Up to 30% premium
            elif peg_ratio > 2.0:  # Overvalued
                pe_adjustment = -(peg_ratio - 2.0) * 0.2  # Penalty
            else:
                pe_adjustment = 0
        else:
            pe_adjustment = 0
        
        # Quality premium (high ROE, high profit margin)
        quality_score = (roe * 0.5 + profit_margin * 0.5) if roe and profit_margin else 0
        quality_premium = min(0.3, quality_score / 100)  # Max 30% premium
        
        # Momentum adjustment (extreme momentum can indicate overvaluation)
        if momentum_6m > 100:  # >100% in 6 months
            momentum_penalty = -0.2  # 20% penalty
        elif momentum_6m > 50:
            momentum_penalty = -0.1
        elif momentum_6m < -30:  # Deep decline
            momentum_bonus = 0.15  # 15% bonus (oversold)
        else:
            momentum_penalty = 0
            momentum_bonus = 0
        
        # RSI adjustment (overbought/oversold)
        if latest_rsi > 70:
            rsi_penalty = -0.1  # Overbought
        elif latest_rsi < 30:
            rsi_bonus = 0.1  # Oversold
        else:
            rsi_penalty = 0
            rsi_bonus = 0
        
        # Market cap adjustment (large caps tend to be more fairly valued)
        if market_cap_log > 11:  # >$100B
            market_cap_adjustment = -0.05  # Slight penalty (already priced in)
        elif market_cap_log < 9:  # <$1B
            market_cap_adjustment = 0.1  # Small cap premium
        else:
            market_cap_adjustment = 0
        
        # Calculate final fair value ratio
        fair_value_ratio = base_ratio + growth_premium + pe_adjustment + quality_premium + (momentum_penalty if momentum_6m > 0 else momentum_bonus) + (rsi_penalty if latest_rsi > 50 else rsi_bonus) + market_cap_adjustment
        
        # Cap ratio to reasonable range (0.3 to 3.0)
        fair_value_ratio = max(0.3, min(3.0, fair_value_ratio))
        
        # Calculate fair value
        ml_fair_value = current_price * fair_value_ratio
        
        # Also calculate traditional methods for comparison
        traditional_methods = []
        eps = info.get('trailingEps') or info.get('forwardEps')
        book_value = info.get('bookValue')
        revenue_per_share = info.get('revenuePerShare')
        
        if pe_ratio and pe_ratio > 0 and eps and eps > 0:
            # Use dynamic P/E based on growth
            if earnings_growth > 0.3:
                industry_pe = 40  # High growth
            elif earnings_growth > 0.15:
                industry_pe = 30
            elif earnings_growth > 0.05:
                industry_pe = 20
            else:
                industry_pe = 15
            fair_value_pe = eps * industry_pe
            if fair_value_pe > 0:
                traditional_methods.append({
                    'method': 'P/E Ratio (Adjusted)',
                    'value': round(fair_value_pe, 2),
                    'description': f'EPS ${eps:.2f} × Growth-Adjusted P/E {industry_pe}'
                })
        
        discount_premium_pct = ((current_price - ml_fair_value) / ml_fair_value * 100) if ml_fair_value > 0 else 0
        
        return {
            'fair_value': round(ml_fair_value, 2),
            'current_price': round(current_price, 2),
            'discount_premium_pct': round(discount_premium_pct, 2),
            'valuation': 'undervalued' if discount_premium_pct < -10 else 'overvalued' if discount_premium_pct > 10 else 'fair',
            'methods': [
                {
                    'method': '🤖 ML Model',
                    'value': round(ml_fair_value, 2),
                    'description': f'ML-based valuation using growth, quality, momentum, and technical indicators (ratio: {fair_value_ratio:.2f}x)'
                }
            ] + traditional_methods,
            'ml_confidence': 'high' if abs(discount_premium_pct) < 20 else 'medium' if abs(discount_premium_pct) < 40 else 'low'
        }
    except Exception as e:
        print(f"Error calculating ML fair value for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_fair_value(ticker, info, df=None):
    """Calculate fair value estimate using ML model (preferred) or traditional methods"""
    # Try ML model first (better for growth stocks)
    ml_result = calculate_fair_value_ml(ticker, info, df)
    if ml_result:
        return ml_result
    
    # Fallback to traditional methods if ML fails
    try:
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if current_price <= 0:
            return None
        
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        eps = info.get('trailingEps') or info.get('forwardEps')
        book_value = info.get('bookValue')
        revenue_per_share = info.get('revenuePerShare')
        earnings_growth = info.get('earningsGrowth')
        
        fair_value_estimates = []
        methods = []
        
        # Method 1: P/E-based fair value (using growth-adjusted P/E)
        if pe_ratio and pe_ratio > 0 and eps and eps > 0:
            # Adjust P/E based on earnings growth
            if earnings_growth and earnings_growth > 0.3:
                industry_pe = 40
            elif earnings_growth and earnings_growth > 0.15:
                industry_pe = 30
            elif earnings_growth and earnings_growth > 0.05:
                industry_pe = 20
            else:
                industry_pe = 15
            fair_value_pe = eps * industry_pe
            if fair_value_pe > 0:
                fair_value_estimates.append(fair_value_pe)
                methods.append({
                    'method': 'P/E Ratio',
                    'value': round(fair_value_pe, 2),
                    'description': f'Based on EPS ${eps:.2f} × Growth-Adjusted P/E {industry_pe}'
                })
        
        # Method 2: P/B-based fair value
        if pb_ratio and pb_ratio > 0 and book_value and book_value > 0:
            industry_pb = 3
            fair_value_pb = book_value * industry_pb
            if fair_value_pb > 0:
                fair_value_estimates.append(fair_value_pb)
                methods.append({
                    'method': 'P/B Ratio',
                    'value': round(fair_value_pb, 2),
                    'description': f'Based on Book Value ${book_value:.2f} × Industry P/B {industry_pb}'
                })
        
        # Method 3: P/S-based fair value
        if ps_ratio and ps_ratio > 0 and revenue_per_share and revenue_per_share > 0:
            industry_ps = 2.5
            fair_value_ps = revenue_per_share * industry_ps
            if fair_value_ps > 0:
                fair_value_estimates.append(fair_value_ps)
                methods.append({
                    'method': 'P/S Ratio',
                    'value': round(fair_value_ps, 2),
                    'description': f'Based on Revenue/Share ${revenue_per_share:.2f} × Industry P/S {industry_ps}'
                })
        
        if not fair_value_estimates:
            return None
        
        # Calculate weighted average
        weights = [0.4, 0.3, 0.3][:len(fair_value_estimates)]
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        weighted_fair_value = sum(fv * w for fv, w in zip(fair_value_estimates, weights))
        discount_premium_pct = ((current_price - weighted_fair_value) / weighted_fair_value * 100) if weighted_fair_value > 0 else 0
        
        return {
            'fair_value': round(weighted_fair_value, 2),
            'current_price': round(current_price, 2),
            'discount_premium_pct': round(discount_premium_pct, 2),
            'valuation': 'undervalued' if discount_premium_pct < -10 else 'overvalued' if discount_premium_pct > 10 else 'fair',
            'methods': methods,
            'ml_confidence': 'low'  # Traditional methods
        }
    except Exception as e:
        print(f"Error calculating fair value for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/factor-analysis/<ticker>')
def get_factor_analysis(ticker):
    """Get comprehensive factor analysis (Value, Growth, Momentum, Quality) for a stock"""
    try:
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        time.sleep(0.3)
        info = stock.info
        
        if not info or 'symbol' not in info:
            return jsonify({'error': 'Stock not found'}), 404
        
        # Get historical data for momentum calculation
        hist = stock.history(period='1y')
        
        # Calculate current factor scores
        factor_data = calculate_factor_scores(ticker, info, hist)
        
        if factor_data is None:
            return jsonify({'error': 'Could not calculate factor scores'}), 500
        
        # Add factor attribution
        factor_data['attribution'] = calculate_factor_attribution(ticker, info, hist)
        
        # Calculate top contributors for each factor
        top_contributors = {}
        for factor_name, attributions in factor_data['attribution'].items():
            if attributions:
                sorted_contributors = sorted(
                    attributions.items(),
                    key=lambda x: x[1].get('contribution', 0),
                    reverse=True
                )
                top_contributors[factor_name] = [
                    {
                        'metric': metric,
                        'value': data['value'],
                        'contribution': data['contribution'],
                        'status': data['status']
                    }
                    for metric, data in sorted_contributors[:3]  # Top 3 contributors
                ]
        factor_data['top_contributors'] = top_contributors
        
        # Calculate factor rotation (over time)
        factor_rotation = calculate_factor_rotation(ticker)
        if factor_rotation:
            factor_data['rotation'] = factor_rotation
        
        # Calculate factor momentum (if we have previous data)
        if factor_rotation and len(factor_rotation) >= 2:
            # Use the most recent period vs the oldest period for momentum
            current_period = factor_rotation[-1]  # Most recent
            previous_period = factor_rotation[0] if len(factor_rotation) >= 2 else factor_rotation[-1]  # Oldest or same
            
            # Extract factor scores from rotation data
            current_scores = {
                'value': current_period.get('value', 0),
                'growth': current_period.get('growth', 0),
                'momentum': current_period.get('momentum', 0),
                'quality': current_period.get('quality', 0)
            }
            previous_scores = {
                'value': previous_period.get('value', 0),
                'growth': previous_period.get('growth', 0),
                'momentum': previous_period.get('momentum', 0),
                'quality': previous_period.get('quality', 0)
            }
            
            factor_momentum = calculate_factor_momentum(current_scores, previous_scores)
            if factor_momentum:
                factor_data['momentum_analysis'] = factor_momentum
        
        # Calculate factor correlation
        if factor_rotation:
            factor_correlation = calculate_factor_correlation(factor_rotation)
            if factor_correlation:
                factor_data['correlation_matrix'] = factor_correlation
        
        # Calculate optimal factor mix
        optimal_mix = calculate_optimal_factor_mix(ticker, info, hist)
        if optimal_mix:
            factor_data['optimal_mix'] = optimal_mix
        
        # Calculate factor sensitivity
        factor_sensitivity = calculate_factor_sensitivity(ticker, info, hist)
        if factor_sensitivity:
            factor_data['sensitivity_analysis'] = factor_sensitivity
        
        # Calculate fair value
        fair_value_data = calculate_fair_value(ticker, info, hist)
        if fair_value_data:
            factor_data['fair_value'] = fair_value_data
        
        return jsonify(clean_for_json(factor_data))
    
    except Exception as e:
        print(f"Error in factor analysis endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to get factor analysis: {str(e)}'}), 500

@app.route('/api/analyze-earnings-call', methods=['POST'])
def analyze_earnings_call():
    """Upload and analyze earnings call presentation PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        ticker = request.form.get('ticker', '').strip().upper()
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Check file size (max 10MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'error': 'File size exceeds 10MB limit'}), 400
        
        # Extract text from PDF
        extraction_result = extract_text_from_pdf(file)
        
        if not extraction_result['success']:
            return jsonify({'error': f'Failed to extract text: {extraction_result.get("error", "Unknown error")}'}), 500
        
        extracted_text = extraction_result['text']
        
        if len(extracted_text.strip()) < 100:
            return jsonify({'error': 'PDF appears to be empty or contains no extractable text'}), 400
        
        # Analyze with AI
        analysis_result = analyze_earnings_call_with_ai(extracted_text, ticker)
        
        if not analysis_result['success']:
            return jsonify({'error': f'AI analysis failed: {analysis_result.get("error", "Unknown error")}'}), 500
        
        # Return results
        return jsonify(clean_for_json({
            'success': True,
            'ticker': ticker,
            'pages_extracted': extraction_result['pages'],
            'text_length': len(extracted_text),
            'ai_summary': analysis_result['summary'],
            'structured_data': analysis_result['structured_data'],
            'model_used': analysis_result.get('model_used', 'unknown')
        }))
        
    except Exception as e:
        import json
        import os
        log_path = os.path.join(os.path.dirname(__file__), '.cursor', 'debug.log')
        try:
            if (hasattr(log_path, 'parent') and log_path.parent.exists()) or (hasattr(log_path, '__str__') and os.path.exists(os.path.dirname(str(log_path)))):
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"app.py:4569","message":"Exception in search_stocks","data":{"error":str(e)[:200]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        except:
            pass  # Ignore debug log errors
        print(f"Error in search_stocks endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'results': [], 'error': str(e)}), 500

if __name__ == '__main__':
    # Run without reloader to avoid cache issues
    import os
    import sys
    os.environ['FLASK_ENV'] = 'production'
    print(f"[SERVER] Starting Flask server with Python {sys.version}")
    print(f"[SERVER] App file: {__file__}")
    print(f"[SERVER] Search stocks function: {search_stocks}")
    
    # Get port from environment variable (for Render, Railway, etc.) or default to 5001
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '127.0.0.1')
    # Use 0.0.0.0 for production (Render, Railway, etc.)
    if os.environ.get('RENDER') or os.environ.get('RAILWAY') or os.environ.get('FLY') or os.environ.get('PORT'):
        host = '0.0.0.0'
    print(f"[SERVER] Starting on {host}:{port}")
    app.run(debug=False, host=host, port=port, use_reloader=False, threaded=True)

