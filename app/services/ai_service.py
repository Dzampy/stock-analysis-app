"""
AI service - Google Gemini integration for earnings calls, news analysis, investment thesis
"""
import os
import requests
from typing import Dict, Optional, List
from app.config import GEMINI_API_KEY, GEMINI_AVAILABLE
from app.utils.logger import logger


def generate_news_summary(news_list: List[Dict], ticker: str) -> Dict:
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
    
    # Generate summary text
    total_articles = len(news_list)
    summary_parts = []
    
    if total_articles > 0:
        summary_parts.append(f"Celkem {total_articles} článků o akcii {ticker}.")
        
        if positive_articles:
            summary_parts.append(f"{len(positive_articles)} pozitivních článků.")
        if negative_articles:
            summary_parts.append(f"{len(negative_articles)} negativních článků.")
        if neutral_articles:
            summary_parts.append(f"{len(neutral_articles)} neutrálních článků.")
        
        # Key points from titles
        key_points = all_titles[:5]  # Top 5 headlines as key points
    
    summary_text = " ".join(summary_parts) if summary_parts else f"No recent news available for {ticker}."
    
    return {
        'summary': summary_text,
        'key_points': key_points if 'key_points' in locals() else [],
        'overall_sentiment': overall_sentiment,
        'sentiment_label': sentiment_label,
        'sentiment_score': round(avg_sentiment_score, 3),
        'total_articles': total_articles,
        'positive_count': len(positive_articles),
        'negative_count': len(negative_articles),
        'neutral_count': len(neutral_articles)
    }


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
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        full_text = '\n\n'.join([page['text'] for page in text_content])
        return {
            'success': True,
            'text': full_text,
            'pages': len(text_content),
            'page_breakdown': text_content
        }
    except Exception as e:
        logger.exception(f"Error extracting text from PDF")
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
            logger.warning(f"Error listing models: {str(e)}")
        
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
        logger.exception(f"Error in AI analysis")
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
        logger.exception(f"Error in AI news impact analysis")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def explain_economic_event_with_ai(event_name: str, event_description: str = '') -> Dict:
    """Generate comprehensive AI explanation for an economic event"""
    from app.config import GEMINI_AVAILABLE, GEMINI_API_KEY
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
        preferred_models = ["gemini-1.5-flash", "gemini-pro"]
        for model_name in preferred_models:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        
        if not model:
            return {'success': False, 'error': 'No available Gemini models found'}
        
        prompt = f"""Vysvětli ekonomický ukazatel/událost "{event_name}" velmi podrobně v češtině.

{event_description if event_description else ''}

VYTVOŘ PODROBNÉ VYSVĚTLENÍ S TĚMITO SEKCEMI:

1. **Co to je a co měří** (3-4 věty)
   - Definice ukazatele/události
   - Co přesně měří
   - Jak se počítá/zjišťuje
   - Kdo ho publikuje a jak často

2. **Kdy je to bullish (pozitivní) pro akcie** (4-5 bodů)
   - Jaké hodnoty/trendy jsou pozitivní
   - Proč to podporuje růst akcií
   - Které sektory/akcie to nejvíce ovlivní
   - Historické příklady

3. **Kdy je to bearish (negativní) pro akcie** (4-5 bodů)
   - Jaké hodnoty/trendy jsou negativní
   - Proč to tlačí akcie dolů
   - Které sektory/akcie to nejvíce ovlivní
   - Historické příklady

4. **Kontext a význam** (2-3 věty)
   - Jak důležitý je tento ukazatel
   - Jak se vztahuje k jiným ekonomickým ukazatelům
   - Co investoři sledují

Buď velmi konkrétní a poskytni praktické příklady."""
        
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.5,
                'max_output_tokens': 2048,
            }
        )
        
        return {
            'success': True,
            'event_name': event_name,
            'what': response.text,
            'bullish': '',  # Will be extracted from response if needed
            'bearish': '',  # Will be extracted from response if needed
            'full_explanation': response.text
        }
    except Exception as e:
        logger.exception(f"Error explaining economic event")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def generate_investment_thesis_with_ai(ticker: str) -> Dict:
    """Generate comprehensive investment thesis using AI with detailed financial data"""
    from app.config import GEMINI_AVAILABLE, GEMINI_API_KEY
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'error': 'Google Gemini API key not configured'
        }
    
    try:
        import google.generativeai as genai
        import yfinance as yf
        import time
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Get stock data
        stock = yf.Ticker(ticker.upper())
        time.sleep(0.2)
        info = stock.info
        
        if not info or 'symbol' not in info:
            return {'success': False, 'error': 'Stock not found'}
        
        # Find available model
        model = None
        preferred_models = ["gemini-1.5-flash", "gemini-pro"]
        for model_name in preferred_models:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        
        if not model:
            return {'success': False, 'error': 'No available Gemini models found'}
        
        # Extract key financial metrics
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        current_price = info.get('currentPrice', 0)
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        price_to_book = info.get('priceToBook')
        profit_margin = info.get('profitMargins')
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        roe = info.get('returnOnEquity')
        debt_to_equity = info.get('debtToEquity')
        beta = info.get('beta')
        dividend_yield = info.get('dividendYield')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
        avg_volume = info.get('averageVolume', 0)
        description = info.get('longBusinessSummary', '')
        
        # Format market cap
        market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
        
        # Build comprehensive prompt with financial data
        financial_data_summary = f"""
KLÍČOVÉ FINANČNÍ ÚDAJE:
- Cena: ${current_price:.2f}
- Market Cap: {market_cap_str}
- P/E Ratio: {pe_ratio if pe_ratio else 'N/A'}
- Forward P/E: {forward_pe if forward_pe else 'N/A'}
- P/B Ratio: {price_to_book if price_to_book else 'N/A'}
- Profit Margin: {profit_margin*100 if profit_margin else 'N/A'}%
- Revenue Growth: {revenue_growth*100 if revenue_growth else 'N/A'}%
- Earnings Growth: {earnings_growth*100 if earnings_growth else 'N/A'}%
- ROE: {roe*100 if roe else 'N/A'}%
- Debt/Equity: {debt_to_equity if debt_to_equity else 'N/A'}
- Beta: {beta if beta else 'N/A'}
- Dividend Yield: {dividend_yield*100 if dividend_yield else 'N/A'}%
- 52W High: ${fifty_two_week_high:.2f}
- 52W Low: ${fifty_two_week_low:.2f}
"""
        
        prompt = f"""Vytvoř velmi podrobnou a komplexní investment thesis pro akcii {ticker} ({company_name}) v sektoru {sector}, odvětví {industry}.

{financial_data_summary}

BUSINESS DESCRIPTION:
{description[:1000] if description else 'N/A'}

VYTVOŘ INVESTMENT THESIS S TĚMITO SEKCEMI:

1. **Executive Summary** (3-4 věty)
   - Klíčové body investičního případu
   - Celkové hodnocení investiční příležitosti

2. **Business Model Analysis** (5-7 bodů)
   - Jak společnost generuje tržby
   - Klíčové produkty/služby
   - Hlavní zdroje příjmů
   - Business model výhody a nevýhody

3. **Competitive Position** (4-6 bodů)
   - Pozice na trhu
   - Konkurenční výhody
   - Bariéry vstupu
   - Moats (ekonomické příkopy)

4. **Financial Health** (5-7 bodů)
   - Analýza výše uvedených finančních metrik
   - Ziskovost a efektivita
   - Finanční síla (debt levels, cash position)
   - Finanční trendy

5. **Growth Prospects** (4-6 bodů)
   - Růstové příležitosti
   - Tržní expanze
   - Produktové inovace
   - Strategické iniciativy

6. **Risks** (5-7 bodů)
   - Hlavní rizika investice
   - Konkurenční rizika
   - Regulační rizika
   - Finanční rizika
   - Tržní rizika

7. **Investment Recommendation** (2-3 věty)
   - Celkové doporučení (Buy/Hold/Sell)
   - Time horizon
   - Klíčové faktory pro úspěch

Formátuj v češtině, buď velmi konkrétní a použij výše uvedená finanční data v analýze."""
        
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 4096,
            }
        )
        
        return {
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'thesis': response.text,
            'financial_metrics': {
                'market_cap': market_cap_str,
                'pe_ratio': pe_ratio,
                'forward_pe': forward_pe,
                'revenue_growth': revenue_growth,
                'profit_margin': profit_margin,
                'roe': roe
            }
        }
    except Exception as e:
        logger.exception(f"Error generating investment thesis")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

