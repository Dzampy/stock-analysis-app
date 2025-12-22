"""
Portfolio and watchlist analysis service
"""
import os
from typing import Dict, List, Optional
from app.config import GEMINI_API_KEY, GEMINI_AVAILABLE
from app.utils.logger import logger


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
        logger.exception(f"Error in AI watchlist news analysis")
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

