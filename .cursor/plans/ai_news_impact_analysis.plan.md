# AI News Impact Analysis

## Overview
Přidání AI analýzy dopadu news na cenu akcie. Pro každou news v sekci "Latest News" bude AI analyzovat, jak může ovlivnit cenu akcie a proč.

## User Requirements
- **Umístění:** V sekci Latest News (Stock Analysis tab), u každé news
- **Výstup:** 
  - Zda news může ovlivnit cenu pozitivně/negativě/neutrálně
  - Konkrétní faktory z news, které jsou relevantní
  - Vysvětlení proč

## Implementation Plan

### 1. Backend Changes (`app.py`)

#### 1.1 Nový endpoint pro AI analýzu news

**Umístění:** `app.py` před `if __name__ == '__main__':` (kolem řádku 4760)

**Implementace:**
```python
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
```

#### 1.2 Funkce pro AI analýzu dopadu news

**Umístění:** `app.py` po `analyze_earnings_call_with_ai()` funkci (kolem řádku 260)

**Implementace:**
```python
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
        except:
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
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0].isdigit()):
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
```

### 2. Frontend Changes (`templates/index.html`)

#### 2.1 Přidání AI Impact Analysis do news zobrazení

**Umístění:** V funkci `displayNews()` nebo tam, kde se zobrazují news (kolem řádku 5500-6000)

**Implementace:**
- Přidat tlačítko "🤖 Analyze Impact" u každé news
- Po kliknutí zobrazit loading state
- Zavolat API endpoint `/api/analyze-news-impact`
- Zobrazit výsledky v expandovatelném boxu pod news

**Struktura zobrazení:**
```javascript
// V displayNews() funkci, u každé news item přidat:
html += `
    <div class="news-item">
        <!-- Existing news content -->
        
        <button onclick="analyzeNewsImpact('${news.link}', '${ticker}', event)" 
                class="ripple-effect" 
                style="margin-top: 10px; padding: 8px 16px; background: rgba(102, 126, 234, 0.1); color: #667eea; border: 1px solid #667eea; border-radius: 6px; font-size: 0.85em; cursor: pointer;">
            🤖 Analyze Impact
        </button>
        
        <div id="newsImpact_${newsIndex}" style="display: none; margin-top: 15px; padding: 15px; background: var(--metric-bg); border-radius: 10px; border-left: 4px solid #667eea;"></div>
    </div>
`;
```

#### 2.2 JavaScript funkce pro analýzu news

**Umístění:** V JavaScript sekci, po `displayNews()` funkci

**Implementace:**
```javascript
async function analyzeNewsImpact(newsLink, ticker, event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    
    // Find the news item
    const newsItem = event ? event.target.closest('.news-item') : null;
    if (!newsItem) return;
    
    // Get news data
    const newsTitle = newsItem.querySelector('.news-title')?.textContent || '';
    const newsSummary = newsItem.querySelector('.news-summary')?.textContent || '';
    const newsContent = newsItem.querySelector('.news-content')?.textContent || '';
    
    // Find impact container
    const newsIndex = Array.from(document.querySelectorAll('.news-item')).indexOf(newsItem);
    const impactContainer = document.getElementById(`newsImpact_${newsIndex}`);
    const analyzeButton = event.target;
    
    if (!impactContainer) {
        // Show loading state
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '⏳ Analyzing...';
        impactContainer.style.display = 'block';
        impactContainer.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: var(--text-secondary); font-size: 0.9em;">Analyzing news impact...</p>
            </div>
        `;
        
        try {
            const response = await fetch('/api/analyze-news-impact', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: newsTitle,
                    summary: newsSummary,
                    content: newsContent,
                    ticker: ticker
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze news impact');
            }
            
            // Display results
            displayNewsImpactAnalysis(data, impactContainer);
            
        } catch (error) {
            console.error('Error analyzing news impact:', error);
            impactContainer.innerHTML = `
                <div style="padding: 15px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border-left: 4px solid #ef4444;">
                    <div style="color: #ef4444; font-weight: 600; margin-bottom: 5px;">❌ Error</div>
                    <div style="color: var(--text-secondary); font-size: 0.9em;">${error.message}</div>
                </div>
            `;
        } finally {
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = '🤖 Analyze Impact';
        }
    }
}

function displayNewsImpactAnalysis(data, container) {
    const impact = data.impact_assessment || 'neutral';
    const level = data.impact_level || 'medium';
    const factors = data.key_factors || [];
    const explanation = data.explanation || '';
    const priceEstimate = data.price_impact_estimate || '';
    
    // Color coding
    const impactColors = {
        'positive': { bg: 'rgba(16, 185, 129, 0.1)', border: '#10b981', emoji: '🟢' },
        'negative': { bg: 'rgba(239, 68, 68, 0.1)', border: '#ef4444', emoji: '🔴' },
        'neutral': { bg: 'rgba(245, 158, 11, 0.1)', border: '#f59e0b', emoji: '🟡' }
    };
    
    const levelLabels = {
        'high': 'Vysoký',
        'medium': 'Střední',
        'low': 'Nízký'
    };
    
    const impactLabels = {
        'positive': 'Pozitivní',
        'negative': 'Negativní',
        'neutral': 'Neutrální'
    };
    
    const colors = impactColors[impact] || impactColors['neutral'];
    
    let html = `
        <div style="padding: 20px; background: ${colors.bg}; border-radius: 10px; border-left: 4px solid ${colors.border};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h5 style="margin: 0; color: var(--text-primary); font-size: 1.1em;">
                    ${colors.emoji} Impact Assessment: ${impactLabels[impact]}
                </h5>
                <span style="font-size: 0.85em; color: var(--text-secondary); padding: 4px 12px; background: ${colors.border}20; border-radius: 6px;">
                    ${levelLabels[level]} dopad
                </span>
            </div>
            
            ${factors.length > 0 ? `
                <div style="margin-bottom: 15px;">
                    <h6 style="margin: 0 0 10px 0; color: var(--text-primary); font-size: 1em; font-weight: 600;">Klíčové faktory:</h6>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        ${factors.map(factor => `
                            <li style="padding: 8px 0; padding-left: 25px; position: relative; color: var(--text-primary); border-bottom: 1px solid var(--border-light);">
                                <span style="position: absolute; left: 0; color: ${colors.border};">•</span>${escapeHtml(factor)}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${explanation ? `
                <div style="margin-bottom: 15px;">
                    <h6 style="margin: 0 0 10px 0; color: var(--text-primary); font-size: 1em; font-weight: 600;">Vysvětlení:</h6>
                    <p style="color: var(--text-primary); line-height: 1.6; margin: 0;">${escapeHtml(explanation)}</p>
                </div>
            ` : ''}
            
            ${priceEstimate ? `
                <div style="margin-top: 15px; padding: 12px; background: rgba(102, 126, 234, 0.05); border-radius: 8px;">
                    <h6 style="margin: 0 0 8px 0; color: var(--text-primary); font-size: 0.95em; font-weight: 600;">💰 Odhad dopadu na cenu:</h6>
                    <p style="color: var(--text-secondary); font-size: 0.9em; margin: 0; line-height: 1.5;">${escapeHtml(priceEstimate)}</p>
                </div>
            ` : ''}
        </div>
    `;
    
    container.innerHTML = html;
}
```

### 3. UI/UX Considerations

#### 3.1 Loading States
- Zobrazit spinner během analýzy
- Disable tlačítko během analýzy
- Zobrazit "Analyzing..." text

#### 3.2 Error Handling
- Zobrazit user-friendly error messages
- Retry možnost při selhání
- Fallback pokud AI není dostupné

#### 3.3 Visual Design
- Barevné kódování podle impact (zelená/červená/žlutá)
- Ikony pro rychlou orientaci
- Collapsible sekce pro úsporu místa
- Clear typography pro čitelnost

### 4. Performance Optimization

#### 4.1 Caching
- Cache výsledky analýzy pro stejné news (cache v localStorage)
- Neanalyzovat stejnou news znovu, pokud už byla analyzována

#### 4.2 Rate Limiting
- Omezit počet analýz na uživatele (např. max 10 za minutu)
- Batch analýza pro více news najednou (volitelné)

### 5. Files to Modify

- `app.py` - přidat endpoint a funkce (řádky ~260, ~4760)
- `templates/index.html` - přidat UI a JavaScript funkce (řádek ~5500-6000 pro news display, nové funkce)

### 6. Testing

- Test s pozitivní news (např. earnings beat)
- Test s negativní news (např. guidance downgrade)
- Test s neutrální news
- Test s chybějícími daty
- Test s dlouhými news texty
- Test bez Gemini API key

### 7. Optional Enhancements

- Batch analýza všech news najednou
- Historie analýz
- Porovnání AI predikce s reálným pohybem ceny (po čase)
- Export analýz
- Share analýzy

