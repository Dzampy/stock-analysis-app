# Earnings Call Presentation Upload & AI Analysis

## Overview

Přidání funkce do Financials sekce, která umožní uživatelům nahrát PDF s earnings call prezentací a získat AI shrnutí důležitých faktorů.

## User Requirements

- **Umístění:** Financials sekce
- **Formát souborů:** PDF pouze
- **AI metoda:** OpenAI API (GPT-4 nebo GPT-3.5)

## Implementation Plan

### 1. Backend Changes (`app.py`)

#### 1.1 Přidání nových závislostí

**Soubor:** `requirements.txt`

**Přidat:**

```
PyPDF2>=3.0.0
openai>=1.0.0
```

**Poznámka:** PyPDF2 pro extrakci textu z PDF, OpenAI pro AI analýzu.

#### 1.2 Konfigurace OpenAI API

**Umístění:** `app.py` kolem řádku 35-45

**Změny:**

```python
# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_AVAILABLE = OPENAI_API_KEY is not None

if not OPENAI_AVAILABLE:
    print("Warning: OpenAI API key not found. Earnings call analysis will not be available.")
```

#### 1.3 Funkce pro extrakci textu z PDF

**Umístění:** `app.py` po `clean_for_json()` funkci (kolem řádku 100)

**Implementace:**

```python
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
```

#### 1.4 Funkce pro AI analýzu earnings call

**Umístění:** `app.py` po `extract_text_from_pdf()` funkci

**Implementace:**

```python
def analyze_earnings_call_with_ai(text_content, ticker=None):
    """Analyze earnings call presentation text using OpenAI API"""
    if not OPENAI_AVAILABLE:
        return {
            'success': False,
            'error': 'OpenAI API key not configured'
        }
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Create prompt for AI analysis
        prompt = f"""Analyze the following earnings call presentation and provide a comprehensive summary in Czech language.

Focus on:
1. Key financial metrics (Revenue, EPS, guidance)
2. Important business updates and strategic initiatives
3. Management commentary and outlook
4. Risks and challenges mentioned
5. Positive highlights and achievements

Format the response as:
- Executive Summary (2-3 sentences)
- Key Financial Highlights (bullet points)
- Strategic Initiatives (bullet points)
- Management Outlook (bullet points)
- Risks & Challenges (bullet points)
- Overall Sentiment (positive/neutral/negative with brief explanation)

Earnings Call Presentation Text:
{text_content[:15000]}  # Limit to avoid token limits
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" for better quality
            messages=[
                {"role": "system", "content": "You are a financial analyst expert at analyzing earnings call presentations. Provide clear, concise summaries in Czech language."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3  # Lower temperature for more factual analysis
        )
        
        ai_summary = response.choices[0].message.content
        
        # Extract structured data from AI response
        summary_data = parse_ai_summary(ai_summary)
        
        return {
            'success': True,
            'summary': ai_summary,
            'structured_data': summary_data,
            'model_used': 'gpt-4o-mini'
        }
        
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def parse_ai_summary(summary_text):
    """Parse AI summary into structured format"""
    structured = {
        'executive_summary': '',
        'financial_highlights': [],
        'strategic_initiatives': [],
        'management_outlook': [],
        'risks_challenges': [],
        'overall_sentiment': 'neutral',
        'sentiment_explanation': ''
    }
    
    # Simple parsing logic - extract sections
    lines = summary_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect section headers
        if 'Executive Summary' in line or 'Shrnutí' in line:
            current_section = 'executive_summary'
        elif 'Financial Highlights' in line or 'Finanční' in line:
            current_section = 'financial_highlights'
        elif 'Strategic' in line or 'Strategické' in line:
            current_section = 'strategic_initiatives'
        elif 'Outlook' in line or 'Výhled' in line:
            current_section = 'management_outlook'
        elif 'Risks' in line or 'Rizika' in line:
            current_section = 'risks_challenges'
        elif 'Sentiment' in line or 'Sentiment' in line:
            current_section = 'sentiment'
        elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
            # Bullet point
            bullet_text = line.lstrip('-•*').strip()
            if current_section and current_section in ['financial_highlights', 'strategic_initiatives', 'management_outlook', 'risks_challenges']:
                structured[current_section].append(bullet_text)
        elif current_section == 'executive_summary':
            structured['executive_summary'] += line + ' '
        elif current_section == 'sentiment':
            if 'positive' in line.lower() or 'pozitivní' in line.lower():
                structured['overall_sentiment'] = 'positive'
            elif 'negative' in line.lower() or 'negativní' in line.lower():
                structured['overall_sentiment'] = 'negative'
            structured['sentiment_explanation'] += line + ' '
    
    return structured
```

#### 1.5 Flask endpoint pro upload a analýzu

**Umístění:** `app.py` před `if __name__ == '__main__':` (kolem řádku 4540)

**Implementace:**

```python
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
        print(f"Error in analyze-earnings-call endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500
```

### 2. Frontend Changes (`templates/index.html`)

#### 2.1 Přidání upload sekce do Financials

**Umístění:** V `displayFinancials()` funkci, po sekci "Risks" (kolem řádku 7570), před `container.innerHTML = html;`

**Implementace:**

```javascript
// 8. 📄 Earnings Call Analysis Section
html += `
    <div class="card" style="margin-bottom: 30px;">
        <h3 style="display: flex; align-items: center; gap: 10px;">
            📄 Earnings Call Analysis
            <span class="info-badge tooltip" style="cursor: help; font-size: 0.9em;">ℹ️
                <span class="tooltiptext" style="width: 300px; white-space: normal;">
                    <strong>Earnings Call Analysis</strong><br>
                    Nahrajte PDF s earnings call prezentací a AI automaticky analyzuje důležité faktory, metriky a management komentáře.<br><br>
                    <strong>Co AI analyzuje:</strong> Finanční metriky, strategické iniciativy, management outlook, rizika a celkový sentiment.
                </span>
            </span>
        </h3>
        
        <div style="margin-top: 20px; padding: 30px; border: 2px dashed var(--border-color); border-radius: 12px; text-align: center; background: var(--metric-bg);">
            <div style="font-size: 3em; margin-bottom: 15px;">📤</div>
            <p style="margin-bottom: 20px; color: var(--text-secondary);">Upload earnings call presentation PDF</p>
            
            <form id="earningsCallUploadForm" onsubmit="uploadEarningsCall(event, '${ticker}'); return false;" style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                <input type="file" id="earningsCallFileInput" accept=".pdf" required style="display: none;" onchange="handleEarningsCallFileSelect(event)" />
                <button type="button" onclick="document.getElementById('earningsCallFileInput').click()" class="ripple-effect" style="padding: 12px 24px; background: #667eea; color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                    📁 Choose PDF File
                </button>
                <div id="earningsCallFileName" style="color: var(--text-secondary); font-size: 0.9em; min-height: 20px;"></div>
                <button type="submit" id="earningsCallUploadBtn" class="ripple-effect" style="padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; display: none;">
                    🤖 Analyze with AI
                </button>
            </form>
            
            <div id="earningsCallAnalysisResult" style="margin-top: 30px; text-align: left;"></div>
        </div>
    </div>
`;
```

#### 2.2 JavaScript funkce pro upload a zobrazení výsledků

**Umístění:** V JavaScript sekci, po `displayFinancials()` funkci (kolem řádku 7586)

**Implementace:**

```javascript
let selectedEarningsCallFile = null;

function handleEarningsCallFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedEarningsCallFile = file;
        const fileNameDiv = document.getElementById('earningsCallFileName');
        const uploadBtn = document.getElementById('earningsCallUploadBtn');
        
        fileNameDiv.innerHTML = `Selected: <strong>${file.name}</strong> (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        uploadBtn.style.display = 'block';
    }
}

async function uploadEarningsCall(event, ticker) {
    event.preventDefault();
    
    if (!selectedEarningsCallFile) {
        showToast('Please select a PDF file first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedEarningsCallFile);
    formData.append('ticker', ticker || '');
    
    const resultDiv = document.getElementById('earningsCallAnalysisResult');
    const uploadBtn = document.getElementById('earningsCallUploadBtn');
    
    // Show loading state
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '⏳ Analyzing...';
    resultDiv.innerHTML = `
        <div style="text-align: center; padding: 40px;">
            <div class="spinner"></div>
            <p style="margin-top: 20px; color: var(--text-secondary);">Analyzing earnings call presentation...</p>
            <p style="font-size: 0.85em; color: var(--text-tertiary); margin-top: 10px;">This may take 30-60 seconds</p>
        </div>
    `;
    
    try {
        const response = await fetch('/api/analyze-earnings-call', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to analyze earnings call');
        }
        
        // Display results
        displayEarningsCallAnalysis(data);
        
        showToast('Earnings call analyzed successfully', 'success');
        
    } catch (error) {
        console.error('Error analyzing earnings call:', error);
        resultDiv.innerHTML = `
            <div style="padding: 20px; background: rgba(239, 68, 68, 0.1); border-radius: 10px; border-left: 4px solid #ef4444;">
                <div style="color: #ef4444; font-weight: 600; margin-bottom: 10px;">❌ Error</div>
                <div style="color: var(--text-secondary);">${error.message}</div>
                <button onclick="uploadEarningsCall(event, '${ticker}')" style="margin-top: 15px; padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer;">Retry</button>
            </div>
        `;
        showToast('Error analyzing earnings call', 'error');
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '🤖 Analyze with AI';
    }
}

function displayEarningsCallAnalysis(data) {
    const resultDiv = document.getElementById('earningsCallAnalysisResult');
    const structured = data.structured_data || {};
    
    let html = `
        <div style="padding: 25px; background: var(--bg-card); border-radius: 12px; border: 2px solid var(--border-color);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h4 style="margin: 0; color: var(--text-primary);">🤖 AI Analysis Results</h4>
                <span style="font-size: 0.85em; color: var(--text-secondary);">${data.pages_extracted} pages analyzed</span>
            </div>
            
            ${structured.executive_summary ? `
                <div style="margin-bottom: 25px; padding: 20px; background: rgba(102, 126, 234, 0.05); border-radius: 10px; border-left: 4px solid #667eea;">
                    <h5 style="margin: 0 0 10px 0; color: var(--text-primary); font-size: 1.1em;">📋 Executive Summary</h5>
                    <p style="color: var(--text-primary); line-height: 1.6; margin: 0;">${structured.executive_summary}</p>
                </div>
            ` : ''}
            
            ${structured.financial_highlights && structured.financial_highlights.length > 0 ? `
                <div style="margin-bottom: 25px;">
                    <h5 style="margin: 0 0 15px 0; color: var(--text-primary); font-size: 1.1em;">💰 Key Financial Highlights</h5>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        ${structured.financial_highlights.map(highlight => `
                            <li style="padding: 10px 0; padding-left: 25px; position: relative; color: var(--text-primary); border-bottom: 1px solid var(--border-light);">
                                <span style="position: absolute; left: 0; color: #10b981;">✓</span>${highlight}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${structured.strategic_initiatives && structured.strategic_initiatives.length > 0 ? `
                <div style="margin-bottom: 25px;">
                    <h5 style="margin: 0 0 15px 0; color: var(--text-primary); font-size: 1.1em;">🚀 Strategic Initiatives</h5>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        ${structured.strategic_initiatives.map(initiative => `
                            <li style="padding: 10px 0; padding-left: 25px; position: relative; color: var(--text-primary); border-bottom: 1px solid var(--border-light);">
                                <span style="position: absolute; left: 0; color: #667eea;">→</span>${initiative}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${structured.management_outlook && structured.management_outlook.length > 0 ? `
                <div style="margin-bottom: 25px;">
                    <h5 style="margin: 0 0 15px 0; color: var(--text-primary); font-size: 1.1em;">🔮 Management Outlook</h5>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        ${structured.management_outlook.map(outlook => `
                            <li style="padding: 10px 0; padding-left: 25px; position: relative; color: var(--text-primary); border-bottom: 1px solid var(--border-light);">
                                <span style="position: absolute; left: 0; color: #f59e0b;">📊</span>${outlook}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${structured.risks_challenges && structured.risks_challenges.length > 0 ? `
                <div style="margin-bottom: 25px;">
                    <h5 style="margin: 0 0 15px 0; color: var(--text-primary); font-size: 1.1em;">⚠️ Risks & Challenges</h5>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        ${structured.risks_challenges.map(risk => `
                            <li style="padding: 10px 0; padding-left: 25px; position: relative; color: var(--text-primary); border-bottom: 1px solid var(--border-light);">
                                <span style="position: absolute; left: 0; color: #ef4444;">⚠</span>${risk}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${structured.overall_sentiment ? `
                <div style="margin-top: 25px; padding: 20px; background: ${structured.overall_sentiment === 'positive' ? 'rgba(16, 185, 129, 0.1)' : structured.overall_sentiment === 'negative' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(245, 158, 11, 0.1)'}; border-radius: 10px; border-left: 4px solid ${structured.overall_sentiment === 'positive' ? '#10b981' : structured.overall_sentiment === 'negative' ? '#ef4444' : '#f59e0b'};">
                    <h5 style="margin: 0 0 10px 0; color: var(--text-primary); font-size: 1.1em;">
                        ${structured.overall_sentiment === 'positive' ? '🟢' : structured.overall_sentiment === 'negative' ? '🔴' : '🟡'} Overall Sentiment: ${structured.overall_sentiment === 'positive' ? 'Positive' : structured.overall_sentiment === 'negative' ? 'Negative' : 'Neutral'}
                    </h5>
                    ${structured.sentiment_explanation ? `
                        <p style="color: var(--text-secondary); margin: 0; line-height: 1.6;">${structured.sentiment_explanation}</p>
                    ` : ''}
                </div>
            ` : ''}
            
            ${data.ai_summary ? `
                <div style="margin-top: 25px; padding: 20px; background: var(--metric-bg); border-radius: 10px;">
                    <h5 style="margin: 0 0 15px 0; color: var(--text-primary); font-size: 1.1em;">📝 Full AI Summary</h5>
                    <div style="color: var(--text-primary); line-height: 1.8; white-space: pre-wrap;">${data.ai_summary}</div>
                </div>
            ` : ''}
        </div>
    `;
    
    resultDiv.innerHTML = html;
}
```

### 3. Environment Configuration

#### 3.1 Přidání OpenAI API key do `.env`

**Soubor:** `.env` (vytvořit pokud neexistuje)

**Přidat:**

```
OPENAI_API_KEY=your_openai_api_key_here
```

**Poznámka:** Uživatel musí získat API key z https://platform.openai.com/api-keys

### 4. Error Handling & Edge Cases

#### 4.1 Backend Error Handling

- Kontrola existence souboru
- Validace formátu (pouze PDF)
- Kontrola velikosti souboru (max 10MB)
- Kontrola dostupnosti OpenAI API
- Error handling při extrakci textu z PDF
- Error handling při AI analýze (rate limits, API errors)

#### 4.2 Frontend Error Handling

- Zobrazení loading stavu během analýzy
- Error messages pro různé typy chyb
- Retry funkce při selhání
- Validace před uploadem (formát, velikost)

### 5. UI/UX Considerations

#### 5.1 Upload Experience

- Drag & drop podpora (volitelné, pro budoucí vylepšení)
- Progress indicator během uploadu
- Zobrazení názvu a velikosti souboru před uploadem
- Clear button pro zrušení výběru

#### 5.2 Results Display

- Strukturované zobrazení s sekcemi
- Barevné kódování sentimentu
- Collapsible sekce pro dlouhé texty (volitelné)
- Možnost exportovat shrnutí (volitelné, pro budoucí vylepšení)

### 6. Testing

#### 6.1 Test Cases

- Upload validního PDF souboru
- Upload příliš velkého souboru (>10MB)
- Upload ne-PDF souboru
- Upload prázdného nebo poškozeného PDF
- Test s různými velikostmi PDF (malý, střední, velký)
- Test bez OpenAI API key
- Test s chybným OpenAI API key
- Test s PDF obsahujícím pouze obrázky (bez textu)

### 7. Implementation Order

1. **Backend - PDF extraction** (PyPDF2, testování extrakce)
2. **Backend - OpenAI integration** (API setup, testování)
3. **Backend - Flask endpoint** (upload handling, error handling)
4. **Frontend - Upload UI** (file input, form)
5. **Frontend - Results display** (strukturované zobrazení)
6. **Testing & refinement**

### 8. Files to Modify

- `requirements.txt` - přidat PyPDF2 a openai
- `app.py` - přidat funkce a endpoint (řádky ~100, ~4540)
- `templates/index.html` - přidat upload sekci a JavaScript funkce (řádek ~7570, ~7586)
- `.env` - přidat OPENAI_API_KEY (vytvořit pokud neexistuje)

### 9. Optional Future Enhancements

- Podpora pro PowerPoint (PPTX) soubory
- Podpora pro textové soubory
- Drag & drop upload
- Historie analyzovaných earnings calls
- Porovnání více earnings calls
- Export shrnutí do PDF/CSV
- Caching výsledků (aby se stejný soubor neanalyzoval znovu)
- Podpora pro audio/video earnings calls (transkripce)

### 10. Notes

- OpenAI API má rate limits a náklady - je důležité to zohlednit
- Pro produkci by bylo dobré přidat rate limiting na endpoint
- PyPDF2 může mít problémy s některými PDF soubory (obrázky místo textu, šifrované PDF) - je dobré to ošetřit
- AI analýza může trvat 30-60 sekund pro větší PDF - je důležité zobrazit loading state
- Pro větší PDF může být nutné rozdělit text na části kvůli token limitům OpenAI API