# Add Forward Estimates and Segment Analysis to Financials

## Overview

Přidání dvou nových sekcí do Financials tabu:

1. **Forward Estimates** - analyst estimates pro příští kvartály/roky s confidence intervals
2. **Segment Analysis** - breakdown podle business segmentů (pokud dostupné z 10-K)

## Implementation Plan

### 1. Backend Changes (`app.py`)

#### 1.1 Rozšíření `get_financials_data()` funkce

**Umístění:** `app.py` kolem řádku 491

**Změny:**

- Přidat získávání forward estimates z `yfinance`:
  - `stock.earnings_estimates` - quarterly/annual estimates
  - `stock.revenue_estimates` - revenue estimates
  - `stock.analyst_price_target` - price targets s high/low/mean
- Přidat získávání segment data:
  - Zkusit získat z `stock.info` (klíče jako `sector`, `industry`, `businessSummary`)
  - Pokud není dostupné, zkusit z `stock.major_holders` nebo jiných zdrojů
  - Segment data mohou být v `stock.info` pod klíči jako `operatingSegments` nebo podobně

**Nové klíče v `financials` dictionary:**

```python
financials = {
    ...
    'forward_estimates': {
        'earnings': {
            'quarterly': [],  # List of {period, estimate, high, low, number_of_analysts}
            'annual': []
        },
        'revenue': {
            'quarterly': [],
            'annual': []
        },
        'price_targets': {
            'mean': None,
            'high': None,
            'low': None,
            'number_of_analysts': None
        }
    },
    'segments': [
        # List of {name, revenue, revenue_pct, operating_income, etc.}
    ],
    ...
}
```

**Implementace:**

- Po řádku 520 (po inicializaci `financials` dictionary) přidat:
  ```python
  # Get Forward Estimates
  try:
      earnings_estimates = stock.earnings_estimates
      revenue_estimates = stock.revenue_estimates
      analyst_targets = stock.analyst_price_target
      
      # Process earnings estimates
      if earnings_estimates is not None and not earnings_estimates.empty:
          # Extract quarterly and annual estimates
          # Calculate confidence intervals (high - low)
          ...
  except Exception as e:
      print(f"Error fetching forward estimates: {str(e)}")
      financials['forward_estimates'] = {'earnings': {'quarterly': [], 'annual': []}, 'revenue': {'quarterly': [], 'annual': []}, 'price_targets': {}}
  
  # Get Segment Data
  try:
      # Try to get from info
      if 'operatingSegments' in info:
          # Parse segment data
          ...
      # Alternative: try to extract from business summary or other sources
      ...
  except Exception as e:
      print(f"Error fetching segment data: {str(e)}")
      financials['segments'] = []
  ```


#### 1.2 Error Handling

- Přidat try-except bloky pro každý nový data source
- Pokud data nejsou dostupná, vrátit prázdné struktury místo None
- Logovat chyby pro debugging

### 2. Frontend Changes (`templates/index.html`)

#### 2.1 Přidání Forward Estimates sekce

**Umístění:** V `displayFinancials()` funkci, po sekci "Cash & Balance Sheet" (kolem řádku 7520), před sekci "Risks"

**Struktura:**

```javascript
// 6. 📈 Forward Estimates Section
html += `
    <div class="card" style="margin-bottom: 30px;">
        <h3 style="display: flex; align-items: center; gap: 10px;">
            📈 Forward Estimates
            <span class="info-badge tooltip" style="cursor: help; font-size: 0.9em;">ℹ️
                <span class="tooltiptext" style="width: 300px; white-space: normal;">
                    <strong>Forward Estimates</strong><br>
                    Analyst estimates pro budoucí kvartály a roky. Zobrazuje consensus (průměr), high/low range a počet analystů.<br><br>
                    <strong>Confidence Interval</strong> = rozsah mezi high a low estimate. Čím širší, tím větší nejistota.
                </span>
            </span>
        </h3>
        
        ${data.forward_estimates && data.forward_estimates.earnings ? `
            <!-- Earnings Estimates Table -->
            <div style="margin-top: 20px;">
                <h4 style="margin-bottom: 15px; color: var(--text-primary);">Earnings Estimates (EPS)</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: var(--table-header-bg);">
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Period</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">Consensus</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">High</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">Low</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">Range</th>
                            <th style="padding: 12px; text-align: center; border-bottom: 2px solid var(--border-color);">Analysts</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.forward_estimates.earnings.quarterly.map(est => `
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 10px;">${est.period}</td>
                                <td style="padding: 10px; text-align: right; font-weight: 600;">$${est.estimate.toFixed(2)}</td>
                                <td style="padding: 10px; text-align: right; color: #10b981;">$${est.high.toFixed(2)}</td>
                                <td style="padding: 10px; text-align: right; color: #ef4444;">$${est.low.toFixed(2)}</td>
                                <td style="padding: 10px; text-align: right; color: var(--text-secondary);">$${est.range.toFixed(2)}</td>
                                <td style="padding: 10px; text-align: center;">${est.number_of_analysts || 'N/A'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            
            <!-- Revenue Estimates Table -->
            <div style="margin-top: 30px;">
                <h4 style="margin-bottom: 15px; color: var(--text-primary);">Revenue Estimates</h4>
                <!-- Similar table structure for revenue -->
            </div>
            
            <!-- Price Targets -->
            ${data.forward_estimates.price_targets && data.forward_estimates.price_targets.mean ? `
                <div style="margin-top: 30px; padding: 20px; background: var(--metric-bg); border-radius: 12px;">
                    <h4 style="margin-bottom: 15px; color: var(--text-primary);">Analyst Price Targets</h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                        <div style="text-align: center; padding: 15px; background: var(--bg-card); border-radius: 8px;">
                            <div style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 5px;">Mean Target</div>
                            <div style="font-size: 1.5em; font-weight: 700; color: var(--text-primary);">$${data.forward_estimates.price_targets.mean.toFixed(2)}</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: var(--bg-card); border-radius: 8px;">
                            <div style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 5px;">High Target</div>
                            <div style="font-size: 1.5em; font-weight: 700; color: #10b981;">$${data.forward_estimates.price_targets.high.toFixed(2)}</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: var(--bg-card); border-radius: 8px;">
                            <div style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 5px;">Low Target</div>
                            <div style="font-size: 1.5em; font-weight: 700; color: #ef4444;">$${data.forward_estimates.price_targets.low.toFixed(2)}</div>
                        </div>
                    </div>
                    ${data.forward_estimates.price_targets.number_of_analysts ? `
                        <div style="text-align: center; margin-top: 15px; color: var(--text-secondary); font-size: 0.9em;">
                            Based on ${data.forward_estimates.price_targets.number_of_analysts} analyst estimates
                        </div>
                    ` : ''}
                </div>
            ` : ''}
        ` : `
            <div style="margin-top: 20px; padding: 20px; background: var(--metric-bg); border-radius: 10px; text-align: center; color: var(--text-secondary);">
                Forward estimates not available for this stock.
            </div>
        `}
    </div>
`;
```

#### 2.2 Přidání Segment Analysis sekce

**Umístění:** Po Forward Estimates sekci, před sekci "Risks"

**Struktura:**

```javascript
// 7. 📊 Segment Analysis Section
html += `
    <div class="card" style="margin-bottom: 30px;">
        <h3 style="display: flex; align-items: center; gap: 10px;">
            📊 Segment Analysis
            <span class="info-badge tooltip" style="cursor: help; font-size: 0.9em;">ℹ️
                <span class="tooltiptext" style="width: 300px; white-space: normal;">
                    <strong>Segment Analysis</strong><br>
                    Breakdown tržeb a zisku podle business segmentů (např. Product A, Product B, Services, atd.).<br><br>
                    <strong>Proč je důležité:</strong> Ukazuje, které části businessu jsou nejziskovější a které rostou nejrychleji.
                </span>
            </span>
        </h3>
        
        ${data.segments && data.segments.length > 0 ? `
            <!-- Segment Table -->
            <div style="margin-top: 20px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: var(--table-header-bg);">
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Segment</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">Revenue</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">% of Total</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">Operating Income</th>
                            <th style="padding: 12px; text-align: right; border-bottom: 2px solid var(--border-color);">Margin %</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.segments.map(segment => `
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 10px; font-weight: 600;">${segment.name}</td>
                                <td style="padding: 10px; text-align: right;">${formatCurrency(segment.revenue)}</td>
                                <td style="padding: 10px; text-align: right;">${segment.revenue_pct.toFixed(1)}%</td>
                                <td style="padding: 10px; text-align: right;">${formatCurrency(segment.operating_income)}</td>
                                <td style="padding: 10px; text-align: right; color: ${segment.margin >= 20 ? '#10b981' : segment.margin >= 10 ? '#f59e0b' : '#ef4444'};">
                                    ${segment.margin.toFixed(1)}%
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            
            <!-- Segment Chart (optional) -->
            <div style="margin-top: 30px;">
                <canvas id="segmentChart" style="max-height: 300px;"></canvas>
            </div>
        ` : `
            <div style="margin-top: 20px; padding: 20px; background: var(--metric-bg); border-radius: 10px; text-align: center; color: var(--text-secondary);">
                Segment data not available for this stock. Segment breakdown is typically available in 10-K annual reports.
            </div>
        `}
    </div>
`;
```

#### 2.3 Přidání Chart.js grafu pro Segment Analysis

**Umístění:** V `setTimeout` bloku na konci `displayFinancials()` (kolem řádku 7575)

```javascript
// Create segment chart if data available
if (data.segments && data.segments.length > 0) {
    createSegmentChart(data.segments);
}

function createSegmentChart(segments) {
    const ctx = document.getElementById('segmentChart');
    if (!ctx) return;
    
    const labels = segments.map(s => s.name);
    const revenues = segments.map(s => s.revenue);
    const colors = ['#667eea', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#3b82f6'];
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: revenues,
                backgroundColor: colors.slice(0, segments.length),
                borderWidth: 2,
                borderColor: 'var(--bg-card)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'right'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const segment = segments[context.dataIndex];
                            return `${context.label}: ${formatCurrency(context.parsed)} (${segment.revenue_pct.toFixed(1)}%)`;
                        }
                    }
                }
            }
        }
    });
}
```

### 3. Data Processing Logic

#### 3.1 Forward Estimates Processing

**Zdroj dat:** `yfinance` poskytuje:

- `stock.earnings_estimates` - DataFrame s estimates pro různé období
- `stock.revenue_estimates` - DataFrame s revenue estimates
- `stock.analyst_price_target` - dictionary s price targets

**Zpracování:**

```python
# Process earnings estimates
if earnings_estimates is not None and not earnings_estimates.empty:
    quarterly_estimates = []
    annual_estimates = []
    
    # Iterate through columns (periods) and rows (estimates)
    for col in earnings_estimates.columns:
        period = str(col)  # e.g., "2024-12-31"
        row_data = earnings_estimates[col]
        
        # Extract consensus, high, low estimates
        consensus = row_data.get('Estimate', None) if hasattr(row_data, 'get') else None
        high = row_data.get('High Estimate', None) if hasattr(row_data, 'get') else None
        low = row_data.get('Low Estimate', None) if hasattr(row_data, 'get') else None
        num_analysts = row_data.get('Number of Analysts', None) if hasattr(row_data, 'get') else None
        
        if consensus is not None:
            estimate_data = {
                'period': period,
                'estimate': float(consensus) if pd.notna(consensus) else None,
                'high': float(high) if pd.notna(high) else None,
                'low': float(low) if pd.notna(low) else None,
                'range': float(high - low) if pd.notna(high) and pd.notna(low) else None,
                'number_of_analysts': int(num_analysts) if pd.notna(num_analysts) else None
            }
            
            # Determine if quarterly or annual based on period
            if 'Q' in period or len(period.split('-')) == 3:  # Quarterly
                quarterly_estimates.append(estimate_data)
            else:  # Annual
                annual_estimates.append(estimate_data)
    
    financials['forward_estimates']['earnings']['quarterly'] = quarterly_estimates
    financials['forward_estimates']['earnings']['annual'] = annual_estimates
```

#### 3.2 Segment Data Processing

**Zdroj dat:** Segment data nejsou přímo v `yfinance`, ale mohou být:

- V `stock.info` pod různými klíči
- V business summary textu (nutné parsování)
- V 10-K reportech (vyžaduje scraping SEC)

**Fallback přístup:**

```python
# Try to get segment data from info
segments = []
try:
    # Check if info contains segment data
    if 'operatingSegments' in info:
        # Parse segment data
        segment_data = info['operatingSegments']
        # Process based on structure
        ...
    elif 'businessSummary' in info:
        # Try to extract segment info from business summary
        # This is a fallback - may not always work
        ...
except:
    pass

# If no segments found, leave empty list
financials['segments'] = segments if segments else []
```

**Poznámka:** Segment data mohou být nedostupná pro mnoho akcií, protože nejsou standardně v `yfinance`. Může být nutné použít alternativní zdroj nebo parsovat z 10-K reportů.

### 4. Testing

#### 4.1 Test Cases

- Test s akcií, která má forward estimates (např. AAPL, MSFT)
- Test s akcií bez forward estimates
- Test s akcií, která má segment data (např. velké konglomeráty)
- Test s akcií bez segment data
- Test error handling při nedostupných datech

#### 4.2 Edge Cases

- Prázdné estimates (žádní analytici)
- Chybějící high/low estimates
- Segment data v nestandardním formátu
- Velký počet segmentů (UI scaling)

### 5. UI/UX Considerations

#### 5.1 Forward Estimates

- Zobrazit confidence interval jako vizuální indikátor (bar chart nebo progress bar)
- Zvýraznit, pokud current price je mimo range estimates
- Zobrazit trend (zlepšující se / zhoršující se estimates)

#### 5.2 Segment Analysis

- Použít doughnut chart pro vizuální reprezentaci
- Zvýraznit nejziskovější segmenty
- Zobrazit growth rate pro každý segment (pokud dostupné)

### 6. Implementation Order

1. **Backend - Forward Estimates** (nejjednodušší, data jsou v yfinance)
2. **Frontend - Forward Estimates UI**
3. **Backend - Segment Analysis** (složitější, může vyžadovat fallback)
4. **Frontend - Segment Analysis UI**
5. **Testing a refinement**

### 7. Files to Modify

- `app.py` - rozšířit `get_financials_data()` funkci (řádek ~491)
- `templates/index.html` - přidat nové sekce do `displayFinancials()` (řádek ~6885)
- `templates/index.html` - přidat `createSegmentChart()` funkci (po `createCashFlowChart()`)

### 8. Notes

- Forward estimates jsou dostupné pro většinu větších akcií
- Segment data mohou být nedostupná pro mnoho akcií - je důležité zobrazit user-friendly zprávu
- Pokud segment data nejsou dostupná z yfinance, může být nutné použít alternativní zdroj (např. scraping z Finviz nebo přímo z SEC 10-K reportů)
- Confidence intervals pro estimates lze vypočítat jako (high - low) / consensus * 100