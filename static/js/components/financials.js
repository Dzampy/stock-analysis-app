/**
 * Financials Component
 * Handles financial data fetching, display, and chart rendering
 */

import { debugLog, debugError, debugWarn } from '../utils/debug.js';
import apiClient from '../services/api-client.js';
import * as formatters from '../utils/formatters.js';

// Global state for financials
let currentFinancialsData = null;
let financialsPeriod = 'quarterly'; // 'quarterly' or 'annual'

// Helper function to safely format numbers with toFixed
function safeToFixed(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value) || typeof value !== 'number') {
        return null;
    }
    return value.toFixed(decimals);
}

// Format currency for display
function formatCurrency(value) {
    if (value === null || value === undefined || isNaN(value) || typeof value !== 'number') return 'N/A';
    const isNegative = value < 0;
    const absValue = Math.abs(value);
    let formatted;
    if (absValue >= 1e12) {
        formatted = `$${(absValue / 1e12).toFixed(2)}T`;
    } else if (absValue >= 1e9) {
        formatted = `$${(absValue / 1e9).toFixed(2)}B`;
    } else if (absValue >= 1e6) {
        formatted = `$${(absValue / 1e6).toFixed(2)}M`;
    } else if (absValue >= 1e3) {
        formatted = `$${(absValue / 1e3).toFixed(2)}K`;
    } else {
        formatted = `$${absValue.toFixed(2)}`;
    }
    return isNegative ? `-${formatted}` : formatted;
}

// Format number for display (millions/billions with commas)
function formatDetailedNumber(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return '-';
    }
    
    const absValue = Math.abs(value);
    let formatted;
    
    if (absValue >= 1e9) {
        formatted = (value / 1e9).toFixed(2) + 'B';
    } else if (absValue >= 1e6) {
        formatted = (value / 1e6).toFixed(2) + 'M';
    } else if (absValue >= 1e3) {
        formatted = (value / 1e3).toFixed(2) + 'K';
    } else {
        formatted = value.toFixed(2);
    }
    
    // Add commas for thousands
    const parts = formatted.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    formatted = parts.join('.');
    
    return formatted;
}

/**
 * Switch between Overview and Detailed tabs
 */
function switchFinancialsTab(tabName) {
    debugLog('🔄 [DEBUG] switchFinancialsTab: Called with tabName:', tabName);
    
    // Update tab buttons
    const buttons = document.querySelectorAll('.financials-tab-btn');
    debugLog('🔄 [DEBUG] switchFinancialsTab: Found buttons:', buttons.length);
    buttons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
            btn.style.color = 'var(--primary-500)';
            btn.style.borderBottomColor = 'var(--primary-500)';
            debugLog('✅ [DEBUG] switchFinancialsTab: Activated button for', tabName);
        } else {
            btn.style.color = 'var(--text-secondary)';
            btn.style.borderBottomColor = 'transparent';
        }
    });
    
    // Update tab content
    document.querySelectorAll('.financials-tab-content').forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });
    
    if (tabName === 'overview') {
        const overviewTab = document.getElementById('financialsOverviewTab');
        if (overviewTab) {
            overviewTab.classList.add('active');
            overviewTab.style.display = 'block';
            debugLog('✅ [DEBUG] switchFinancialsTab: Overview tab shown');
        } else {
            debugError('❌ [DEBUG] switchFinancialsTab: Overview tab not found!');
        }
    } else if (tabName === 'detailed') {
        const detailedTab = document.getElementById('financialsDetailedTab');
        if (detailedTab) {
            detailedTab.classList.add('active');
            detailedTab.style.display = 'block';
            debugLog('✅ [DEBUG] switchFinancialsTab: Detailed tab shown');
            
            // Check if content exists and re-render if needed
            const detailedContent = document.getElementById('detailedFinancialsContent');
            if (detailedContent) {
                debugLog('✅ [DEBUG] switchFinancialsTab: detailedFinancialsContent found, innerHTML length:', detailedContent.innerHTML.length);
                const contentText = detailedContent.innerHTML.trim();
                const isEmpty = !contentText || contentText === '';
                const isPlaceholder = contentText.includes('Click "Load Financials"') || contentText.includes('Loading overview data') || contentText.includes('⚠️');
                
                if (isEmpty || isPlaceholder) {
                    debugWarn('⚠️ [DEBUG] switchFinancialsTab: detailedFinancialsContent is empty or placeholder, trying to re-render...');
                    
                    // Try to re-render if we have currentFinancialsData
                    if (currentFinancialsData) {
                        debugLog('🔄 [DEBUG] switchFinancialsTab: Re-rendering with currentFinancialsData');
                        const ticker = document.getElementById('financialsTickerInput')?.value.trim().toUpperCase() || 'UNKNOWN';
                        renderDetailedFinancials(currentFinancialsData, ticker);
                    } else {
                        debugWarn('⚠️ [DEBUG] switchFinancialsTab: currentFinancialsData not available');
                    }
                }
            } else {
                debugError('❌ [DEBUG] switchFinancialsTab: detailedFinancialsContent container not found!');
            }
        } else {
            debugError('❌ [DEBUG] switchFinancialsTab: Detailed tab not found!');
        }
    }
}

/**
 * Change financials period (quarterly/annual)
 */
function changeFinancialsPeriod(period) {
    financialsPeriod = period;
    document.querySelectorAll('.financials-period-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-period') === period) {
            btn.classList.add('active');
        }
    });
    if (currentFinancialsData) {
        // Get ticker from input or use stored ticker
        const ticker = document.getElementById('financialsTickerInput')?.value.trim().toUpperCase() || 'UNKNOWN';
        displayFinancials(currentFinancialsData, ticker);
    }
}

/**
 * Render detailed financials table and charts
 */
function renderDetailedFinancials(data, ticker) {
    debugLog('📋📋📋 [DEBUG] renderDetailedFinancials: FUNCTION CALLED for ticker:', ticker || 'NO TICKER');
    debugLog('📋📋📋 [DEBUG] renderDetailedFinancials: Data check:', {
        hasData: !!data,
        hasDetailedIncomeStatement: !!data?.detailed_income_statement,
        hasQuarterly: !!(data?.detailed_income_statement?.quarterly),
        quarterlyLength: data?.detailed_income_statement?.quarterly?.length || 0,
        annualLength: data?.detailed_income_statement?.annual?.length || 0,
        allKeys: data ? Object.keys(data).slice(0, 20) : []
    });
    
    const container = document.getElementById('detailedFinancialsContent');
    if (!container) {
        debugError('❌❌❌ [DEBUG] renderDetailedFinancials: detailedFinancialsContent container not found!');
        return;
    }
    debugLog('✅✅✅ [DEBUG] renderDetailedFinancials: Container found');
    
    if (!data || !data.detailed_income_statement) {
        debugWarn('⚠️⚠️⚠️ [DEBUG] renderDetailedFinancials: No detailed_income_statement in data');
        container.innerHTML = '<p style="color: var(--text-secondary); padding: 20px;">⚠️ Detailed financial data not available. Backend may not have returned detailed_income_statement. Check server logs.</p>';
        return;
    }
    
    debugLog('✅✅✅ [DEBUG] renderDetailedFinancials: detailed_income_statement found, processing...');
    
    let quarterlyData = data.detailed_income_statement.quarterly || [];
    let annualData = data.detailed_income_statement.annual || [];
    
    // If no quarterly data, try to use annual data as fallback
    if (quarterlyData.length === 0 && annualData.length > 0) {
        debugLog('⚠️ [DEBUG] renderDetailedFinancials: No quarterly data, using annual data as fallback');
        quarterlyData = annualData;
    }
    
    if (quarterlyData.length === 0) {
        debugWarn('⚠️ [DEBUG] renderDetailedFinancials: No financial data available (neither quarterly nor annual)');
        container.innerHTML = '<div style="padding: 40px; text-align: center; color: var(--text-secondary);"><p>⚠️ No quarterly or annual financial data available for this ticker.</p><p style="font-size: 0.9em; margin-top: 10px;">This may be because:</p><ul style="text-align: left; display: inline-block; margin-top: 10px;"><li>The ticker is too new or delisted</li><li>Yahoo Finance does not have financial data for this company</li><li>There was an error fetching the data</li></ul><p style="font-size: 0.9em; margin-top: 15px;">Check the browser console (F12) and server logs for more details.</p></div>';
        return;
    }
    
    debugLog('✅ [DEBUG] renderDetailedFinancials: Processing', quarterlyData.length, 'metrics');
    
    // Define only the metrics we want to show (in order)
    const allowedMetrics = [
        { 
            search: ['total revenue'], 
            exact: ['total revenue'],
            exclude: ['cost of revenue', 'cost of goods'],
            label: 'Total Revenue (Tržby)' 
        },
        { 
            search: ['cost of revenue', 'cost of goods sold', 'cogs'], 
            exact: ['cost of revenue', 'cost of goods sold'],
            exclude: [],
            label: 'Cost of Revenue' 
        },
        { 
            search: ['gross profit'], 
            exact: ['gross profit'],
            exclude: [],
            label: 'Gross Profit (Hrubý zisk)' 
        },
        { 
            search: ['operating expense', 'operating expenses', 'total operating expenses'], 
            exact: ['operating expense', 'operating expenses'],
            exclude: [],
            label: 'Operating Expenses (Provozní náklady)' 
        },
        { 
            search: ['operating income', 'operating income loss'], 
            exact: ['operating income'],
            exclude: [],
            label: 'Operating Income / Loss (Provozní zisk / ztráta)' 
        },
        { 
            search: ['ebitda'], 
            exact: ['ebitda'],
            exclude: [],
            label: 'EBITDA' 
        },
        { 
            search: ['net income common', 'net income from continuing'], 
            exact: ['net income common', 'net income from continuing'],
            exclude: ['pretax income', 'normalized income'],
            label: 'Net Income (Čistý zisk / ztráta)' 
        },
        { 
            search: ['diluted eps'], 
            exact: ['diluted eps'],
            exclude: ['basic eps'],
            label: 'EPS – Earnings Per Share (Zisk na akcii)' 
        }
    ];
    
    // Filter and map metrics to display
    const sortedMetrics = [];
    for (const allowedMetric of allowedMetrics) {
        const found = quarterlyData.find(metric => {
            const metricName = (metric.metric || '').toString().toLowerCase();
            
            // First try exact match
            const exactMatch = allowedMetric.exact.some(term => 
                metricName === term || metricName === term + 's'
            );
            if (exactMatch) {
                // Check if it should be excluded
                const shouldExclude = allowedMetric.exclude.some(excludeTerm => 
                    metricName.includes(excludeTerm)
                );
                if (!shouldExclude) {
                    return true;
                }
            }
            
            // Then try search terms (but exclude if needed)
            const shouldExclude = allowedMetric.exclude.some(excludeTerm => 
                metricName.includes(excludeTerm)
            );
            if (shouldExclude) {
                return false;
            }
            
            return allowedMetric.search.some(term => {
                // Prefer exact match or starts with
                return metricName === term || 
                       metricName.startsWith(term + ' ') ||
                       metricName.includes(' ' + term + ' ') ||
                       metricName.endsWith(' ' + term);
            });
        });
        if (found) {
            // Use the display label instead of original metric name
            sortedMetrics.push({
                ...found,
                metric: allowedMetric.label
            });
        } else {
            debugWarn('⚠️ [DEBUG] renderDetailedFinancials: Metric not found:', allowedMetric.label, 'Available metrics:', quarterlyData.map(m => m.metric).slice(0, 10));
        }
    }
    
    debugLog('✅ [DEBUG] renderDetailedFinancials: Filtered to', sortedMetrics.length, 'metrics from', quarterlyData.length, 'total');
    
    // Get all unique quarter dates from all metrics
    const allDates = new Set();
    sortedMetrics.forEach(metric => {
        if (metric.values && typeof metric.values === 'object') {
            Object.keys(metric.values).forEach(date => allDates.add(date));
        }
    });
    
    debugLog('📅 [DEBUG] renderDetailedFinancials: Found', allDates.size, 'unique dates');
    
    // Sort dates (most recent first) - same as Yahoo Finance
    const sortedDates = Array.from(allDates).sort((a, b) => {
        try {
            const dateA = new Date(a);
            const dateB = new Date(b);
            return dateB - dateA; // Descending order (newest first)
        } catch (e) {
            return 0;
        }
    });
    
    // Take first 4 quarters/years for display (most recent 4)
    const displayDates = sortedDates.slice(0, 4);
    debugLog('📅 [DEBUG] renderDetailedFinancials: Display dates:', displayDates);
    
    // Prepare data for charts (use all available dates, not just 4)
    const allSortedDates = sortedDates; // All available quarters for charts
    
    // Find metrics for charts
    const revenueMetric = sortedMetrics.find(m => m.metric.includes('Total Revenue'));
    const grossProfitMetric = sortedMetrics.find(m => m.metric.includes('Gross Profit'));
    const operatingIncomeMetric = sortedMetrics.find(m => m.metric.includes('Operating Income'));
    const netIncomeMetric = sortedMetrics.find(m => m.metric.includes('Net Income'));
    const costOfRevenueMetric = sortedMetrics.find(m => m.metric.includes('Cost of Revenue'));
    const operatingExpensesMetric = sortedMetrics.find(m => m.metric.includes('Operating Expenses'));
    
    // Build HTML with charts section first, then table
    let html = `
        <!-- Charts Section -->
        <div style="margin-bottom: 30px;">
            <!-- Revenue Trend Chart -->
            <div class="card" style="margin-bottom: 20px;">
                <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; font-size: 1.1em;">
                    📈 Revenue Trend
                    <span style="font-size: 0.7em; font-weight: normal; color: var(--text-secondary);">(All Available Quarters)</span>
                </h3>
                <div style="position: relative; height: 300px;">
                    <canvas id="detailedRevenueChart-${ticker}"></canvas>
                </div>
            </div>
            
            <!-- Multi-Metric Chart -->
            <div class="card" style="margin-bottom: 20px;">
                <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; font-size: 1.1em;">
                    📊 Key Metrics Overview
                    <span style="font-size: 0.7em; font-weight: normal; color: var(--text-secondary);">(Revenue, Gross Profit, Operating Income, Net Income)</span>
                </h3>
                <div style="position: relative; height: 350px;">
                    <canvas id="detailedMultiMetricChart-${ticker}"></canvas>
                </div>
            </div>
            
            <!-- Margins Chart -->
            <div class="card" style="margin-bottom: 20px;">
                <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; font-size: 1.1em;">
                    💰 Margin Trends
                    <span style="font-size: 0.7em; font-weight: normal; color: var(--text-secondary);">(Gross, Operating, Net Margin %)</span>
                </h3>
                <div style="position: relative; height: 300px;">
                    <canvas id="detailedMarginsChart-${ticker}"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Table Section -->
        <div class="detailed-financials-scroll">
            <table class="detailed-financials-table">
                <thead>
                    <tr>
                        <th>Breakdown</th>
                        <th>TTM</th>
                        ${displayDates.map(date => {
                            try {
                                const d = new Date(date);
                                const month = d.toLocaleString('en-US', { month: 'short' }).toUpperCase();
                                const day = d.getDate();
                                const year = d.getFullYear();
                                return `<th>${month} ${day}, ${year}</th>`;
                            } catch (e) {
                                return `<th>${date}</th>`;
                            }
                        }).join('')}
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Render each metric as a row (in sorted order)
    debugLog('🔄 [DEBUG] renderDetailedFinancials: Rendering', sortedMetrics.length, 'rows');
    sortedMetrics.forEach((metric, index) => {
        const metricName = metric.metric || 'Unknown';
        const ttm = metric.ttm;
        const values = metric.values || {};
        
        if (index < 3) {
            debugLog(`📊 [DEBUG] renderDetailedFinancials: Row ${index}: ${metricName}, TTM: ${ttm}, has ${Object.keys(values).length} values`);
        }
        
        html += '<tr>';
        html += `<td style="font-weight: 500;">${metricName}</td>`;
        html += `<td class="metric-value ${ttm !== null && ttm !== undefined && ttm < 0 ? 'negative' : ''}">${ttm !== null && ttm !== undefined ? formatDetailedNumber(ttm) : '-'}</td>`;
        
        // Add values for each quarter/year
        displayDates.forEach(date => {
            const value = values[date];
            const cellClass = value !== null && value !== undefined && value < 0 ? 'negative' : '';
            html += `<td class="metric-value ${cellClass}">${value !== null && value !== undefined ? formatDetailedNumber(value) : '-'}</td>`;
        });
        
        html += '</tr>';
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = html;
    
    // Create charts after HTML is rendered
    setTimeout(() => {
        // Destroy previous chart instances if they exist
        if (window.detailedRevenueChartInstances) {
            Object.values(window.detailedRevenueChartInstances).forEach(chart => {
                if (chart) chart.destroy();
            });
        }
        if (window.detailedMultiMetricChartInstances) {
            Object.values(window.detailedMultiMetricChartInstances).forEach(chart => {
                if (chart) chart.destroy();
            });
        }
        if (window.detailedMarginsChartInstances) {
            Object.values(window.detailedMarginsChartInstances).forEach(chart => {
                if (chart) chart.destroy();
            });
        }
        
        // Initialize chart instances objects if they don't exist
        if (!window.detailedRevenueChartInstances) window.detailedRevenueChartInstances = {};
        if (!window.detailedMultiMetricChartInstances) window.detailedMultiMetricChartInstances = {};
        if (!window.detailedMarginsChartInstances) window.detailedMarginsChartInstances = {};
        
        // Create Revenue Chart
        if (revenueMetric) {
            createDetailedRevenueChart(revenueMetric, allSortedDates, ticker);
        }
        
        // Create Multi-Metric Chart
        if (revenueMetric || grossProfitMetric || operatingIncomeMetric || netIncomeMetric) {
            createDetailedMultiMetricChart({
                revenue: revenueMetric,
                grossProfit: grossProfitMetric,
                operatingIncome: operatingIncomeMetric,
                netIncome: netIncomeMetric
            }, allSortedDates, ticker);
        }
        
        // Create Margins Chart
        if (revenueMetric && (grossProfitMetric || operatingIncomeMetric || netIncomeMetric)) {
            createDetailedMarginsChart({
                revenue: revenueMetric,
                grossProfit: grossProfitMetric,
                operatingIncome: operatingIncomeMetric,
                netIncome: netIncomeMetric
            }, allSortedDates, ticker);
        }
    }, 100);
}

/**
 * Create detailed revenue chart
 */
function createDetailedRevenueChart(revenueMetric, sortedDates, ticker) {
    const canvasId = `detailedRevenueChart-${ticker}`;
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        debugWarn(`Canvas ${canvasId} not found`);
        return;
    }
    
    // Destroy previous instance
    if (window.detailedRevenueChartInstances && window.detailedRevenueChartInstances[canvasId]) {
        window.detailedRevenueChartInstances[canvasId].destroy();
    }
    
    const values = revenueMetric.values || {};
    
    // Prepare data - reverse dates for chronological order (oldest to newest)
    const dates = [...sortedDates].reverse();
    const revenueData = dates.map(date => {
        const value = values[date];
        return value !== null && value !== undefined && !isNaN(value) ? value : null;
    });
    
    // Format labels (Q1 '24 format)
    const labels = dates.map(date => {
        try {
            const d = new Date(date);
            const quarter = Math.floor((d.getMonth()) / 3) + 1;
            const year = d.getFullYear().toString().slice(-2);
            return `Q${quarter} '${year}`;
        } catch (e) {
            return date;
        }
    });
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Revenue',
                data: revenueData,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#667eea',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            if (value === null) return 'N/A';
                            return `Revenue: ${formatDetailedNumber(value)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return formatDetailedNumber(value);
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
    
    // Store instance
    if (!window.detailedRevenueChartInstances) window.detailedRevenueChartInstances = {};
    window.detailedRevenueChartInstances[canvasId] = chart;
}

/**
 * Create detailed multi-metric chart
 */
function createDetailedMultiMetricChart(metrics, sortedDates, ticker) {
    const canvasId = `detailedMultiMetricChart-${ticker}`;
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        debugWarn(`Canvas ${canvasId} not found`);
        return;
    }
    
    // Destroy previous instance
    if (window.detailedMultiMetricChartInstances && window.detailedMultiMetricChartInstances[canvasId]) {
        window.detailedMultiMetricChartInstances[canvasId].destroy();
    }
    
    // Prepare data - reverse dates for chronological order
    const dates = [...sortedDates].reverse();
    const labels = dates.map(date => {
        try {
            const d = new Date(date);
            const quarter = Math.floor((d.getMonth()) / 3) + 1;
            const year = d.getFullYear().toString().slice(-2);
            return `Q${quarter} '${year}`;
        } catch (e) {
            return date;
        }
    });
    
    const datasets = [];
    
    // Revenue (bar chart, primary axis)
    if (metrics.revenue && metrics.revenue.values) {
        const revenueData = dates.map(date => {
            const value = metrics.revenue.values[date];
            return value !== null && value !== undefined && !isNaN(value) ? value : null;
        });
        datasets.push({
            label: 'Revenue',
            data: revenueData,
            type: 'bar',
            backgroundColor: 'rgba(102, 126, 234, 0.6)',
            borderColor: '#667eea',
            borderWidth: 2,
            yAxisID: 'y',
            order: 4
        });
    }
    
    // Gross Profit (line, secondary axis)
    if (metrics.grossProfit && metrics.grossProfit.values) {
        const grossProfitData = dates.map(date => {
            const value = metrics.grossProfit.values[date];
            return value !== null && value !== undefined && !isNaN(value) ? value : null;
        });
        datasets.push({
            label: 'Gross Profit',
            data: grossProfitData,
            type: 'line',
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.3,
            yAxisID: 'y1',
            pointRadius: 3,
            pointHoverRadius: 5,
            order: 1
        });
    }
    
    // Operating Income (line, secondary axis)
    if (metrics.operatingIncome && metrics.operatingIncome.values) {
        const operatingIncomeData = dates.map(date => {
            const value = metrics.operatingIncome.values[date];
            return value !== null && value !== undefined && !isNaN(value) ? value : null;
        });
        datasets.push({
            label: 'Operating Income',
            data: operatingIncomeData,
            type: 'line',
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.3,
            yAxisID: 'y1',
            pointRadius: 3,
            pointHoverRadius: 5,
            order: 2
        });
    }
    
    // Net Income (line, secondary axis) - color based on sign
    if (metrics.netIncome && metrics.netIncome.values) {
        const netIncomeData = dates.map(date => {
            const value = metrics.netIncome.values[date];
            return value !== null && value !== undefined && !isNaN(value) ? value : null;
        });
        // Determine color based on values (green if mostly positive, red if mostly negative)
        const hasPositive = netIncomeData.some(v => v !== null && v > 0);
        const hasNegative = netIncomeData.some(v => v !== null && v < 0);
        const netIncomeColor = hasPositive && !hasNegative ? '#10b981' : hasNegative && !hasPositive ? '#ef4444' : '#8b5cf6';
        
        datasets.push({
            label: 'Net Income',
            data: netIncomeData,
            type: 'line',
            borderColor: netIncomeColor,
            backgroundColor: hasPositive && !hasNegative ? 'rgba(16, 185, 129, 0.1)' : hasNegative && !hasPositive ? 'rgba(239, 68, 68, 0.1)' : 'rgba(139, 92, 246, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.3,
            yAxisID: 'y1',
            pointRadius: 3,
            pointHoverRadius: 5,
            order: 3
        });
    }
    
    if (datasets.length === 0) {
        debugWarn('No data available for multi-metric chart');
        return;
    }
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            if (value === null) return context.dataset.label + ': N/A';
                            return context.dataset.label + ': ' + formatDetailedNumber(value);
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return formatDetailedNumber(value);
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    title: {
                        display: true,
                        text: 'Revenue'
                    }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return formatDetailedNumber(value);
                        }
                    },
                    grid: {
                        drawOnChartArea: false
                    },
                    title: {
                        display: true,
                        text: 'Income Metrics'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
    
    // Store instance
    if (!window.detailedMultiMetricChartInstances) window.detailedMultiMetricChartInstances = {};
    window.detailedMultiMetricChartInstances[canvasId] = chart;
}

/**
 * Create detailed margins chart
 */
function createDetailedMarginsChart(metrics, sortedDates, ticker) {
    const canvasId = `detailedMarginsChart-${ticker}`;
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        debugWarn(`Canvas ${canvasId} not found`);
        return;
    }
    
    // Destroy previous instance
    if (window.detailedMarginsChartInstances && window.detailedMarginsChartInstances[canvasId]) {
        window.detailedMarginsChartInstances[canvasId].destroy();
    }
    
    // Prepare data - reverse dates for chronological order
    const dates = [...sortedDates].reverse();
    const labels = dates.map(date => {
        try {
            const d = new Date(date);
            const quarter = Math.floor((d.getMonth()) / 3) + 1;
            const year = d.getFullYear().toString().slice(-2);
            return `Q${quarter} '${year}`;
        } catch (e) {
            return date;
        }
    });
    
    const datasets = [];
    
    // Calculate Gross Margin
    if (metrics.revenue && metrics.revenue.values && metrics.grossProfit && metrics.grossProfit.values) {
        const grossMarginData = dates.map(date => {
            const revenue = metrics.revenue.values[date];
            const grossProfit = metrics.grossProfit.values[date];
            if (revenue !== null && revenue !== undefined && !isNaN(revenue) && 
                grossProfit !== null && grossProfit !== undefined && !isNaN(grossProfit) && revenue !== 0) {
                return (grossProfit / revenue) * 100;
            }
            return null;
        });
        datasets.push({
            label: 'Gross Margin (%)',
            data: grossMarginData,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6
        });
    }
    
    // Calculate Operating Margin
    if (metrics.revenue && metrics.revenue.values && metrics.operatingIncome && metrics.operatingIncome.values) {
        const operatingMarginData = dates.map(date => {
            const revenue = metrics.revenue.values[date];
            const operatingIncome = metrics.operatingIncome.values[date];
            if (revenue !== null && revenue !== undefined && !isNaN(revenue) && 
                operatingIncome !== null && operatingIncome !== undefined && !isNaN(operatingIncome) && revenue !== 0) {
                return (operatingIncome / revenue) * 100;
            }
            return null;
        });
        datasets.push({
            label: 'Operating Margin (%)',
            data: operatingMarginData,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6
        });
    }
    
    // Calculate Net Margin
    if (metrics.revenue && metrics.revenue.values && metrics.netIncome && metrics.netIncome.values) {
        const netMarginData = dates.map(date => {
            const revenue = metrics.revenue.values[date];
            const netIncome = metrics.netIncome.values[date];
            if (revenue !== null && revenue !== undefined && !isNaN(revenue) && 
                netIncome !== null && netIncome !== undefined && !isNaN(netIncome) && revenue !== 0) {
                return (netIncome / revenue) * 100;
            }
            return null;
        });
        datasets.push({
            label: 'Net Margin (%)',
            data: netMarginData,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6
        });
    }
    
    if (datasets.length === 0) {
        debugWarn('No data available for margins chart');
        return;
    }
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            if (value === null) return context.dataset.label + ': N/A';
                            return context.dataset.label + ': ' + value.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
    
    // Store instance
    if (!window.detailedMarginsChartInstances) window.detailedMarginsChartInstances = {};
    window.detailedMarginsChartInstances[canvasId] = chart;
}

// Note: displayFinancials and loadFinancials are very large functions (2000+ lines)
// They will be added in a follow-up commit due to size constraints
// For now, we export the helper functions and chart creation functions

// Global exports for backward compatibility during refactoring
window.switchFinancialsTab = switchFinancialsTab;
window.changeFinancialsPeriod = changeFinancialsPeriod;
window.renderDetailedFinancials = renderDetailedFinancials;
window.createDetailedRevenueChart = createDetailedRevenueChart;
window.createDetailedMultiMetricChart = createDetailedMultiMetricChart;
window.createDetailedMarginsChart = createDetailedMarginsChart;
window.formatCurrency = formatCurrency;
window.formatDetailedNumber = formatDetailedNumber;

export {
    switchFinancialsTab,
    changeFinancialsPeriod,
    renderDetailedFinancials,
    createDetailedRevenueChart,
    createDetailedMultiMetricChart,
    createDetailedMarginsChart,
    formatCurrency,
    formatDetailedNumber,
    safeToFixed
};
