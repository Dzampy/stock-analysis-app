/**
 * Stock Analysis Component
 * Handles stock data fetching, display, and chart rendering
 */

import { debugLog, debugError } from '../utils/debug.js';
import apiClient from '../services/api-client.js';
import * as formatters from '../utils/formatters.js';

// Global state for stock analysis
let currentData = null;
let currentTimeframe = '1y';
let currentChartType = 'line';
let priceChart = null;
let rsiChart = null;
let macdChart = null;

// Chart features state
const chartFeatures = {
    volumeOverlay: false,
    supportResistance: false,
    annotationMode: false
};

/**
 * Search for stock data
 * @param {string} ticker - Stock ticker symbol
 * @param {string} period - Time period (1d, 1w, 1m, 3m, 6m, 1y, ytd)
 */
export async function searchStock(ticker, period = '1y') {
    if (!ticker) {
        throw new Error('Please enter a stock ticker');
    }

    ticker = ticker.trim().toUpperCase();
    currentTimeframe = period || currentTimeframe;

    debugLog(`[STOCK ANALYSIS] Fetching data for ${ticker} with period ${currentTimeframe}`);

    try {
        const data = await apiClient.get(`/api/stock/${ticker}`, { period: currentTimeframe });
        
        debugLog(`[STOCK ANALYSIS] Data received:`, { 
            hasData: !!data, 
            hasChartData: !!data.chart_data,
            hasMetrics: !!data.metrics,
            ticker: data.ticker 
        });

        // Ensure news is an array
        if (!data.news || !Array.isArray(data.news)) {
            data.news = [];
        }

        // Add timestamp to data
        data.lastUpdated = new Date().toISOString();
        currentData = data;

        // Make sure results section is visible
        const resultsSection = document.getElementById('results');
        if (resultsSection) {
            resultsSection.classList.remove('hidden');
            debugLog('[STOCK ANALYSIS] Results section made visible');
        } else {
            debugError('[STOCK ANALYSIS] Results section not found!');
        }

        // Display stock data
        setTimeout(() => {
            try {
                debugLog('[STOCK ANALYSIS] Calling displayStockData...');
                displayStockData(data);
                debugLog('[STOCK ANALYSIS] displayStockData completed');
            } catch (displayError) {
                debugError('[STOCK ANALYSIS] Error displaying stock data:', displayError);
                debugError('[STOCK ANALYSIS] Error stack:', displayError.stack);
                throw displayError;
            }
        }, 100);

        return data;
    } catch (error) {
        debugError('[STOCK ANALYSIS] Error in searchStock:', error);
        debugError('[STOCK ANALYSIS] Error stack:', error.stack);
        throw error;
    }
}

/**
 * Display stock data in UI
 * @param {Object} data - Stock data from API
 */
function displayStockData(data) {
    const ticker = data.ticker || document.getElementById('tickerInput')?.value.trim().toUpperCase();
    
    // Update header
    const stockNameEl = document.getElementById('stockName');
    if (stockNameEl) {
        stockNameEl.innerHTML = `${data.company_info?.name || data.ticker || ticker}`;
    }
    
    const stockSectorEl = document.getElementById('stockSector');
    if (stockSectorEl) {
        stockSectorEl.textContent = 
            `${data.company_info?.sector || 'N/A'} • ${data.company_info?.industry || 'N/A'}`;
    }

    // Update price
    const price = data.metrics?.current_price;
    const change = data.metrics?.price_change;
    const changePct = data.metrics?.price_change_pct;

    const currentPriceEl = document.getElementById('currentPrice');
    if (currentPriceEl && price !== undefined) {
        currentPriceEl.textContent = `$${price.toFixed(2)}`;
    }

    const changeEl = document.getElementById('priceChange');
    if (changeEl && change !== undefined && changePct !== undefined) {
        changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%)`;
        changeEl.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
    }

    // Create charts
    try {
        if (currentChartType === 'line') {
            createPriceChart(data);
        } else {
            createCandlestickChart(data);
        }
        createRSIChart(data);
        createMACDChart(data);
    } catch (chartError) {
        debugError('[STOCK ANALYSIS] Error creating charts:', chartError);
    }

    // Update metrics, company info, news, etc.
    if (typeof displayMetrics === 'function') {
        displayMetrics(data.metrics);
    }

    const companyInfoEl = document.getElementById('companyInfo');
    if (companyInfoEl) {
        companyInfoEl.innerHTML = `<p>${data.company_info?.description || 'No description available.'}</p>`;
    }

    // Display news if available
    if (data.news && Array.isArray(data.news) && data.news.length > 0) {
        if (typeof displayNews === 'function') {
            displayNews(data.news);
        }
    }
}

/**
 * Create price chart (line chart)
 * @param {Object} data - Stock data
 */
function createPriceChart(data) {
    const canvas = document.getElementById('priceChart');
    if (!canvas) {
        debugError('Price chart canvas not found');
        return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        debugError('Could not get 2d context from price chart canvas');
        return;
    }

    // Destroy existing chart
    if (priceChart) {
        priceChart.destroy();
    }

    // Show line chart, hide candlestick
    canvas.style.display = 'block';
    const candlestickContainer = document.getElementById('candlestickChart');
    if (candlestickContainer) {
        candlestickContainer.style.display = 'none';
    }

    // Chart configuration
    const chartData = data.chart_data;
    const datasets = [{
        label: '💰 Close Price',
        data: chartData.close,
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 5
    }];

    // Add indicators if available
    if (data.indicators?.sma_20) {
        datasets.push({
            label: 'SMA 20',
            data: data.indicators.sma_20,
            borderColor: '#f59e0b',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0
        });
    }

    if (data.indicators?.sma_50) {
        datasets.push({
            label: 'SMA 50',
            data: data.indicators.sma_50,
            borderColor: '#ef4444',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0
        });
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        }
    });
}

/**
 * Create candlestick chart
 * @param {Object} data - Stock data
 */
function createCandlestickChart(data) {
    // Implementation would use lightweight-charts library
    // This is a placeholder - full implementation would be in charts/candlestick-chart.js
    debugLog('Creating candlestick chart - implementation in charts module');
}

/**
 * Create RSI chart
 * @param {Object} data - Stock data
 */
function createRSIChart(data) {
    const canvas = document.getElementById('rsiChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (rsiChart) {
        rsiChart.destroy();
    }

    const rsiData = data.indicators?.rsi || [];
    if (rsiData.length === 0) return;

    rsiChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.chart_data.dates,
            datasets: [{
                label: 'RSI',
                data: rsiData,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            }
        }
    });
}

/**
 * Create MACD chart
 * @param {Object} data - Stock data
 */
function createMACDChart(data) {
    const canvas = document.getElementById('macdChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (macdChart) {
        macdChart.destroy();
    }

    const macdData = data.indicators?.macd || [];
    const signalData = data.indicators?.macd_signal || [];
    const histogramData = data.indicators?.macd_histogram || [];

    if (macdData.length === 0) return;

    macdChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.chart_data.dates,
            datasets: [
                {
                    label: 'MACD',
                    data: macdData,
                    borderColor: '#667eea',
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'Signal',
                    data: signalData,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'Histogram',
                    data: histogramData,
                    type: 'bar',
                    backgroundColor: 'rgba(102, 126, 234, 0.3)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

/**
 * Change timeframe
 * @param {string} period - New timeframe period
 */
export function changeTimeframe(period) {
    currentTimeframe = period;
    
    // Update active button
    document.querySelectorAll('.timeframe-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-period') === period) {
            btn.classList.add('active');
        }
    });
    
    // Reload stock data if we have a ticker
    const tickerInput = document.getElementById('tickerInput');
    if (tickerInput) {
        const ticker = tickerInput.value.trim().toUpperCase();
        if (ticker) {
            searchStock(ticker, period);
        }
    }
}

/**
 * Switch chart type
 * @param {string} type - 'line' or 'candlestick'
 */
export function switchChartType(type) {
    currentChartType = type;
    
    if (currentData) {
        if (type === 'line') {
            createPriceChart(currentData);
        } else {
            createCandlestickChart(currentData);
        }
    }
}

// Export for global access
export { currentData, currentTimeframe, currentChartType, chartFeatures };

// Make functions globally available for backwards compatibility
if (typeof window !== 'undefined') {
    window.searchStock = (ticker) => searchStock(ticker, currentTimeframe);
    window.changeTimeframe = changeTimeframe;
    window.switchChartType = switchChartType;
}
