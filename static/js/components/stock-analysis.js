/**
 * Stock Analysis Component
 * Handles stock data fetching, display, and chart rendering
 */

import { debugLog, debugError } from '../utils/debug.js';
import apiClient from '../services/api-client.js';
import * as formatters from '../utils/formatters.js';

// Global state (will be moved to state management later)
let currentTimeframe = '1y';
let currentChartType = 'line';
let currentData = null;
let priceChart = null;
let rsiChart = null;
let macdChart = null;

/**
 * Stock Analysis Class
 */
class StockAnalysis {
    constructor() {
        this.currentTicker = null;
        this.currentTimeframe = '1y';
        this.currentChartType = 'line';
        this.currentData = null;
        this.charts = {
            price: null,
            rsi: null,
            macd: null
        };
    }

    /**
     * Search for stock data
     * @param {string} ticker - Stock ticker symbol
     * @param {string} period - Time period (1d, 1w, 1m, 3m, 6m, 1y, ytd)
     */
    async search(ticker, period = '1y') {
        if (!ticker) {
            throw new Error('Please enter a stock ticker');
        }

        this.currentTicker = ticker.toUpperCase();
        this.currentTimeframe = period;

        debugLog(`[STOCK ANALYSIS] Fetching data for ${this.currentTicker} with period ${period}`);

        try {
            const data = await apiClient.get(`/api/stock/${this.currentTicker}`, { period });

            if (!data || !data.chart_data) {
                throw new Error('Invalid data received from server');
            }

            // Ensure news is an array
            if (!data.news || !Array.isArray(data.news)) {
                data.news = [];
            }

            // Add timestamp
            data.lastUpdated = new Date().toISOString();
            this.currentData = data;

            debugLog(`[STOCK ANALYSIS] Data received:`, {
                hasData: !!data,
                hasChartData: !!data.chart_data,
                hasMetrics: !!data.metrics,
                ticker: data.ticker
            });

            // Display the data
            this.display(data);

            return data;
        } catch (error) {
            debugError('[STOCK ANALYSIS] Error in search:', error);
            throw error;
        }
    }

    /**
     * Display stock data
     * @param {Object} data - Stock data from API
     */
    display(data) {
        try {
            debugLog('[STOCK ANALYSIS] Calling displayStockData...');

            // Make sure results section is visible
            const resultsSection = document.getElementById('results');
            if (resultsSection) {
                resultsSection.classList.remove('hidden');
                debugLog('[STOCK ANALYSIS] Results section made visible');
            } else {
                debugError('[STOCK ANALYSIS] Results section not found!');
            }

            // Update header
            this.updateHeader(data);

            // Update price
            this.updatePrice(data);

            // Update metrics
            if (typeof displayMetrics === 'function') {
                displayMetrics(data.metrics);
            }

            // Update company info
            this.updateCompanyInfo(data);

            // Create charts
            this.createCharts(data);

            // Display news
            if (data.news && Array.isArray(data.news) && data.news.length > 0) {
                if (typeof displayNews === 'function') {
                    displayNews(data.news);
                }
            }

            debugLog('[STOCK ANALYSIS] displayStockData completed');
        } catch (error) {
            debugError('[STOCK ANALYSIS] Error displaying stock data:', error);
            debugError('[STOCK ANALYSIS] Error stack:', error.stack);
            throw error;
        }
    }

    /**
     * Update stock header
     */
    updateHeader(data) {
        const stockNameEl = document.getElementById('stockName');
        const stockSectorEl = document.getElementById('stockSector');

        if (stockNameEl) {
            stockNameEl.innerHTML = `${data.company_info?.name || data.ticker}`;
        }

        if (stockSectorEl) {
            stockSectorEl.textContent = 
                `${data.company_info?.sector || 'N/A'} • ${data.company_info?.industry || 'N/A'}`;
        }
    }

    /**
     * Update price display
     */
    updatePrice(data) {
        const price = data.metrics?.current_price;
        const change = data.metrics?.price_change;
        const changePct = data.metrics?.price_change_pct;

        const priceEl = document.getElementById('currentPrice');
        const changeEl = document.getElementById('priceChange');

        if (priceEl && price !== undefined) {
            priceEl.textContent = `$${price.toFixed(2)}`;
        }

        if (changeEl && change !== undefined && changePct !== undefined) {
            changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%)`;
            changeEl.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    /**
     * Update company info
     */
    updateCompanyInfo(data) {
        const companyInfoEl = document.getElementById('companyInfo');
        if (companyInfoEl) {
            companyInfoEl.innerHTML = `<p>${data.company_info?.description || 'No description available.'}</p>`;
        }
    }

    /**
     * Create all charts
     */
    createCharts(data) {
        try {
            if (this.currentChartType === 'line') {
                if (typeof createPriceChart === 'function') {
                    createPriceChart(data);
                }
            } else {
                if (typeof createCandlestickChart === 'function') {
                    createCandlestickChart(data);
                }
            }

            if (typeof createRSIChart === 'function') {
                createRSIChart(data);
            }

            if (typeof createMACDChart === 'function') {
                createMACDChart(data);
            }
        } catch (error) {
            debugError('[STOCK ANALYSIS] Error creating charts:', error);
        }
    }

    /**
     * Change timeframe
     */
    async changeTimeframe(period) {
        this.currentTimeframe = period;

        // Update active button
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.getAttribute('data-period') === period) {
                btn.classList.add('active');
            }
        });

        // Reload data if we have a ticker
        if (this.currentTicker) {
            await this.search(this.currentTicker, period);
        }
    }

    /**
     * Change chart type
     */
    changeChartType(type) {
        this.currentChartType = type;

        if (this.currentData) {
            this.createCharts(this.currentData);
        }
    }
}

// Create singleton instance
const stockAnalysis = new StockAnalysis();

// Export for use in other modules
export default stockAnalysis;

// Make available globally for backwards compatibility
window.stockAnalysis = stockAnalysis;
