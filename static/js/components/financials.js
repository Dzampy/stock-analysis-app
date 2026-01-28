/**
 * Financials Component
 * Handles financial data loading and display
 */

import { debugLog, debugError, debugWarn } from '../utils/debug.js';
import apiClient from '../services/api-client.js';

/**
 * Financials Class
 */
class Financials {
    constructor() {
        this.currentTicker = null;
        this.currentData = null;
    }

    /**
     * Load financials data
     * @param {string} ticker - Stock ticker symbol
     */
    async load(ticker) {
        if (!ticker) {
            throw new Error('Ticker is required');
        }

        this.currentTicker = ticker.toUpperCase();
        debugLog('[DEBUG] loadFinancials: Function called');
        debugLog('[DEBUG] loadFinancials: Ticker extracted', { ticker: this.currentTicker });

        const container = document.getElementById('financialsContent');
        if (!container) {
            debugError('Financials container not found');
            return;
        }

        // Show loading state
        container.innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <div class="spinner"></div>
                <p style="margin-top: 20px; color: var(--text-secondary);">Loading financial data for ${this.currentTicker}...</p>
            </div>
        `;

        try {
            debugLog('[DEBUG] loadFinancials: Fetching financials data for', this.currentTicker);
            
            const data = await apiClient.get(`/api/financials/${this.currentTicker}`);

            debugLog('[DEBUG] loadFinancials: Data received', {
                hasData: !!data,
                hasDetailedIncomeStatement: !!data?.detailed_income_statement
            });

            this.currentData = data;

            // Display financials
            if (typeof displayFinancials === 'function') {
                debugLog('🔍 [DEBUG] loadFinancials: About to call displayFinancials');
                debugLog('🔍 [DEBUG] loadFinancials: Data keys:', Object.keys(data || {}));
                debugLog('🔍 [DEBUG] loadFinancials: Has detailed_income_statement?', !!data?.detailed_income_statement);

                if (data?.detailed_income_statement) {
                    debugLog('🔍 [DEBUG] loadFinancials: quarterly length:', data.detailed_income_statement.quarterly?.length || 0);
                    debugLog('🔍 [DEBUG] loadFinancials: annual length:', data.detailed_income_statement.annual?.length || 0);
                }

                debugLog('🎯🎯🎯 [DEBUG] loadFinancials: About to call displayFinancials with ticker:', this.currentTicker);
                debugLog('🎯🎯🎯 [DEBUG] loadFinancials: Data has keys:', Object.keys(data || {}).slice(0, 10));

                displayFinancials(data, this.currentTicker);

                debugLog('🎯🎯🎯 [DEBUG] loadFinancials: displayFinancials call completed');

                // Check if overview tab is empty and try to re-render
                setTimeout(() => {
                    const overviewTab = document.getElementById('financialsOverviewTab');
                    if (overviewTab && (!overviewTab.innerHTML || overviewTab.innerHTML.trim() === '' || overviewTab.innerHTML.includes('Click "Load Financials"'))) {
                        debugWarn('⚠️⚠️⚠️ [DEBUG] loadFinancials: overviewTab is empty after displayFinancials, trying to re-render...');
                        // Try calling displayFinancials again
                        displayFinancials(data, this.currentTicker);
                    }
                }, 500);
            } else {
                debugError('displayFinancials function not found');
            }
        } catch (error) {
            debugError('[DEBUG] loadFinancials: Error loading financials:', error);
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                    <p>❌ Error loading financial data: ${error.message}</p>
                    <button onclick="if(typeof loadFinancials === 'function') loadFinancials('${this.currentTicker}')" 
                            style="margin-top: 20px; padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer;">
                        Retry
                    </button>
                </div>
            `;
        }
    }

    /**
     * Load advanced financial analyses
     * @param {string} ticker - Stock ticker symbol
     */
    async loadAdvancedAnalyses(ticker) {
        if (!ticker) {
            return;
        }

        debugLog('📊 [DEBUG] loadAdvancedFinancialsAnalyses: Loading advanced analyses for', ticker);

        try {
            const advancedData = await apiClient.get(`/api/financials/${ticker}/advanced`);

            debugLog('📊 [DEBUG] loadAdvancedFinancialsAnalyses: Advanced data received', Object.keys(advancedData));

            // Display advanced analyses if function exists
            if (typeof displayAdvancedFinancialsAnalyses === 'function') {
                displayAdvancedFinancialsAnalyses(advancedData, ticker);
            }
        } catch (error) {
            debugWarn('⚠️ [DEBUG] loadAdvancedFinancialsAnalyses: Failed to load advanced analyses', error);
        }
    }
}

// Create singleton instance
const financials = new Financials();

// Export for use in other modules
export default financials;

// Make available globally for backwards compatibility
window.financials = financials;
