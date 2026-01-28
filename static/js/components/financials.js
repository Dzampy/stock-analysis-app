/**
 * Financials Component
 * Handles all financial data loading, display, and chart rendering
 */

import { debugLog, debugError, debugWarn } from '../utils/debug.js';
import apiClient from '../services/api-client.js';
import * as formatters from '../utils/formatters.js';

// State
let currentFinancialsData = null;
let financialsPeriod = 'quarterly'; // 'quarterly' or 'annual'

/**
 * Load financials data for a ticker
 */
export async function loadFinancials() {
    debugLog('[FINANCIALS] loadFinancials: Function called');
    const ticker = document.getElementById('financialsTickerInput')?.value.trim().toUpperCase();
    debugLog('[FINANCIALS] loadFinancials: Ticker extracted', {ticker});
    
    if (!ticker) {
        if (typeof showError === 'function') {
            showError('Please enter a stock ticker');
        }
        return;
    }

    const container = document.getElementById('financialsContent');
    if (!container) {
        debugError('[FINANCIALS] financialsContent container not found');
        return;
    }

    // Show loading skeleton
    container.innerHTML = `
        <div class="skeleton-card">
            <div class="skeleton skeleton-title"></div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                <div class="skeleton skeleton-metric"></div>
                <div class="skeleton skeleton-metric"></div>
                <div class="skeleton skeleton-metric"></div>
                <div class="skeleton skeleton-metric"></div>
            </div>
            <div class="skeleton skeleton-chart" style="margin-top: 30px;"></div>
            <div style="margin-top: 20px;">
                <div class="skeleton skeleton-table-row"></div>
                <div class="skeleton skeleton-table-row"></div>
                <div class="skeleton skeleton-table-row"></div>
            </div>
        </div>
    `;

    try {
        debugLog('[FINANCIALS] loadFinancials: Fetching financials data for', ticker);
        const result = await apiClient.get(`/api/financials/${ticker}`);
        
        if (!result.success) {
            if (typeof showError === 'function') {
                showError(result.error || 'Failed to load financials data');
            }
            return;
        }
        
        const data = result.data;
        
        // Track data timestamp if function exists
        if (typeof trackDataTimestamp === 'function') {
            trackDataTimestamp(`/api/financials/${ticker}`, result.timestamp || Date.now());
        }
        if (typeof updateDataFreshnessIndicator === 'function') {
            updateDataFreshnessIndicator(result.timestamp || new Date().toISOString(), 'financialsContent');
        }
        
        debugLog('[FINANCIALS] loadFinancials: Data received', {
            hasData: !!data,
            hasForwardEstimates: !!data.forward_estimates,
            dataKeys: Object.keys(data || {})
        });

        // Add to recent searches if function exists
        if (typeof addToRecentSearches === 'function') {
            const companyName = data.company_name || ticker;
            addToRecentSearches(ticker, companyName);
        }
        
        currentFinancialsData = data;
        
        // Call displayFinancials with basic data
        displayFinancials(data, ticker);
        
        // Ensure tabs container is visible after displayFinancials
        setTimeout(() => {
            const tabsContainer = document.getElementById('financialsTabsContainer');
            if (tabsContainer) {
                tabsContainer.style.display = 'block';
                tabsContainer.style.visibility = 'visible';
            }
            
            // Double-check that overviewTab has content
            const overviewTab = document.getElementById('financialsOverviewTab');
            if (overviewTab && (!overviewTab.innerHTML || overviewTab.innerHTML.trim() === '' || overviewTab.innerHTML.includes('Loading overview data'))) {
                debugWarn('[FINANCIALS] loadFinancials: overviewTab is empty after displayFinancials, trying to re-render...');
                displayFinancials(data, ticker);
            }
        }, 100);
        
        // Load advanced analyses asynchronously in background
        loadAdvancedFinancialsAnalyses(ticker, data);
        
        debugLog('[FINANCIALS] loadFinancials: displayFinancials call completed');
    } catch (error) {
        debugError('[FINANCIALS] loadFinancials: Error', error);
        container.innerHTML = `
            <div class="error">
                Error loading financials: ${error.message}<br><br>
                <button onclick="if(typeof window.loadFinancials === 'function') { window.loadFinancials(); }" style="margin-top: 10px; padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer;">Retry</button>
            </div>
        `;
    } finally {
        const button = document.querySelector('button[onclick*="loadFinancials"]');
        if (button && button.classList.contains('loading')) {
            button.classList.remove('loading');
            button.innerHTML = button.dataset.originalText || '🔍 Load Financials';
        }
    }
}

/**
 * Load advanced financial analyses asynchronously
 */
export async function loadAdvancedFinancialsAnalyses(ticker, basicData) {
    debugLog('[FINANCIALS] loadAdvancedFinancialsAnalyses: Loading advanced analyses for', ticker);
    
    try {
        // Show loading indicators for advanced sections
        const overviewTab = document.getElementById('financialsOverviewTab');
        if (overviewTab) {
            const advancedSectionsPlaceholder = `
                <div id="advancedAnalysesPlaceholder" style="margin-top: 30px; padding: 20px; background: var(--metric-bg); border-radius: 12px; border: 1px dashed var(--border-color);">
                    <div style="text-align: center; color: var(--text-secondary);">
                        <div class="spinner" style="margin: 0 auto 15px;"></div>
                        <p style="margin: 0; font-size: 0.9em;">Loading advanced analyses...</p>
                    </div>
                </div>
            `;
            if (!overviewTab.querySelector('#advancedAnalysesPlaceholder')) {
                overviewTab.insertAdjacentHTML('beforeend', advancedSectionsPlaceholder);
            }
        }
        
        // Fetch advanced analyses
        const response = await apiClient.get(`/api/financials/${ticker}/advanced`);
        const advancedData = response.data || response;
        
        debugLog('[FINANCIALS] loadAdvancedFinancialsAnalyses: Advanced data received', Object.keys(advancedData));
        
        // Merge advanced data with basic data
        const mergedData = { ...basicData, ...advancedData };
        
        // Merge Finviz estimates into forward_estimates if available
        if (advancedData.finviz_estimates) {
            const finvizEst = advancedData.finviz_estimates;
            if (!mergedData.forward_estimates) {
                mergedData.forward_estimates = { revenue: {}, eps: {} };
            }
            if (finvizEst.estimates) {
                if (finvizEst.estimates.revenue) {
                    Object.assign(mergedData.forward_estimates.revenue, finvizEst.estimates.revenue);
                }
                if (finvizEst.estimates.eps) {
                    Object.assign(mergedData.forward_estimates.eps, finvizEst.estimates.eps);
                }
            }
            if (finvizEst.actuals) {
                if (!mergedData.quarterly_actuals) {
                    mergedData.quarterly_actuals = { revenue: {}, eps: {} };
                }
                if (finvizEst.actuals.revenue) {
                    Object.assign(mergedData.quarterly_actuals.revenue, finvizEst.actuals.revenue);
                }
                if (finvizEst.actuals.eps) {
                    Object.assign(mergedData.quarterly_actuals.eps, finvizEst.actuals.eps);
                }
            }
        }
        
        currentFinancialsData = mergedData;
        
        // Destroy charts before re-rendering
        if (window.revenueChartInstance) {
            window.revenueChartInstance.destroy();
            window.revenueChartInstance = null;
        }
        if (window.epsChartInstance) {
            window.epsChartInstance.destroy();
            window.epsChartInstance = null;
        }
        
        // Re-render financials with advanced data
        displayFinancials(mergedData, ticker);
        
        // Remove loading placeholder
        const placeholder = document.getElementById('advancedAnalysesPlaceholder');
        if (placeholder) {
            placeholder.remove();
        }
        
    } catch (error) {
        debugWarn('[FINANCIALS] loadAdvancedFinancialsAnalyses: Failed to load advanced analyses', error);
        const placeholder = document.getElementById('advancedAnalysesPlaceholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <div style="text-align: center; color: var(--text-secondary); padding: 10px;">
                    <p style="margin: 0; font-size: 0.85em;">Advanced analyses unavailable</p>
                </div>
            `;
        }
    }
}

/**
 * Change financials period (quarterly/annual)
 */
export function changeFinancialsPeriod(period) {
    financialsPeriod = period;
    document.querySelectorAll('.financials-period-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-period') === period) {
            btn.classList.add('active');
        }
    });
    if (currentFinancialsData) {
        const ticker = document.getElementById('financialsTickerInput')?.value.trim().toUpperCase() || 'UNKNOWN';
        displayFinancials(currentFinancialsData, ticker);
    }
}

/**
 * Display financials data
 * NOTE: This is a large function that will be kept in index.html for now
 * due to its size and dependencies. It will be gradually refactored.
 */
export function displayFinancials(data, ticker) {
    // This function is too large to move immediately
    // It will remain in index.html and be called from here
    // TODO: Refactor this function into smaller pieces
    if (typeof window.displayFinancials === 'function') {
        window.displayFinancials(data, ticker);
    } else {
        debugError('[FINANCIALS] displayFinancials: window.displayFinancials not found');
    }
}

/**
 * Switch between financials tabs (overview/detailed)
 */
export function switchFinancialsTab(tabName) {
    if (typeof window.switchFinancialsTab === 'function') {
        window.switchFinancialsTab(tabName);
    } else {
        debugError('[FINANCIALS] switchFinancialsTab: window.switchFinancialsTab not found');
    }
}

/**
 * Render detailed financials table
 */
export function renderDetailedFinancials(data, ticker) {
    if (typeof window.renderDetailedFinancials === 'function') {
        window.renderDetailedFinancials(data, ticker);
    } else {
        debugError('[FINANCIALS] renderDetailedFinancials: window.renderDetailedFinancials not found');
    }
}

// Global exports for backward compatibility during refactoring
window.loadFinancials = loadFinancials;
window.loadAdvancedFinancialsAnalyses = loadAdvancedFinancialsAnalyses;
window.changeFinancialsPeriod = changeFinancialsPeriod;
window.displayFinancials = displayFinancials;
window.switchFinancialsTab = switchFinancialsTab;
window.renderDetailedFinancials = renderDetailedFinancials;

export { currentFinancialsData, financialsPeriod };
