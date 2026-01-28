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
 * Load financials data for a ticker
 */
export async function loadFinancials() {
    debugLog('[DEBUG] loadFinancials: Function called');
    const ticker = document.getElementById('financialsTickerInput').value.trim().toUpperCase();
    debugLog('[DEBUG] loadFinancials: Ticker extracted', {ticker});
    if (!ticker) {
        if (typeof window.showError === 'function') {
            window.showError('Please enter a stock ticker');
        }
        return;
    }

    const container = document.getElementById('financialsContent');
    if (!container) {
        debugError('[DEBUG] loadFinancials: financialsContent container not found');
        return;
    }

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
        debugLog('[DEBUG] loadFinancials: Fetching financials data for', ticker);
        
        // Use fetchJSON if available (for backward compatibility), otherwise use apiClient
        let result;
        if (typeof window.fetchJSON === 'function') {
            result = await window.fetchJSON(`/api/financials/${ticker}`, {}, container);
        } else {
            const response = await apiClient.get(`/api/financials/${ticker}`);
            result = { success: true, data: response };
        }
        
        if (!result.success) {
            return; // Error already displayed by fetchJSON
        }
        
        const data = result.data;
        
        // Track data timestamp if function exists
        if (typeof window.trackDataTimestamp === 'function') {
            window.trackDataTimestamp(`/api/financials/${ticker}`, result.timestamp || Date.now());
        }
        if (typeof window.updateDataFreshnessIndicator === 'function') {
            window.updateDataFreshnessIndicator(result.timestamp || new Date().toISOString(), 'financialsContent');
        }
        
        debugLog('[DEBUG] loadFinancials: Data received', {
            hasData: !!data,
            hasForwardEstimates: !!data.forward_estimates,
            dataKeys: Object.keys(data || {})
        });

        // Add to recent searches
        const companyName = data.company_name || ticker;
        if (typeof window.addToRecentSearches === 'function') {
            window.addToRecentSearches(ticker, companyName);
        }
        
        currentFinancialsData = data;
        
        debugLog('🔍 [DEBUG] loadFinancials: About to call displayFinancials');
        debugLog('🔍 [DEBUG] loadFinancials: Data keys:', Object.keys(data || {}));
        debugLog('🔍 [DEBUG] loadFinancials: Has detailed_income_statement?', !!data?.detailed_income_statement);
        if (data?.detailed_income_statement) {
            debugLog('🔍 [DEBUG] loadFinancials: quarterly length:', data.detailed_income_statement.quarterly?.length || 0);
            debugLog('🔍 [DEBUG] loadFinancials: annual length:', data.detailed_income_statement.annual?.length || 0);
        }
        
        debugLog('🎯🎯🎯 [DEBUG] loadFinancials: About to call displayFinancials with ticker:', ticker);
        debugLog('🎯🎯🎯 [DEBUG] loadFinancials: Data has keys:', Object.keys(data || {}).slice(0, 10));
        
        // Call displayFinancials with basic data (fast)
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
                debugWarn('⚠️⚠️⚠️ [DEBUG] loadFinancials: overviewTab is empty after displayFinancials, trying to re-render...');
                // Re-call displayFinancials if content is missing
                displayFinancials(data, ticker);
            }
        }, 100);
        
        // Load advanced analyses asynchronously in background (doesn't block UI)
        loadAdvancedFinancialsAnalyses(ticker, data);
        
        debugLog('🎯🎯🎯 [DEBUG] loadFinancials: displayFinancials call completed');
    } catch (error) {
        debugError('Error loading financials:', error);
        if (container) {
            container.innerHTML = `
                <div class="error">
                    Error loading financials: ${error.message}<br><br>
                    <button onclick="if(typeof window.loadFinancials === 'function') { window.loadFinancials(); }" style="margin-top: 10px; padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer;">Retry</button>
                </div>
            `;
        }
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
    debugLog('📊 [DEBUG] loadAdvancedFinancialsAnalyses: Loading advanced analyses for', ticker);
    
    try {
        // Show loading indicators for advanced sections
        const overviewTab = document.getElementById('financialsOverviewTab');
        if (overviewTab) {
            // Add loading placeholders for advanced sections
            const advancedSectionsPlaceholder = `
                <div id="advancedAnalysesPlaceholder" style="margin-top: 30px; padding: 20px; background: var(--metric-bg); border-radius: 12px; border: 1px dashed var(--border-color);">
                    <div style="text-align: center; color: var(--text-secondary);">
                        <div class="spinner" style="margin: 0 auto 15px;"></div>
                        <p style="margin: 0; font-size: 0.9em;">Loading advanced analyses...</p>
                    </div>
                </div>
            `;
            // Insert before the end of overviewTab
            if (!overviewTab.querySelector('#advancedAnalysesPlaceholder')) {
                overviewTab.insertAdjacentHTML('beforeend', advancedSectionsPlaceholder);
            }
        }
        
        // Fetch advanced analyses
        const response = await fetch(`/api/financials/${ticker}/advanced`);
        if (!response.ok) {
            throw new Error(`Failed to load advanced analyses: ${response.statusText}`);
        }
        
        const advancedData = await response.json();
        debugLog('📊 [DEBUG] loadAdvancedFinancialsAnalyses: Advanced data received', Object.keys(advancedData));
        
        // Merge advanced data with basic data
        const mergedData = { ...basicData, ...advancedData };
        
        // Merge Finviz estimates into forward_estimates if available
        if (advancedData.finviz_estimates) {
            const finvizEst = advancedData.finviz_estimates;
            if (!mergedData.forward_estimates) {
                mergedData.forward_estimates = { revenue: {}, eps: {} };
            }
            // Merge Finviz estimates into forward_estimates
            // CRITICAL: Ensure all Finviz estimates (especially future quarters) are merged
            if (finvizEst.estimates) {
                if (finvizEst.estimates.revenue) {
                    // Merge Finviz revenue estimates - Finviz data takes precedence for overlapping quarters
                    // This ensures future quarters from Finviz are always included
                    Object.assign(mergedData.forward_estimates.revenue, finvizEst.estimates.revenue);
                }
                if (finvizEst.estimates.eps) {
                    // Merge Finviz EPS estimates - Finviz data takes precedence for overlapping quarters
                    Object.assign(mergedData.forward_estimates.eps, finvizEst.estimates.eps);
                }
            }
            // Merge Finviz actuals (for better accuracy in charts)
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
        
        // CRITICAL: Verify forward_estimates are properly merged before re-rendering
        if (mergedData.forward_estimates && mergedData.forward_estimates.revenue) {
            const revenueKeys = Object.keys(mergedData.forward_estimates.revenue);
            const futureRevenueCount = revenueKeys.filter(q => {
                if (!q.includes('-Q')) return false;
                const [year, qNum] = q.split('-Q');
                const now = new Date();
                const currentYear = now.getFullYear();
                const currentMonth = now.getMonth() + 1;
                const currentQuarter = Math.ceil(currentMonth / 3);
                const quarterYear = parseInt(year);
                const quarterNum = parseInt(qNum);
                return quarterYear > currentYear || (quarterYear === currentYear && quarterNum > currentQuarter);
            }).length;
            
            if (futureRevenueCount > 0) {
                // Force chart recreation with updated forward_estimates
                // Destroy charts immediately before calling displayFinancials
                if (window.revenueChartInstance) {
                    window.revenueChartInstance.destroy();
                    window.revenueChartInstance = null;
                }
                if (window.epsChartInstance) {
                    window.epsChartInstance.destroy();
                    window.epsChartInstance = null;
                }
            }
        }
        
        // Re-render financials with advanced data (this will recreate charts with updated forward_estimates)
        displayFinancials(mergedData, ticker);
        
        // Remove loading placeholder
        const placeholder = document.getElementById('advancedAnalysesPlaceholder');
        if (placeholder) {
            placeholder.remove();
        }
        
    } catch (error) {
        debugWarn('⚠️ [DEBUG] loadAdvancedFinancialsAnalyses: Failed to load advanced analyses', error);
        // Remove loading placeholder on error
        const placeholder = document.getElementById('advancedAnalysesPlaceholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <div style="text-align: center; color: var(--text-secondary); padding: 10px;">
                    <p style="margin: 0; font-size: 0.85em;">Advanced analyses unavailable</p>
                </div>
            `;
        }
        // Don't fail the whole page if advanced analyses fail
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
        // Get ticker from input or use stored ticker
        const ticker = document.getElementById('financialsTickerInput')?.value.trim().toUpperCase() || 'UNKNOWN';
        displayFinancials(currentFinancialsData, ticker);
    }
}

/**
 * Switch between Overview and Detailed tabs
 */
export function switchFinancialsTab(tabName) {
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

// Note: displayFinancials and renderDetailedFinancials are very large functions
// They will be exported but kept in index.html for now due to size
// TODO: Extract these functions in a future refactoring step

// Global exports for backward compatibility during refactoring
if (typeof window !== 'undefined') {
    window.loadFinancials = loadFinancials;
    window.loadAdvancedFinancialsAnalyses = loadAdvancedFinancialsAnalyses;
    window.changeFinancialsPeriod = changeFinancialsPeriod;
    window.switchFinancialsTab = switchFinancialsTab;
    window.formatCurrency = formatCurrency;
    window.formatDetailedNumber = formatDetailedNumber;
    window.safeToFixed = safeToFixed;
}
