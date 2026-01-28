/**
 * Watchlist Component
 * Handles watchlist management, display, and sidebar updates
 */

import { debugLog, debugError, debugWarn } from '../utils/debug.js';
import storageService from '../services/storage-service.js';
import apiClient from '../services/api-client.js';

// Watchlist state
let watchlist = {
    groups: {},
    order: []
};

/**
 * Get watchlist from storage
 */
export function getWatchlist() {
    try {
        const stored = storageService.get('watchlist');
        if (!stored) {
            return { groups: {}, order: [] };
        }
        
        if (Array.isArray(stored)) {
            // Old format - convert to new format
            return {
                groups: { 'Uncategorized': stored },
                order: ['Uncategorized']
            };
        }
        
        return stored;
    } catch (error) {
        debugError('Error loading watchlist:', error);
        return { groups: {}, order: [] };
    }
}

/**
 * Save watchlist to storage
 */
export function saveWatchlist(data) {
    try {
        storageService.set('watchlist', data);
        watchlist = data;
    } catch (error) {
        debugError('Error saving watchlist:', error);
    }
}

/**
 * Add ticker to watchlist
 * @param {string} ticker - Stock ticker
 * @param {string} groupName - Group name (default: 'Uncategorized')
 */
export function addToWatchlist(ticker, groupName = 'Uncategorized') {
    ticker = ticker.trim().toUpperCase();
    if (!ticker) return false;

    const wl = getWatchlist();
    
    // Check if already exists
    for (const group of Object.values(wl.groups)) {
        if (Array.isArray(group) && group.includes(ticker)) {
            return false; // Already exists
        }
    }

    // Initialize group if needed
    if (!wl.groups[groupName]) {
        wl.groups[groupName] = [];
    }
    if (!wl.order.includes(groupName)) {
        wl.order.push(groupName);
    }

    // Add ticker
    wl.groups[groupName].push(ticker);
    saveWatchlist(wl);
    
    // Update UI
    updateWatchlistSidebar();
    loadWatchlistFromStorage();
    
    return true;
}

/**
 * Remove ticker from watchlist
 * @param {string} ticker - Stock ticker
 */
export function removeFromWatchlist(ticker) {
    ticker = ticker.trim().toUpperCase();
    const wl = getWatchlist();
    let removed = false;

    for (const groupName in wl.groups) {
        const group = wl.groups[groupName];
        if (Array.isArray(group)) {
            const index = group.indexOf(ticker);
            if (index > -1) {
                group.splice(index, 1);
                removed = true;
                
                // Remove empty groups
                if (group.length === 0) {
                    delete wl.groups[groupName];
                    const orderIndex = wl.order.indexOf(groupName);
                    if (orderIndex > -1) {
                        wl.order.splice(orderIndex, 1);
                    }
                }
            }
        }
    }

    if (removed) {
        saveWatchlist(wl);
        updateWatchlistSidebar();
        loadWatchlistFromStorage();
    }

    return removed;
}

/**
 * Get group for a ticker
 * @param {string} ticker - Stock ticker
 * @returns {string|null} Group name or null
 */
export function getTickerGroup(ticker) {
    ticker = ticker.trim().toUpperCase();
    const wl = getWatchlist();

    for (const [groupName, group] of Object.entries(wl.groups)) {
        if (Array.isArray(group) && group.includes(ticker)) {
            return groupName;
        }
    }

    return null;
}

/**
 * Update watchlist sidebar
 */
export function updateWatchlistSidebar() {
    const container = document.getElementById('watchlistSidebarContent');
    if (!container) {
        debugWarn('watchlistSidebarContent container not found');
        return;
    }

    const wl = getWatchlist();
    debugLog('updateWatchlistSidebar - watchlist:', JSON.stringify(wl));

    if (Object.keys(wl.groups).length === 0 || wl.order.length === 0) {
        container.innerHTML = `
            <div class="watchlist-empty">
                <div class="watchlist-empty-icon">⭐</div>
                <p>Your watchlist is empty</p>
                <p style="font-size: 0.85em; margin-top: 10px; opacity: 0.7;">Add stocks to track them here</p>
            </div>
        `;
        return;
    }

    let html = '';
    
    // Render groups in order
    for (const groupName of wl.order) {
        const group = wl.groups[groupName];
        if (!Array.isArray(group) || group.length === 0) continue;

        html += `
            <div class="watchlist-group" data-group="${groupName}">
                <div class="watchlist-group-header">
                    <span class="watchlist-group-name">${groupName}</span>
                    <span class="watchlist-group-count">(${group.length})</span>
                </div>
                <div class="watchlist-group-items">
        `;

        // Load and display data for each ticker
        for (const ticker of group) {
            html += `
                <div class="watchlist-item" data-ticker="${ticker}">
                    <div class="watchlist-item-ticker">${ticker}</div>
                    <div class="watchlist-item-price">Loading...</div>
                    <div class="watchlist-item-change">-</div>
                </div>
            `;
        }

        html += `
                </div>
            </div>
        `;
    }

    container.innerHTML = html;

    // Load data for all tickers
    loadWatchlistItemData();
}

/**
 * Load data for watchlist items
 */
async function loadWatchlistItemData() {
    const wl = getWatchlist();
    const allTickers = [];
    
    for (const group of Object.values(wl.groups)) {
        if (Array.isArray(group)) {
            allTickers.push(...group);
        }
    }

    // Remove duplicates
    const uniqueTickers = [...new Set(allTickers)];

    // Load data for each ticker
    for (const ticker of uniqueTickers) {
        try {
            const data = await apiClient.get(`/api/stock/${ticker}`, { period: '1d' });
            
            const itemEl = document.querySelector(`.watchlist-item[data-ticker="${ticker}"]`);
            if (itemEl && data.metrics) {
                const priceEl = itemEl.querySelector('.watchlist-item-price');
                const changeEl = itemEl.querySelector('.watchlist-item-change');
                
                if (priceEl) {
                    priceEl.textContent = `$${data.metrics.current_price.toFixed(2)}`;
                }
                
                if (changeEl) {
                    const changePct = data.metrics.price_change_pct || 0;
                    changeEl.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
                    changeEl.className = `watchlist-item-change ${changePct >= 0 ? 'positive' : 'negative'}`;
                }
            }
        } catch (error) {
            debugError(`Error loading data for ${ticker}:`, error);
        }
    }
}

/**
 * Load watchlist from storage and display
 */
export function loadWatchlistFromStorage() {
    watchlist = getWatchlist();
    updateWatchlistSidebar();
}

// Initialize on load
if (typeof window !== 'undefined') {
    // Make functions globally available
    window.addToWatchlist = addToWatchlist;
    window.removeFromWatchlist = removeFromWatchlist;
    window.getWatchlist = getWatchlist;
    window.updateWatchlistSidebar = updateWatchlistSidebar;
    window.loadWatchlistFromStorage = loadWatchlistFromStorage;
}

export default {
    getWatchlist,
    saveWatchlist,
    addToWatchlist,
    removeFromWatchlist,
    getTickerGroup,
    updateWatchlistSidebar,
    loadWatchlistFromStorage
};
