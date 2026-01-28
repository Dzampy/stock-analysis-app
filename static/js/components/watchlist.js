/**
 * Watchlist Component
 * Handles watchlist management (add, remove, groups, sidebar display)
 */

import { debugLog, debugError, debugWarn } from '../utils/debug.js';
import storageService from '../services/storage-service.js';
import apiClient from '../services/api-client.js';

/**
 * Watchlist Class
 */
class Watchlist {
    constructor() {
        this.watchlist = this.load();
    }

    /**
     * Load watchlist from storage
     */
    load() {
        try {
            const stored = storageService.get('watchlist');
            if (!stored) {
                return {
                    groups: { 'Uncategorized': [] },
                    order: ['Uncategorized']
                };
            }

            // Handle old format (array)
            if (Array.isArray(stored)) {
                return {
                    groups: { 'Uncategorized': stored },
                    order: ['Uncategorized']
                };
            }

            // New format with groups
            return stored;
        } catch (error) {
            debugError('Error loading watchlist:', error);
            return {
                groups: { 'Uncategorized': [] },
                order: ['Uncategorized']
            };
        }
    }

    /**
     * Save watchlist to storage
     */
    save() {
        storageService.set('watchlist', this.watchlist);
    }

    /**
     * Add ticker to watchlist
     * @param {string} ticker - Stock ticker
     * @param {string} groupName - Group name (default: 'Uncategorized')
     */
    add(ticker, groupName = 'Uncategorized') {
        ticker = ticker.toUpperCase().trim();

        if (!ticker) {
            return false;
        }

        // Check if already exists
        if (this.has(ticker)) {
            return false;
        }

        // Ensure group exists
        if (!this.watchlist.groups[groupName]) {
            this.watchlist.groups[groupName] = [];
            if (!this.watchlist.order.includes(groupName)) {
                this.watchlist.order.push(groupName);
            }
        }

        // Add to group
        this.watchlist.groups[groupName].push(ticker);
        this.save();

        return true;
    }

    /**
     * Remove ticker from watchlist
     * @param {string} ticker - Stock ticker
     */
    remove(ticker) {
        ticker = ticker.toUpperCase().trim();

        for (const groupName in this.watchlist.groups) {
            const index = this.watchlist.groups[groupName].indexOf(ticker);
            if (index >= 0) {
                this.watchlist.groups[groupName].splice(index, 1);
                
                // Remove empty groups (except Uncategorized)
                if (this.watchlist.groups[groupName].length === 0 && groupName !== 'Uncategorized') {
                    delete this.watchlist.groups[groupName];
                    const orderIndex = this.watchlist.order.indexOf(groupName);
                    if (orderIndex >= 0) {
                        this.watchlist.order.splice(orderIndex, 1);
                    }
                }
                
                this.save();
                return true;
            }
        }

        return false;
    }

    /**
     * Check if ticker is in watchlist
     * @param {string} ticker - Stock ticker
     */
    has(ticker) {
        ticker = ticker.toUpperCase().trim();

        for (const group of Object.values(this.watchlist.groups)) {
            if (group.includes(ticker)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Get group for ticker
     * @param {string} ticker - Stock ticker
     * @returns {string|null} Group name or null
     */
    getGroup(ticker) {
        ticker = ticker.toUpperCase().trim();

        for (const [groupName, tickers] of Object.entries(this.watchlist.groups)) {
            if (tickers.includes(ticker)) {
                return groupName;
            }
        }

        return null;
    }

    /**
     * Get all tickers
     * @returns {string[]} Array of all tickers
     */
    getAllTickers() {
        const allTickers = [];
        for (const group of Object.values(this.watchlist.groups)) {
            allTickers.push(...group);
        }
        return allTickers;
    }

    /**
     * Create new group
     * @param {string} groupName - Group name
     */
    createGroup(groupName) {
        if (!this.watchlist.groups[groupName]) {
            this.watchlist.groups[groupName] = [];
            if (!this.watchlist.order.includes(groupName)) {
                this.watchlist.order.push(groupName);
            }
            this.save();
            return true;
        }
        return false;
    }

    /**
     * Delete group
     * @param {string} groupName - Group name
     */
    deleteGroup(groupName) {
        if (groupName === 'Uncategorized') {
            return false; // Cannot delete Uncategorized
        }

        if (this.watchlist.groups[groupName]) {
            // Move tickers to Uncategorized
            const tickers = this.watchlist.groups[groupName];
            if (!this.watchlist.groups['Uncategorized']) {
                this.watchlist.groups['Uncategorized'] = [];
            }
            this.watchlist.groups['Uncategorized'].push(...tickers);

            delete this.watchlist.groups[groupName];
            const orderIndex = this.watchlist.order.indexOf(groupName);
            if (orderIndex >= 0) {
                this.watchlist.order.splice(orderIndex, 1);
            }
            this.save();
            return true;
        }

        return false;
    }

    /**
     * Move ticker to different group
     * @param {string} ticker - Stock ticker
     * @param {string} targetGroup - Target group name
     */
    moveToGroup(ticker, targetGroup) {
        ticker = ticker.toUpperCase().trim();

        // Remove from current group
        this.remove(ticker);

        // Add to target group
        return this.add(ticker, targetGroup);
    }
}

// Create singleton instance
const watchlist = new Watchlist();

// Export for use in other modules
export default watchlist;

// Make available globally for backwards compatibility
window.watchlist = watchlist;
