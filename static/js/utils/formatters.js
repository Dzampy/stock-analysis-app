/**
 * Formatters - Utility functions for formatting data
 */

/**
 * Format currency
 * @param {number} value - Value to format
 * @param {string} currency - Currency symbol (default: '$')
 * @param {number} decimals - Number of decimals (default: 2)
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (value, currency = '$', decimals = 2) => {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const numValue = typeof value === 'string' ? parseFloat(value) : value;

    if (isNaN(numValue)) {
        return 'N/A';
    }

    // Format large numbers with abbreviations
    if (Math.abs(numValue) >= 1e12) {
        return `${currency}${(numValue / 1e12).toFixed(decimals)}T`;
    } else if (Math.abs(numValue) >= 1e9) {
        return `${currency}${(numValue / 1e9).toFixed(decimals)}B`;
    } else if (Math.abs(numValue) >= 1e6) {
        return `${currency}${(numValue / 1e6).toFixed(decimals)}M`;
    } else if (Math.abs(numValue) >= 1e3) {
        return `${currency}${(numValue / 1e3).toFixed(decimals)}K`;
    } else {
        return `${currency}${numValue.toFixed(decimals)}`;
    }
};

/**
 * Format percentage
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimals (default: 2)
 * @param {boolean} showSign - Show + sign for positive values (default: false)
 * @returns {string} Formatted percentage string
 */
export const formatPercentage = (value, decimals = 2, showSign = false) => {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const numValue = typeof value === 'string' ? parseFloat(value) : value;

    if (isNaN(numValue)) {
        return 'N/A';
    }

    const sign = showSign && numValue >= 0 ? '+' : '';
    return `${sign}${numValue.toFixed(decimals)}%`;
};

/**
 * Format number with thousand separators
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimals (default: 0)
 * @returns {string} Formatted number string
 */
export const formatNumber = (value, decimals = 0) => {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const numValue = typeof value === 'string' ? parseFloat(value) : value;

    if (isNaN(numValue)) {
        return 'N/A';
    }

    return numValue.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
};

/**
 * Format date
 * @param {Date|string|number} date - Date to format
 * @param {string} format - Format style ('short', 'long', 'time', 'datetime')
 * @returns {string} Formatted date string
 */
export const formatDate = (date, format = 'short') => {
    if (!date) {
        return 'N/A';
    }

    const dateObj = date instanceof Date ? date : new Date(date);

    if (isNaN(dateObj.getTime())) {
        return 'N/A';
    }

    const options = {
        short: { year: 'numeric', month: 'short', day: 'numeric' },
        long: { year: 'numeric', month: 'long', day: 'numeric' },
        time: { hour: '2-digit', minute: '2-digit' },
        datetime: { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }
    };

    return dateObj.toLocaleDateString('en-US', options[format] || options.short);
};

/**
 * Format relative time (e.g., "2 hours ago")
 * @param {Date|string|number} date - Date to format
 * @returns {string} Relative time string
 */
export const formatRelativeTime = (date) => {
    if (!date) {
        return 'N/A';
    }

    const dateObj = date instanceof Date ? date : new Date(date);

    if (isNaN(dateObj.getTime())) {
        return 'N/A';
    }

    const now = new Date();
    const diffMs = now - dateObj;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffSecs < 60) {
        return 'just now';
    } else if (diffMins < 60) {
        return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
    } else if (diffHours < 24) {
        return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    } else if (diffDays < 7) {
        return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
    } else {
        return formatDate(dateObj, 'short');
    }
};

/**
 * Format large number with abbreviation
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimals (default: 1)
 * @returns {string} Formatted number with abbreviation
 */
export const formatLargeNumber = (value, decimals = 1) => {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }

    const numValue = typeof value === 'string' ? parseFloat(value) : value;

    if (isNaN(numValue)) {
        return 'N/A';
    }

    const absValue = Math.abs(numValue);
    const sign = numValue < 0 ? '-' : '';

    if (absValue >= 1e12) {
        return `${sign}${(absValue / 1e12).toFixed(decimals)}T`;
    } else if (absValue >= 1e9) {
        return `${sign}${(absValue / 1e9).toFixed(decimals)}B`;
    } else if (absValue >= 1e6) {
        return `${sign}${(absValue / 1e6).toFixed(decimals)}M`;
    } else if (absValue >= 1e3) {
        return `${sign}${(absValue / 1e3).toFixed(decimals)}K`;
    } else {
        return numValue.toFixed(decimals);
    }
};

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML string
 */
export const escapeHtml = (text) => {
    if (typeof text !== 'string') {
        return String(text);
    }

    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };

    return text.replace(/[&<>"']/g, m => map[m]);
};

export default {
    formatCurrency,
    formatPercentage,
    formatNumber,
    formatDate,
    formatRelativeTime,
    formatLargeNumber,
    escapeHtml
};
