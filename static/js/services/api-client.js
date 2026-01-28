/**
 * API Client - Centralized API calls with error handling and retry logic
 */

import { debugLog, debugError } from '../utils/debug.js';

const API_BASE_URL = ''; // Relative URLs

/**
 * Default fetch options
 */
const defaultOptions = {
    headers: {
        'Content-Type': 'application/json',
    },
    credentials: 'same-origin'
};

/**
 * Retry configuration
 */
const RETRY_CONFIG = {
    maxRetries: 3,
    retryDelay: 1000, // 1 second
    retryableStatuses: [408, 429, 500, 502, 503, 504]
};

/**
 * Sleep utility for retry delays
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Check if error is retryable
 */
const isRetryable = (error, status) => {
    if (status && RETRY_CONFIG.retryableStatuses.includes(status)) {
        return true;
    }
    // Network errors are retryable
    if (error instanceof TypeError && error.message.includes('fetch')) {
        return true;
    }
    return false;
};

/**
 * Fetch with retry logic
 * @param {string} url - API endpoint
 * @param {RequestInit} options - Fetch options
 * @param {number} retryCount - Current retry count
 * @returns {Promise<Response>}
 */
const fetchWithRetry = async (url, options = {}, retryCount = 0) => {
    try {
        const response = await fetch(url, { ...defaultOptions, ...options });

        // If successful or non-retryable error, return immediately
        if (response.ok || !isRetryable(null, response.status)) {
            return response;
        }

        // Retry on retryable errors
        if (retryCount < RETRY_CONFIG.maxRetries) {
            const delay = RETRY_CONFIG.retryDelay * Math.pow(2, retryCount); // Exponential backoff
            debugLog(`Retrying request to ${url} (attempt ${retryCount + 1}/${RETRY_CONFIG.maxRetries}) after ${delay}ms`);
            await sleep(delay);
            return fetchWithRetry(url, options, retryCount + 1);
        }

        return response;
    } catch (error) {
        // Network errors are retryable
        if (isRetryable(error) && retryCount < RETRY_CONFIG.maxRetries) {
            const delay = RETRY_CONFIG.retryDelay * Math.pow(2, retryCount);
            debugLog(`Retrying request to ${url} after network error (attempt ${retryCount + 1}/${RETRY_CONFIG.maxRetries})`);
            await sleep(delay);
            return fetchWithRetry(url, options, retryCount + 1);
        }
        throw error;
    }
};

/**
 * API Client Class
 */
class APIClient {
    /**
     * GET request
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {Promise<any>}
     */
    async get(endpoint, params = {}) {
        const url = new URL(endpoint, window.location.origin);
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined) {
                url.searchParams.append(key, params[key]);
            }
        });

        debugLog(`API GET: ${url.pathname}${url.search}`);

        try {
            const response = await fetchWithRetry(url.toString());
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            debugError(`API GET error for ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * POST request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body
     * @returns {Promise<any>}
     */
    async post(endpoint, data = {}) {
        const url = new URL(endpoint, window.location.origin);

        debugLog(`API POST: ${url.pathname}`, data);

        try {
            const response = await fetchWithRetry(url.toString(), {
                method: 'POST',
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            debugError(`API POST error for ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * PUT request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body
     * @returns {Promise<any>}
     */
    async put(endpoint, data = {}) {
        const url = new URL(endpoint, window.location.origin);

        debugLog(`API PUT: ${url.pathname}`, data);

        try {
            const response = await fetchWithRetry(url.toString(), {
                method: 'PUT',
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            debugError(`API PUT error for ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * DELETE request
     * @param {string} endpoint - API endpoint
     * @returns {Promise<any>}
     */
    async delete(endpoint) {
        const url = new URL(endpoint, window.location.origin);

        debugLog(`API DELETE: ${url.pathname}`);

        try {
            const response = await fetchWithRetry(url.toString(), {
                method: 'DELETE'
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            debugError(`API DELETE error for ${endpoint}:`, error);
            throw error;
        }
    }
}

// Create singleton instance
const apiClient = new APIClient();

export default apiClient;
