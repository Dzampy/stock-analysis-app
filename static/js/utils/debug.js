/**
 * Debug utility - conditional logging based on environment
 * Only logs in development mode, removed in production builds
 */

// Check if we're in development mode
const isDevelopment = () => {
    // Check for development indicators
    return (
        window.location.hostname === 'localhost' ||
        window.location.hostname === '127.0.0.1' ||
        window.location.hostname.includes('localhost') ||
        document.documentElement.hasAttribute('data-debug') ||
        localStorage.getItem('debug') === 'true'
    );
};

const DEBUG_ENABLED = isDevelopment();

/**
 * Debug logger - only logs in development
 * @param {...any} args - Arguments to log
 */
export const debugLog = (...args) => {
    if (DEBUG_ENABLED) {
        console.log(...args);
    }
};

/**
 * Debug error logger - always logs errors (even in production)
 * @param {...any} args - Arguments to log
 */
export const debugError = (...args) => {
    // Errors should always be logged, but can be filtered in production
    if (DEBUG_ENABLED) {
        console.error(...args);
    } else {
        // In production, only log critical errors
        // Could send to error tracking service here
        console.error(...args);
    }
};

/**
 * Debug warn logger - only logs in development
 * @param {...any} args - Arguments to log
 */
export const debugWarn = (...args) => {
    if (DEBUG_ENABLED) {
        console.warn(...args);
    }
};

/**
 * Debug info logger - only logs in development
 * @param {...any} args - Arguments to log
 */
export const debugInfo = (...args) => {
    if (DEBUG_ENABLED) {
        console.info(...args);
    }
};

/**
 * Debug group - only groups in development
 * @param {string} label - Group label
 */
export const debugGroup = (label) => {
    if (DEBUG_ENABLED) {
        console.group(label);
    }
};

/**
 * Debug group end - only ends group in development
 */
export const debugGroupEnd = () => {
    if (DEBUG_ENABLED) {
        console.groupEnd();
    }
};

// Export default object for convenience
export default {
    log: debugLog,
    error: debugError,
    warn: debugWarn,
    info: debugInfo,
    group: debugGroup,
    groupEnd: debugGroupEnd,
    isEnabled: DEBUG_ENABLED
};
