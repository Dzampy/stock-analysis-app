/**
 * Main.js - Application initialization and global setup
 */

import { debugLog, debugError, debugWarn } from './utils/debug.js';
import storageService from './services/storage-service.js';
import apiClient from './services/api-client.js';
import * as formatters from './utils/formatters.js';

// Make utilities globally available for backwards compatibility
window.debugLog = debugLog;
window.debugError = debugError;
window.debugWarn = debugWarn;
window.storageService = storageService;
window.apiClient = apiClient;
window.formatters = formatters;

// Global state
window.appState = {
    currentTicker: null,
    currentSection: 'welcomeSection',
    darkMode: true
};

/**
 * Initialize application
 */
function init() {
    debugLog('🚀 Application initializing...');
    
    // Load dark mode preference
    loadDarkMode();
    
    // Initialize sidebar state
    if (typeof initSidebarState === 'function') {
        initSidebarState();
    }
    
    // Initialize lazy loading
    if (typeof initLazyLoading === 'function') {
        initLazyLoading();
    }
    
    // Load latest updates
    if (typeof loadLatestUpdates === 'function') {
        loadLatestUpdates();
    }
    
    debugLog('✅ Application initialized');
}

/**
 * Load dark mode preference (default: dark for data-first / terminal feel)
 */
function loadDarkMode() {
    const darkMode = storageService.get('darkMode', true);
    if (darkMode) {
        document.documentElement.classList.add('dark-mode');
        document.body.classList.add('dark-mode');
        window.appState.darkMode = true;
    } else {
        document.documentElement.classList.remove('dark-mode');
        document.body.classList.remove('dark-mode');
        window.appState.darkMode = false;
    }
}

/**
 * Toggle dark mode
 */
function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark-mode');
    document.body.classList.toggle('dark-mode', isDark);
    storageService.set('darkMode', isDark);
    window.appState.darkMode = isDark;
    debugLog('Dark mode:', isDark ? 'enabled' : 'disabled');
}

// Export for use in other modules
export { init, toggleDarkMode, storageService, apiClient, formatters };

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
