/**
 * Storage Service - Centralized localStorage management with validation and error handling
 */

const STORAGE_PREFIX = 'stockapp_';
const MAX_STORAGE_SIZE = 5 * 1024 * 1024; // 5MB (conservative estimate)

/**
 * Get full key with prefix
 */
const getKey = (key) => `${STORAGE_PREFIX}${key}`;

/**
 * Check if storage is available
 */
const isStorageAvailable = () => {
    try {
        const test = '__storage_test__';
        localStorage.setItem(test, test);
        localStorage.removeItem(test);
        return true;
    } catch (e) {
        return false;
    }
};

/**
 * Get storage size in bytes (approximate)
 */
const getStorageSize = () => {
    let total = 0;
    for (let key in localStorage) {
        if (localStorage.hasOwnProperty(key)) {
            total += localStorage[key].length + key.length;
        }
    }
    return total;
};

/**
 * Storage Service Class
 */
class StorageService {
    constructor() {
        this.available = isStorageAvailable();
        if (!this.available) {
            console.warn('localStorage is not available');
        }
    }

    /**
     * Set item in storage
     * @param {string} key - Storage key
     * @param {any} value - Value to store (will be JSON stringified)
     * @param {number} expiration - Expiration time in milliseconds (optional)
     * @returns {boolean} Success status
     */
    set(key, value, expiration = null) {
        if (!this.available) {
            return false;
        }

        try {
            const item = {
                value: value,
                timestamp: Date.now(),
                expiration: expiration ? Date.now() + expiration : null
            };

            const serialized = JSON.stringify(item);
            const fullKey = getKey(key);

            // Check storage size
            const currentSize = getStorageSize();
            const newSize = currentSize - (localStorage.getItem(fullKey)?.length || 0) + serialized.length;

            if (newSize > MAX_STORAGE_SIZE) {
                console.warn(`Storage quota exceeded for key: ${key}`);
                this.cleanup(); // Try to free up space
                // Check again after cleanup
                const sizeAfterCleanup = getStorageSize();
                if (sizeAfterCleanup + serialized.length > MAX_STORAGE_SIZE) {
                    console.error('Storage quota still exceeded after cleanup');
                    return false;
                }
            }

            localStorage.setItem(fullKey, serialized);
            return true;
        } catch (e) {
            if (e.name === 'QuotaExceededError') {
                console.error('Storage quota exceeded');
                this.cleanup();
                return false;
            }
            console.error(`Error setting storage key ${key}:`, e);
            return false;
        }
    }

    /**
     * Get item from storage
     * @param {string} key - Storage key
     * @param {any} defaultValue - Default value if key doesn't exist or expired
     * @returns {any} Stored value or default
     */
    get(key, defaultValue = null) {
        if (!this.available) {
            return defaultValue;
        }

        try {
            const fullKey = getKey(key);
            const item = localStorage.getItem(fullKey);

            if (!item) {
                return defaultValue;
            }

            const parsed = JSON.parse(item);

            // Check expiration
            if (parsed.expiration && Date.now() > parsed.expiration) {
                this.remove(key);
                return defaultValue;
            }

            return parsed.value;
        } catch (e) {
            console.error(`Error getting storage key ${key}:`, e);
            return defaultValue;
        }
    }

    /**
     * Remove item from storage
     * @param {string} key - Storage key
     * @returns {boolean} Success status
     */
    remove(key) {
        if (!this.available) {
            return false;
        }

        try {
            localStorage.removeItem(getKey(key));
            return true;
        } catch (e) {
            console.error(`Error removing storage key ${key}:`, e);
            return false;
        }
    }

    /**
     * Check if key exists
     * @param {string} key - Storage key
     * @returns {boolean} Whether key exists
     */
    has(key) {
        if (!this.available) {
            return false;
        }

        try {
            const fullKey = getKey(key);
            const item = localStorage.getItem(fullKey);
            if (!item) {
                return false;
            }

            const parsed = JSON.parse(item);
            // Check expiration
            if (parsed.expiration && Date.now() > parsed.expiration) {
                this.remove(key);
                return false;
            }

            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Clear all items with our prefix
     */
    clear() {
        if (!this.available) {
            return;
        }

        const keys = Object.keys(localStorage);
        keys.forEach(key => {
            if (key.startsWith(STORAGE_PREFIX)) {
                localStorage.removeItem(key);
            }
        });
    }

    /**
     * Cleanup expired items and old items if storage is getting full
     */
    cleanup() {
        if (!this.available) {
            return;
        }

        const now = Date.now();
        const keys = Object.keys(localStorage);
        let cleaned = 0;

        keys.forEach(key => {
            if (key.startsWith(STORAGE_PREFIX)) {
                try {
                    const item = localStorage.getItem(key);
                    if (item) {
                        const parsed = JSON.parse(item);
                        if (parsed.expiration && now > parsed.expiration) {
                            localStorage.removeItem(key);
                            cleaned++;
                        }
                    }
                } catch (e) {
                    // Invalid item, remove it
                    localStorage.removeItem(key);
                    cleaned++;
                }
            }
        });

        if (cleaned > 0) {
            console.log(`Cleaned up ${cleaned} expired storage items`);
        }
    }

    /**
     * Get all keys with our prefix
     * @returns {string[]} Array of keys (without prefix)
     */
    keys() {
        if (!this.available) {
            return [];
        }

        const allKeys = Object.keys(localStorage);
        return allKeys
            .filter(key => key.startsWith(STORAGE_PREFIX))
            .map(key => key.substring(STORAGE_PREFIX.length));
    }
}

// Create singleton instance
const storageService = new StorageService();

// Cleanup on load
storageService.cleanup();

export default storageService;
