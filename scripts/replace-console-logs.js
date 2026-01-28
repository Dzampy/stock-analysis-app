#!/usr/bin/env node
/**
 * Script to replace console.log/error/warn with debug logging
 * Usage: node scripts/replace-console-logs.js
 */

const fs = require('fs');
const path = require('path');

const INDEX_HTML_PATH = path.join(__dirname, '..', 'templates', 'index.html');

// Read the file
let content = fs.readFileSync(INDEX_HTML_PATH, 'utf8');

// Count original console statements
const originalLogCount = (content.match(/console\.log\(/g) || []).length;
const originalErrorCount = (content.match(/console\.error\(/g) || []).length;
const originalWarnCount = (content.match(/console\.warn\(/g) || []).length;

console.log(`Found ${originalLogCount} console.log, ${originalErrorCount} console.error, ${originalWarnCount} console.warn`);

// Replace console.log with debugLog (but keep error handling console.error)
// Pattern: console.log(...) -> debugLog(...)
// But we need to be careful with multiline statements

// First, add debug import at the top of script tags if not present
if (!content.includes('import { debugLog') && content.includes('<script')) {
    // Find the first <script> tag that's not a CDN link
    const scriptTagMatch = content.match(/<script[^>]*>(?!.*src)/);
    if (scriptTagMatch) {
        const insertPos = scriptTagMatch.index + scriptTagMatch[0].length;
        const debugImport = `
        // Import debug utilities
        import { debugLog, debugError, debugWarn } from '/static/js/utils/debug.js';
        `;
        content = content.slice(0, insertPos) + debugImport + content.slice(insertPos);
    }
}

// Replace console.log with debugLog
content = content.replace(/console\.log\(/g, 'debugLog(');

// Replace console.warn with debugWarn
content = content.replace(/console\.warn\(/g, 'debugWarn(');

// Keep console.error as is for now (errors should always be logged)
// But we can replace with debugError if needed
// content = content.replace(/console\.error\(/g, 'debugError(');

// Write back
fs.writeFileSync(INDEX_HTML_PATH, content, 'utf8');

const newLogCount = (content.match(/debugLog\(/g) || []).length;
const newWarnCount = (content.match(/debugWarn\(/g) || []).length;

console.log(`✅ Replaced ${originalLogCount} console.log with debugLog`);
console.log(`✅ Replaced ${originalWarnCount} console.warn with debugWarn`);
console.log(`⚠️  Kept ${originalErrorCount} console.error (errors should always be logged)`);
console.log(`\nTotal replacements: ${originalLogCount + originalWarnCount}`);
