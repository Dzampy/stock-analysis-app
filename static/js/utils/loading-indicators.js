/**
 * Loading indicator helpers - used across Stock Analysis, Financials, News, AI, Backtest.
 * Exposed on window for use by inline script until full modular refactor.
 */
(function () {
    'use strict';

    function createLoadingIndicator(message, includeProgress) {
        message = message || 'Loading...';
        includeProgress = !!includeProgress;
        var progressHtml = includeProgress
            ? '<div style="margin-top: 15px; width: 100%; max-width: 300px; margin-left: auto; margin-right: auto;">'
                + '<div style="background: rgba(102, 126, 234, 0.1); border-radius: 8px; height: 8px; overflow: hidden;">'
                + '<div class="loading-progress-bar" style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: 0%; transition: width 0.3s ease; animation: loading-pulse 1.5s ease-in-out infinite;"></div>'
                + '</div><p style="margin-top: 10px; font-size: 0.85em; color: var(--text-tertiary);">This may take a moment...</p></div>'
            : '';
        return '<div style="text-align: center; padding: 40px;">'
            + '<div class="spinner" style="margin: 0 auto;"></div>'
            + '<p style="margin-top: 20px; color: var(--text-secondary); font-size: 1.05em;">' + message + '</p>'
            + progressHtml + '</div>';
    }

    function createLongOperationIndicator(title, subtitle) {
        subtitle = subtitle || 'This may take 30–60 seconds...';
        return '<div style="text-align: center; padding: 60px 40px;">'
            + '<div class="spinner spinner-lg" style="margin: 0 auto;" role="status" aria-live="polite" aria-label="Loading"></div>'
            + '<p style="margin-top: 30px; color: var(--text-primary); font-size: 1.2em; font-weight: 600;">' + title + '</p>'
            + '<p style="margin-top: 15px; color: var(--text-secondary);">' + subtitle + '</p>'
            + '<div style="margin-top: 25px; width: 100%; max-width: 400px; margin-left: auto; margin-right: auto;">'
            + '<div style="background: rgba(102, 126, 234, 0.1); border-radius: 10px; height: 10px; overflow: hidden;">'
            + '<div class="loading-progress-bar" style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: 0%; transition: width 0.5s ease; animation: loading-pulse 2s ease-in-out infinite;"></div>'
            + '</div></div></div>';
    }

    function createMLTrainingIndicator(ticker) {
        return '<div style="text-align: center; padding: 60px 40px;">'
            + '<div class="spinner spinner-lg" style="margin: 0 auto;" role="status" aria-live="polite" aria-label="Training model"></div>'
            + '<p style="margin-top: 30px; color: var(--text-primary); font-size: 1.2em; font-weight: 600;">Training ML Model for ' + ticker + '</p>'
            + '<p style="margin-top: 15px; color: var(--text-secondary);">This may take 30–60 seconds...</p>'
            + '<div style="margin-top: 25px; width: 100%; max-width: 400px; margin-left: auto; margin-right: auto;">'
            + '<div style="background: rgba(102, 126, 234, 0.1); border-radius: 10px; height: 10px; overflow: hidden;">'
            + '<div class="loading-progress-bar" style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: 0%; transition: width 0.5s ease; animation: loading-pulse 2s ease-in-out infinite;"></div>'
            + '</div><div style="margin-top: 15px; font-size: 0.9em; color: var(--text-tertiary);">'
            + '<p>📊 Downloading historical data...</p><p style="margin-top: 5px;">🤖 Training model...</p><p style="margin-top: 5px;">✅ Calculating predictions...</p>'
            + '</div></div></div>';
    }

    window.createLoadingIndicator = createLoadingIndicator;
    window.createLongOperationIndicator = createLongOperationIndicator;
    window.createMLTrainingIndicator = createMLTrainingIndicator;
})();
