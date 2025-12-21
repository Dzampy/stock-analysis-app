#!/usr/bin/env python3
"""
Comprehensive backtest report for Fáze 2 improvements
"""
import requests
import json
import sys
sys.path.insert(0, '.')

from app import extract_ml_features, calculate_technical_indicators
import yfinance as yf

def comprehensive_backtest():
    """Comprehensive backtest of all Fáze 2 improvements"""
    print("=" * 80)
    print("COMPREHENSIVE BACKTEST REPORT - FÁZE 2")
    print("=" * 80)
    
    results = {
        'feature_extraction': {'passed': 0, 'failed': 0, 'details': []},
        'model_training': {'passed': 0, 'failed': 0, 'details': []},
        'confidence_intervals': {'passed': 0, 'failed': 0, 'details': []},
        'predictions': {'passed': 0, 'failed': 0, 'details': []}
    }
    
    test_tickers = ["AAPL", "MSFT", "TSLA", "GOOGL"]
    base_url = "http://127.0.0.1:5001"
    
    # Test 1: Feature Extraction
    print("\n" + "=" * 80)
    print("TEST 1: Feature Extraction")
    print("=" * 80)
    
    ticker = "AAPL"
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='1y')
        info = stock.info
        indicators = calculate_technical_indicators(df)
        features = extract_ml_features(ticker, df, info, indicators, {}, [])
        
        # Check all required features
        required_features = {
            'market_context': ['sp500_correlation_30d', 'sp500_correlation_90d', 'relative_strength_vs_sp500', 'market_regime'],
            'interactions': ['momentum_volatility_interaction', 'rsi_momentum_interaction', 'adx_momentum_interaction', 
                           'volume_momentum_interaction', 'beta_volatility_interaction', 'pe_momentum_interaction'],
            'lags': ['momentum_1m_lag', 'volume_ratio_lag'],
            'rolling': ['volatility_7d', 'volatility_ratio_7d_30d', 'momentum_acceleration']
        }
        
        all_present = True
        for category, feature_list in required_features.items():
            for feature in feature_list:
                if feature in features:
                    value = features.get(feature)
                    if value is not None:
                        print(f"  ✅ {category}.{feature}: {value:.4f}")
                        results['feature_extraction']['passed'] += 1
                    else:
                        print(f"  ⚠️  {category}.{feature}: None")
                        results['feature_extraction']['failed'] += 1
                        all_present = False
                else:
                    print(f"  ❌ {category}.{feature}: MISSING")
                    results['feature_extraction']['failed'] += 1
                    all_present = False
        
        if all_present:
            print(f"\n  ✅ All {sum(len(f) for f in required_features.values())} new features present!")
            results['feature_extraction']['details'].append(f"{ticker}: All features present")
        else:
            results['feature_extraction']['details'].append(f"{ticker}: Some features missing")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results['feature_extraction']['failed'] += 1
    
    # Test 2: Model Training & Predictions
    print("\n" + "=" * 80)
    print("TEST 2: Model Training & Predictions")
    print("=" * 80)
    
    for ticker in test_tickers:
        try:
            requests.post(f"{base_url}/api/clear-ml-cache")
            response = requests.get(f"{base_url}/api/ai-recommendations/{ticker}")
            
            if response.status_code == 200:
                data = response.json()
                pp = data.get('ml_models', {}).get('price_prediction', {})
                
                # Check model type
                model_type = pp.get('model_type', '')
                if model_type == 'random_forest':
                    print(f"  ✅ {ticker}: Model type = {model_type}")
                    results['model_training']['passed'] += 1
                    results['model_training']['details'].append(f"{ticker}: {model_type}")
                else:
                    print(f"  ⚠️  {ticker}: Model type = {model_type} (expected random_forest)")
                    results['model_training']['failed'] += 1
                
                # Check predictions
                predictions = pp.get('predictions', {})
                if predictions and all(p > 0 for p in predictions.values()):
                    print(f"  ✅ {ticker}: All predictions valid")
                    results['predictions']['passed'] += 1
                    results['predictions']['details'].append(f"{ticker}: Valid predictions")
                else:
                    print(f"  ❌ {ticker}: Invalid predictions")
                    results['predictions']['failed'] += 1
            else:
                print(f"  ❌ {ticker}: API error {response.status_code}")
                results['model_training']['failed'] += 1
        
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")
            results['model_training']['failed'] += 1
    
    # Test 3: Confidence Intervals
    print("\n" + "=" * 80)
    print("TEST 3: Confidence Intervals Calibration")
    print("=" * 80)
    
    max_widths = {'1m': 20, '3m': 28, '6m': 30, '12m': 38}
    
    for ticker in test_tickers:
        try:
            response = requests.get(f"{base_url}/api/ai-recommendations/{ticker}")
            if response.status_code == 200:
                data = response.json()
                pp = data.get('ml_models', {}).get('price_prediction', {})
                predictions = pp.get('predictions', {})
                ci = pp.get('confidence_intervals', {})
                
                ticker_passed = True
                for period in ['1m', '3m', '6m', '12m']:
                    if period in predictions and period in ci:
                        pred = predictions[period]
                        ci_lower = ci[period].get('lower', 0)
                        ci_upper = ci[period].get('upper', 0)
                        width = ((ci_upper - ci_lower) / pred * 100) if pred > 0 else 0
                        max_width = max_widths.get(period, 50)
                        
                        # Allow values at or below the limit (with small tolerance for floating point)
                        if width <= max_width + 0.01:  # Small tolerance for floating point precision
                            results['confidence_intervals']['passed'] += 1
                        else:
                            results['confidence_intervals']['failed'] += 1
                            ticker_passed = False
                            results['confidence_intervals']['details'].append(
                                f"{ticker} {period}: {width:.1f}% > {max_width}%"
                            )
                
                if ticker_passed:
                    print(f"  ✅ {ticker}: All CIs within limits")
                else:
                    print(f"  ⚠️  {ticker}: Some CIs exceed limits")
        
        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")
            results['confidence_intervals']['failed'] += 1
    
    # Final Report
    print("\n" + "=" * 80)
    print("FINAL BACKTEST REPORT")
    print("=" * 80)
    
    total_tests = 0
    total_passed = 0
    
    for test_name, result in results.items():
        passed = result['passed']
        failed = result['failed']
        total = passed + failed
        total_tests += total
        total_passed += passed
        
        if total > 0:
            success_rate = (passed / total) * 100
            status = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
            print(f"\n{status} {test_name.replace('_', ' ').title()}:")
            print(f"   Passed: {passed}/{total} ({success_rate:.1f}%)")
            if failed > 0:
                print(f"   Failed: {failed}")
                if result['details']:
                    for detail in result['details'][:3]:
                        print(f"     - {detail}")
    
    overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\n" + "=" * 80)
    print(f"OVERALL SUCCESS RATE: {total_passed}/{total_tests} ({overall_success:.1f}%)")
    print("=" * 80)
    
    if overall_success >= 90:
        print("✅ BACKTEST PASSED - All Fáze 2 improvements working correctly!")
    elif overall_success >= 70:
        print("⚠️  BACKTEST PARTIAL - Most improvements working, some issues detected")
    else:
        print("❌ BACKTEST FAILED - Significant issues detected")
    
    print("=" * 80)

if __name__ == "__main__":
    comprehensive_backtest()

