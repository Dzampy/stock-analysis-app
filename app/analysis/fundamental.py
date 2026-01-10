"""
Fundamental analysis - Financials scoring, cash flow, profitability, balance sheet, industry ranking
"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, List
from app.utils.constants import RATE_LIMIT_DELAY
from app.utils.logger import logger


def calculate_metrics(df: pd.DataFrame, info: Dict) -> Dict:
    """
    Calculate investment metrics
    
    Args:
        df: DataFrame with price data
        info: Stock info dict from yfinance
        
    Returns:
        Dict with calculated metrics
    """
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
    
    # Price change
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0
    
    # Volatility (30-day)
    if len(df) >= 30:
        returns = df['Close'].pct_change().dropna()
        volatility = returns.tail(30).std() * np.sqrt(252) * 100  # Annualized
    else:
        volatility = None
    
    # 52-week range
    year_high = df['High'].max()
    year_low = df['Low'].min()
    
    # Metrics from info
    metrics = {
        'current_price': round(current_price, 2),
        'price_change': round(price_change, 2),
        'price_change_pct': round(price_change_pct, 2),
        'volume': int(df['Volume'].iloc[-1]),
        'avg_volume': int(df['Volume'].tail(30).mean()) if len(df) >= 30 else int(df['Volume'].mean()),
        'volatility': round(volatility, 2) if volatility else None,
        'year_high': round(year_high, 2),
        'year_low': round(year_low, 2),
        'pe_ratio': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'market_cap': info.get('marketCap'),
        'dividend_yield': info.get('dividendYield'),
        'beta': info.get('beta'),
        'eps': info.get('trailingEps'),
        'book_value': info.get('bookValue'),
        'price_to_book': info.get('priceToBook'),
        'profit_margin': info.get('profitMargins'),
        'revenue_growth': info.get('revenueGrowth'),
        'earnings_growth': info.get('earningsGrowth'),
    }
    
    return metrics


def calculate_financials_score(financials: Dict, info: Dict, company_stage: str = 'unknown') -> Dict:
    """
    Calculate overall financials score (0-100) based on multiple metrics
    
    For growth companies with growing revenue, applies more lenient criteria.
    
    Args:
        financials: Financial data dict
        info: Stock info dict
        company_stage: Company stage ('growth', 'early_stage', 'mature', 'unknown')
        
    Returns:
        Dict with score, grade, and breakdown
    """
    try:
        score = 0
        max_score = 100
        breakdown = {}
        
        snapshot = financials.get('executive_snapshot', {})
        balance_sheet = financials.get('balance_sheet', {})
        margins = financials.get('margins', {})
        quarterly_income = financials.get('income_statement', {}).get('quarterly', [])
        
        # Check if this is a growth company with growing revenue
        revenue_yoy = snapshot.get('revenue_yoy', 0) if snapshot.get('revenue_yoy') else 0
        is_growth_with_revenue = (company_stage == 'growth' or company_stage == 'early_stage') and revenue_yoy > 0
        
        # 1. Revenue Growth (0-20 points)
        revenue_growth_score = 0
        if revenue_yoy >= 20:
            revenue_growth_score = 20
        elif revenue_yoy >= 10:
            revenue_growth_score = 15
        elif revenue_yoy >= 5:
            revenue_growth_score = 10
        elif revenue_yoy >= 0:
            revenue_growth_score = 5
        elif revenue_yoy >= -5:
            revenue_growth_score = 2
        
        if is_growth_with_revenue and revenue_yoy >= 15:
            revenue_growth_score = min(20, revenue_growth_score + 2)
        
        score += revenue_growth_score
        breakdown['revenue_growth'] = revenue_growth_score
        
        # 2. Profitability (0-25 points)
        net_margin = None
        if margins.get('quarterly') and len(margins['quarterly']) > 0:
            net_margin = margins['quarterly'][0].get('net_margin')
        
        profitability_score = 0
        if net_margin is not None:
            if net_margin >= 0.20:
                profitability_score = 25
            elif net_margin >= 0.10:
                profitability_score = 20
            elif net_margin >= 0.05:
                profitability_score = 15
            elif net_margin >= 0:
                profitability_score = 10
            elif net_margin >= -0.05:
                profitability_score = 5
            elif is_growth_with_revenue and net_margin >= -0.15:
                profitability_score = 3  # More lenient for growth companies
        else:
            # Try to infer from info
            profit_margin = info.get('profitMargins')
            if profit_margin:
                net_margin = profit_margin
                if net_margin >= 0.20:
                    profitability_score = 25
                elif net_margin >= 0.10:
                    profitability_score = 20
                elif net_margin >= 0.05:
                    profitability_score = 15
                elif net_margin >= 0:
                    profitability_score = 10
        
        score += profitability_score
        breakdown['profitability'] = profitability_score
        
        # 3. Cash Flow (0-20 points)
        cash_flow_score = 0
        # TODO: Add cash flow analysis when get_cash_flow_analysis is moved
        score += cash_flow_score
        breakdown['cash_flow'] = cash_flow_score
        
        # 4. Debt Levels (0-15 points)
        debt_score = 0
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None:
            if debt_to_equity <= 0.3:
                debt_score = 15
            elif debt_to_equity <= 0.5:
                debt_score = 12
            elif debt_to_equity <= 1.0:
                debt_score = 8
            elif debt_to_equity <= 2.0:
                debt_score = 4
        
        score += debt_score
        breakdown['debt'] = debt_score
        
        # 5. Stability (0-10 points)
        stability_score = 0
        beta = info.get('beta')
        if beta is not None:
            if 0.8 <= beta <= 1.2:
                stability_score = 10
            elif 0.6 <= beta <= 1.4:
                stability_score = 7
            elif 0.4 <= beta <= 1.6:
                stability_score = 5
        
        score += stability_score
        breakdown['stability'] = stability_score
        
        # 6. Efficiency (0-10 points)
        efficiency_score = 0
        roe = info.get('returnOnEquity')
        if roe is not None:
            if roe >= 0.20:
                efficiency_score = 10
            elif roe >= 0.15:
                efficiency_score = 8
            elif roe >= 0.10:
                efficiency_score = 6
            elif roe >= 0.05:
                efficiency_score = 4
        
        score += efficiency_score
        breakdown['efficiency'] = efficiency_score
        
        # Determine grade
        if score >= 80:
            grade = 'Excellent'
        elif score >= 65:
            grade = 'Strong'
        elif score >= 50:
            grade = 'Good'
        elif score >= 35:
            grade = 'Fair'
        elif score >= 20:
            grade = 'Weak'
        else:
            grade = 'Very Weak'
        
        return {
            'score': round(score, 1),
            'max_score': max_score,
            'grade': grade,
            'breakdown': breakdown
        }
    
    except Exception as e:
        logger.exception(f"Error calculating financials score")
        import traceback
        traceback.print_exc()
        return {
            'score': 0,
            'max_score': 100,
            'grade': 'Unknown',
            'breakdown': {}
        }


def get_balance_sheet_health(ticker):
    """Get comprehensive balance sheet health analysis"""
    try:
        import yfinance as yf
        import time
        import pandas as pd
        
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        
        if balance_sheet is None or balance_sheet.empty:
            return None
        
        def find_row(df, keywords):
            for idx in df.index:
                idx_lower = str(idx).lower()
                if any(kw.lower() in idx_lower for kw in keywords):
                    return df.loc[idx]
            return None
        
        # Get key balance sheet items
        current_assets_row = find_row(balance_sheet, ['total current assets', 'current assets'])
        current_liabilities_row = find_row(balance_sheet, ['total current liabilities', 'current liabilities'])
        cash_row = find_row(balance_sheet, ['cash and cash equivalents', 'cash'])
        inventory_row = find_row(balance_sheet, ['inventory', 'inventories'])
        total_debt_row = find_row(balance_sheet, ['total debt', 'long term debt', 'total liabilities'])
        equity_row = find_row(balance_sheet, ['total stockholders equity', 'total equity', 'stockholders equity'])
        total_assets_row = find_row(balance_sheet, ['total assets'])
        
        revenue_row = find_row(income_stmt, ['total revenue', 'revenue']) if income_stmt is not None else None
        
        # Build trends
        trends = []
        for i, col in enumerate(balance_sheet.columns[:8]):
            try:
                quarter_date = pd.Timestamp(col)
                quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                
                ca = float(current_assets_row.iloc[i]) if current_assets_row is not None and i < len(current_assets_row) else None
                cl = float(current_liabilities_row.iloc[i]) if current_liabilities_row is not None and i < len(current_liabilities_row) else None
                cash = float(cash_row.iloc[i]) if cash_row is not None and i < len(cash_row) else None
                inventory = float(inventory_row.iloc[i]) if inventory_row is not None and i < len(inventory_row) else None
                debt = float(total_debt_row.iloc[i]) if total_debt_row is not None and i < len(total_debt_row) else None
                equity = float(equity_row.iloc[i]) if equity_row is not None and i < len(equity_row) else None
                total_assets = float(total_assets_row.iloc[i]) if total_assets_row is not None and i < len(total_assets_row) else None
                
                revenue = float(revenue_row.iloc[i]) if revenue_row is not None and i < len(revenue_row) else None
                
                # Calculate ratios
                current_ratio = (ca / cl) if ca and cl and cl > 0 else None
                quick_ratio = ((ca - inventory) / cl) if ca and inventory is not None and cl and cl > 0 else (ca / cl if ca and cl and cl > 0 else None)
                debt_to_equity = (debt / equity) if debt and equity and equity > 0 else None
                working_capital = (ca - cl) if ca and cl else None
                working_capital_efficiency = (working_capital / revenue * 100) if working_capital and revenue and revenue > 0 else None
                asset_turnover = (revenue / total_assets) if revenue and total_assets and total_assets > 0 else None
                
                trends.append({
                    'quarter': quarter_str,
                    'date': quarter_date.strftime('%Y-%m-%d'),
                    'current_ratio': round(current_ratio, 2) if current_ratio is not None else None,
                    'quick_ratio': round(quick_ratio, 2) if quick_ratio is not None else None,
                    'debt_to_equity': round(debt_to_equity, 2) if debt_to_equity is not None else None,
                    'working_capital': working_capital,
                    'working_capital_efficiency': round(working_capital_efficiency, 2) if working_capital_efficiency is not None else None,
                    'asset_turnover': round(asset_turnover, 2) if asset_turnover is not None else None
                })
            except (IndexError, ValueError, TypeError):
                continue
        
        # Calculate Health Score (0-100)
        health_score = 50  # Base score
        breakdown = {}
        
        if trends and len(trends) > 0:
            latest = trends[0]
            
            # Current Ratio Score (0-20 points)
            if latest.get('current_ratio') is not None:
                cr = latest['current_ratio']
                if cr >= 2.0:
                    breakdown['current_ratio'] = 20
                elif cr >= 1.5:
                    breakdown['current_ratio'] = 15
                elif cr >= 1.0:
                    breakdown['current_ratio'] = 10
                elif cr >= 0.5:
                    breakdown['current_ratio'] = 5
                else:
                    breakdown['current_ratio'] = 0
            else:
                breakdown['current_ratio'] = 0
            
            # Quick Ratio Score (0-15 points)
            if latest.get('quick_ratio') is not None:
                qr = latest['quick_ratio']
                if qr >= 1.5:
                    breakdown['quick_ratio'] = 15
                elif qr >= 1.0:
                    breakdown['quick_ratio'] = 12
                elif qr >= 0.5:
                    breakdown['quick_ratio'] = 8
                else:
                    breakdown['quick_ratio'] = 3
            else:
                breakdown['quick_ratio'] = 0
            
            # Debt-to-Equity Score (0-25 points, lower is better)
            if latest.get('debt_to_equity') is not None:
                dte = latest['debt_to_equity']
                if dte <= 0.3:
                    breakdown['debt_to_equity'] = 25
                elif dte <= 0.5:
                    breakdown['debt_to_equity'] = 20
                elif dte <= 1.0:
                    breakdown['debt_to_equity'] = 15
                elif dte <= 2.0:
                    breakdown['debt_to_equity'] = 10
                else:
                    breakdown['debt_to_equity'] = 5
            else:
                breakdown['debt_to_equity'] = 10
            
            # Working Capital Efficiency Score (0-20 points)
            if latest.get('working_capital_efficiency') is not None:
                wce = latest['working_capital_efficiency']
                if wce <= 10:  # Efficient (low WC relative to revenue)
                    breakdown['working_capital_efficiency'] = 20
                elif wce <= 20:
                    breakdown['working_capital_efficiency'] = 15
                elif wce <= 30:
                    breakdown['working_capital_efficiency'] = 10
                else:
                    breakdown['working_capital_efficiency'] = 5
            else:
                breakdown['working_capital_efficiency'] = 10
            
            # Asset Turnover Score (0-20 points, higher is better)
            if latest.get('asset_turnover') is not None:
                at = latest['asset_turnover']
                if at >= 1.0:
                    breakdown['asset_turnover'] = 20
                elif at >= 0.5:
                    breakdown['asset_turnover'] = 15
                elif at >= 0.3:
                    breakdown['asset_turnover'] = 10
                else:
                    breakdown['asset_turnover'] = 5
            else:
                breakdown['asset_turnover'] = 10
            
            health_score = sum(breakdown.values())
        
        return {
            'health_score': min(100, max(0, health_score)),
            'breakdown': breakdown,
            'trends': trends
        }
    except Exception as e:
        logger.exception(f"Error in balance sheet health analysis for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def get_management_guidance_tracking(ticker):
    """Track management guidance vs actual results"""
    try:
        import yfinance as yf
        import time
        import pandas as pd
        
        # Get earnings calendar and actual results
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        # Get earnings calendar for guidance estimates
        calendar = stock.calendar
        earnings_history = stock.earnings_history if hasattr(stock, 'earnings_history') else None
        
        guidance_tracking = []
        guidance_accuracy_scores = []
        
        # Try to extract guidance from earnings calendar
        if calendar is not None:
            try:
                if isinstance(calendar, dict):
                    for date_key, data in calendar.items():
                        if isinstance(data, dict):
                            # Look for guidance fields
                            guidance_revenue = data.get('Revenue Average') or data.get('revenueEstimate')
                            guidance_eps = data.get('Earnings Average') or data.get('epsEstimate')
                            
                            if guidance_revenue or guidance_eps:
                                # Try to match with actual results
                                # This is simplified - in reality would need to match dates
                                guidance_tracking.append({
                                    'date': str(date_key),
                                    'revenue_guidance': guidance_revenue,
                                    'eps_guidance': guidance_eps,
                                    'actual_revenue': None,  # Would need to match from actuals
                                    'actual_eps': None,
                                    'revenue_beat': None,
                                    'eps_beat': None
                                })
                elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
                    for idx, row in calendar.iterrows():
                        try:
                            guidance_revenue = row.get('Revenue Average') if 'Revenue Average' in row else None
                            guidance_eps = row.get('Earnings Average') if 'Earnings Average' in row else None
                            
                            if guidance_revenue is not None or guidance_eps is not None:
                                guidance_tracking.append({
                                    'date': str(idx),
                                    'revenue_guidance': float(guidance_revenue) if guidance_revenue is not None else None,
                                    'eps_guidance': float(guidance_eps) if guidance_eps is not None else None,
                                    'actual_revenue': None,
                                    'actual_eps': None,
                                    'revenue_beat': None,
                                    'eps_beat': None
                                })
                        except:
                            continue
            except Exception as e:
                logger.warning(f"Error parsing calendar for {ticker}: {e}")
        
        # Calculate accuracy score if we have guidance vs actuals
        if guidance_tracking:
            for entry in guidance_tracking:
                if entry.get('revenue_guidance') and entry.get('actual_revenue'):
                    guidance = entry['revenue_guidance']
                    actual = entry['actual_revenue']
                    if guidance > 0:
                        error_pct = abs(actual - guidance) / guidance * 100
                        accuracy = max(0, 100 - error_pct)  # 100% if perfect, decreases with error
                        guidance_accuracy_scores.append(accuracy)
                        entry['revenue_accuracy'] = round(accuracy, 1)
                
                if entry.get('eps_guidance') and entry.get('actual_eps'):
                    guidance = entry['eps_guidance']
                    actual = entry['actual_eps']
                    if guidance != 0:
                        error_pct = abs(actual - guidance) / abs(guidance) * 100
                        accuracy = max(0, 100 - error_pct)
                        guidance_accuracy_scores.append(accuracy)
                        entry['eps_accuracy'] = round(accuracy, 1)
        
        overall_accuracy = sum(guidance_accuracy_scores) / len(guidance_accuracy_scores) if guidance_accuracy_scores else None
        
        # Get forward guidance if available
        forward_guidance = None
        if calendar is not None:
            try:
                # Get next earnings date and estimates
                if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                    next_earnings = calendar.iloc[0] if len(calendar) > 0 else None
                    if next_earnings is not None:
                        forward_guidance = {
                            'revenue_guidance': float(next_earnings.get('Revenue Average', 0)) if 'Revenue Average' in next_earnings else None,
                            'eps_guidance': float(next_earnings.get('Earnings Average', 0)) if 'Earnings Average' in next_earnings else None,
                            'date': str(calendar.index[0]) if len(calendar) > 0 else None
                        }
            except:
                pass
        
        return {
            'guidance_history': guidance_tracking[:8],  # Last 8 quarters
            'accuracy_score': round(overall_accuracy, 1) if overall_accuracy is not None else None,
            'forward_guidance': forward_guidance,
            'revisions_count': 0  # Would need to track revisions over time
        }
    except Exception as e:
        logger.exception(f"Error in management guidance tracking for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def get_segment_breakdown(ticker):
    """Get segment and geography breakdown"""
    try:
        import yfinance as yf
        import time
        
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        info = stock.info
        
        # Try to get segment data from info
        segment_data = {}
        geography_data = {}
        
        # Check for segment revenue in info (yfinance sometimes has this)
        segment_keys = [k for k in info.keys() if 'segment' in k.lower() or 'business' in k.lower() or 'geography' in k.lower()]
        
        # Try to extract from majorHoldersBreakdown or other fields
        # This is a simplified version - segment data is often in 10-K filings
        
        # For now, return structure (will be populated if data available)
        # In production, would scrape from SEC filings or use specialized API
        return {
            'segments': segment_data,
            'geography': geography_data,
            'available': len(segment_data) > 0 or len(geography_data) > 0
        }
    except Exception as e:
        logger.exception(f"Error in segment breakdown for {ticker}")
        return None


def get_peer_comparison_data(ticker: str, industry_category: str, sector: str, limit: int = 4) -> List[Dict]:
    """
    Get peer comparison data for similar companies in the same industry/sector.
    
    Args:
        ticker: Target stock ticker
        industry_category: Industry category (e.g., 'data centers', 'technology')
        sector: Sector (e.g., 'Technology', 'Financial Services')
        limit: Maximum number of peers to return
        
    Returns:
        List of peer comparison dictionaries with metrics
    """
    try:
        import yfinance as yf
        import time
        from app.utils.constants import MAX_PEER_COMPARISON, RATE_LIMIT_DELAY
        
        # Get target stock info for comparison
        target_stock = yf.Ticker(ticker.upper())
        time.sleep(RATE_LIMIT_DELAY)
        target_info = target_stock.info
        
        if not target_info or 'symbol' not in target_info:
            logger.warning(f"Could not get info for target ticker {ticker}")
            return []
        
        target_sector = target_info.get('sector', sector) or sector
        target_industry = target_info.get('industry', industry_category) or industry_category
        target_market_cap = target_info.get('marketCap')
        
        logger.debug(f"Looking for peers for {ticker} in sector '{target_sector}', industry '{target_industry}'")
        
        # Known peer lists for common sectors/industries
        # This is a simplified approach - in production, would use a stock screener API
        peer_candidates = []
        
        # Try to find peers using common tickers in the same sector
        # For data centers (special case mentioned in code)
        data_center_tickers = ['CIFR', 'IREN', 'NBIS', 'EQIX', 'DLR', 'AMT', 'PLD']
        if ticker.upper() in data_center_tickers or 'data center' in target_industry.lower():
            peer_candidates = [t for t in data_center_tickers if t != ticker.upper()]
        
        # For technology sector
        elif 'Technology' in target_sector or 'Software' in target_industry:
            tech_peers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC']
            peer_candidates = [t for t in tech_peers if t != ticker.upper()]
        
        # For financial services
        elif 'Financial' in target_sector:
            financial_peers = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW']
            peer_candidates = [t for t in financial_peers if t != ticker.upper()]
        
        # For healthcare
        elif 'Healthcare' in target_sector:
            healthcare_peers = ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR']
            peer_candidates = [t for t in healthcare_peers if t != ticker.upper()]
        
        # For consumer
        elif 'Consumer' in target_sector:
            consumer_peers = ['WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'TGT', 'COST', 'LOW']
            peer_candidates = [t for t in consumer_peers if t != ticker.upper()]
        
        # If no specific peers found, try to use screener to find similar companies
        if not peer_candidates:
            logger.debug(f"No predefined peers found, trying to find similar companies by market cap and sector")
            # For now, return empty - would need screener integration
            return []
        
        # Limit candidates
        peer_candidates = peer_candidates[:limit * 2]  # Get more candidates, then filter
        
        # Fetch data for peer candidates
        peers_data = []
        for peer_ticker in peer_candidates:
            try:
                time.sleep(RATE_LIMIT_DELAY)
                peer_stock = yf.Ticker(peer_ticker)
                peer_info = peer_stock.info
                
                if not peer_info or 'symbol' not in peer_info:
                    continue
                
                # Filter by sector/industry match
                peer_sector = peer_info.get('sector', '')
                peer_industry = peer_info.get('industry', '')
                
                # Check if sector matches (or industry if sector is N/A)
                sector_match = (target_sector and peer_sector and 
                               target_sector.lower() in peer_sector.lower()) or \
                              (not target_sector or target_sector == 'N/A')
                
                industry_match = (target_industry and peer_industry and 
                                 any(keyword in peer_industry.lower() 
                                     for keyword in target_industry.lower().split())) or \
                                (not target_industry or target_industry == 'N/A')
                
                # Prefer sector match, but accept if industry matches
                if not (sector_match or industry_match):
                    continue
                
                # Get key metrics for comparison
                peer_market_cap = peer_info.get('marketCap')
                peer_pe = peer_info.get('trailingPE')
                peer_forward_pe = peer_info.get('forwardPE')
                peer_price_to_book = peer_info.get('priceToBook')
                peer_profit_margin = peer_info.get('profitMargins')
                peer_revenue_growth = peer_info.get('revenueGrowth')
                peer_earnings_growth = peer_info.get('earningsGrowth')
                peer_roe = peer_info.get('returnOnEquity')
                peer_debt_to_equity = peer_info.get('debtToEquity')
                peer_current_price = peer_info.get('currentPrice')
                
                # Calculate similarity score based on market cap proximity
                similarity_score = 0
                if target_market_cap and peer_market_cap:
                    # Prefer peers with similar market cap (within 10x)
                    cap_ratio = max(target_market_cap, peer_market_cap) / min(target_market_cap, peer_market_cap)
                    if cap_ratio <= 2:
                        similarity_score = 100
                    elif cap_ratio <= 5:
                        similarity_score = 80
                    elif cap_ratio <= 10:
                        similarity_score = 60
                    else:
                        similarity_score = 40
                else:
                    similarity_score = 50
                
                peers_data.append({
                    'ticker': peer_ticker,
                    'name': peer_info.get('longName', peer_ticker),
                    'sector': peer_sector,
                    'industry': peer_industry,
                    'market_cap': peer_market_cap,
                    'current_price': peer_current_price,
                    'pe_ratio': peer_pe,
                    'forward_pe': peer_forward_pe,
                    'price_to_book': peer_price_to_book,
                    'profit_margin': peer_profit_margin,
                    'revenue_growth': peer_revenue_growth,
                    'earnings_growth': peer_earnings_growth,
                    'roe': peer_roe,
                    'debt_to_equity': peer_debt_to_equity,
                    'similarity_score': similarity_score
                })
                
                if len(peers_data) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Error fetching data for {peer_ticker}: {str(e)}")
                continue
        
        # Sort by similarity score (market cap proximity)
        peers_data.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Limit results
        peers_data = peers_data[:limit]
        
        logger.info(f"Found {len(peers_data)} peers for {ticker}")
        return peers_data
        
    except Exception as e:
        logger.exception(f"Error getting peer comparison data for {ticker}")
        import traceback
        traceback.print_exc()
        return []


def get_industry_ranking(ticker, industry_category, sector, market_cap):
    """Get industry ranking for a stock based on market cap"""
    if not market_cap or market_cap <= 0:
        logger.debug(f"get_industry_ranking: Invalid market_cap for {ticker}: {market_cap}")
        return None
    
    # Define peer companies by industry category
    peer_tickers = {
        'data centers': ['EQIX', 'DLR', 'AMT', 'CCI', 'SBAC', 'IREN', 'CIFR', 'NBIS', 'GDS', 'VNET', 'PD', 'CONE', 'QTS', 'COR', 'LAND'],
        'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'ABBV', 'MRK', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA'],
        'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'MA', 'V', 'PYPL', 'COF', 'USB', 'TFC'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'HAL', 'NOV', 'FANG', 'MRO', 'OVV', 'CTRA', 'MTDR'],
        'consumer': ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'COST', 'AMZN', 'TSCO', 'BBY', 'FIVE', 'DKS'],
        'industrial': ['BA', 'CAT', 'GE', 'HON', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'SWK', 'TXT', 'AME', 'CMI', 'DE', 'FTV'],
        'real estate': ['AMT', 'PLD', 'PSA', 'EQIX', 'WELL', 'SPG', 'O', 'DLR', 'EXPI', 'CBRE', 'JLL', 'CWK', 'MMC', 'BAM', 'BXP']
    }
    
    # Map sector to category if industry_category is "Other"
    sector_to_category = {
        'Technology': 'technology',
        'Healthcare': 'healthcare',
        'Financial Services': 'finance',
        'Energy': 'energy',
        'Consumer Cyclical': 'consumer',
        'Consumer Defensive': 'consumer',
        'Industrials': 'industrial',
        'Real Estate': 'real estate'
    }
    
    # Get peers for this category
    peers = peer_tickers.get(industry_category, [])
    
    # If category is "Other", try to use sector-based peers
    if not peers and industry_category == 'Other' and sector and sector != 'N/A':
        category_from_sector = sector_to_category.get(sector)
        if category_from_sector:
            peers = peer_tickers.get(category_from_sector, [])
            industry_category = category_from_sector
            logger.debug(f"get_industry_ranking: Using sector '{sector}' to map to category '{category_from_sector}' for {ticker}")
    
    if not peers:
        logger.debug(f"get_industry_ranking: No peers found for category '{industry_category}' (ticker: {ticker}, sector: {sector})")
        return None
    
    # Fetch market caps for peers (limit to avoid too many API calls)
    peer_data = []
    for peer_ticker in peers[:20]:  # Limit to 20 peers
        if peer_ticker == ticker:
            continue
        try:
            peer_stock = yf.Ticker(peer_ticker)
            time.sleep(0.1)  # Rate limiting
            peer_info = peer_stock.info
            peer_market_cap = peer_info.get('marketCap', 0)
            if peer_market_cap and peer_market_cap > 0:
                peer_data.append({
                    'ticker': peer_ticker,
                    'market_cap': peer_market_cap
                })
        except Exception as e:
            logger.debug(f"Error fetching market cap for peer {peer_ticker}: {str(e)}")
            continue
    
    # Add current ticker
    peer_data.append({
        'ticker': ticker,
        'market_cap': market_cap
    })
    
    # Sort by market cap (descending)
    peer_data.sort(key=lambda x: x['market_cap'], reverse=True)
    
    # Find position
    position = next((i + 1 for i, p in enumerate(peer_data) if p['ticker'] == ticker), None)
    total = len(peer_data)
    
    if position and total > 1:
        return {
            'position': position,
            'total': total,
            'category': industry_category.title()
        }
    
    return None


def get_cash_flow_analysis(ticker):
    """Get comprehensive cash flow statement analysis"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        # Get cash flow statements
        cf_quarterly = stock.quarterly_cashflow
        cf_annual = stock.cashflow
        
        if cf_quarterly is None or cf_quarterly.empty:
            logger.debug(f"[CASH FLOW] No quarterly cashflow data for {ticker}")
            return None
        
        logger.debug(f"[CASH FLOW] Found quarterly cashflow data for {ticker}: {len(cf_quarterly.columns)} quarters")
        
        # Extract key cash flow components
        def find_cf_row(df, keywords):
            for idx in df.index:
                idx_lower = str(idx).lower()
                if any(kw.lower() in idx_lower for kw in keywords):
                    return df.loc[idx]
            return None
        
        # Operating Cash Flow
        ocf_row = find_cf_row(cf_quarterly, ['operating cash flow', 'operating activities', 'cash from operations', 'net cash provided by operating activities'])
        # Investing Cash Flow
        icf_row = find_cf_row(cf_quarterly, ['investing cash flow', 'investing activities', 'cash from investing', 'net cash used in investing activities'])
        # Financing Cash Flow
        fcf_row = find_cf_row(cf_quarterly, ['financing cash flow', 'financing activities', 'cash from financing', 'net cash used in financing activities'])
        # Capital Expenditures
        capex_row = find_cf_row(cf_quarterly, ['capital expenditure', 'capex', 'purchase of property', 'capital expenditures'])
        
        # Build quarterly cash flow breakdown
        quarterly_breakdown = []
        fcf_trend = []
        
        # Get last 8 quarters
        for i, col in enumerate(cf_quarterly.columns[:8]):
            try:
                quarter_date = pd.Timestamp(col)
                quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                
                ocf = float(ocf_row.iloc[i]) if ocf_row is not None and i < len(ocf_row) else None
                icf = float(icf_row.iloc[i]) if icf_row is not None and i < len(icf_row) else None
                fcf_val = float(fcf_row.iloc[i]) if fcf_row is not None and i < len(fcf_row) else None
                capex = float(capex_row.iloc[i]) if capex_row is not None and i < len(capex_row) else None
                
                # Calculate FCF if not directly available
                if ocf is not None and capex is not None:
                    calculated_fcf = ocf - abs(capex) if capex < 0 else ocf + capex
                else:
                    calculated_fcf = None
                
                fcf_final = calculated_fcf if fcf_val is None else fcf_val
                
                quarterly_breakdown.append({
                    'quarter': quarter_str,
                    'date': quarter_date.strftime('%Y-%m-%d'),
                    'operating_cf': ocf,
                    'investing_cf': icf,
                    'financing_cf': fcf_val,
                    'capex': abs(capex) if capex is not None and capex < 0 else (capex if capex is not None else None),
                    'fcf': fcf_final
                })
                
                if fcf_final is not None:
                    fcf_trend.append({
                        'quarter': quarter_str,
                        'date': quarter_date.strftime('%Y-%m-%d'),
                        'fcf': fcf_final
                    })
            except (IndexError, ValueError, TypeError) as e:
                continue
        
        # Calculate Cash Conversion Cycle (requires income statement and balance sheet)
        try:
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            
            # Get latest quarter data
            if income_stmt is not None and not income_stmt.empty and balance_sheet is not None and not balance_sheet.empty:
                # Accounts Receivable
                ar_row = find_cf_row(balance_sheet, ['accounts receivable', 'receivables', 'trade receivables'])
                # Inventory
                inv_row = find_cf_row(balance_sheet, ['inventory', 'inventories'])
                # Accounts Payable
                ap_row = find_cf_row(balance_sheet, ['accounts payable', 'payables', 'trade payables'])
                # Revenue
                rev_row = find_cf_row(income_stmt, ['total revenue', 'revenue', 'net sales', 'sales'])
                # COGS
                cogs_row = find_cf_row(income_stmt, ['cost of revenue', 'cost of goods sold', 'cogs', 'cost of sales'])
                
                if ar_row is not None and rev_row is not None and len(ar_row) > 0 and len(rev_row) > 0:
                    ar = float(ar_row.iloc[0])
                    revenue = float(rev_row.iloc[0])
                    
                    # Days Sales Outstanding
                    dso = (ar / revenue * 365) if revenue > 0 else None
                else:
                    dso = None
                
                if inv_row is not None and cogs_row is not None and len(inv_row) > 0 and len(cogs_row) > 0:
                    inventory = float(inv_row.iloc[0])
                    cogs = float(cogs_row.iloc[0])
                    
                    # Days Inventory Outstanding
                    dio = (inventory / cogs * 365) if cogs > 0 else None
                else:
                    dio = None
                
                if ap_row is not None and cogs_row is not None and len(ap_row) > 0 and len(cogs_row) > 0:
                    ap = float(ap_row.iloc[0])
                    cogs = float(cogs_row.iloc[0])
                    
                    # Days Payable Outstanding
                    dpo = (ap / cogs * 365) if cogs > 0 else None
                else:
                    dpo = None
                
                # Cash Conversion Cycle
                if dso is not None and dio is not None and dpo is not None:
                    ccc = dso + dio - dpo
                else:
                    ccc = None
            else:
                ccc = None
                dso = None
                dio = None
                dpo = None
        except Exception as e:
            logger.debug(f"Error calculating CCC for {ticker}: {e}")
            ccc = None
            dso = None
            dio = None
            dpo = None
        
        # Calculate Cash Runway for growth companies (if negative FCF)
        cash_runway = None
        if fcf_trend and len(fcf_trend) > 0:
            latest_fcf = fcf_trend[0]['fcf']
            if latest_fcf is not None and latest_fcf < 0:
                # Get cash from balance sheet
                try:
                    bs = stock.balance_sheet
                    if bs is not None and not bs.empty:
                        cash_row = find_cf_row(bs, ['cash and cash equivalents', 'cash', 'cash and short term investments'])
                        if cash_row is not None and len(cash_row) > 0:
                            cash = float(cash_row.iloc[0])
                            monthly_burn = abs(latest_fcf) / 3  # Quarterly to monthly
                            if monthly_burn > 0:
                                cash_runway = cash / monthly_burn  # Months
                except:
                    pass
        
        # FCF Trend Projection (simple linear trend)
        fcf_projection = None
        if len(fcf_trend) >= 4:
            try:
                # Use last 4 quarters for trend
                recent_fcf = [q['fcf'] for q in fcf_trend[:4] if q['fcf'] is not None]
                if len(recent_fcf) >= 3:
                    # Simple average growth rate
                    growth_rates = []
                    for i in range(len(recent_fcf) - 1):
                        if recent_fcf[i+1] != 0:
                            growth = (recent_fcf[i] - recent_fcf[i+1]) / abs(recent_fcf[i+1])
                            growth_rates.append(growth)
                    
                    if growth_rates:
                        avg_growth = sum(growth_rates) / len(growth_rates)
                        next_fcf = recent_fcf[0] * (1 + avg_growth)
                        fcf_projection = {
                            'next_quarter': next_fcf,
                            'growth_rate': avg_growth * 100,
                            'method': 'linear_trend'
                        }
            except:
                pass
        
        return {
            'quarterly_breakdown': quarterly_breakdown,
            'fcf_trend': fcf_trend,
            'fcf_projection': fcf_projection,
            'cash_conversion_cycle': {
                'ccc': round(ccc, 1) if ccc is not None else None,
                'dso': round(dso, 1) if dso is not None else None,
                'dio': round(dio, 1) if dio is not None else None,
                'dpo': round(dpo, 1) if dpo is not None else None
            },
            'cash_runway': {
                'months': round(cash_runway, 1) if cash_runway is not None else None,
                'status': 'critical' if cash_runway is not None and cash_runway < 6 else 'warning' if cash_runway is not None and cash_runway < 12 else 'ok' if cash_runway is not None else None
            }
        }
    except Exception as e:
        logger.exception(f"Error in cash flow analysis for {ticker}: {str(e)}")
        return None


def scrape_macrotrends_margins(ticker: str, margin_type: str) -> List[Dict]:
    """
    Scrape margin data from Macrotrends
    
    Args:
        ticker: Stock ticker symbol
        margin_type: 'gross-profit-margin', 'operating-profit-margin', or 'net-profit-margin'
        
    Returns:
        List of dicts with date, margin value, and other data
    """
    try:
        # Get company name from yfinance for URL construction
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        info = stock.info
        company_name = info.get('longName', ticker.lower()).lower()
        # Clean company name for URL (replace spaces with hyphens, remove special chars)
        company_name = re.sub(r'[^a-z0-9\s-]', '', company_name)
        company_name = re.sub(r'\s+', '-', company_name).strip()
        
        # Construct URL - try different formats
        # Format 1: /stocks/charts/TICKER/company-name/margin-type
        # Format 2: /stocks/charts/TICKER/margin-type (fallback)
        url = f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/{company_name}/{margin_type}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"Scraping Macrotrends margin data from: {url}")
        
        # Try requests first, then cloudscraper if it fails (Macrotrends may block regular requests)
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"Regular requests failed for Macrotrends, trying cloudscraper: {e}")
            try:
                import cloudscraper
                scraper = cloudscraper.create_scraper()
                response = scraper.get(url, headers=headers, timeout=15)
                response.raise_for_status()
            except Exception as e2:
                logger.error(f"Both requests and cloudscraper failed: {e2}")
                raise
        
        soup = BeautifulSoup(response.content, 'html.parser')
        logger.debug(f"Macrotrends response status: {response.status_code}, content length: {len(response.content)}")
        
        # Find the data table - it has class "historical_data_table" or is in a table with specific structure
        table = soup.find('table', {'class': lambda x: x and 'historical' in x.lower()})
        if not table:
            # Try to find any table with margin data
            tables = soup.find_all('table')
            logger.debug(f"Found {len(tables)} tables on page, searching for margin table...")
            for t in tables:
                # Look for table with "Date" header or "Net Margin" / "Gross Margin" / "Operating Margin" text
                table_text = t.get_text().lower()
                if 'date' in table_text and ('margin' in table_text or margin_type.split('-')[0] in table_text):
                    table = t
                    logger.debug(f"Found margin table with text: {table_text[:200]}")
                    break
        
        if not table:
            logger.warning(f"Could not find margin table on Macrotrends for {ticker} {margin_type}. URL: {url}")
            # Try alternative URL without company name
            try:
                alt_url = f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/{margin_type}"
                logger.info(f"Trying alternative URL: {alt_url}")
                alt_response = requests.get(alt_url, headers=headers, timeout=10)
                alt_response.raise_for_status()
                alt_soup = BeautifulSoup(alt_response.content, 'html.parser')
                alt_table = alt_soup.find('table', {'class': lambda x: x and 'historical' in x.lower()})
                if alt_table:
                    logger.info(f"Found table using alternative URL")
                    table = alt_table
                    soup = alt_soup
                else:
                    logger.warning(f"Alternative URL also failed")
                    return []
            except Exception as e:
                logger.warning(f"Alternative URL failed: {e}")
                return []
        
        # Parse table rows (skip header)
        all_rows = table.find_all('tr')
        logger.debug(f"Found {len(all_rows)} rows in table")
        
        # Find header row to determine column index for margin
        header_row = all_rows[0] if all_rows else None
        margin_col_idx = 3  # Default to 4th column (0-indexed: 3)
        date_col_idx = 0  # Date is typically first column
        
        if header_row:
            header_cells = header_row.find_all(['th', 'td'])
            logger.debug(f"Header cells: {[cell.get_text(strip=True) for cell in header_cells]}")
            
            for idx, cell in enumerate(header_cells):
                header_text = cell.get_text(strip=True).lower()
                # Look for margin column more specifically
                if 'margin' in header_text:
                    margin_col_idx = idx
                    logger.info(f"Found margin column at index {margin_col_idx}: '{cell.get_text(strip=True)}'")
                    break
                # Also check for date column
                if 'date' in header_text:
                    date_col_idx = idx
            
            logger.info(f"Using date column index: {date_col_idx}, margin column index: {margin_col_idx}")
        
        rows = all_rows[1:] if len(all_rows) > 1 else []  # Skip header row
        margin_data = []
        
        logger.debug(f"Processing {len(rows)} data rows, margin column index: {margin_col_idx}")
        
        for row_idx, row in enumerate(rows[:20]):  # Process first 20 rows for debugging
            cells = row.find_all(['td', 'th'])
            if len(cells) <= max(date_col_idx, margin_col_idx):
                logger.debug(f"Row {row_idx}: Not enough cells ({len(cells)}), skipping")
                continue
                
            try:
                # Get date from correct column
                date_str = cells[date_col_idx].get_text(strip=True)
                # Get margin from correct column
                margin_str = cells[margin_col_idx].get_text(strip=True)
                
                # Log raw values for debugging
                all_cell_values = [c.get_text(strip=True) for c in cells]
                logger.debug(f"Row {row_idx} all cells: {all_cell_values}")
                logger.debug(f"Row {row_idx}: date_str='{date_str}', margin_str='{margin_str}'")
                
                # Parse date (format: YYYY-MM-DD from Macrotrends)
                date_obj = None
                try:
                    # Macrotrends uses YYYY-MM-DD format
                    date_obj = pd.Timestamp(date_str)
                except:
                    # Try alternative date formats
                    try:
                        date_obj = pd.to_datetime(date_str)
                    except:
                        logger.debug(f"Could not parse date: '{date_str}'")
                        continue
                
                # Parse margin - Macrotrends shows values like "33.33%" or "-200.00%"
                margin_value = None
                if margin_str and margin_str.strip():
                    # Remove % sign and any commas/spaces
                    margin_clean = margin_str.replace('%', '').replace(',', '').replace('$', '').strip()
                    
                    # Skip invalid values
                    if margin_clean.lower() in ['n/a', 'na', '-', '', 'null', 'none']:
                        logger.debug(f"Skipping invalid margin value: '{margin_str}'")
                        continue
                    
                    try:
                        margin_value = float(margin_clean)
                        # Macrotrends values are already in percentage form (e.g., 33.33 means 33.33%)
                        # Don't multiply by 100, just use as-is
                        logger.debug(f"Parsed margin value: {margin_value} (from '{margin_str}')")
                    except ValueError as e:
                        logger.debug(f"Could not parse margin value '{margin_str}': {e}")
                        continue
                    
                    # Sanity check: margins should be reasonable (between -1000% and 1000%)
                    if abs(margin_value) > 1000:
                        logger.warning(f"Suspicious margin value {margin_value}% for {date_str}, skipping")
                        continue
                
                if margin_value is not None and date_obj is not None:
                    margin_data.append({
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'margin': margin_value,  # Already in percentage (e.g., 33.33 means 33.33%)
                        'quarter': f"{date_obj.year}-Q{(date_obj.month - 1) // 3 + 1}"
                    })
                    logger.info(f"✓ Added margin data: {date_obj.strftime('%Y-%m-%d')} = {margin_value}%")
                else:
                    logger.debug(f"Row {row_idx}: Missing date or margin value")
                    
            except Exception as e:
                logger.warning(f"Error parsing row {row_idx} in Macrotrends table: {e}", exc_info=True)
                continue
        
        # Sort by date descending (most recent first)
        margin_data.sort(key=lambda x: x['date'], reverse=True)
        
        logger.info(f"Successfully scraped {len(margin_data)} margin data points for {ticker} {margin_type}")
        if len(margin_data) > 0:
            logger.debug(f"Sample data: {margin_data[:3]}")
        
        return margin_data
        
    except Exception as e:
        logger.warning(f"Error scraping Macrotrends margins for {ticker} {margin_type}: {str(e)}")
        return []


def get_profitability_analysis(ticker, financials_data):
    """Get deep dive profitability analysis using yfinance data for margins"""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(0.2)
        
        # Use quarterly income statement from yfinance to calculate margins
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is None or income_stmt.empty:
            logger.warning(f"No quarterly income statement available for {ticker}")
            return None
        
        def find_row(df, keywords):
            for idx in df.index:
                idx_lower = str(idx).lower()
                if any(kw.lower() in idx_lower for kw in keywords):
                    return df.loc[idx]
            return None
        
        # Get all necessary rows
        revenue_row = find_row(income_stmt, ['total revenue', 'revenue', 'net sales', 'total net sales'])
        gross_profit_row = find_row(income_stmt, ['gross profit'])
        operating_income_row = find_row(income_stmt, ['operating income', 'income from operations', 'operating income or loss'])
        net_income_row = find_row(income_stmt, ['net income', 'net earnings', 'net income common stockholders'])
        
        if revenue_row is None:
            logger.warning(f"Could not find revenue row for {ticker}")
            return None
        
        logger.info(f"Found rows for {ticker}: revenue={revenue_row is not None}, gross_profit={gross_profit_row is not None}, operating_income={operating_income_row is not None}, net_income={net_income_row is not None}")
        
        # Calculate margins from yfinance quarterly income statement
        margin_trends = []
        
        # Process each quarter (yfinance returns newest first)
        for i, col in enumerate(income_stmt.columns[:12]):  # Get up to 12 quarters
            try:
                quarter_date = pd.Timestamp(col)
                date_str = quarter_date.strftime('%Y-%m-%d')
                quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                
                # Get values for this quarter
                revenue = float(revenue_row.iloc[i]) if i < len(revenue_row) and pd.notna(revenue_row.iloc[i]) else None
                gross_profit = float(gross_profit_row.iloc[i]) if gross_profit_row is not None and i < len(gross_profit_row) and pd.notna(gross_profit_row.iloc[i]) else None
                operating_income = float(operating_income_row.iloc[i]) if operating_income_row is not None and i < len(operating_income_row) and pd.notna(operating_income_row.iloc[i]) else None
                net_income = float(net_income_row.iloc[i]) if net_income_row is not None and i < len(net_income_row) and pd.notna(net_income_row.iloc[i]) else None
                
                # Calculate margins as percentages (only if we have revenue)
                gross_margin = None
                operating_margin = None
                net_margin = None
                
                if revenue is not None and revenue != 0:
                    if gross_profit is not None:
                        gross_margin = (gross_profit / revenue) * 100
                    if operating_income is not None:
                        operating_margin = (operating_income / revenue) * 100
                    if net_income is not None:
                        net_margin = (net_income / revenue) * 100
                
                # Sanity check: margins should be reasonable
                if gross_margin is not None and abs(gross_margin) > 1000:
                    logger.warning(f"Unrealistic gross margin {gross_margin}% for {date_str}, setting to None")
                    gross_margin = None
                if operating_margin is not None and abs(operating_margin) > 1000:
                    logger.warning(f"Unrealistic operating margin {operating_margin}% for {date_str}, setting to None")
                    operating_margin = None
                if net_margin is not None and abs(net_margin) > 1000:
                    logger.warning(f"Unrealistic net margin {net_margin}% for {date_str}, setting to None")
                    net_margin = None
                
                # Only add if we have at least revenue and one calculated margin
                if revenue is not None and (gross_margin is not None or operating_margin is not None or net_margin is not None):
                    margin_trends.append({
                        'quarter': quarter_str,
                        'date': date_str,
                        'gross_margin': round(gross_margin, 2) if gross_margin is not None else None,
                        'operating_margin': round(operating_margin, 2) if operating_margin is not None else None,
                        'net_margin': round(net_margin, 2) if net_margin is not None else None,
                        'revenue': revenue,
                        'operating_income': operating_income,
                        'net_income': net_income
                    })
                    logger.debug(f"Added margin for {date_str} ({quarter_str}): gross={gross_margin}%, operating={operating_margin}%, net={net_margin}%, revenue={revenue}")
                    
            except (IndexError, ValueError, TypeError) as e:
                logger.debug(f"Error processing quarter {i} for {ticker}: {e}")
                continue
        
        if len(margin_trends) == 0:
            logger.error(f"No margin trends calculated for {ticker}")
            return None
        
        logger.info(f"Calculated {len(margin_trends)} margin trends for {ticker}")
        
        # Calculate margin expansion/contraction
        margin_expansion = {}
        if len(margin_trends) >= 2:
            latest = margin_trends[0]
            previous = margin_trends[1]
            
            if latest.get('gross_margin') and previous.get('gross_margin'):
                margin_expansion['gross'] = latest['gross_margin'] - previous['gross_margin']
            if latest.get('operating_margin') and previous.get('operating_margin'):
                margin_expansion['operating'] = latest['operating_margin'] - previous['operating_margin']
            if latest.get('net_margin') and previous.get('net_margin'):
                margin_expansion['net'] = latest['net_margin'] - previous['net_margin']
        
        # Calculate Operating Leverage
        operating_leverage = None
        if len(margin_trends) >= 2:
            latest = margin_trends[0]
            previous = margin_trends[1]
            
            if (latest.get('revenue') and previous.get('revenue') and 
                latest.get('operating_income') and previous.get('operating_income') and
                previous['revenue'] > 0 and previous['operating_income'] != 0):
                
                revenue_growth = (latest['revenue'] - previous['revenue']) / previous['revenue']
                operating_income_growth = (latest['operating_income'] - previous['operating_income']) / abs(previous['operating_income'])
                
                if revenue_growth != 0:
                    operating_leverage = operating_income_growth / revenue_growth
        
        # Break-even analysis for growth companies
        break_even_analysis = None
        company_stage = financials_data.get('company_stage', 'unknown')
        if company_stage in ['growth', 'early_stage']:
            if len(margin_trends) >= 4:
                # Calculate average revenue growth
                revenue_growth_rates = []
                for i in range(len(margin_trends) - 1):
                    if margin_trends[i].get('revenue') and margin_trends[i+1].get('revenue') and margin_trends[i+1]['revenue'] > 0:
                        growth = (margin_trends[i]['revenue'] - margin_trends[i+1]['revenue']) / margin_trends[i+1]['revenue']
                        revenue_growth_rates.append(growth)
                
                if revenue_growth_rates:
                    avg_growth = sum(revenue_growth_rates) / len(revenue_growth_rates)
                    latest_revenue = margin_trends[0].get('revenue', 0)
                    latest_net_income = margin_trends[0].get('net_income', 0)
                    
                    if latest_net_income < 0 and avg_growth > 0:
                        # Project when they'll break even
                        current_loss = abs(latest_net_income)
                        # Estimate break-even revenue (simplified)
                        if latest_revenue > 0:
                            loss_margin = current_loss / latest_revenue
                            # Assume loss margin decreases with scale
                            break_even_revenue = latest_revenue * (1 + loss_margin / 0.1)  # Simplified
                            
                            # Estimate quarters to break-even
                            if avg_growth > 0:
                                quarters_to_breakeven = (break_even_revenue / latest_revenue - 1) / avg_growth
                                break_even_analysis = {
                                    'estimated_quarters': max(1, int(quarters_to_breakeven)),
                                    'estimated_revenue': break_even_revenue,
                                    'current_revenue': latest_revenue,
                                    'avg_growth_rate': avg_growth * 100
                                }
        
        return {
            'margin_trends': margin_trends,
            'margin_expansion': margin_expansion,
            'operating_leverage': round(operating_leverage, 2) if operating_leverage is not None else None,
            'break_even_analysis': break_even_analysis
        }
    except Exception as e:
        logger.exception(f"Error in profitability analysis for {ticker}: {str(e)}")
        return None

