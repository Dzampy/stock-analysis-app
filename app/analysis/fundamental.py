"""
Fundamental analysis - Financials scoring, cash flow, profitability, balance sheet, industry ranking
"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
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

