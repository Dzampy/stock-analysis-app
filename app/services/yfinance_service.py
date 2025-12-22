"""
yfinance service - Stock data, financials, earnings, institutional data
"""
import yfinance as yf
import pandas as pd
import time
from typing import Dict, Optional, List
from app.utils.constants import RATE_LIMIT_DELAY, DEFAULT_PERIOD
from app.utils.logger import logger


def get_stock_data(ticker: str, period: str = DEFAULT_PERIOD) -> Optional[Dict]:
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1m, 5m, 15m, 1h, 4h, 1d, 1w, 5d, 1mo, 3mo, 6mo, 1y, 5y)
        
    Returns:
        Dict with 'history' (DataFrame) and 'info' (dict) or None
    """
    try:
        # Map custom timeframes to yfinance period and interval
        timeframe_map = {
            '1m': {'period': '1d', 'interval': '1m'},
            '5m': {'period': '5d', 'interval': '5m'},
            '15m': {'period': '5d', 'interval': '15m'},
            '1h': {'period': '1mo', 'interval': '1h'},
            '4h': {'period': '3mo', 'interval': '1h'},
            '1d': {'period': '1y', 'interval': '1d'},
            '1w': {'period': '5y', 'interval': '1wk'},
        }
        
        if period in timeframe_map:
            tf_config = timeframe_map[period]
            use_period = tf_config['period']
            use_interval = tf_config['interval']
        else:
            use_period = period
            use_interval = None
        
        # Try using Ticker method first
        try:
            if use_interval:
                logger.debug(f"Fetching intraday data for {ticker}: period={use_period}, interval={use_interval}")
                stock = yf.Ticker(ticker)
                hist = stock.history(period=use_period, interval=use_interval, auto_adjust=True, prepost=False)
                logger.debug(f"Got {len(hist)} rows of intraday data")
            else:
                hist = yf.download(ticker, period=use_period, progress=False, show_errors=False, auto_adjust=True)
                if hist.empty:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=use_period, auto_adjust=True)
        except Exception as e:
            logger.warning(f"Error downloading data for {ticker}: {str(e)}")
            try:
                stock = yf.Ticker(ticker)
                if use_interval:
                    hist = stock.history(period=use_period, interval=use_interval, auto_adjust=True, prepost=False)
                else:
                    hist = stock.history(period=use_period, auto_adjust=True)
            except Exception as e2:
                logger.warning(f"Error with Ticker method: {str(e2)}")
                if use_interval:
                    logger.debug(f"Intraday data may not be available for {ticker} with interval {use_interval}")
                raise
        
        if hist.empty:
            return None
        
        # Handle multi-level columns from download()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        
        # Get additional info with error handling
        info = {}
        try:
            stock = yf.Ticker(ticker)
            time.sleep(0.5)
            info = stock.info
            if not info or 'symbol' not in info:
                info = {
                    'longName': ticker,
                    'symbol': ticker,
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'longBusinessSummary': 'Data temporarily unavailable from Yahoo Finance API.'
                }
        except Exception as e:
            logger.warning(f"Could not fetch info for {ticker}: {str(e)}")
            info = {
                'longName': ticker,
                'symbol': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'longBusinessSummary': 'Data temporarily unavailable from Yahoo Finance API.'
            }
        
        return {
            'history': hist,
            'info': info
        }
    except Exception as e:
        logger.exception(f"Error fetching data for {ticker}")
        return None


def get_institutional_ownership(ticker: str) -> Optional[Dict]:
    """
    Get institutional ownership % and top holders from Finviz/yfinance
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with ownership data or None
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        import json
        import re
        
        # Try yfinance first
        stock = yf.Ticker(ticker)
        time.sleep(RATE_LIMIT_DELAY)
        info = stock.info
        
        institutional_holders = stock.institutional_holders
        current_ownership_pct = info.get('heldPercentInstitutions', 0) * 100 if info.get('heldPercentInstitutions') else None
        
        # Get data from Finviz
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                snapshot_table = soup.find('table', class_='snapshot-table2')
                
                if snapshot_table:
                    rows = snapshot_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True)
                            if 'Inst Own' in label:
                                value_cell = cells[1]
                                b_tag = value_cell.find('b')
                                value_text = b_tag.get_text(strip=True) if b_tag else value_cell.get_text(strip=True)
                                value_text = value_text.replace('%', '').replace(',', '').strip()
                                try:
                                    ownership_pct = float(value_text)
                                    if ownership_pct > (current_ownership_pct or 0) or current_ownership_pct is None:
                                        current_ownership_pct = ownership_pct
                                except (ValueError, TypeError):
                                    pass
                
                # Try to get top holders from JSON data
                json_match = re.search(r'id="institutional-ownership-init-data-0"[^>]*>([^<]+)', response.text)
                if json_match:
                    try:
                        holders_data = json.loads(json_match.group(1))
                        top_holders = []
                        if 'managersOwnership' in holders_data:
                            for holder in holders_data['managersOwnership'][:10]:
                                top_holders.append({
                                    'holder': holder.get('name', 'N/A'),
                                    'pct_ownership': round(holder.get('percOwnership', 0), 2)
                                })
                        if top_holders:
                            return {
                                'ownership_pct': round(current_ownership_pct, 2) if current_ownership_pct else None,
                                'top_holders': top_holders
                            }
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Error scraping Finviz for institutional data for {ticker}: {e}")
        
        # Get top holders from yfinance
        top_holders = []
        if institutional_holders is not None and not institutional_holders.empty:
            for _, row in institutional_holders.head(10).iterrows():
                top_holders.append({
                    'holder': row.get('Holder', 'N/A'),
                    'shares': int(row.get('Shares', 0)) if pd.notna(row.get('Shares')) else 0,
                    'value': int(row.get('Value', 0)) if pd.notna(row.get('Value')) else 0,
                    'pct_ownership': float(row.get('% Out', 0)) if pd.notna(row.get('% Out')) else 0
                })
        
        return {
            'ownership_pct': round(current_ownership_pct, 2) if current_ownership_pct else None,
            'top_holders': top_holders[:10]
        }
    
    except Exception as e:
        logger.exception(f"Error fetching institutional ownership for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def get_financials_data(ticker: str) -> Optional[Dict]:
    """
    Get comprehensive financial data for Financials tab
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with financial data or None
    """
    try:
        from app.services.finviz_service import get_quarterly_estimates_from_finviz
        
        stock = yf.Ticker(ticker)
        time.sleep(0.3)  # Rate limiting
        
        # Get quarterly estimates and actuals from Finviz
        finviz_data = None
        try:
            finviz_data = get_quarterly_estimates_from_finviz(ticker)
        except Exception as finviz_error:
            logger.warning(f"Finviz scraping failed for {ticker}: {str(finviz_error)}")
            finviz_data = {'estimates': {}, 'actuals': {}}
        
        quarterly_estimates = finviz_data.get('estimates', {}) if isinstance(finviz_data, dict) else {}
        quarterly_actuals = finviz_data.get('actuals', {}) if isinstance(finviz_data, dict) else {}
        
        # Get company info
        try:
            info = stock.info
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
        except:
            info = {}
            sector = 'N/A'
            industry = 'N/A'
        
        # Map industry to category
        data_center_tickers = ['CIFR', 'IREN', 'NBIS', 'EQIX', 'DLR', 'AMT', 'CCI', 'SBAC', 'GDS', 'VNET', 'PD', 'CONE', 'QTS', 'COR', 'LAND']
        
        industry_category_map = {
            'data centers': ['data center', 'data center reit', 'reit—data center', 'data center reits', 'data centers', 'datacenter', 'datacenters'],
            'technology': ['technology', 'software', 'semiconductors', 'internet', 'telecommunications', 'tech'],
            'healthcare': ['healthcare', 'biotechnology', 'pharmaceuticals', 'medical devices', 'biotech'],
            'finance': ['financial services', 'banking', 'insurance', 'financial', 'banks'],
            'energy': ['energy', 'oil', 'gas', 'renewable energy', 'utilities'],
            'consumer': ['consumer', 'retail', 'consumer goods', 'consumer cyclical', 'consumer defensive'],
            'industrial': ['industrial', 'manufacturing', 'aerospace', 'defense'],
            'real estate': ['real estate', 'reit', 'real estate investment trust']
        }
        
        industry_category = 'Other'
        if ticker.upper() in data_center_tickers:
            industry_category = 'data centers'
        else:
            industry_lower = industry.lower()
            for category, keywords in industry_category_map.items():
                if any(keyword in industry_lower for keyword in keywords):
                    industry_category = category
                    break
        
        # Get comprehensive financial statements
        income_stmt_q = stock.quarterly_income_stmt
        income_stmt_a = stock.income_stmt
        cash_flow_q = stock.quarterly_cashflow
        cash_flow_a = stock.cashflow
        balance_sheet_q = stock.quarterly_balance_sheet
        balance_sheet_a = stock.balance_sheet
        
        # Process income statement (quarterly)
        income_statement_quarterly = []
        if income_stmt_q is not None and not income_stmt_q.empty:
            quarters = income_stmt_q.columns.tolist()
            for i, quarter_date in enumerate(quarters):
                # Format quarter as YYYY-QN (e.g., 2024-Q1)
                if hasattr(quarter_date, 'year') and hasattr(quarter_date, 'month'):
                    quarter_num = (quarter_date.month - 1) // 3 + 1
                    quarter_str = f"{quarter_date.year}-Q{quarter_num}"
                elif hasattr(quarter_date, 'strftime'):
                    quarter_str = quarter_date.strftime('%Y-Q%q') if '%q' in quarter_date.strftime('%Y-Q%q') else f"{quarter_date.year}-Q{(quarter_date.month-1)//3+1}"
                else:
                    quarter_str = str(quarter_date)
                
                # Extract key metrics
                revenue = None
                net_income = None
                eps = None
                
                for idx in income_stmt_q.index:
                    idx_str = str(idx).lower()
                    if 'total revenue' in idx_str or 'revenue' in idx_str:
                        val = income_stmt_q.loc[idx].iloc[i]
                        if pd.notna(val):
                            revenue = float(val)
                    if 'net income' in idx_str:
                        val = income_stmt_q.loc[idx].iloc[i]
                        if pd.notna(val):
                            net_income = float(val)
                    if 'basic earnings per share' in idx_str or 'diluted earnings per share' in idx_str:
                        val = income_stmt_q.loc[idx].iloc[i]
                        if pd.notna(val):
                            eps = float(val)
                
                income_statement_quarterly.append({
                    'quarter': quarter_str,
                    'date': quarter_date.strftime('%Y-%m-%d') if hasattr(quarter_date, 'strftime') else str(quarter_date),
                    'revenue': revenue,
                    'net_income': net_income,
                    'eps': eps
                })
        
        # Process income statement (annual)
        income_statement_annual = []
        if income_stmt_a is not None and not income_stmt_a.empty:
            years = income_stmt_a.columns.tolist()
            for i, year_date in enumerate(years):
                year_str = year_date.strftime('%Y') if hasattr(year_date, 'strftime') else str(year_date)
                
                revenue = None
                net_income = None
                eps = None
                
                for idx in income_stmt_a.index:
                    idx_str = str(idx).lower()
                    if 'total revenue' in idx_str or 'revenue' in idx_str:
                        val = income_stmt_a.loc[idx].iloc[i]
                        if pd.notna(val):
                            revenue = float(val)
                    if 'net income' in idx_str:
                        val = income_stmt_a.loc[idx].iloc[i]
                        if pd.notna(val):
                            net_income = float(val)
                    if 'basic earnings per share' in idx_str or 'diluted earnings per share' in idx_str:
                        val = income_stmt_a.loc[idx].iloc[i]
                        if pd.notna(val):
                            eps = float(val)
                
                income_statement_annual.append({
                    'year': year_str,
                    'date': year_date.strftime('%Y-%m-%d') if hasattr(year_date, 'strftime') else str(year_date),
                    'revenue': revenue,
                    'net_income': net_income,
                    'eps': eps
                })
        
        # Process cash flow (quarterly)
        cash_flow_quarterly = []
        if cash_flow_q is not None and not cash_flow_q.empty:
            quarters = cash_flow_q.columns.tolist()
            for i, quarter_date in enumerate(quarters):
                # Format quarter as YYYY-QN (e.g., 2024-Q1)
                if hasattr(quarter_date, 'year') and hasattr(quarter_date, 'month'):
                    quarter_num = (quarter_date.month - 1) // 3 + 1
                    quarter_str = f"{quarter_date.year}-Q{quarter_num}"
                elif hasattr(quarter_date, 'strftime'):
                    quarter_str = quarter_date.strftime('%Y-Q%q') if '%q' in quarter_date.strftime('%Y-Q%q') else f"{quarter_date.year}-Q{(quarter_date.month-1)//3+1}"
                else:
                    quarter_str = str(quarter_date)
                
                operating_cf = None
                capex = None
                fcf = None
                
                for idx in cash_flow_q.index:
                    idx_str = str(idx).lower()
                    if 'operating cash flow' in idx_str or 'cash from operating activities' in idx_str:
                        val = cash_flow_q.loc[idx].iloc[i]
                        if pd.notna(val):
                            operating_cf = float(val)
                    if 'capital expenditure' in idx_str or 'capex' in idx_str:
                        val = cash_flow_q.loc[idx].iloc[i]
                        if pd.notna(val):
                            capex = float(val)
                
                if operating_cf is not None and capex is not None:
                    fcf = operating_cf - capex
                elif operating_cf is not None:
                    fcf = operating_cf
                
                cash_flow_quarterly.append({
                    'quarter': quarter_str,
                    'date': quarter_date.strftime('%Y-%m-%d') if hasattr(quarter_date, 'strftime') else str(quarter_date),
                    'operating_cf': operating_cf,
                    'capex': capex,
                    'fcf': fcf
                })
        
        # Process balance sheet
        balance_sheet = {}
        if balance_sheet_q is not None and not balance_sheet_q.empty:
            latest_quarter = balance_sheet_q.columns[0] if len(balance_sheet_q.columns) > 0 else None
            if latest_quarter is not None:
                for idx in balance_sheet_q.index:
                    idx_str = str(idx).lower()
                    val = balance_sheet_q.loc[idx].iloc[0]
                    if pd.notna(val):
                        if 'cash' in idx_str and 'equivalents' in idx_str:
                            balance_sheet['cash'] = float(val)
                        if 'total debt' in idx_str:
                            balance_sheet['total_debt'] = float(val)
                        if 'total equity' in idx_str or 'stockholders equity' in idx_str:
                            balance_sheet['equity'] = float(val)
                
                # Calculate net debt
                if 'cash' in balance_sheet and 'total_debt' in balance_sheet:
                    balance_sheet['net_debt'] = balance_sheet['total_debt'] - balance_sheet['cash']
        
        # Calculate margins
        margins_quarterly = []
        if income_statement_quarterly:
            for item in income_statement_quarterly:
                margin_data = {}
                if item.get('revenue') and item.get('revenue') > 0:
                    # Gross margin (simplified - would need COGS)
                    # Operating margin (simplified - would need operating income)
                    if item.get('net_income'):
                        margin_data['net_margin'] = (item['net_income'] / item['revenue']) * 100
                if margin_data:
                    margin_data['quarter'] = item['quarter']
                    margins_quarterly.append(margin_data)
        
        # Create executive snapshot
        executive_snapshot = {}
        if income_statement_quarterly:
            latest = income_statement_quarterly[0] if income_statement_quarterly else None
            if latest:
                # Calculate TTM (Trailing Twelve Months) - sum of last 4 quarters
                ttm_revenue = sum([q.get('revenue', 0) or 0 for q in income_statement_quarterly[:4]])
                ttm_net_income = sum([q.get('net_income', 0) or 0 for q in income_statement_quarterly[:4]])
                executive_snapshot['revenue_ttm'] = ttm_revenue if ttm_revenue > 0 else None
                executive_snapshot['net_income_ttm'] = ttm_net_income if ttm_net_income != 0 else None
        
        if cash_flow_quarterly:
            ttm_fcf = sum([q.get('fcf', 0) or 0 for q in cash_flow_quarterly[:4]])
            executive_snapshot['fcf_ttm'] = ttm_fcf if ttm_fcf != 0 else None
        
        # Calculate financials score (will be enhanced by calculate_financials_score)
        from app.analysis.fundamental import calculate_financials_score
        financials_score = calculate_financials_score({
            'income_statement': {'quarterly': income_statement_quarterly},
            'cash_flow': {'quarterly': cash_flow_quarterly},
            'balance_sheet': balance_sheet,
            'executive_snapshot': executive_snapshot
        }, info, 'unknown')
        
        return {
            'ticker': ticker.upper(),
            'company_name': info.get('longName', ticker.upper()),
            'sector': sector,
            'industry': industry,
            'industry_category': industry_category,
            'income_statement': {
                'quarterly': income_statement_quarterly,
                'annual': income_statement_annual
            },
            'cash_flow': {
                'quarterly': cash_flow_quarterly
            },
            'balance_sheet': balance_sheet,
            'margins': {
                'quarterly': margins_quarterly
            },
            'executive_snapshot': executive_snapshot,
            'financials_score': financials_score,
            'quarterly_estimates': quarterly_estimates,
            'quarterly_actuals': quarterly_actuals,
            'info': info
        }
    
    except Exception as e:
        logger.exception(f"Error fetching financials data for {ticker}")
        return None


def get_earnings_qoq(ticker: str) -> Optional[Dict]:
    """
    Get quarterly earnings, EPS, revenue and compare with expectations
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with quarterly earnings data or None
    """
    try:
        from app.services.finviz_service import get_quarterly_estimates_from_finviz
        
        # Get Finviz actuals first
        finviz_data = get_quarterly_estimates_from_finviz(ticker)
        quarterly_actuals = finviz_data.get('actuals', {}) if isinstance(finviz_data, dict) else {}
        
        stock = yf.Ticker(ticker.upper())
        time.sleep(RATE_LIMIT_DELAY)
        income_stmt = stock.quarterly_income_stmt
        
        if income_stmt is None or income_stmt.empty:
            return None
        
        # Find Net Income row
        net_income_row = None
        search_terms = [
            'net income from continuing operation',
            'net income from continuing operations',
            'net income',
            'total net income',
            'income from continuing operations',
            'normalized income'
        ]
        
        for term in search_terms:
            for idx in income_stmt.index:
                idx_str = str(idx).lower()
                if term in idx_str:
                    net_income_row = income_stmt.loc[idx]
                    break
            if net_income_row is not None:
                break
        
        # Find Revenue row
        revenue_row = None
        for idx in income_stmt.index:
            idx_str = str(idx).lower()
            if 'total revenue' in idx_str:
                revenue_row = income_stmt.loc[idx]
                break
        
        # Find EPS row - try multiple variations
        eps_row = None
        eps_search_terms = [
            'basic earnings per share',
            'diluted earnings per share',
            'earnings per share',
            'eps',
            'net income per share'
        ]
        
        for term in eps_search_terms:
            for idx in income_stmt.index:
                idx_str = str(idx).lower()
                if term in idx_str:
                    eps_row = income_stmt.loc[idx]
                    break
            if eps_row is not None:
                break
        
        if net_income_row is None and revenue_row is None and eps_row is None:
            return None
        
        # Get quarters (columns are dates)
        quarters = []
        if not income_stmt.empty:
            quarters = income_stmt.columns.tolist()
        
        earnings_data = []
        
        for i, quarter_date in enumerate(quarters):
            quarter_str = quarter_date.strftime('%Y-Q%q') if hasattr(quarter_date, 'strftime') else str(quarter_date)
            
            # Extract values
            net_income = float(net_income_row.iloc[i]) if net_income_row is not None and i < len(net_income_row) else None
            revenue = float(revenue_row.iloc[i]) if revenue_row is not None and i < len(revenue_row) else None
            eps = float(eps_row.iloc[i]) if eps_row is not None and i < len(eps_row) else None
            
            # Get actuals from Finviz if available
            finviz_eps_actual = None
            finviz_revenue_actual = None
            
            # Try to match quarter
            for q_key, q_value in quarterly_actuals.get('eps', {}).items():
                if quarter_str in q_key or q_key in quarter_str:
                    finviz_eps_actual = q_value
                    break
            
            for q_key, q_value in quarterly_actuals.get('revenue', {}).items():
                if quarter_str in q_key or q_key in quarter_str:
                    finviz_revenue_actual = q_value
                    break
            
            # Use Finviz actuals if available, otherwise use yfinance
            eps_actual = finviz_eps_actual if finviz_eps_actual is not None else eps
            revenue_actual = finviz_revenue_actual if finviz_revenue_actual is not None else revenue
            
            # Calculate QoQ changes
            if i > 0:
                prev_net_income = float(net_income_row.iloc[i-1]) if net_income_row is not None and i-1 < len(net_income_row) else None
                prev_revenue = float(revenue_row.iloc[i-1]) if revenue_row is not None and i-1 < len(revenue_row) else None
                prev_eps = float(eps_row.iloc[i-1]) if eps_row is not None and i-1 < len(eps_row) else None
                
                net_income_qoq = ((net_income - prev_net_income) / prev_net_income * 100) if net_income and prev_net_income and prev_net_income != 0 else None
                revenue_qoq = ((revenue - prev_revenue) / prev_revenue * 100) if revenue and prev_revenue and prev_revenue != 0 else None
                eps_qoq = ((eps_actual - prev_eps) / prev_eps * 100) if eps_actual and prev_eps and prev_eps != 0 else None
            else:
                net_income_qoq = None
                revenue_qoq = None
                eps_qoq = None
            
            earnings_data.append({
                'quarter': quarter_str,
                'date': quarter_date.strftime('%Y-%m-%d') if hasattr(quarter_date, 'strftime') else str(quarter_date),
                'net_income': net_income,
                'revenue': revenue_actual,
                'eps': eps_actual,
                'net_income_qoq': round(net_income_qoq, 2) if net_income_qoq is not None else None,
                'revenue_qoq': round(revenue_qoq, 2) if revenue_qoq is not None else None,
                'eps_qoq': round(eps_qoq, 2) if eps_qoq is not None else None
            })
        
        return {
            'ticker': ticker.upper(),
            'earnings': earnings_data
        }
    
    except Exception as e:
        logger.exception(f"Error fetching earnings QoQ for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def get_retail_activity_indicators(ticker: str) -> Optional[Dict]:
    """
    Get retail activity indicators (estimated retail activity from volume patterns)
    
    Analyzes volume patterns, price movements, and trading characteristics to estimate retail vs institutional activity.
    """
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import time
        from app.utils.constants import RATE_LIMIT_DELAY
        
        stock = yf.Ticker(ticker.upper())
        time.sleep(RATE_LIMIT_DELAY)
        hist = stock.history(period='3mo')
        
        if hist.empty or len(hist) < 20:
            return None
        
        # Calculate volume metrics
        volume = hist['Volume']
        price = hist['Close']
        price_changes = price.pct_change()
        price_changes_abs = price_changes.abs()
        
        # Volume statistics
        avg_volume = volume.mean()
        median_volume = volume.median()
        current_volume = volume.iloc[-1]
        recent_avg_volume = volume.tail(5).mean()
        
        volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        recent_volume_ratio = (recent_avg_volume / avg_volume) if avg_volume > 0 else 1.0
        
        # Price volatility
        volatility = price_changes_abs.mean()
        recent_volatility = price_changes_abs.tail(5).mean()
        
        # Volume-price relationship
        # Retail activity often shows: high volume with smaller price movements
        # Institutional activity often shows: large price movements with moderate volume
        volume_price_correlation = volume.corr(price_changes_abs)
        
        # Intraday patterns (if available)
        # Retail traders often trade more during market open/close
        # This is simplified - would need intraday data for full analysis
        
        # Estimate retail activity percentage
        # Factors that suggest retail activity:
        # 1. High volume relative to average (retail often creates volume spikes)
        # 2. Lower correlation between volume and price changes (retail trades more randomly)
        # 3. Higher volatility relative to volume (retail creates noise)
        
        retail_score = 0
        
        # Volume factor (0-40 points)
        if volume_ratio > 2.0:
            retail_score += 40  # Very high volume = likely retail
        elif volume_ratio > 1.5:
            retail_score += 30
        elif volume_ratio > 1.2:
            retail_score += 20
        elif volume_ratio > 1.0:
            retail_score += 10
        
        # Volume-price correlation factor (0-30 points)
        # Lower correlation suggests more retail activity
        if volume_price_correlation < 0.3:
            retail_score += 30
        elif volume_price_correlation < 0.5:
            retail_score += 20
        elif volume_price_correlation < 0.7:
            retail_score += 10
        
        # Volatility factor (0-30 points)
        # Higher volatility relative to volume suggests retail
        volatility_ratio = recent_volatility / volatility if volatility > 0 else 1.0
        if volatility_ratio > 1.5:
            retail_score += 30
        elif volatility_ratio > 1.2:
            retail_score += 20
        elif volatility_ratio > 1.0:
            retail_score += 10
        
        # Normalize to 0-100
        retail_activity_pct = min(100, max(0, retail_score))
        
        # Determine retail sentiment based on recent price action
        recent_returns = price_changes.tail(5).mean()
        if recent_returns > 0.02:  # >2% average return
            retail_sentiment = 'bullish'
        elif recent_returns < -0.02:  # <-2% average return
            retail_sentiment = 'bearish'
        else:
            retail_sentiment = 'neutral'
        
        # Calculate retail vs institutional ratio
        institutional_activity_pct = 100 - retail_activity_pct
        retail_vs_institutional_ratio = (retail_activity_pct / institutional_activity_pct) if institutional_activity_pct > 0 else 1.0
        
        # Activity level classification
        if retail_activity_pct >= 70:
            activity_level = 'high'
        elif retail_activity_pct >= 50:
            activity_level = 'moderate'
        elif retail_activity_pct >= 30:
            activity_level = 'low'
        else:
            activity_level = 'very_low'
        
        return {
            'estimated_retail_activity': activity_level,
            'retail_activity_pct': round(retail_activity_pct, 2),
            'institutional_activity_pct': round(institutional_activity_pct, 2),
            'retail_vs_institutional_volume_ratio': round(retail_vs_institutional_ratio, 2),
            'volume_ratio': round(volume_ratio, 2),
            'recent_volume_ratio': round(recent_volume_ratio, 2),
            'volume_price_correlation': round(volume_price_correlation, 3),
            'retail_sentiment': retail_sentiment,
            'volatility_ratio': round(volatility_ratio, 3),
            'current_volume': int(current_volume),
            'avg_volume': int(avg_volume)
        }
    except Exception as e:
        logger.exception(f"Error getting retail activity indicators for {ticker}")
        import traceback
        traceback.print_exc()
        return None

