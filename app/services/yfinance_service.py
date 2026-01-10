"""
yfinance service - Stock data, financials, earnings, institutional data
"""
import yfinance as yf
import pandas as pd
import time
from typing import Dict, Optional, List
from functools import wraps
from app.utils.constants import RATE_LIMIT_DELAY, DEFAULT_PERIOD
from app.utils.logger import logger
from app.config import CACHE_TIMEOUTS

# Import cache - will be initialized when app starts
try:
    from app import cache
    CACHE_AVAILABLE = True
except (ImportError, RuntimeError):
    CACHE_AVAILABLE = False
    cache = None


def get_stock_data(ticker: str, period: str = DEFAULT_PERIOD) -> Optional[Dict]:
    """
    Fetch stock data from Yahoo Finance (cached)
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1m, 5m, 15m, 1h, 4h, 1d, 1w, 5d, 1mo, 3mo, 6mo, 1y, 5y)
        
    Returns:
        Dict with 'history' (DataFrame) and 'info' (dict) or None
    """
    # Check cache first
    if CACHE_AVAILABLE and cache:
        cache_key = f"yfinance_stock_data_{ticker}_{period}"
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {ticker} ({period})")
            return cached_data
    
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
        
        result = {
            'history': hist,
            'info': info
        }
        
        # Cache the result
        if CACHE_AVAILABLE and cache:
            cache_key = f"yfinance_stock_data_{ticker}_{period}"
            cache.set(cache_key, result, timeout=CACHE_TIMEOUTS['yfinance'])
            logger.debug(f"Cached data for {ticker} ({period})")
        
        return result
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
    Get comprehensive financial data for Financials tab - RESTORED FROM ORIGINAL APP.PY
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with financial data or None
    """
    try:
        from app.services.finviz_service import get_quarterly_estimates_from_finviz
        from app.analysis.fundamental import (
            calculate_financials_score, 
            get_industry_ranking,
            get_cash_flow_analysis,
            get_profitability_analysis,
            get_balance_sheet_health,
            get_management_guidance_tracking,
            get_segment_breakdown
        )
        from app.utils.json_utils import clean_for_json
        from datetime import datetime
        
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
        
        if quarterly_estimates:
            rev_count = len(quarterly_estimates.get('revenue', {}))
            eps_count = len(quarterly_estimates.get('eps', {}))
            if rev_count > 0 or eps_count > 0:
                logger.info(f"Finviz: Found {rev_count} revenue and {eps_count} EPS estimates for {ticker}")
        
        if quarterly_actuals:
            rev_actual_count = len(quarterly_actuals.get('revenue', {}))
            eps_actual_count = len(quarterly_actuals.get('eps', {}))
            if rev_actual_count > 0 or eps_actual_count > 0:
                logger.info(f"Finviz: Found {rev_actual_count} revenue and {eps_actual_count} EPS actual values for {ticker}")
        
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
            'finance': ['financial services', 'banks', 'insurance', 'capital markets', 'banking'],
            'energy': ['oil & gas', 'renewable energy', 'utilities', 'energy', 'oil'],
            'consumer': ['consumer goods', 'retail', 'consumer services', 'consumer'],
            'industrial': ['industrial', 'manufacturing', 'machinery', 'industrials'],
            'real estate': ['real estate', 'reit', 'real estate investment trust', 'reits']
        }
        
        industry_category = 'Other'
        if ticker.upper() in data_center_tickers:
            industry_category = 'data centers'
        else:
            industry_lower = industry.lower() if industry else ''
            for category, keywords in industry_category_map.items():
                if any(keyword in industry_lower for keyword in keywords):
                    industry_category = category
                    break
        
        # Get industry ranking
        industry_ranking = get_industry_ranking(ticker, industry_category, sector, info.get('marketCap', 0))
        
        # Start with Finviz estimates
        forward_revenue_estimates = quarterly_estimates.get('revenue', {}).copy()
        forward_eps_estimates = quarterly_estimates.get('eps', {}).copy()
        
        # Check if we need to supplement with yfinance calendar for future quarters
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        current_quarter = (current_month - 1) // 3 + 1
        
        # Check if Finviz has future quarters (e.g., 2026)
        finviz_has_future = False
        for q in list(forward_revenue_estimates.keys()) + list(forward_eps_estimates.keys()):
            if '2026' in q or '2027' in q:
                finviz_has_future = True
                break
        
        # If Finviz doesn't have future quarters, try yfinance calendar as fallback
        if not finviz_has_future:
            try:
                calendar = stock.calendar
                if calendar is not None:
                    # yfinance calendar can be either DataFrame or dict
                    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                        # Parse DataFrame format
                        for idx, row in calendar.iterrows():
                            if pd.isna(idx):
                                continue
                            
                            # Get quarter info from the index (usually a date)
                            if isinstance(idx, pd.Timestamp):
                                cal_year = idx.year
                                cal_month = idx.month
                                cal_quarter = (cal_month - 1) // 3 + 1
                                quarter_str = f"{cal_year}-Q{cal_quarter}"
                                
                                # Only add if it's a future quarter
                                is_future = (cal_year > current_year) or (cal_year == current_year and cal_quarter > current_quarter)
                                
                                if is_future:
                                    # Try to get revenue estimate
                                    revenue_est = None
                                    if 'Revenue Average' in row.index:
                                        revenue_est = row['Revenue Average']
                                    elif 'Revenue' in row.index:
                                        revenue_est = row['Revenue']
                                    
                                    if revenue_est is not None and pd.notna(revenue_est):
                                        try:
                                            revenue_val = float(revenue_est)
                                            if revenue_val > 0:
                                                # yfinance returns revenue in actual dollars (not millions)
                                                if quarter_str not in forward_revenue_estimates:
                                                    forward_revenue_estimates[quarter_str] = revenue_val
                                        except (ValueError, TypeError):
                                            pass
                                    
                                    # Try to get EPS estimate
                                    eps_est = None
                                    if 'Earnings Average' in row.index:
                                        eps_est = row['Earnings Average']
                                    elif 'Earnings' in row.index:
                                        eps_est = row['Earnings']
                                    
                                    if eps_est is not None and pd.notna(eps_est):
                                        try:
                                            eps_val = float(eps_est)
                                            if quarter_str not in forward_eps_estimates:
                                                forward_eps_estimates[quarter_str] = eps_val
                                        except (ValueError, TypeError):
                                            pass
                    elif isinstance(calendar, dict):
                        # Parse dict format (yfinance returns dict with keys like 'Revenue Average', 'Earnings Average', 'Earnings Date')
                        # yfinance calendar typically contains only one quarter, but we'll generate Q2, Q3, Q4 based on Q1
                        
                        # Get earnings date to determine Q1
                        earnings_date = None
                        if 'Earnings Date' in calendar:
                            earnings_date = calendar['Earnings Date']
                            if isinstance(earnings_date, (list, tuple)) and len(earnings_date) > 0:
                                earnings_date = earnings_date[0]
                        
                        # Determine Q1 from earnings date
                        q1_quarter_str = None
                        if earnings_date:
                            if isinstance(earnings_date, pd.Timestamp):
                                cal_year = earnings_date.year
                                cal_month = earnings_date.month
                                cal_quarter = (cal_month - 1) // 3 + 1
                                q1_quarter_str = f"{cal_year}-Q{cal_quarter}"
                            elif isinstance(earnings_date, str):
                                # Try to parse date string
                                try:
                                    from dateutil import parser
                                    parsed_date = parser.parse(earnings_date)
                                    cal_year = parsed_date.year
                                    cal_month = parsed_date.month
                                    cal_quarter = (cal_month - 1) // 3 + 1
                                    q1_quarter_str = f"{cal_year}-Q{cal_quarter}"
                                except:
                                    pass
                        
                        # If no date, use next quarter as Q1
                        if not q1_quarter_str:
                            if current_quarter < 4:
                                q1_quarter_str = f"{current_year}-Q{current_quarter + 1}"
                            else:
                                q1_quarter_str = f"{current_year + 1}-Q1"
                        
                        # Get Q1 estimates
                        q1_revenue = None
                        q1_eps = None
                        
                        if 'Revenue Average' in calendar:
                            revenue_est = calendar['Revenue Average']
                            if revenue_est is not None:
                                try:
                                    q1_revenue = float(revenue_est)
                                except (ValueError, TypeError):
                                    pass
                        
                        if 'Earnings Average' in calendar:
                            eps_est = calendar['Earnings Average']
                            if eps_est is not None:
                                try:
                                    q1_eps = float(eps_est)
                                except (ValueError, TypeError):
                                    pass
                        
                        # Process Q1, Q2, Q3, Q4 (generate subsequent quarters)
                        for i in range(4):  # Q1, Q2, Q3, Q4
                            # Calculate quarter string
                            q1_year = int(q1_quarter_str.split('-Q')[0])
                            q1_num = int(q1_quarter_str.split('-Q')[1])
                            
                            target_q = q1_num + i
                            target_year = q1_year
                            while target_q > 4:
                                target_q -= 4
                                target_year += 1
                            
                            quarter_str = f"{target_year}-Q{target_q}"
                            
                            # Check if it's a future quarter
                            q_year = int(quarter_str.split('-Q')[0])
                            q_num = int(quarter_str.split('-Q')[1])
                            is_future = (q_year > current_year) or (q_year == current_year and q_num > current_quarter)
                            
                            if is_future and quarter_str not in forward_revenue_estimates and quarter_str not in forward_eps_estimates:
                                # For Q1, use actual estimates from yfinance
                                # For Q2, use Q1 values as placeholder (yfinance calendar typically has only Q1)
                                if i == 0:
                                    # Q1: use actual estimates
                                    if q1_revenue and q1_revenue > 0:
                                        forward_revenue_estimates[quarter_str] = q1_revenue
                                    if q1_eps is not None:
                                        forward_eps_estimates[quarter_str] = q1_eps
                                elif i == 1:
                                    # Q2: use Q1 values as placeholder (since yfinance calendar usually has only Q1)
                                    if q1_revenue and q1_revenue > 0:
                                        forward_revenue_estimates[quarter_str] = q1_revenue
                                    if q1_eps is not None:
                                        forward_eps_estimates[quarter_str] = q1_eps
            except Exception as e:
                logger.exception(f"Error processing yfinance calendar for {ticker}")
        
        financials = {
            'executive_snapshot': {},
            'income_statement': {'quarterly': [], 'annual': []},
            'margins': {'quarterly': [], 'annual': []},
            'cash_flow': {'quarterly': [], 'annual': []},
            'balance_sheet': {},
            'segments': [],
            'red_flags': [],
            'fundamentals_verdict': 'neutral',
            'main_verdict_sentence': '',
            'company_stage': 'unknown',
            'sector': sector,
            'industry': industry,
            'industry_category': industry_category,
            'industry_ranking': industry_ranking,
            'forward_estimates': {
                'revenue': forward_revenue_estimates,
                'eps': forward_eps_estimates
            }
        }
        
        # Get income statements
        quarterly_income = stock.quarterly_income_stmt
        annual_income = stock.financials
        
        # Get cash flow statements
        quarterly_cf = stock.quarterly_cashflow
        annual_cf = stock.cashflow
        
        # Get balance sheet
        quarterly_bs = stock.quarterly_balance_sheet
        annual_bs = stock.balance_sheet
        
        # Helper function to find row in statement
        def find_row(df, search_terms):
            if df is None or df.empty:
                return None
            for term in search_terms:
                for idx in df.index:
                    if term in str(idx).lower():
                        return df.loc[idx]
            return None
        
        # Get Revenue
        revenue_row_q = find_row(quarterly_income, ['total revenue', 'revenue', 'net sales', 'sales'])
        revenue_row_a = find_row(annual_income, ['total revenue', 'revenue', 'net sales', 'sales'])
        
        # Get Net Income
        net_income_row_q = find_row(quarterly_income, ['net income', 'total net income', 'income from continuing operations'])
        net_income_row_a = find_row(annual_income, ['net income', 'total net income', 'income from continuing operations'])
        
        # Get Operating Cash Flow
        ocf_row_q = find_row(quarterly_cf, ['operating cash flow', 'total cash from operating activities', 'operating activities'])
        ocf_row_a = find_row(annual_cf, ['operating cash flow', 'total cash from operating activities', 'operating activities'])
        
        # Get CapEx
        capex_row_q = find_row(quarterly_cf, ['capital expenditures', 'capex', 'capital expenditure'])
        capex_row_a = find_row(annual_cf, ['capital expenditures', 'capex', 'capital expenditure'])
        
        # Get Free Cash Flow (Operating CF - CapEx)
        def calculate_fcf(ocf_row, capex_row):
            if ocf_row is None or capex_row is None:
                return None
            try:
                # Both are pandas Series with same index (quarterly dates)
                fcf = ocf_row.copy()
                for i in range(min(len(fcf), len(capex_row))):
                    if pd.notna(fcf.iloc[i]) and pd.notna(capex_row.iloc[i]):
                        fcf.iloc[i] = float(fcf.iloc[i]) - abs(float(capex_row.iloc[i]))
                    else:
                        fcf.iloc[i] = None
                return fcf
            except:
                return None
        
        fcf_row_q = calculate_fcf(ocf_row_q, capex_row_q) if ocf_row_q is not None and capex_row_q is not None else None
        fcf_row_a = calculate_fcf(ocf_row_a, capex_row_a) if ocf_row_a is not None and capex_row_a is not None else None
        
        # Calculate TTM values (sum of last 4 quarters)
        revenue_ttm = None
        net_income_ttm = None
        fcf_ttm = None
        
        if revenue_row_q is not None and len(revenue_row_q) >= 4:
            try:
                revenue_ttm = float(revenue_row_q.iloc[:4].sum())
            except:
                try:
                    revenue_ttm = sum([float(v) for v in list(revenue_row_q.values)[:4] if pd.notna(v)])
                except:
                    pass
        
        if net_income_row_q is not None and len(net_income_row_q) >= 4:
            try:
                net_income_ttm = float(net_income_row_q.iloc[:4].sum())
            except:
                try:
                    net_income_ttm = sum([float(v) for v in list(net_income_row_q.values)[:4] if pd.notna(v)])
                except:
                    pass
        
        if fcf_row_q is not None and len(fcf_row_q) >= 4:
            try:
                fcf_ttm = float(fcf_row_q.iloc[:4].sum())
            except:
                try:
                    fcf_ttm = sum([float(v) for v in list(fcf_row_q.values)[:4] if pd.notna(v)])
                except:
                    pass
        
        # Calculate YoY growth (compare current quarter to same quarter last year)
        def calculate_yoy_growth(current_row, periods_ago=4):
            if current_row is None or len(current_row) < periods_ago + 1:
                return None
            try:
                # Get most recent (first) and 4 quarters ago
                current = float(current_row.iloc[0])
                previous = float(current_row.iloc[periods_ago])
                if previous != 0 and not pd.isna(current) and not pd.isna(previous):
                    return ((current - previous) / abs(previous)) * 100
            except (IndexError, ValueError, TypeError):
                pass
            return None
        
        revenue_yoy = calculate_yoy_growth(revenue_row_q)
        net_income_yoy = calculate_yoy_growth(net_income_row_q)
        
        # Get Gross Margin
        gross_profit_row_q = find_row(quarterly_income, ['gross profit', 'total gross profit'])
        gross_margin_q = None
        if gross_profit_row_q is not None and revenue_row_q is not None:
            try:
                if len(gross_profit_row_q) > 0 and len(revenue_row_q) > 0:
                    gross_profit = float(gross_profit_row_q.iloc[0] if hasattr(gross_profit_row_q, 'iloc') else list(gross_profit_row_q.values())[0])
                    revenue = float(revenue_row_q.iloc[0] if hasattr(revenue_row_q, 'iloc') else list(revenue_row_q.values())[0])
                    if revenue != 0:
                        gross_margin_q = (gross_profit / revenue) * 100
            except:
                pass
        
        # Get Operating Margin
        operating_income_row_q = find_row(quarterly_income, ['operating income', 'income from operations', 'operating profit'])
        operating_margin_q = None
        if operating_income_row_q is not None and revenue_row_q is not None:
            try:
                if len(operating_income_row_q) > 0 and len(revenue_row_q) > 0:
                    op_income = float(operating_income_row_q.iloc[0] if hasattr(operating_income_row_q, 'iloc') else list(operating_income_row_q.values())[0])
                    revenue = float(revenue_row_q.iloc[0] if hasattr(revenue_row_q, 'iloc') else list(revenue_row_q.values())[0])
                    if revenue != 0:
                        operating_margin_q = (op_income / revenue) * 100
            except:
                pass
        
        # Get Net Margin
        net_margin_q = None
        if net_income_row_q is not None and revenue_row_q is not None:
            try:
                if len(net_income_row_q) > 0 and len(revenue_row_q) > 0:
                    net_income = float(net_income_row_q.iloc[0] if hasattr(net_income_row_q, 'iloc') else list(net_income_row_q.values())[0])
                    revenue = float(revenue_row_q.iloc[0] if hasattr(revenue_row_q, 'iloc') else list(revenue_row_q.values())[0])
                    if revenue != 0:
                        net_margin_q = (net_income / revenue) * 100
            except:
                pass
        
        # Get Debt
        total_debt_row = find_row(quarterly_bs, ['total debt', 'total liabilities', 'long term debt'])
        total_debt = None
        if total_debt_row is not None and len(total_debt_row) > 0:
            try:
                total_debt = float(total_debt_row.iloc[0] if hasattr(total_debt_row, 'iloc') else list(total_debt_row.values())[0])
            except:
                pass
        
        # Calculate Debt/FCF ratio
        debt_fcf_ratio = None
        if total_debt is not None and fcf_ttm is not None and fcf_ttm != 0:
            debt_fcf_ratio = abs(total_debt / fcf_ttm)
        
        # FCF Margin
        fcf_margin = None
        if fcf_ttm is not None and revenue_ttm is not None and revenue_ttm != 0:
            fcf_margin = (fcf_ttm / revenue_ttm) * 100
        
        # Build Executive Snapshot
        financials['executive_snapshot'] = {
            'revenue_ttm': revenue_ttm,
            'revenue_yoy': revenue_yoy,
            'net_income_ttm': net_income_ttm,
            'net_income_yoy': net_income_yoy,
            'fcf_ttm': fcf_ttm,
            'fcf_margin': fcf_margin,
            'gross_margin': gross_margin_q,
            'debt_fcf_ratio': debt_fcf_ratio
        }
        
        # Build Income Statement data for charts
        if revenue_row_q is not None and quarterly_income is not None:
            for i, col in enumerate(quarterly_income.columns[:8]):  # Last 8 quarters
                try:
                    quarter_date = pd.Timestamp(col)
                    quarter_calc = (quarter_date.month - 1) // 3 + 1
                    quarter_str = f"{quarter_date.year}-Q{quarter_calc}"
                    
                    # Prefer Finviz actuals over yfinance for revenue (same as EPS)
                    revenue_val = None
                    
                    # Try Finviz actuals first - try exact match by quarter string first, then date matching
                    if quarterly_actuals and 'revenue' in quarterly_actuals:
                        best_match_rev = None
                        best_match_q = None
                        min_date_diff = float('inf')
                        
                        # FIRST: Try exact match by quarter string (e.g., "2024-Q3" == "2024-Q3")
                        if quarter_str in quarterly_actuals['revenue']:
                            best_match_rev = quarterly_actuals['revenue'][quarter_str]
                            best_match_q = quarter_str
                            min_date_diff = 0
                        else:
                            # SECOND: Find closest Finviz revenue by date (fallback for fiscal vs calendar quarter differences)
                            for finviz_q, finviz_rev in quarterly_actuals['revenue'].items():
                                try:
                                    if '-Q' in finviz_q:
                                        fv_year, fv_num = finviz_q.split('-Q')
                                        fv_num = int(fv_num)
                                        fv_year = int(fv_year)
                                        fv_month = (fv_num - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                                        fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                                        date_diff = abs((quarter_date - fv_date).days)
                                        
                                        # Accept match within 120 days (allows for fiscal vs calendar quarter differences)
                                        if date_diff < min_date_diff and date_diff <= 120:
                                            min_date_diff = date_diff
                                            best_match_rev = finviz_rev
                                            best_match_q = finviz_q
                                except Exception as e:
                                    pass
                        
                        if best_match_rev is not None:
                            revenue_val = best_match_rev
                            logger.debug(f"Using Finviz revenue actual for {ticker} {quarter_str}: {revenue_val} (matched with {best_match_q}, date_diff={min_date_diff} days)")
                    
                    # Fallback to yfinance if Finviz not available
                    if revenue_val is None:
                        revenue_val = float(revenue_row_q.iloc[i]) if i < len(revenue_row_q) else None
                        if revenue_val is not None and not pd.isna(revenue_val):
                            logger.debug(f"Using yfinance revenue for {ticker} {quarter_str}: {revenue_val}")
                    
                    net_income_val = float(net_income_row_q.iloc[i]) if net_income_row_q is not None and i < len(net_income_row_q) else None
                    
                    if revenue_val is not None and not pd.isna(revenue_val):
                        # Try to get EPS from income statement
                        eps_val = None
                        
                        # Prefer Finviz actuals over yfinance for EPS if available
                        if quarterly_actuals and isinstance(quarterly_actuals, dict):
                            if 'eps' in quarterly_actuals:
                                # Try exact match first
                                if quarter_str in quarterly_actuals['eps']:
                                    eps_val = quarterly_actuals['eps'][quarter_str]
                                    logger.debug(f"Using Finviz EPS for {ticker} {quarter_str}: {eps_val}")
                                else:
                                    # Try to find matching quarter with different format
                                    for finviz_q, finviz_eps in quarterly_actuals['eps'].items():
                                        if finviz_q == quarter_str:
                                            eps_val = finviz_eps
                                            logger.debug(f"Matched Finviz EPS for {ticker} {quarter_str}: {eps_val}")
                                            break
                        
                        # Fallback to yfinance if Finviz not available
                        if eps_val is None:
                            try:
                                eps_row = find_row(quarterly_income, ['diluted eps', 'basic eps', 'earnings per share', 'eps'])
                                if eps_row is not None and i < len(eps_row):
                                    eps_val = float(eps_row.iloc[i]) if hasattr(eps_row, 'iloc') else float(list(eps_row.values())[i])
                                    if pd.isna(eps_val):
                                        eps_val = None
                                    else:
                                        logger.debug(f"Using yfinance EPS for {ticker} {quarter_str}: {eps_val}")
                            except Exception:
                                pass
                        
                        # Get estimates from Finviz
                        revenue_estimate = None
                        eps_estimate = None
                        
                        if quarterly_estimates and isinstance(quarterly_estimates, dict):
                            # Revenue estimate - try exact match first, then try date-based matching
                            if 'revenue' in quarterly_estimates:
                                # Try exact match first
                                if quarter_str in quarterly_estimates['revenue']:
                                    revenue_estimate = quarterly_estimates['revenue'][quarter_str]
                                    logger.debug(f"Using Finviz revenue estimate (exact match) for {ticker} {quarter_str}: {revenue_estimate}")
                                else:
                                    # Try to find closest match by date
                                    best_match_est = None
                                    best_match_q = None
                                    min_date_diff = float('inf')
                                    
                                    for finviz_q, finviz_est in quarterly_estimates['revenue'].items():
                                        try:
                                            if '-Q' in finviz_q:
                                                fv_year, fv_num = finviz_q.split('-Q')
                                                fv_num = int(fv_num)
                                                fv_year = int(fv_year)
                                                fv_month = (fv_num - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                                                fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                                                date_diff = abs((quarter_date - fv_date).days)
                                                
                                                # Accept match within 120 days
                                                if date_diff < min_date_diff and date_diff <= 120:
                                                    min_date_diff = date_diff
                                                    best_match_est = finviz_est
                                                    best_match_q = finviz_q
                                        except Exception as e:
                                            pass
                                    
                                    if best_match_est is not None:
                                        revenue_estimate = best_match_est
                                        logger.debug(f"Using Finviz revenue estimate (date match) for {ticker} {quarter_str}: {revenue_estimate} (matched with {best_match_q}, date_diff={min_date_diff} days)")
                            
                            # EPS estimate - try exact match first, then try date-based matching
                            if 'eps' in quarterly_estimates:
                                # Try exact match first
                                if quarter_str in quarterly_estimates['eps']:
                                    eps_estimate = quarterly_estimates['eps'][quarter_str]
                                else:
                                    # Try to find closest match by date
                                    best_match_eps_est = None
                                    min_date_diff = float('inf')
                                    
                                    for finviz_q, finviz_eps_est in quarterly_estimates['eps'].items():
                                        try:
                                            if '-Q' in finviz_q:
                                                fv_year, fv_num = finviz_q.split('-Q')
                                                fv_num = int(fv_num)
                                                fv_year = int(fv_year)
                                                fv_month = (fv_num - 1) * 3 + 1
                                                fv_date = pd.Timestamp(year=fv_year, month=fv_month, day=1)
                                                date_diff = abs((quarter_date - fv_date).days)
                                                
                                                if date_diff < min_date_diff and date_diff <= 120:
                                                    min_date_diff = date_diff
                                                    best_match_eps_est = finviz_eps_est
                                        except Exception as e:
                                            pass
                                    
                                    if best_match_eps_est is not None:
                                        eps_estimate = best_match_eps_est
                        
                        financials['income_statement']['quarterly'].append({
                            'quarter': quarter_str,
                            'date': quarter_date.strftime('%Y-%m-%d'),
                            'revenue': revenue_val,
                            'revenue_estimate': revenue_estimate,  # Analyst estimate or None
                            'net_income': net_income_val if net_income_val is not None and not pd.isna(net_income_val) else None,
                            'eps': eps_val,
                            'eps_estimate': eps_estimate  # Analyst estimate or None
                        })
                except (IndexError, ValueError, TypeError):
                    continue
        
        # Build Margins data
        if gross_margin_q is not None or operating_margin_q is not None or net_margin_q is not None:
            financials['margins']['quarterly'].append({
                'gross_margin': gross_margin_q,
                'operating_margin': operating_margin_q,
                'net_margin': net_margin_q
            })
        
        # Build Cash Flow data
        if ocf_row_q is not None and quarterly_cf is not None:
            for i, col in enumerate(quarterly_cf.columns[:8]):
                try:
                    quarter_date = pd.Timestamp(col)
                    quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                    ocf_val = float(ocf_row_q.iloc[i]) if i < len(ocf_row_q) else None
                    capex_val = float(capex_row_q.iloc[i]) if capex_row_q is not None and i < len(capex_row_q) else None
                    fcf_val = float(fcf_row_q.iloc[i]) if fcf_row_q is not None and i < len(fcf_row_q) else None
                    
                    if ocf_val is not None and not pd.isna(ocf_val):
                        financials['cash_flow']['quarterly'].append({
                            'quarter': quarter_str,
                            'date': quarter_date.strftime('%Y-%m-%d'),
                            'operating_cf': ocf_val,
                            'capex': abs(capex_val) if capex_val is not None and not pd.isna(capex_val) else None,
                            'fcf': fcf_val if fcf_val is not None and not pd.isna(fcf_val) else None
                        })
                except (IndexError, ValueError, TypeError):
                    continue
        
        # Build Balance Sheet (simplified)
        cash_row = find_row(quarterly_bs, ['cash and cash equivalents', 'cash', 'cash and short term investments'])
        equity_row = find_row(quarterly_bs, ['total stockholders equity', 'total equity', 'stockholders equity'])
        current_assets_row = find_row(quarterly_bs, ['total current assets'])
        current_liabilities_row = find_row(quarterly_bs, ['total current liabilities'])
        
        # Get total_debt if not already defined (it should be from earlier, but ensure it's available)
        if 'total_debt' not in locals() or total_debt is None:
            total_debt_row = find_row(quarterly_bs, ['total debt', 'total liabilities', 'long term debt', 'total long term debt'])
            total_debt = None
            if total_debt_row is not None and len(total_debt_row) > 0:
                try:
                    total_debt = float(total_debt_row.iloc[0] if hasattr(total_debt_row, 'iloc') else list(total_debt_row.values())[0])
                except:
                    pass
        
        cash = None
        equity = None
        current_ratio = None
        
        if cash_row is not None and len(cash_row) > 0:
            try:
                cash = float(cash_row.iloc[0] if hasattr(cash_row, 'iloc') else list(cash_row.values())[0])
            except:
                pass
        
        if equity_row is not None and len(equity_row) > 0:
            try:
                equity = float(equity_row.iloc[0] if hasattr(equity_row, 'iloc') else list(equity_row.values())[0])
            except:
                pass
        
        current_assets = None
        current_liabilities = None
        if current_assets_row is not None and current_liabilities_row is not None:
            try:
                if len(current_assets_row) > 0 and len(current_liabilities_row) > 0:
                    current_assets = float(current_assets_row.iloc[0] if hasattr(current_assets_row, 'iloc') else list(current_assets_row.values())[0])
                    current_liabilities = float(current_liabilities_row.iloc[0] if hasattr(current_liabilities_row, 'iloc') else list(current_liabilities_row.values())[0])
                    if current_liabilities != 0:
                        current_ratio = current_assets / current_liabilities
            except:
                pass
        
        # Get Total Assets and Total Liabilities
        total_assets_row = find_row(quarterly_bs, ['total assets'])
        total_liabilities_row = find_row(quarterly_bs, ['total liabilities', 'total liabilities and stockholders equity'])
        
        total_assets = None
        total_liabilities = None
        if total_assets_row is not None and len(total_assets_row) > 0:
            try:
                total_assets = float(total_assets_row.iloc[0] if hasattr(total_assets_row, 'iloc') else list(total_assets_row.values())[0])
            except:
                pass
        
        if total_liabilities_row is not None and len(total_liabilities_row) > 0:
            try:
                total_liabilities = float(total_liabilities_row.iloc[0] if hasattr(total_liabilities_row, 'iloc') else list(total_liabilities_row.values())[0])
            except:
                pass
        
        # Get Long-term and Short-term Debt
        long_term_debt_row = find_row(quarterly_bs, ['long term debt', 'long-term debt', 'long term debt and capital lease obligation'])
        short_term_debt_row = find_row(quarterly_bs, ['short term debt', 'short-term debt', 'current debt', 'current portion of long term debt'])
        
        long_term_debt = None
        short_term_debt = None
        if long_term_debt_row is not None and len(long_term_debt_row) > 0:
            try:
                long_term_debt = float(long_term_debt_row.iloc[0] if hasattr(long_term_debt_row, 'iloc') else list(long_term_debt_row.values())[0])
            except:
                pass
        
        if short_term_debt_row is not None and len(short_term_debt_row) > 0:
            try:
                short_term_debt = float(short_term_debt_row.iloc[0] if hasattr(short_term_debt_row, 'iloc') else list(short_term_debt_row.values())[0])
            except:
                pass
        
        # Get Inventory for Quick Ratio
        inventory_row = find_row(quarterly_bs, ['inventory', 'total inventory'])
        inventory = None
        if inventory_row is not None and len(inventory_row) > 0:
            try:
                inventory = float(inventory_row.iloc[0] if hasattr(inventory_row, 'iloc') else list(inventory_row.values())[0])
            except:
                pass
        
        # Calculate derived metrics
        net_debt = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
        
        working_capital = None
        if current_assets is not None and current_liabilities is not None:
            working_capital = current_assets - current_liabilities
        
        quick_ratio = None
        if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
            inventory_value = inventory if inventory is not None else 0
            quick_ratio = (current_assets - inventory_value) / current_liabilities
        
        debt_to_equity = None
        if total_debt is not None and equity is not None and equity != 0:
            debt_to_equity = total_debt / equity
        
        debt_to_assets = None
        if total_debt is not None and total_assets is not None and total_assets != 0:
            debt_to_assets = total_debt / total_assets
        
        financials['balance_sheet'] = {
            'cash': cash,
            'total_debt': total_debt,
            'net_debt': net_debt,
            'equity': equity,
            'current_ratio': current_ratio,
            'current_assets': current_assets,
            'current_liabilities': current_liabilities,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'long_term_debt': long_term_debt,
            'short_term_debt': short_term_debt,
            'inventory': inventory,
            'working_capital': working_capital,
            'quick_ratio': quick_ratio,
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets
        }
        
        # Generate Red Flags
        red_flags = []
        
        # Check for declining revenue
        if len(financials['income_statement']['quarterly']) >= 3:
            revenues = [q['revenue'] for q in financials['income_statement']['quarterly'][:3] if q.get('revenue') is not None]
            if len(revenues) >= 3:
                # Check if revenue is declining: newest > previous > older
                is_declining = all(revenues[i] > revenues[i+1] for i in range(len(revenues)-1))
                if is_declining:
                    red_flags.append({
                        'type': 'revenue_decline',
                        'severity': 'high',
                        'message': 'Tržby klesají 3 kvartály po sobě'
                    })
        
        # Check FCF < Net Income
        if fcf_ttm is not None and net_income_ttm is not None and fcf_ttm < net_income_ttm:
            red_flags.append({
                'type': 'fcf_quality',
                'severity': 'medium',
                'message': 'FCF < Net Income (možné accounting issues)'
            })
        
        # Check rising debt + falling margins
        if debt_fcf_ratio is not None and debt_fcf_ratio > 3 and gross_margin_q is not None:
            red_flags.append({
                'type': 'debt_margin',
                'severity': 'high',
                'message': 'Rostoucí dluh + potenciálně klesající marže'
            })
        
        financials['red_flags'] = red_flags
        
        # Detect Company Stage
        company_stage = 'unknown'
        
        # Early-stage: Very low/no revenue, negative earnings, high burn rate
        if revenue_ttm is not None and revenue_ttm < 100_000_000:  # Less than $100M
            if net_income_ttm is not None and net_income_ttm < 0:
                if fcf_ttm is not None and fcf_ttm < 0:
                    company_stage = 'early_stage'
        
        # Growth: Growing revenue, may be unprofitable but improving
        if revenue_yoy and revenue_yoy > 20:
            if net_income_ttm is None or net_income_ttm < 0:
                if net_income_yoy and net_income_yoy > 0:  # Improving losses
                    company_stage = 'growth'
        
        # Mature: Stable revenue, positive earnings, positive FCF
        if revenue_ttm and revenue_ttm > 1_000_000_000:  # Over $1B
            if net_income_ttm and net_income_ttm > 0:
                if fcf_ttm and fcf_ttm > 0:
                    if revenue_yoy and -5 < revenue_yoy < 15:  # Moderate growth
                        company_stage = 'mature'
        
        # Turnaround: Declining revenue, negative earnings, trying to recover
        if revenue_yoy and revenue_yoy < -10:
            if net_income_ttm and net_income_ttm < 0:
                company_stage = 'turnaround'
        
        # Default to growth if revenue is growing
        if company_stage == 'unknown' and revenue_yoy and revenue_yoy > 10:
            company_stage = 'growth'
        
        financials['company_stage'] = company_stage
        
        # Generate Fundamentals Verdict
        verdict_score = 0
        if revenue_yoy and revenue_yoy > 0:
            verdict_score += 1
        if net_income_yoy and net_income_yoy > 0:
            verdict_score += 1
        if fcf_ttm and fcf_ttm > 0:
            verdict_score += 1
        if gross_margin_q and gross_margin_q > 30:
            verdict_score += 1
        if debt_fcf_ratio and debt_fcf_ratio < 2:
            verdict_score += 1
        
        if verdict_score >= 4:
            financials['fundamentals_verdict'] = 'strong'
        elif verdict_score >= 2:
            financials['fundamentals_verdict'] = 'neutral'
        else:
            financials['fundamentals_verdict'] = 'weak'
        
        # Generate Main Verdict Sentence
        verdict_parts = []
        
        # Revenue assessment
        if revenue_yoy:
            if revenue_yoy > 15:
                verdict_parts.append('Silné tržby')
            elif revenue_yoy > 5:
                verdict_parts.append('Rostoucí tržby')
            elif revenue_yoy > 0:
                verdict_parts.append('Mírně rostoucí tržby')
            else:
                verdict_parts.append('Klesající tržby')
        
        # Earnings assessment
        if net_income_ttm:
            if net_income_ttm > 0:
                if net_income_yoy and net_income_yoy > 10:
                    verdict_parts.append('zisk roste rychle')
                elif net_income_yoy and net_income_yoy > 0:
                    verdict_parts.append('zisk roste')
                else:
                    verdict_parts.append('zisk je stabilní')
            else:
                if company_stage == 'early_stage':
                    verdict_parts.append('ztráty jsou očekávané (early-stage)')
                elif net_income_yoy and net_income_yoy > 0:
                    verdict_parts.append('ztráty se zmenšují')
                else:
                    verdict_parts.append('ztráty pokračují')
        
        # Cash flow assessment
        if fcf_ttm:
            if fcf_ttm > 0:
                if fcf_ttm >= net_income_ttm if net_income_ttm else False:
                    verdict_parts.append('výborný cash flow')
                else:
                    verdict_parts.append('pozitivní cash flow')
            else:
                if company_stage == 'early_stage':
                    verdict_parts.append('burn rate je očekávaný')
                else:
                    verdict_parts.append('negativní cash flow')
        
        # Combine into sentence
        if len(verdict_parts) >= 2:
            main_sentence = f"{verdict_parts[0]}, {verdict_parts[1]}"
            if len(verdict_parts) >= 3:
                main_sentence += f" → {verdict_parts[2]}"
        elif len(verdict_parts) == 1:
            main_sentence = verdict_parts[0]
        else:
            main_sentence = "Finanční data vyžadují další analýzu."
        
        # Add context based on stage
        if company_stage == 'early_stage':
            main_sentence += " (Pre-revenue / early-stage company - ztráty jsou očekávané)"
        elif company_stage == 'growth':
            if net_income_ttm and net_income_ttm < 0:
                main_sentence += " (Growth phase - investice do růstu)"
        elif company_stage == 'turnaround':
            main_sentence += " (Turnaround - firma se snaží zotavit)"
        
        financials['main_verdict_sentence'] = main_sentence
        
        # Add trend indicators to executive snapshot
        financials['executive_snapshot']['revenue_trend'] = 'improving' if revenue_yoy and revenue_yoy > 0 else 'deteriorating' if revenue_yoy and revenue_yoy < 0 else 'stable'
        financials['executive_snapshot']['earnings_trend'] = 'improving' if net_income_yoy and net_income_yoy > 0 else 'deteriorating' if net_income_yoy and net_income_yoy < 0 else 'stable'
        financials['executive_snapshot']['fcf_trend'] = 'improving' if fcf_ttm and fcf_ttm > 0 else 'deteriorating' if fcf_ttm and fcf_ttm < 0 else 'stable'
        
        # Calculate overall financials score (pass company_stage for growth adjustments)
        financials_score = calculate_financials_score(financials, info, company_stage)
        financials['financials_score'] = financials_score
        
        # Add advanced analyses
        # 1. Cash Flow Statement Analysis
        try:
            cash_flow_analysis = get_cash_flow_analysis(ticker)
            if cash_flow_analysis:
                financials['cash_flow_analysis'] = cash_flow_analysis
        except Exception as e:
            logger.warning(f"Failed to get cash flow analysis for {ticker}: {str(e)}")
        
        # 2. Profitability Deep Dive
        try:
            profitability_analysis = get_profitability_analysis(ticker, financials)
            if profitability_analysis:
                financials['profitability_analysis'] = profitability_analysis
        except Exception as e:
            logger.warning(f"Failed to get profitability analysis for {ticker}: {str(e)}")
        
        # 3. Balance Sheet Health Score
        try:
            balance_sheet_health = get_balance_sheet_health(ticker)
            if balance_sheet_health:
                financials['balance_sheet_health'] = balance_sheet_health
        except Exception as e:
            logger.warning(f"Failed to get balance sheet health for {ticker}: {str(e)}")
        
        # 4. Management Guidance Tracking
        try:
            guidance_tracking = get_management_guidance_tracking(ticker)
            if guidance_tracking:
                financials['management_guidance'] = guidance_tracking
        except Exception as e:
            logger.warning(f"Failed to get management guidance for {ticker}: {str(e)}")
        
        # 5. Segment/Geography Breakdown
        try:
            segment_breakdown = get_segment_breakdown(ticker)
            if segment_breakdown:
                financials['segment_breakdown'] = segment_breakdown
        except Exception as e:
            logger.warning(f"Failed to get segment breakdown for {ticker}: {str(e)}")
        
        # Add detailed financial statements for Detailed Financials tab
        def convert_dataframe_to_detailed_format(df, is_quarterly=True):
            """
            Convert pandas DataFrame to JSON-serializable format with TTM calculation for quarterly data.
            
            Args:
                df: pandas DataFrame with financial statement data
                is_quarterly: True for quarterly data (calculate TTM), False for annual data
                
            Returns:
                List of dicts with metric name, values per period, and TTM (if quarterly)
            """
            if df is None:
                logger.warning(f"convert_dataframe_to_detailed_format: DataFrame is None (is_quarterly={is_quarterly})")
                return []
            if df.empty:
                logger.warning(f"convert_dataframe_to_detailed_format: DataFrame is empty (is_quarterly={is_quarterly})")
                return []
            
            logger.debug(f"convert_dataframe_to_detailed_format: Processing DataFrame with {len(df)} rows, {len(df.columns)} columns (is_quarterly={is_quarterly})")
            
            result = []
            
            # Iterate through each row (metric) in the DataFrame
            for metric_name in df.index:
                try:
                    metric_row = df.loc[metric_name]
                    values_dict = {}
                    
                    # Process each column (quarter/year)
                    for col in df.columns:
                        try:
                            value = metric_row[col]
                            # Convert to float if possible, otherwise keep as None
                            if pd.notna(value):
                                try:
                                    float_value = float(value)
                                    # Format date as string
                                    if isinstance(col, pd.Timestamp):
                                        col_str = col.strftime('%Y-%m-%d')
                                    else:
                                        col_str = str(col)
                                    values_dict[col_str] = float_value
                                except (ValueError, TypeError):
                                    pass
                        except (KeyError, IndexError):
                            pass
                    
                    # Calculate TTM for quarterly data (sum of last 4 quarters)
                    ttm_value = None
                    if is_quarterly and len(values_dict) >= 4:
                        try:
                            # Get sorted dates (most recent first typically)
                            sorted_values = sorted(values_dict.items(), key=lambda x: x[0], reverse=True)
                            # Take first 4 quarters and sum
                            last_4_quarters = [v for _, v in sorted_values[:4]]
                            if all(v is not None for v in last_4_quarters):
                                ttm_value = sum(last_4_quarters)
                        except Exception:
                            pass
                    
                    # Only add if we have at least one value
                    if values_dict:
                        metric_data = {
                            'metric': str(metric_name),
                            'values': values_dict
                        }
                        if ttm_value is not None:
                            metric_data['ttm'] = ttm_value
                        result.append(metric_data)
                        
                except Exception as e:
                    logger.warning(f"Error processing metric {metric_name}: {str(e)}", exc_info=True)
                    continue
            
            logger.info(f"convert_dataframe_to_detailed_format: Converted {len(result)} metrics (is_quarterly={is_quarterly})")
            return result
        
        # Convert detailed financial statements
        try:
            logger.info(f"Converting detailed income statement for {ticker}: quarterly_income is {'None' if quarterly_income is None else ('empty' if quarterly_income.empty else 'has data')}, annual_income is {'None' if annual_income is None else ('empty' if annual_income.empty else 'has data')}")
            financials['detailed_income_statement'] = {
                'quarterly': convert_dataframe_to_detailed_format(quarterly_income, is_quarterly=True) if quarterly_income is not None and not quarterly_income.empty else [],
                'annual': convert_dataframe_to_detailed_format(annual_income, is_quarterly=False) if annual_income is not None and not annual_income.empty else []
            }
            logger.info(f"Converted income statement: quarterly={len(financials['detailed_income_statement']['quarterly'])}, annual={len(financials['detailed_income_statement']['annual'])}")
        except Exception as e:
            logger.warning(f"Failed to convert detailed income statement for {ticker}: {str(e)}", exc_info=True)
            financials['detailed_income_statement'] = {'quarterly': [], 'annual': []}
        
        try:
            logger.info(f"Converting detailed balance sheet for {ticker}: quarterly_bs is {'None' if quarterly_bs is None else ('empty' if quarterly_bs.empty else 'has data')}, annual_bs is {'None' if annual_bs is None else ('empty' if annual_bs.empty else 'has data')}")
            financials['detailed_balance_sheet'] = {
                'quarterly': convert_dataframe_to_detailed_format(quarterly_bs, is_quarterly=True) if quarterly_bs is not None and not quarterly_bs.empty else [],
                'annual': convert_dataframe_to_detailed_format(annual_bs, is_quarterly=False) if annual_bs is not None and not annual_bs.empty else []
            }
        except Exception as e:
            logger.warning(f"Failed to convert detailed balance sheet for {ticker}: {str(e)}", exc_info=True)
            financials['detailed_balance_sheet'] = {'quarterly': [], 'annual': []}
        
        try:
            logger.info(f"Converting detailed cash flow for {ticker}: quarterly_cf is {'None' if quarterly_cf is None else ('empty' if quarterly_cf.empty else 'has data')}, annual_cf is {'None' if annual_cf is None else ('empty' if annual_cf.empty else 'has data')}")
            financials['detailed_cash_flow'] = {
                'quarterly': convert_dataframe_to_detailed_format(quarterly_cf, is_quarterly=True) if quarterly_cf is not None and not quarterly_cf.empty else [],
                'annual': convert_dataframe_to_detailed_format(annual_cf, is_quarterly=False) if annual_cf is not None and not annual_cf.empty else []
            }
        except Exception as e:
            logger.warning(f"Failed to convert detailed cash flow for {ticker}: {str(e)}", exc_info=True)
            financials['detailed_cash_flow'] = {'quarterly': [], 'annual': []}
        
        # Clean financials data to ensure all Timestamps are converted before return
        return clean_for_json(financials)
    
    except Exception as e:
        logger.exception(f"Error fetching financials for {ticker}")
        return None


def get_earnings_qoq(ticker: str) -> Optional[Dict]:
    """
    Get quarterly earnings, EPS, revenue and compare with expectations (cached)
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with quarterly earnings data or None
    """
    # Check cache first
    if CACHE_AVAILABLE and cache:
        cache_key = f"yfinance_earnings_qoq_{ticker}"
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for earnings QoQ {ticker}")
            return cached_data
    
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
        
        result = {
            'ticker': ticker.upper(),
            'earnings': earnings_data
        }
        
        # Cache the result
        if CACHE_AVAILABLE and cache:
            cache_key = f"yfinance_earnings_qoq_{ticker}"
            cache.set(cache_key, result, timeout=CACHE_TIMEOUTS['yfinance'])
            logger.debug(f"Cached earnings QoQ for {ticker}")
        
        return result
    
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

