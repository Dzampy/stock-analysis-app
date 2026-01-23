"""
Sector comparison service - Get sector averages for financial metrics
"""
import yfinance as yf
import pandas as pd
import time
from typing import Dict, Optional, List
from app.utils.logger import logger
from app.utils.constants import RATE_LIMIT_DELAY


def get_sector_historical_data(sector: str, metric: str = 'revenue_growth', quarters: int = 8) -> Optional[Dict]:
    """
    Get historical sector averages for a specific metric over time.
    
    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare')
        metric: Metric to track ('revenue_growth', 'profit_margin', 'operating_margin', 'net_margin', 'roe', 'roa')
        quarters: Number of quarters to retrieve
        
    Returns:
        Dict with historical sector averages by quarter or None
    """
    if not sector or sector == 'N/A':
        return None
    
    try:
        # Define representative tickers for each sector
        sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'ORCL', 'NFLX', 'ADBE'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'ABBV', 'MRK', 'BMY', 'AMGN', 'GILD'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'MA'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'HAL', 'NOV'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'COST'],
            'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'MDT', 'CL', 'GIS'],
            'Industrials': ['BA', 'CAT', 'GE', 'HON', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'SWK'],
            'Real Estate': ['AMT', 'PLD', 'PSA', 'EQIX', 'WELL', 'SPG', 'O', 'DLR', 'EXPI', 'CBRE'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'EA', 'ATVI', 'TTWO'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ES'],
            'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'PPG', 'DD', 'FCX', 'NEM', 'VALE', 'RIO'],
        }
        
        tickers = sector_tickers.get(sector, [])
        if not tickers:
            return None
        
        # Limit to top 8 to avoid too many API calls
        tickers = tickers[:8]
        
        # Map metric names to data sources
        metric_map = {
            'revenue_growth': ('income_stmt', 'total revenue', lambda rev, prev: ((rev - prev) / prev * 100) if prev and prev > 0 else None),
            'profit_margin': ('income_stmt', ['net income', 'total revenue'], lambda ni, rev: (ni / rev * 100) if rev and rev > 0 else None),
            'operating_margin': ('income_stmt', ['operating income', 'total revenue'], lambda oi, rev: (oi / rev * 100) if rev and rev > 0 else None),
            'net_margin': ('income_stmt', ['net income', 'total revenue'], lambda ni, rev: (ni / rev * 100) if rev and rev > 0 else None),
        }
        
        if metric not in metric_map:
            # For metrics that come from info (not historical), return None
            return None
        
        metric_source, metric_keys, metric_calc = metric_map[metric]
        
        # Collect historical data for each ticker
        historical_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                time.sleep(RATE_LIMIT_DELAY)
                
                if metric_source == 'income_stmt':
                    income_stmt = stock.income_stmt
                    if income_stmt is None or income_stmt.empty:
                        continue
                    
                    # Helper to find row
                    def find_row(df, keywords):
                        for idx in df.index:
                            idx_lower = str(idx).lower()
                            if isinstance(keywords, list):
                                if any(kw.lower() in idx_lower for kw in keywords):
                                    return df.loc[idx]
                            else:
                                if keywords.lower() in idx_lower:
                                    return df.loc[idx]
                        return None
                    
                    # Get data for last N quarters
                    quarter_data = []
                    for i, col in enumerate(income_stmt.columns[:quarters]):
                        try:
                            quarter_date = pd.Timestamp(col)
                            quarter_str = f"{quarter_date.year}-Q{(quarter_date.month - 1) // 3 + 1}"
                            
                            if metric == 'revenue_growth':
                                revenue_row = find_row(income_stmt, 'total revenue')
                                if revenue_row is not None and i < len(revenue_row):
                                    revenue = float(revenue_row.iloc[i]) if pd.notna(revenue_row.iloc[i]) else None
                                    prev_revenue = float(revenue_row.iloc[i+1]) if i+1 < len(revenue_row) and pd.notna(revenue_row.iloc[i+1]) else None
                                    if revenue and prev_revenue:
                                        growth = metric_calc(revenue, prev_revenue)
                                        quarter_data.append({'quarter': quarter_str, 'value': growth})
                            else:
                                # For margin metrics
                                revenue_row = find_row(income_stmt, 'total revenue')
                                income_row = find_row(income_stmt, metric_keys[0])
                                
                                if revenue_row is not None and income_row is not None and i < len(revenue_row) and i < len(income_row):
                                    revenue = float(revenue_row.iloc[i]) if pd.notna(revenue_row.iloc[i]) else None
                                    income = float(income_row.iloc[i]) if pd.notna(income_row.iloc[i]) else None
                                    if revenue and income:
                                        margin = metric_calc(income, revenue)
                                        quarter_data.append({'quarter': quarter_str, 'value': margin})
                        except (IndexError, ValueError, TypeError):
                            continue
                    
                    if quarter_data:
                        historical_data[ticker] = quarter_data
                        
            except Exception as e:
                logger.debug(f"Error fetching historical data for {ticker}: {str(e)}")
                continue
        
        if not historical_data:
            return None
        
        # Calculate sector averages by quarter
        quarter_averages = {}
        for ticker, data in historical_data.items():
            for entry in data:
                quarter = entry['quarter']
                value = entry['value']
                if value is not None:
                    if quarter not in quarter_averages:
                        quarter_averages[quarter] = []
                    quarter_averages[quarter].append(value)
        
        # Calculate averages
        sector_historical = {}
        for quarter, values in sorted(quarter_averages.items()):
            if values:
                avg = sum(values) / len(values)
                sector_historical[quarter] = round(avg, 2)
        
        return {
            'sector': sector,
            'metric': metric,
            'historical_averages': sector_historical,
            'sample_size': len(historical_data)
        }
        
    except Exception as e:
        logger.exception(f"Error getting sector historical data for {sector}")
        return None


def get_sector_averages(sector: str, industry: str = None) -> Optional[Dict]:
    """
    Get average financial metrics for a sector
    
    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare')
        industry: Optional industry name for more specific comparison
        
    Returns:
        Dict with sector average metrics or None
    """
    if not sector or sector == 'N/A':
        return None
    
    try:
        # Define representative tickers for each sector (top companies by market cap)
        sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'ORCL', 'NFLX', 'ADBE'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'ABBV', 'MRK', 'BMY', 'AMGN', 'GILD'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'MA'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'HAL', 'NOV'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'COST'],
            'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'MDT', 'CL', 'GIS'],
            'Industrials': ['BA', 'CAT', 'GE', 'HON', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'SWK'],
            'Real Estate': ['AMT', 'PLD', 'PSA', 'EQIX', 'WELL', 'SPG', 'O', 'DLR', 'EXPI', 'CBRE'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'EA', 'ATVI', 'TTWO'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ES'],
            'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'PPG', 'DD', 'FCX', 'NEM', 'VALE', 'RIO'],
        }
        
        # Get tickers for this sector
        tickers = sector_tickers.get(sector, [])
        if not tickers:
            logger.warning(f"No representative tickers defined for sector: {sector}")
            return None
        
        # Limit to top 10 to avoid too many API calls
        tickers = tickers[:10]
        
        metrics_list = []
        successful_fetches = 0
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                time.sleep(0.2)  # Rate limiting
                info = stock.info
                
                # Extract key metrics
                metrics = {
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'price_to_book': info.get('priceToBook'),
                    'profit_margin': info.get('profitMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'return_on_equity': info.get('returnOnEquity'),
                    'return_on_assets': info.get('returnOnAssets'),
                    'dividend_yield': info.get('dividendYield'),
                }
                
                # Only add if we have at least some metrics
                if any(v is not None for v in metrics.values()):
                    metrics_list.append(metrics)
                    successful_fetches += 1
                
            except Exception as e:
                logger.debug(f"Error fetching data for {ticker} (sector {sector}): {str(e)}")
                continue
        
        if not metrics_list:
            logger.warning(f"No metrics collected for sector: {sector}")
            return None
        
        # Calculate averages (excluding None values)
        def safe_average(values):
            """Calculate average excluding None values"""
            valid_values = [v for v in values if v is not None and not pd.isna(v)]
            if not valid_values:
                return None
            return sum(valid_values) / len(valid_values)
        
        averages = {}
        metric_names = metrics_list[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if m.get(metric_name) is not None]
            if values:
                avg = safe_average(values)
                if avg is not None:
                    averages[metric_name] = round(avg, 4)
        
        logger.info(f"Calculated sector averages for {sector} from {successful_fetches} companies")
        
        return {
            'sector': sector,
            'averages': averages,
            'sample_size': successful_fetches
        }
        
    except Exception as e:
        logger.exception(f"Error getting sector averages for {sector}")
        return None

