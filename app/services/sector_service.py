"""
Sector comparison service - Get sector averages for financial metrics
"""
import yfinance as yf
import pandas as pd
import time
from typing import Dict, Optional, List
from app.utils.logger import logger


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

