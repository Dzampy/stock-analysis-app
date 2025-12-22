"""
SEC API service - Insider trading, institutional flow, whale watching
"""
import os
import requests
from typing import Dict, Optional, List
from app.config import SEC_API_KEY
from app.utils.constants import SEC_API_TIMEOUT
from app.utils.logger import logger


def get_institutional_flow(ticker: str) -> Optional[Dict]:
    """
    Get institutional flow data (13F filings, net flow, top buyers/sellers)
    
    Uses SEC API to query 13F filings and calculate institutional flow.
    """
    try:
        if not SEC_API_KEY:
            return None
        
        # SEC API endpoint for 13F filings
        url = "https://api.sec-api.io/form-13f"
        
        headers = {
            'Authorization': SEC_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Query for 13F filings containing this ticker
        # 13F filings report institutional holdings
        query = {
            "query": {
                "query_string": {
                    "query": f"holdings.ticker:{ticker.upper()}"
                }
            },
            "from": 0,
            "size": 50,  # Get more filings to calculate flow
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = requests.post(url, headers=headers, json=query, timeout=SEC_API_TIMEOUT)
        
        if response.status_code != 200:
            if response.status_code == 401:
                logger.warning(f"SEC API authentication failed for institutional flow")
            elif response.status_code == 429:
                logger.warning(f"SEC API rate limit exceeded for institutional flow")
            else:
                logger.warning(f"SEC API returned status {response.status_code} for institutional flow")
            return None
        
        data = response.json()
        filings = data.get('filings', [])
        
        if not filings or len(filings) < 2:
            # Need at least 2 filings to calculate flow
            return None
        
        # Group filings by institution (filer)
        institution_changes = {}
        
        for filing in filings:
            try:
                filer_name = filing.get('filer', {}).get('name', 'Unknown')
                filed_at = filing.get('filedAt', '')
                
                # Get holdings for this ticker
                holdings = filing.get('holdings', [])
                ticker_holding = None
                
                for holding in holdings:
                    if holding.get('ticker', '').upper() == ticker.upper():
                        ticker_holding = holding
                        break
                
                if ticker_holding:
                    shares = ticker_holding.get('shares', 0)
                    value = ticker_holding.get('value', 0)
                    
                    if filer_name not in institution_changes:
                        institution_changes[filer_name] = []
                    
                    institution_changes[filer_name].append({
                        'date': filed_at[:10] if filed_at else 'N/A',
                        'shares': shares,
                        'value': value
                    })
            except Exception as e:
                logger.warning(f"Error processing 13F filing: {str(e)}")
                continue
        
        # Calculate net flow for each institution
        net_flows = []
        top_buyers = []
        top_sellers = []
        
        for institution, changes in institution_changes.items():
            if len(changes) < 2:
                continue  # Need at least 2 data points
            
            # Sort by date
            changes.sort(key=lambda x: x['date'])
            
            # Calculate change between most recent and previous
            latest = changes[-1]
            previous = changes[-2]
            
            shares_change = latest['shares'] - previous['shares']
            value_change = latest['value'] - previous['value']
            
            if abs(shares_change) > 0:  # Only include if there was actual change
                net_flows.append({
                    'institution': institution,
                    'shares_change': shares_change,
                    'value_change': value_change,
                    'current_shares': latest['shares'],
                    'current_value': latest['value'],
                    'date': latest['date']
                })
        
        # Sort by absolute value change
        net_flows.sort(key=lambda x: abs(x['value_change']), reverse=True)
        
        # Separate buyers and sellers
        for flow in net_flows:
            if flow['value_change'] > 0:
                top_buyers.append(flow)
            elif flow['value_change'] < 0:
                top_sellers.append(flow)
        
        # Calculate total net flow
        total_net_flow = sum(f['value_change'] for f in net_flows)
        
        # Get most recent filing date
        last_updated = filings[0].get('filedAt', '')[:10] if filings else None
        
        return {
            'net_flow': round(total_net_flow, 2) if total_net_flow else None,
            'top_buyers': top_buyers[:10],  # Top 10 buyers
            'top_sellers': top_sellers[:10],  # Top 10 sellers
            'last_updated': last_updated,
            'total_institutions': len(net_flows)
        }
        
    except Exception as e:
        logger.exception(f"Error getting institutional flow for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def get_whale_watching(ticker: str) -> Optional[Dict]:
    """
    Get whale watching data (top institutional positions, changes in whale positions)
    
    Uses SEC API to find largest institutional holders (whales) and track their position changes.
    """
    try:
        if not SEC_API_KEY:
            return None
        
        # SEC API endpoint for 13F filings
        url = "https://api.sec-api.io/form-13f"
        
        headers = {
            'Authorization': SEC_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Query for most recent 13F filings containing this ticker
        query = {
            "query": {
                "query_string": {
                    "query": f"holdings.ticker:{ticker.upper()}"
                }
            },
            "from": 0,
            "size": 100,  # Get more filings to find whales
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = requests.post(url, headers=headers, json=query, timeout=SEC_API_TIMEOUT)
        
        if response.status_code != 200:
            if response.status_code == 401:
                logger.warning(f"SEC API authentication failed for whale watching")
            elif response.status_code == 429:
                logger.warning(f"SEC API rate limit exceeded for whale watching")
            else:
                logger.warning(f"SEC API returned status {response.status_code} for whale watching")
            return None
        
        data = response.json()
        filings = data.get('filings', [])
        
        if not filings:
            return None
        
        # Get most recent holdings for each institution
        current_holdings = {}
        previous_holdings = {}
        
        for filing in filings:
            try:
                filer_name = filing.get('filer', {}).get('name', 'Unknown')
                filed_at = filing.get('filedAt', '')
                date_str = filed_at[:10] if filed_at else 'N/A'
                
                # Get holdings for this ticker
                holdings = filing.get('holdings', [])
                ticker_holding = None
                
                for holding in holdings:
                    if holding.get('ticker', '').upper() == ticker.upper():
                        ticker_holding = holding
                        break
                
                if ticker_holding:
                    shares = ticker_holding.get('shares', 0)
                    value = ticker_holding.get('value', 0)
                    
                    # Store most recent and previous holdings
                    if filer_name not in current_holdings:
                        current_holdings[filer_name] = {
                            'shares': shares,
                            'value': value,
                            'date': date_str
                        }
                    elif filer_name not in previous_holdings:
                        previous_holdings[filer_name] = {
                            'shares': shares,
                            'value': value,
                            'date': date_str
                        }
            except Exception as e:
                logger.warning(f"Error processing 13F filing for whale watching: {str(e)}")
                continue
        
        # Sort by value to find whales (largest positions)
        top_positions = []
        for institution, holding in current_holdings.items():
            top_positions.append({
                'institution': institution,
                'shares': holding['shares'],
                'value': holding['value'],
                'date': holding['date']
            })
        
        # Sort by value (descending)
        top_positions.sort(key=lambda x: x['value'], reverse=True)
        
        # Calculate position changes
        position_changes = []
        for institution, current in current_holdings.items():
            previous = previous_holdings.get(institution)
            if previous:
                shares_change = current['shares'] - previous['shares']
                value_change = current['value'] - previous['value']
                change_pct = (value_change / previous['value'] * 100) if previous['value'] > 0 else 0
                
                # Only include significant changes (>5% or >$1M)
                if abs(change_pct) > 5 or abs(value_change) > 1000000:
                    position_changes.append({
                        'institution': institution,
                        'shares_change': shares_change,
                        'value_change': value_change,
                        'change_pct': round(change_pct, 2),
                        'current_value': current['value'],
                        'date': current['date']
                    })
        
        # Sort position changes by absolute value change
        position_changes.sort(key=lambda x: abs(x['value_change']), reverse=True)
        
        # Define whale threshold (top 10% by value or >$100M)
        whale_threshold = top_positions[0]['value'] * 0.1 if top_positions else 100000000
        whales = [p for p in top_positions if p['value'] >= whale_threshold]
        
        return {
            'top_positions': top_positions[:20],  # Top 20 positions
            'position_changes': position_changes[:15],  # Top 15 significant changes
            'whale_count': len(whales),
            'whales': whales[:10],  # Top 10 whales
            'last_updated': top_positions[0]['date'] if top_positions else None
        }
        
    except Exception as e:
        logger.exception(f"Error getting whale watching for {ticker}")
        import traceback
        traceback.print_exc()
        return None
