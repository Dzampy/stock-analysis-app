"""
Analyst ratings and insider trading service
"""
import requests
from bs4 import BeautifulSoup
import os
import re
from app.utils.logger import logger

# Get SEC API key from environment
SEC_API_KEY = os.getenv('SEC_API_KEY')


def get_marketbeat_insider_trading(ticker):
    """Scrape insider trading data from MarketBeat"""
    try:
        # MarketBeat uses different URL format - try multiple variations
        urls = [
            f"https://www.marketbeat.com/stocks/NASDAQ/{ticker.upper()}/insider-trades/",
            f"https://www.marketbeat.com/stocks/NASDAQ/{ticker.upper()}/insiders/",
            f"https://www.marketbeat.com/stocks/NYSE/{ticker.upper()}/insider-trades/",
            f"https://www.marketbeat.com/stocks/NYSE/{ticker.upper()}/insiders/",
            f"https://www.marketbeat.com/stocks/{ticker.upper()}/insider-trades/"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
        }
        
        transactions = []
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for insider trading table
                    tables = soup.find_all('table')
                    insider_table = None
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        if len(rows) > 1:
                            header_text = rows[0].get_text().lower()
                            # MarketBeat typically has headers like "Date", "Insider", "Transaction", "Shares", "Value"
                            if ('insider' in header_text or 'transaction' in header_text) and \
                               ('date' in header_text or 'shares' in header_text):
                                insider_table = table
                                break
                    
                    if insider_table:
                        rows = insider_table.find_all('tr')
                        for row in rows[1:31]:  # Skip header, limit to 30
                            try:
                                cells = row.find_all(['td', 'th'])
                                if len(cells) < 4:
                                    continue
                                
                                # MarketBeat format varies, try to extract common fields
                                # Typical: Date, Insider, Position, Transaction Type, Shares, Value
                                date_str = 'N/A'
                                insider = 'N/A'
                                position = 'N/A'
                                transaction_type = None
                                shares = None
                                value = None
                                
                                # Try to parse cells - MarketBeat structure may vary
                                for i, cell in enumerate(cells):
                                    text = cell.get_text(strip=True)
                                    text_lower = text.lower()
                                    
                                    # Date detection
                                    if any(x in text for x in ['2024', '2025', '2023']) or \
                                       ('/' in text and len(text) < 15) or \
                                       ('-' in text and len(text) < 15):
                                        date_str = text
                                    
                                    # Transaction type
                                    if 'sale' in text_lower or 'sell' in text_lower:
                                        transaction_type = 'sell'
                                    elif ('purchase' in text_lower or 'buy' in text_lower or 'acquisition' in text_lower or
                                          'option exercise' in text_lower or 'exercise' in text_lower or
                                          'grant' in text_lower or 'award' in text_lower or
                                          'conversion' in text_lower or 'convert' in text_lower):
                                        transaction_type = 'buy'
                                    
                                    # Value (contains $ or large numbers with commas)
                                    if '$' in text or (',' in text and len(text) > 5 and any(c.isdigit() for c in text)):
                                        try:
                                            clean_value = text.replace('$', '').replace(',', '').replace(' ', '')
                                            if clean_value and clean_value.replace('.', '').isdigit():
                                                value = float(clean_value)
                                        except:
                                            pass
                                    
                                    # Shares (numbers, possibly with K/M suffixes)
                                    if not '$' in text and any(c.isdigit() for c in text):
                                        try:
                                            clean_shares = text.replace(',', '').replace(' ', '').upper()
                                            if 'K' in clean_shares:
                                                shares = int(float(clean_shares.replace('K', '')) * 1000)
                                            elif 'M' in clean_shares:
                                                shares = int(float(clean_shares.replace('M', '')) * 1000000)
                                            else:
                                                if clean_shares.replace('.', '').isdigit():
                                                    shares = int(float(clean_shares))
                                        except:
                                            pass
                                    
                                    # Insider name (longer text, not a number, not a date)
                                    if len(text) > 5 and not any(c in text for c in ['$', '/', '-']) and \
                                       not text.replace(',', '').replace('.', '').isdigit() and \
                                       not any(x in text for x in ['2024', '2025', '2023']):
                                        if insider == 'N/A':
                                            insider = text
                                
                                if transaction_type and value and value > 0:
                                    transactions.append({
                                        'date': date_str,
                                        'transaction_type': transaction_type,
                                        'value': value,
                                        'shares': shares,
                                        'insider': insider,
                                        'position': position,
                                        'text': transaction_type
                                    })
                            except Exception as e:
                                logger.warning(f"Error parsing MarketBeat row: {str(e)}")
                                continue
                        
                        if transactions:
                            break  # Found data, no need to try other URLs
            except Exception as e:
                logger.warning(f"Error accessing {url}: {str(e)}")
                continue
        
        return transactions if transactions else None
        
    except Exception as e:
        logger.exception(f"Error scraping MarketBeat for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def get_tipranks_insider_trading(ticker):
    """Scrape insider trading data from TipRanks"""
    try:
        url = f"https://www.tipranks.com/stocks/{ticker.upper()}/insider-trading"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.warning(f"TipRanks returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        transactions = []
        
        # TipRanks uses tables or divs with specific classes for insider transactions
        # Look for transaction rows
        transaction_rows = soup.find_all('tr', class_=lambda x: x and ('transaction' in str(x).lower() or 'insider' in str(x).lower()))
        
        # If no specific class, try to find table rows
        if not transaction_rows:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > 1:  # Has header + data rows
                    transaction_rows = rows[1:]  # Skip header
                    break
        
        # Alternative: look for div-based structure
        if not transaction_rows:
            transaction_divs = soup.find_all('div', class_=lambda x: x and ('transaction' in str(x).lower() or 'insider' in str(x).lower() or 'row' in str(x).lower()))
            if transaction_divs:
                transaction_rows = transaction_divs
        
        for row in transaction_rows[:30]:  # Limit to 30 most recent
            try:
                # Extract data from row
                cells = row.find_all(['td', 'th', 'div'])
                if len(cells) < 3:
                    continue
                
                # Try to extract: Date, Insider, Position, Transaction Type, Shares, Value
                date_str = 'N/A'
                insider = 'N/A'
                position = 'N/A'
                transaction_type = None
                shares = None
                value = None
                
                # Parse cells - TipRanks structure may vary
                for i, cell in enumerate(cells):
                    text = cell.get_text(strip=True)
                    text_lower = text.lower()
                    
                    # Date detection
                    if any(x in text for x in ['2024', '2025', '2023']) or '/' in text or '-' in text:
                        if len(text) < 20:  # Likely a date
                            date_str = text
                    
                    # Transaction type detection
                    if 'sale' in text_lower or 'sell' in text_lower:
                        transaction_type = 'sell'
                    elif 'purchase' in text_lower or 'buy' in text_lower or 'acquisition' in text_lower:
                        transaction_type = 'buy'
                    
                    # Value detection (contains $ or numbers with commas)
                    if '$' in text or (',' in text and any(c.isdigit() for c in text)):
                        try:
                            # Remove $ and commas, convert to float
                            clean_text = text.replace('$', '').replace(',', '').replace(' ', '')
                            if clean_text:
                                value = float(clean_text)
                        except:
                            pass
                    
                    # Shares detection (numbers without $)
                    if not '$' in text and any(c.isdigit() for c in text) and (',' in text or int(text.replace(',', '').replace(' ', '')) > 0):
                        try:
                            clean_text = text.replace(',', '').replace(' ', '')
                            if clean_text.isdigit():
                                shares = int(clean_text)
                        except:
                            pass
                    
                    # Insider name (usually longer text, not a number)
                    if len(text) > 5 and not any(c in text for c in ['$', '/', '-']) and not text.replace(',', '').replace('.', '').isdigit():
                        if insider == 'N/A' and 'insider' not in text_lower:
                            insider = text
                
                # Only add if we have valid transaction type and value
                if transaction_type and value and value > 0:
                    transactions.append({
                        'date': date_str,
                        'transaction_type': transaction_type,
                        'value': value,
                        'shares': shares,
                        'insider': insider,
                        'position': position
                    })
            except Exception as e:
                logger.warning(f"Error parsing TipRanks row: {str(e)}")
                continue
        
        return transactions if transactions else None
        
    except Exception as e:
        logger.exception(f"Error scraping TipRanks for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def get_sec_api_insider_trading(ticker):
    """Get insider trading data from SEC API (sec-api.io) - official SEC Form 3, 4, 5 filings"""
    if not SEC_API_KEY:
        logger.debug("SEC_API_KEY not configured, skipping SEC API")
        return None
    
    try:
        # SEC API endpoint for Form 4 (most common insider trading form)
        url = "https://api.sec-api.io/form-4"
        
        headers = {
            'Authorization': SEC_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Query for searching by ticker symbol
        # SEC API uses Elasticsearch query format
        query = {
            "query": {
                "query_string": {
                    "query": f"issuer.tradingSymbol:{ticker.upper()}"
                }
            },
            "from": 0,
            "size": 30,
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = requests.post(url, headers=headers, json=query, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            transactions = []
            
            filings = data.get('filings', [])
            if not filings:
                logger.info(f"SEC API returned no filings for {ticker}")
                return None
            
            for filing in filings:
                try:
                    # Get filing date
                    filed_at = filing.get('filedAt', '')
                    date_str = filed_at[:10] if filed_at else 'N/A'  # Extract YYYY-MM-DD
                    
                    # Get reporting owner (insider) info
                    reporting_owner = filing.get('reportingOwner', {})
                    insider_name = reporting_owner.get('name', 'N/A')
                    
                    # Get relationship
                    relationship = reporting_owner.get('relationship', {})
                    position = 'N/A'
                    if relationship.get('isOfficer', False):
                        position = 'Officer'
                    elif relationship.get('isDirector', False):
                        position = 'Director'
                    elif relationship.get('isTenPercentOwner', False):
                        position = '10% Owner'
                    elif relationship.get('isOther', False):
                        position = 'Other'
                    
                    # Process non-derivative transactions (direct stock purchases/sales)
                    non_derivative = filing.get('nonDerivativeTable', {}).get('holdings', [])
                    for holding in non_derivative:
                        transactions_list = holding.get('transactions', [])
                        for trans in transactions_list:
                            transaction_code = trans.get('transactionCode', '')
                            shares = trans.get('shares', 0)
                            price = trans.get('pricePerShare', 0)
                            
                            # Calculate value
                            value = 0
                            if shares and price:
                                try:
                                    value = float(shares) * float(price)
                                except:
                                    pass
                            
                            # Determine transaction type based on SEC transaction codes
                            # P = Open market purchase, A = Grant/award, I = Discretionary transaction (acquisition)
                            # S = Open market sale, D = Disposition to issuer, F = Payment of exercise price
                            transaction_type = None
                            if transaction_code in ['P', 'A', 'I', 'M', 'X', 'C', 'L']:  # Purchase codes
                                transaction_type = 'buy'
                            elif transaction_code in ['S', 'D', 'F', 'E', 'H', 'U']:  # Sale codes
                                transaction_type = 'sell'
                            
                            if transaction_type and value > 0:
                                logger.debug(f"SEC API: Adding {transaction_type} transaction: {insider_name}, value={value}, shares={shares}, code={transaction_code}")
                                transactions.append({
                                    'date': date_str,
                                    'transaction_type': transaction_type,
                                    'value': value,
                                    'shares': int(shares) if shares else 0,
                                    'insider': insider_name,
                                    'position': position,
                                    'transaction_code': transaction_code
                                })
                            elif transaction_type:
                                logger.debug(f"SEC API: Skipping {transaction_type} transaction (value={value}, shares={shares}, code={transaction_code})")
                    
                    # Process derivative transactions (options, warrants, etc.)
                    derivative = filing.get('derivativeTable', {}).get('holdings', [])
                    for holding in derivative:
                        transactions_list = holding.get('transactions', [])
                        for trans in transactions_list:
                            transaction_code = trans.get('transactionCode', '')
                            shares = trans.get('shares', 0)
                            price = trans.get('pricePerShare', 0)
                            
                            value = 0
                            if shares and price:
                                try:
                                    value = float(shares) * float(price)
                                except:
                                    pass
                            
                            transaction_type = None
                            if transaction_code in ['P', 'A', 'I', 'M', 'X', 'C']:
                                transaction_type = 'buy'
                            elif transaction_code in ['S', 'D', 'F', 'E', 'H']:
                                transaction_type = 'sell'
                            
                            if transaction_type and value > 0:
                                transactions.append({
                                    'date': date_str,
                                    'transaction_type': transaction_type,
                                    'value': value,
                                    'shares': int(shares) if shares else 0,
                                    'insider': insider_name,
                                    'position': f'Derivative ({position})',
                                    'transaction_code': transaction_code
                                })
                
                except Exception as filing_error:
                    logger.warning(f"Error processing SEC filing: {str(filing_error)}")
                    continue
            
            return transactions if transactions else None
            
        elif response.status_code == 401:
            logger.warning(f"SEC API authentication failed - check API key")
            return None
        elif response.status_code == 429:
            logger.warning(f"SEC API rate limit exceeded")
            return None
        else:
            logger.warning(f"SEC API returned status {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        logger.exception(f"Error fetching SEC API insider trading for {ticker}")
        import traceback
        traceback.print_exc()
        return None

