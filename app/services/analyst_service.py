"""
Analyst ratings and insider trading service
"""
import requests
from bs4 import BeautifulSoup
import os
import re
import json
from datetime import datetime
from app.utils.logger import logger

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# Get SEC API key from environment
SEC_API_KEY = os.getenv('SEC_API_KEY')


def get_yahoo_insider_trading(ticker):
    """Get insider trading data from Yahoo Finance using yfinance API (same data as Yahoo Finance web)"""
    try:
        import yfinance as yf
        import pandas as pd
        
        stock = yf.Ticker(ticker.upper())
        insider_df = stock.insider_transactions
        
        if insider_df is None or insider_df.empty:
            logger.warning(f"No insider transactions data from yfinance for {ticker}")
            return None
        
        transactions = []
        
        # Process all rows (yfinance already has the correct data from Yahoo Finance)
        for idx, row in insider_df.iterrows():
            try:
                row_dict = row.to_dict()
                
                # Parse transaction type from Text field
                transaction_type = None
                text = str(row_dict.get('Text', '')).lower() if row_dict.get('Text') else ''
                
                if any(word in text for word in ['sale', 'sell', 'dispose', 'disposition']):
                    transaction_type = 'sell'
                elif any(word in text for word in ['purchase', 'buy', 'acquisition', 'acquire', 'option', 'exercise', 'grant', 'award', 'convert', 'conversion']):
                    transaction_type = 'buy'
                
                # Parse value
                value = None
                val = row_dict.get('Value')
                if val is not None and pd.notna(val):
                    try:
                        value = float(val)
                    except:
                        pass
                
                # Parse shares
                shares = None
                sh = row_dict.get('Shares')
                if sh is not None and pd.notna(sh):
                    try:
                        shares = int(float(sh))  # Handle decimals
                    except:
                        pass
                
                # Get insider name
                insider = 'N/A'
                ins = row_dict.get('Insider')
                if ins is not None and pd.notna(ins):
                    insider = str(ins)
                
                # Parse date - yfinance uses index or Start Date
                date_str = 'N/A'
                # Try index first (usually the date)
                if hasattr(idx, 'strftime'):
                    date_str = idx.strftime('%Y-%m-%d')
                elif hasattr(idx, 'date'):
                    date_str = idx.date().strftime('%Y-%m-%d')
                else:
                    # Fallback to Start Date field
                    date_val = row_dict.get('Start Date')
                    if date_val is not None and pd.notna(date_val):
                        if hasattr(date_val, 'strftime'):
                            date_str = date_val.strftime('%Y-%m-%d')
                        elif hasattr(date_val, 'date'):
                            date_str = date_val.date().strftime('%Y-%m-%d')
                        else:
                            date_str = str(date_val)
                
                # Only add if we have valid transaction type
                if transaction_type:
                    transactions.append({
                        'date': date_str,
                        'transaction_type': transaction_type,
                        'value': value or 0,
                        'shares': shares or 0,
                        'insider': insider,
                        'position': row_dict.get('Position') or 'N/A',
                        'text': str(row_dict.get('Text', ''))
                    })
            except Exception as row_error:
                logger.debug(f"Error parsing yfinance insider row: {str(row_error)}")
                continue
        
        if transactions:
            logger.info(f"yfinance API returned {len(transactions)} insider transactions for {ticker}")
            return transactions
        else:
            logger.warning(f"No valid transactions parsed for {ticker}")
            return None
            
    except Exception as e:
        logger.exception(f"Error getting yfinance insider data for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_marketbeat_insider_trading(ticker):
    """Scrape insider trading data from MarketBeat"""
    try:
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        # Use cloudscraper if available (for Cloudflare protection)
        if CLOUDSCRAPER_AVAILABLE:
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, headers=headers, timeout=15)
        else:
            response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            logger.warning(f"Yahoo Finance returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        transactions = []
        
        # First, try to find JSON data in script tags (Yahoo Finance uses React)
        script_tags = soup.find_all('script')
        for script in script_tags:
            if not script.string:
                continue
            
            script_text = script.string
            
            # Look for root.App.main or similar JSON structures
            # Yahoo Finance often stores data in window.__PRELOADED_STATE__ or root.App.main
            json_patterns = [
                r'root\.App\.main\s*=\s*(\{.*?\});',
                r'window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});',
                r'"insiderTransactions":\s*(\[.*?\])',
                r'"InsiderTransactions":\s*(\[.*?\])',
            ]
            
            for pattern in json_patterns:
                try:
                    matches = re.finditer(pattern, script_text, re.DOTALL)
                    for match in matches:
                        try:
                            json_str = match.group(1)
                            # Try to parse the JSON
                            data = json.loads(json_str)
                            
                            # Recursively search for insider transaction data
                            def find_insider_data(obj, path=[]):
                                if isinstance(obj, dict):
                                    for key, value in obj.items():
                                        if 'insider' in key.lower() and isinstance(value, list):
                                            return value
                                        result = find_insider_data(value, path + [key])
                                        if result:
                                            return result
                                elif isinstance(obj, list):
                                    for item in obj:
                                        result = find_insider_data(item, path)
                                        if result:
                                            return result
                                return None
                            
                            insider_data = find_insider_data(data)
                            if insider_data:
                                logger.info(f"Found insider data in JSON for {ticker}")
                                # Process the JSON data
                                for item in insider_data:
                                    try:
                                        # Extract fields from JSON structure
                                        date_str = item.get('filedDate') or item.get('date') or item.get('transactionDate') or 'N/A'
                                        name = item.get('name') or item.get('insider') or item.get('filerName') or 'N/A'
                                        transaction_text = item.get('transaction') or item.get('transactionType') or item.get('transactionText') or ''
                                        value = item.get('value') or item.get('transactionValue') or 0
                                        shares = item.get('shares') or item.get('transactionShares') or 0
                                        
                                        # Parse transaction type
                                        transaction_type = None
                                        transaction_lower = str(transaction_text).lower()
                                        
                                        if any(word in transaction_lower for word in ['sale', 'sell', 'dispose', 'disposition']):
                                            transaction_type = 'sell'
                                        elif any(word in transaction_lower for word in ['purchase', 'buy', 'acquisition', 'acquire', 'option', 'exercise', 'grant', 'award', 'convert', 'conversion']):
                                            transaction_type = 'buy'
                                        
                                        # Convert value and shares to proper types
                                        try:
                                            if isinstance(value, str):
                                                value = float(re.sub(r'[^\d.]', '', value))
                                            else:
                                                value = float(value) if value else 0
                                        except:
                                            value = 0
                                        
                                        try:
                                            if isinstance(shares, str):
                                                shares = int(float(re.sub(r'[^\d.]', '', shares)))
                                            else:
                                                shares = int(shares) if shares else 0
                                        except:
                                            shares = 0
                                        
                                        # Parse date
                                        if date_str and date_str != 'N/A':
                                            try:
                                                if isinstance(date_str, int):
                                                    # Unix timestamp
                                                    date_str = datetime.fromtimestamp(date_str).strftime('%Y-%m-%d')
                                                elif isinstance(date_str, str):
                                                    # Try various date formats
                                                    date_formats = [
                                                        '%b %d, %Y', '%B %d, %Y', '%m/%d/%Y', 
                                                        '%Y-%m-%d', '%d %b %Y', '%Y-%m-%dT%H:%M:%S'
                                                    ]
                                                    parsed = False
                                                    for fmt in date_formats:
                                                        try:
                                                            date_str = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                                                            parsed = True
                                                            break
                                                        except:
                                                            continue
                                                    if not parsed:
                                                        date_str = str(date_str)
                                            except:
                                                date_str = str(date_str)
                                        
                                        if transaction_type and value and value > 0:
                                            transactions.append({
                                                'date': date_str,
                                                'transaction_type': transaction_type,
                                                'value': value,
                                                'shares': shares,
                                                'insider': name,
                                                'position': item.get('position') or 'N/A',
                                                'text': str(transaction_text)
                                            })
                                    except Exception as e:
                                        logger.debug(f"Error parsing JSON insider item: {str(e)}")
                                        continue
                                
                                if transactions:
                                    break
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.debug(f"Error parsing JSON from script: {str(e)}")
                            continue
                    
                    if transactions:
                        break
                except Exception as e:
                    logger.debug(f"Error in JSON pattern matching: {str(e)}")
                    continue
            
            if transactions:
                break
        
        # Yahoo Finance uses a table with insider transactions
        # Look for the main table containing insider data
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:  # Need at least header + data row
                continue
            
            # Check if this looks like an insider transactions table
            header_row = rows[0]
            header_text = header_row.get_text().lower()
            
            # Yahoo Finance insider table typically has: Date, Name, Transaction, Value, Shares
            if any(keyword in header_text for keyword in ['date', 'name', 'transaction', 'insider', 'value', 'shares']):
                # Process data rows
                for row in rows[1:]:
                    try:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) < 4:
                            continue
                        
                        # Extract data from cells
                        # Typical structure: Date, Name, Transaction, Value, Shares
                        date_text = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                        name_text = cells[1].get_text(strip=True) if len(cells) > 1 else 'N/A'
                        transaction_text = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                        value_text = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                        shares_text = cells[4].get_text(strip=True) if len(cells) > 4 else ''
                        
                        # Parse transaction type
                        transaction_type = None
                        transaction_lower = transaction_text.lower()
                        
                        if any(word in transaction_lower for word in ['sale', 'sell', 'dispose', 'disposition']):
                            transaction_type = 'sell'
                        elif any(word in transaction_lower for word in ['purchase', 'buy', 'acquisition', 'acquire', 'option', 'exercise', 'grant', 'award', 'convert', 'conversion']):
                            transaction_type = 'buy'
                        
                        # Parse value
                        value = None
                        if value_text:
                            # Remove currency symbols and commas
                            value_clean = re.sub(r'[^\d.]', '', value_text)
                            try:
                                value = float(value_clean)
                            except:
                                pass
                        
                        # Parse shares
                        shares = None
                        if shares_text:
                            # Remove commas and other non-numeric chars except decimal point
                            shares_clean = re.sub(r'[^\d.]', '', shares_text)
                            try:
                                shares = int(float(shares_clean))  # Convert to int, handling decimals
                            except:
                                pass
                        
                        # Parse date
                        date_str = 'N/A'
                        if date_text:
                            try:
                                # Try to parse various date formats
                                date_formats = [
                                    '%b %d, %Y',  # Jan 15, 2024
                                    '%B %d, %Y',   # January 15, 2024
                                    '%m/%d/%Y',    # 01/15/2024
                                    '%Y-%m-%d',    # 2024-01-15
                                    '%d %b %Y',    # 15 Jan 2024
                                ]
                                
                                parsed_date = None
                                for fmt in date_formats:
                                    try:
                                        parsed_date = datetime.strptime(date_text, fmt)
                                        break
                                    except:
                                        continue
                                
                                if parsed_date:
                                    date_str = parsed_date.strftime('%Y-%m-%d')
                                else:
                                    date_str = date_text  # Use as-is if can't parse
                            except:
                                date_str = date_text
                        
                        # Only add if we have valid transaction type and value
                        if transaction_type and value and value > 0:
                            transactions.append({
                                'date': date_str,
                                'transaction_type': transaction_type,
                                'value': value,
                                'shares': shares,
                                'insider': name_text,
                                'position': 'N/A',  # Yahoo Finance doesn't always show position
                                'text': transaction_text
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing Yahoo Finance insider row: {str(e)}")
                        continue
                
                # If we found transactions, break (we found the right table)
                if transactions:
                    break
        
        # If no transactions found in tables, try alternative selectors
        if not transactions:
            # Look for div-based structures or other patterns
            transaction_divs = soup.find_all(['div', 'span'], class_=lambda x: x and ('insider' in str(x).lower() or 'transaction' in str(x).lower()))
            
            # Also try looking for data attributes or specific Yahoo Finance patterns
            # Yahoo Finance might use React components, so we might need to look for data in script tags
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'insider' in script.string.lower():
                    # Try to extract JSON data from script
                    try:
                        import json
                        # Look for JSON objects in the script
                        json_match = re.search(r'\{.*"insider".*\}', script.string, re.DOTALL)
                        if json_match:
                            # This is a simplified approach - might need more sophisticated parsing
                            pass
                    except:
                        pass
        
        if transactions:
            logger.info(f"Yahoo Finance returned {len(transactions)} insider transactions for {ticker}")
            return transactions
        
        # If no transactions found via scraping, try yfinance API as fallback
        logger.debug(f"No transactions found via scraping for {ticker}, trying yfinance API")
        try:
            import yfinance as yf
            import pandas as pd
            
            stock = yf.Ticker(ticker.upper())
            insider_df = stock.insider_transactions
            
            if insider_df is not None and not insider_df.empty:
                for idx, row in insider_df.tail(50).iterrows():
                    try:
                        row_dict = row.to_dict()
                        
                        transaction_type = None
                        text = str(row_dict.get('Text', '')).lower() if row_dict.get('Text') else ''
                        
                        if 'sale' in text or 'sell' in text:
                            transaction_type = 'sell'
                        elif any(word in text for word in ['purchase', 'buy', 'acquisition', 'option', 'exercise', 'grant', 'award', 'convert']):
                            transaction_type = 'buy'
                        
                        value = None
                        val = row_dict.get('Value')
                        if val is not None and pd.notna(val):
                            try:
                                value = float(val)
                            except:
                                pass
                        
                        shares = None
                        sh = row_dict.get('Shares')
                        if sh is not None and pd.notna(sh):
                            try:
                                shares = int(sh)
                            except:
                                pass
                        
                        insider = 'N/A'
                        ins = row_dict.get('Insider')
                        if ins is not None and pd.notna(ins):
                            insider = str(ins)
                        
                        date_str = 'N/A'
                        date_val = row_dict.get('Start Date')
                        if date_val is not None and pd.notna(date_val):
                            if hasattr(date_val, 'strftime'):
                                date_str = date_val.strftime('%Y-%m-%d')
                            else:
                                date_str = str(date_val)
                        
                        if transaction_type and value and value > 0:
                            transactions.append({
                                'date': date_str,
                                'transaction_type': transaction_type,
                                'value': value,
                                'shares': shares,
                                'insider': insider,
                                'position': 'N/A',
                                'text': str(row_dict.get('Text', ''))
                            })
                    except Exception as row_error:
                        logger.debug(f"Error parsing yfinance row: {str(row_error)}")
                        continue
                
                if transactions:
                    logger.info(f"yfinance API returned {len(transactions)} insider transactions for {ticker}")
                    return transactions
        except Exception as yf_error:
            logger.debug(f"Error getting yfinance insider data: {str(yf_error)}")
        
        logger.warning(f"No insider transactions found for {ticker}")
        return None
            
    except Exception as e:
        logger.exception(f"Error scraping Yahoo Finance insider trading for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


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

