"""
Finviz scraping service - Quarterly estimates, short interest, analyst ratings, insider trading
"""
import re
import time
import json
from datetime import datetime
from typing import Dict, Optional, List
import requests
from bs4 import BeautifulSoup
import yfinance as yf

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

from app.utils.constants import FINVIZ_TIMEOUT, RATE_LIMIT_DELAY
from app.utils.logger import logger


def get_quarterly_estimates_from_finviz(ticker: str) -> Dict:
    """
    Get quarterly revenue and EPS estimates AND actual values from Finviz - PARSING JSON DATA FROM HTML
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with 'estimates' and 'actuals' keys, each containing 'revenue' and 'eps' dicts
    """
    estimates = {'revenue': {}, 'eps': {}}
    actuals = {'revenue': {}, 'eps': {}}  # Store actual reported values
    
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # Try using cloudscraper first (for Cloudflare protection on Render)
        try:
            if CLOUDSCRAPER_AVAILABLE:
                scraper = cloudscraper.create_scraper()
                response = scraper.get(url, headers=headers, timeout=FINVIZ_TIMEOUT)
            else:
                response = requests.get(url, headers=headers, timeout=FINVIZ_TIMEOUT)
        except requests.exceptions.Timeout:
            logger.warning(f" Finviz request timeout for {ticker}")
            return {'estimates': estimates, 'actuals': actuals}
        except requests.exceptions.RequestException as e:
            logger.warning(f" Finviz request failed for {ticker}: {str(e)}")
            return {'estimates': estimates, 'actuals': actuals}
        except Exception as e:
            # If cloudscraper fails, try regular requests
            try:
                response = requests.get(url, headers=headers, timeout=FINVIZ_TIMEOUT)
            except requests.exceptions.Timeout:
                logger.warning(f" Finviz request timeout for {ticker}")
                return {'estimates': estimates, 'actuals': actuals}
            except requests.exceptions.RequestException as req_e:
                logger.warning(f" Finviz request failed for {ticker}: {str(req_e)}")
                return {'estimates': estimates, 'actuals': actuals}
        
        if response.status_code != 200:
            logger.warning(f" Finviz returned status {response.status_code} for {ticker}")
            return {'estimates': estimates, 'actuals': actuals}
        
        # Finviz stores estimates in JSON data embedded in HTML
        # Look for var data = {...} with chartEvents containing earnings data
        html_text = response.text
        
        # Find JSON data object
        json_match = re.search(r'var\s+data\s*=\s*({[^;]+});', html_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                data = json.loads(json_str)
                # Extract chartEvents which contain earnings estimates
                if 'chartEvents' in data:
                    earnings_events = [e for e in data['chartEvents'] if e.get('eventType') == 'chartEvent/earnings']
                    for event in data['chartEvents']:
                        if event.get('eventType') == 'chartEvent/earnings':
                            fiscal_period = event.get('fiscalPeriod', '')
                            fiscal_end_date_ts = event.get('fiscalEndDate')
                            # Parse fiscal period (e.g., "2025Q3" -> "2025-Q3")
                            period_match = re.search(r'(\d{4})Q(\d)', fiscal_period)
                            if period_match:
                                year = period_match.group(1)
                                quarter = period_match.group(2)
                                
                                # Convert fiscal quarter to calendar quarter using fiscalEndDate
                                if fiscal_end_date_ts:
                                    fiscal_end_date = datetime.fromtimestamp(fiscal_end_date_ts)
                                    calendar_quarter = (fiscal_end_date.month - 1) // 3 + 1
                                    calendar_year = fiscal_end_date.year
                                    quarter_str = f"{calendar_year}-Q{calendar_quarter}"
                                else:
                                    quarter_str = f"{year}-Q{quarter}"
                                
                                # Get EPS estimate
                                eps_est = event.get('epsEstimate')
                                if eps_est is not None and eps_est != 0:
                                    estimates['eps'][quarter_str] = float(eps_est)
                                
                                # Get revenue estimate - Finviz returns in millions, convert to dollars
                                revenue_est = event.get('salesEstimate')
                                if revenue_est is None or revenue_est == 0:
                                    revenue_est_keys = ['revenueEstimate', 'sales', 'revenue', 'estimatedRevenue', 'estimatedSales']
                                    for key in revenue_est_keys:
                                        if key in event and event[key] is not None:
                                            try:
                                                val = float(event[key])
                                                if val != 0:
                                                    revenue_est = val
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                
                                if revenue_est is not None and revenue_est != 0:
                                    revenue_est_dollars = float(revenue_est) * 1_000_000
                                    estimates['revenue'][quarter_str] = revenue_est_dollars
                                
                                # Get actual EPS
                                eps_actual = None
                                priority_keys = ['epsActual', 'reportedEPS', 'epsReported', 'actualEPS', 'epsActualValue', 'actualEpsValue']
                                fallback_keys = ['eps', 'reportedEps']
                                
                                for key in priority_keys:
                                    if key in event and event[key] is not None:
                                        try:
                                            val = float(event[key])
                                            eps_actual = val
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                
                                if eps_actual is None:
                                    for key in fallback_keys:
                                        if key in event and event[key] is not None:
                                            try:
                                                val = float(event[key])
                                                if val != 0:
                                                    eps_actual = val
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                
                                if eps_actual is None and fiscal_end_date_ts:
                                    fiscal_end_date = datetime.fromtimestamp(fiscal_end_date_ts)
                                    if fiscal_end_date < datetime.now():
                                        eps_est = event.get('epsEstimate')
                                        if eps_est is not None and eps_est != 0:
                                            eps_actual = float(eps_est)
                                
                                if eps_actual is not None:
                                    actuals['eps'][quarter_str] = eps_actual
                                
                                # Get actual revenue - same pattern as EPS
                                revenue_actual = None
                                revenue_priority_keys = [
                                    'revenueActual', 'salesActual', 'reportedRevenue', 'actualRevenue', 
                                    'revenueActualValue', 'actualRevenueValue', 'reportedSales', 
                                    'salesReported', 'actualSales', 'salesActualValue'
                                ]
                                revenue_fallback_keys = ['sales', 'revenue']
                                
                                for key in revenue_priority_keys:
                                    if key in event and event[key] is not None:
                                        try:
                                            val = float(event[key])
                                            revenue_actual = val * 1_000_000  # Convert millions to dollars
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                
                                if revenue_actual is None:
                                    for key in revenue_fallback_keys:
                                        if key in event and event[key] is not None:
                                            try:
                                                val = float(event[key])
                                                if val != 0:
                                                    if val < 1e12:
                                                        revenue_actual = val * 1_000_000
                                                    else:
                                                        revenue_actual = val
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                
                                if revenue_actual is None and fiscal_end_date_ts:
                                    fiscal_end_date = datetime.fromtimestamp(fiscal_end_date_ts)
                                    if fiscal_end_date < datetime.now():
                                        revenue_est_for_actual = None
                                        estimate_keys_to_try = ['salesEstimate', 'revenueEstimate', 'sales', 'revenue']
                                        for est_key in estimate_keys_to_try:
                                            if est_key in event and event[est_key] is not None:
                                                try:
                                                    val = float(event[est_key])
                                                    if val != 0:
                                                        revenue_est_for_actual = val
                                                        break
                                                except (ValueError, TypeError):
                                                    continue
                                        if revenue_est_for_actual is not None:
                                            revenue_actual = float(revenue_est_for_actual) * 1_000_000
                                
                                if revenue_actual is not None:
                                    actuals['revenue'][quarter_str] = revenue_actual
                                
            except json.JSONDecodeError as e:
                logger.exception(f"Error parsing Finviz JSON data: {str(e)}")
        
        # Fallback: Also check snapshot table for EPS next Q
        if not estimates['eps']:
            soup = BeautifulSoup(html_text, 'html.parser')
            snapshot_table = soup.find('table', class_='snapshot-table2')
            if snapshot_table:
                rows = snapshot_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).upper()
                        value = cells[1].get_text(strip=True)
                        
                        if 'EPS NEXT Q' in label:
                            try:
                                eps_val = float(value.replace(',', ''))
                                now = datetime.now()
                                next_quarter = (now.month - 1) // 3 + 1
                                next_year = now.year
                                if next_quarter == 4:
                                    next_quarter = 1
                                    next_year += 1
                                else:
                                    next_quarter += 1
                                quarter_str = f"{next_year}-Q{next_quarter}"
                                estimates['eps'][quarter_str] = eps_val
                            except (ValueError, TypeError):
                                pass
        
        if estimates['revenue'] or estimates['eps']:
            rev_count = len(estimates['revenue'])
            eps_count = len(estimates['eps'])
            logger.info(f"Finviz: Found {rev_count} revenue and {eps_count} EPS estimates for {ticker}")
        
        if actuals['revenue'] or actuals['eps']:
            rev_actual_count = len(actuals['revenue'])
            eps_actual_count = len(actuals['eps'])
            logger.info(f"Finviz: Found {rev_actual_count} revenue and {eps_actual_count} EPS actual values for {ticker}")
        
    except Exception as e:
        logger.exception(f"Error fetching estimates from Finviz for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return {
        'estimates': estimates,
        'actuals': actuals
    }


def get_short_interest_from_finviz(ticker: str) -> Optional[Dict]:
    """
    Get short interest data from Finviz (Short Float %, Short Ratio, Short Interest)
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with short interest data or None
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.warning(f"Finviz returned status {response.status_code} for {ticker} short interest")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the snapshot table
        snapshot_table = soup.find('table', class_='snapshot-table2')
        if not snapshot_table:
            snapshot_table = soup.find('table', class_='screener_snapshot-table-body')
        if not snapshot_table:
            all_tables = soup.find_all('table')
            for table in all_tables:
                classes = table.get('class', [])
                if any('snapshot' in str(c).lower() for c in classes):
                    snapshot_table = table
                    break
        
        if not snapshot_table:
            logger.info(f"[SHORT INTEREST] No snapshot table found for {ticker}")
            return None
        
        short_interest_data = {}
        
        # Parse table rows
        rows = snapshot_table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True)
                value_cell = cells[1]
                
                b_tag = value_cell.find('b')
                value_text = b_tag.get_text(strip=True) if b_tag else value_cell.get_text(strip=True)
                value_text = value_text.replace('\xa0', ' ').strip()
                
                # Extract Short Float %
                if 'Short Float' in label:
                    try:
                        short_float_pct = float(value_text.replace('%', '').replace(',', ''))
                        short_interest_data['short_float_pct'] = short_float_pct
                    except (ValueError, AttributeError):
                        pass
                
                # Extract Short Ratio
                elif 'Short Ratio' in label:
                    try:
                        short_ratio = float(value_text.replace(',', ''))
                        short_interest_data['short_ratio'] = short_ratio
                    except (ValueError, AttributeError):
                        pass
                
                # Extract Short Interest (absolute number)
                elif 'Short Interest' in label and 'Short Float' not in label and 'Short Ratio' not in label:
                    try:
                        value_clean = value_text.replace(',', '').replace(' ', '')
                        if 'M' in value_clean:
                            short_interest = float(value_clean.replace('M', '')) * 1_000_000
                        elif 'B' in value_clean:
                            short_interest = float(value_clean.replace('B', '')) * 1_000_000_000
                        elif 'K' in value_clean:
                            short_interest = float(value_clean.replace('K', '')) * 1_000
                        else:
                            short_interest = float(value_clean)
                        short_interest_data['short_interest'] = short_interest
                    except (ValueError, AttributeError):
                        pass
        
        # Fallback to yfinance
        if not short_interest_data:
            try:
                stock = yf.Ticker(ticker)
                time.sleep(RATE_LIMIT_DELAY)
                info = stock.info
                
                shares_short = info.get('sharesShort')
                shares_outstanding = info.get('sharesOutstanding')
                float_shares = info.get('floatShares') or shares_outstanding
                
                if shares_short and float_shares:
                    short_float_pct = (shares_short / float_shares) * 100
                    short_interest_data['short_float_pct'] = round(short_float_pct, 2)
                    short_interest_data['short_interest'] = shares_short
                    
                    avg_volume = info.get('averageVolume')
                    if avg_volume and avg_volume > 0:
                        short_ratio = shares_short / avg_volume
                        short_interest_data['short_ratio'] = round(short_ratio, 2)
            except Exception as e:
                logger.info(f"[SHORT INTEREST] yfinance fallback failed for {ticker}: {e}")
        
        if not short_interest_data:
            return None
        
        # Calculate short squeeze score (0-100)
        squeeze_score = 0
        
        if 'short_float_pct' in short_interest_data:
            short_float = short_interest_data['short_float_pct']
            if short_float > 20:
                squeeze_score += 40
            elif short_float > 10:
                squeeze_score += 25
            elif short_float > 5:
                squeeze_score += 10
        
        if 'short_ratio' in short_interest_data:
            short_ratio = short_interest_data['short_ratio']
            if short_ratio > 5:
                squeeze_score += 30
            elif short_ratio > 3:
                squeeze_score += 20
            elif short_ratio > 1:
                squeeze_score += 10
        
        short_interest_data['squeeze_score'] = min(100, squeeze_score)
        
        # Try to get historical short interest data from Finviz chart
        try:
            json_match = re.search(r'var\s+data\s*=\s*({[^;]+});', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    finviz_data = json.loads(json_str)
                    if 'chartEvents' in finviz_data:
                        short_history = []
                        for event in finviz_data['chartEvents']:
                            if event.get('eventType') == 'chartEvent/shortInterest':
                                date_ts = event.get('date')
                                short_val = event.get('shortInterest') or event.get('value')
                                if date_ts and short_val:
                                    date_str = datetime.fromtimestamp(date_ts).strftime('%Y-%m-%d')
                                    short_history.append({
                                        'date': date_str,
                                        'short_interest': float(short_val),
                                        'short_float_pct': event.get('shortFloatPct')
                                    })
                        
                        if short_history:
                            short_history.sort(key=lambda x: x['date'])
                            short_interest_data['history'] = short_history
                except:
                    pass
        except:
            pass
        
        return short_interest_data
    
    except Exception as e:
        logger.exception(f"Error fetching short interest for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_short_interest_history(ticker: str) -> Optional[List[Dict]]:
    """
    Get historical short interest data from MarketBeat (reported every 15 days)
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of historical short interest data points or None
    """
    try:
        if not CLOUDSCRAPER_AVAILABLE:
            logger.warning("cloudscraper not installed. Install with: pip install cloudscraper")
            return None
        
        import csv
        from io import StringIO
        
        # Determine exchange for MarketBeat URL
        stock = yf.Ticker(ticker)
        time.sleep(RATE_LIMIT_DELAY)
        info = stock.info
        exchange = info.get('exchange', 'NASDAQ').upper()
        
        url = f'https://www.marketbeat.com/stocks/{exchange}/{ticker.upper()}/short-interest/'
        
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=30)
        
        if response.status_code != 200:
            logger.info(f"[SHORT INTEREST HISTORY] MarketBeat returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        
        history_data = []
        
        # Parse shortInterestSeries
        for script in scripts:
            if not script.string:
                continue
            
            # Extract shortInterestSeries CSV
            match = re.search(r'var shortInterestSeries = "([^"]+)"', script.string)
            if match:
                csv_data = match.group(1)
                csv_data = csv_data.replace('\\n', '\n').replace('\\,', ',')
                try:
                    reader = csv.DictReader(StringIO(csv_data))
                    for row in reader:
                        date_str = row.get('Date', '')
                        amount_str = row.get('Amount', '').replace(',', '').replace('$', '').strip()
                        
                        if date_str and amount_str:
                            try:
                                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                date_formatted = date_obj.strftime('%Y-%m-%d')
                                amount = float(amount_str)
                                
                                history_data.append({
                                    'date': date_formatted,
                                    'short_interest': int(amount),
                                    'short_float_pct': None,
                                    'short_ratio': None
                                })
                            except (ValueError, TypeError):
                                continue
                except Exception as e:
                    logger.info(f"[SHORT INTEREST HISTORY] Error parsing shortInterestSeries: {e}")
            
            # Extract shortInterestFloatSeries
            match = re.search(r'var shortInterestFloatSeries = "([^"]+)"', script.string)
            if match:
                csv_data = match.group(1)
                csv_data = csv_data.replace('\\n', '\n').replace('\\,', ',')
                try:
                    reader = csv.DictReader(StringIO(csv_data))
                    float_data = {}
                    for row in reader:
                        date_str = row.get('Date', '')
                        percent_str = row.get('Percent', '').strip()
                        
                        if date_str and percent_str:
                            try:
                                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                date_formatted = date_obj.strftime('%Y-%m-%d')
                                percent = float(percent_str) * 100
                                float_data[date_formatted] = round(percent, 2)
                            except (ValueError, TypeError):
                                continue
                    
                    for entry in history_data:
                        if entry['date'] in float_data:
                            entry['short_float_pct'] = float_data[entry['date']]
                except Exception as e:
                    logger.info(f"[SHORT INTEREST HISTORY] Error parsing shortInterestFloatSeries: {e}")
            
            # Extract shortInterestRatioSeries
            match = re.search(r'var shortInterestRatioSeries = "([^"]+)"', script.string)
            if match:
                csv_data = match.group(1)
                csv_data = csv_data.replace('\\n', '\n').replace('\\,', ',')
                try:
                    reader = csv.DictReader(StringIO(csv_data))
                    ratio_data = {}
                    for row in reader:
                        date_str = row.get('Date', '')
                        ratio_str = row.get('Ratio', '').strip()
                        
                        if date_str and ratio_str:
                            try:
                                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                date_formatted = date_obj.strftime('%Y-%m-%d')
                                ratio = float(ratio_str)
                                ratio_data[date_formatted] = round(ratio, 2)
                            except (ValueError, TypeError):
                                continue
                    
                    for entry in history_data:
                        if entry['date'] in ratio_data:
                            entry['short_ratio'] = ratio_data[entry['date']]
                except Exception as e:
                    logger.info(f"[SHORT INTEREST HISTORY] Error parsing shortInterestRatioSeries: {e}")
        
        history_data.sort(key=lambda x: x['date'])
        return history_data[-30:] if history_data else None
    
    except Exception as e:
        logger.exception(f"Error fetching short interest history from MarketBeat for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_finviz_analyst_ratings(ticker: str) -> Optional[List[Dict]]:
    """
    Scrape individual analyst ratings from Finviz
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of analyst recommendations or None
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.warning(f"Finviz returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        recommendations = []
        
        all_tables = soup.find_all('table')
        
        for table in all_tables:
            rows = table.find_all('tr')
            if len(rows) > 1:
                header_row = rows[0]
                header_cells = header_row.find_all(['td', 'th'])
                header_texts = [cell.get_text(strip=True).lower() for cell in header_cells]
                
                if 'date' in header_texts and 'analyst' in header_texts and ('action' in header_texts or 'rating' in header_texts) and len(header_cells) == 5:
                    if len(rows) > 1:
                        first_data_row = rows[1]
                        first_data_cells = first_data_row.find_all(['td', 'th'])
                        if len(first_data_cells) == 5:
                            first_row_text = ' '.join([cell.get_text(strip=True).lower() for cell in first_data_cells])
                            if any(word in first_row_text for word in ['upgrade', 'downgrade', 'initiate', 'maintain', 'reiterate', 'perform', 'outperform', 'underperform']):
                                for row in rows[1:31]:
                                    try:
                                        cells = row.find_all(['td', 'th'])
                                        if len(cells) < 3:
                                            continue
                                        
                                        date_str = cells[0].get_text(strip=True) if len(cells) > 0 else 'N/A'
                                        action = cells[1].get_text(strip=True) if len(cells) > 1 else 'N/A'
                                        firm = cells[2].get_text(strip=True) if len(cells) > 2 else 'N/A'
                                        rating_change = cells[3].get_text(strip=True) if len(cells) > 3 else 'N/A'
                                        target_change = cells[4].get_text(strip=True) if len(cells) > 4 else ''
                                        
                                        rating = 'N/A'
                                        if '→' in rating_change:
                                            parts = rating_change.split('→')
                                            if len(parts) > 1:
                                                rating = parts[1].strip()
                                        elif rating_change and rating_change != 'N/A':
                                            rating = rating_change
                                        
                                        target_price = None
                                        if target_change and '$' in target_change:
                                            try:
                                                match = re.search(r'\$(\d+\.?\d*)', target_change)
                                                if match:
                                                    target_price = float(match.group(1))
                                            except:
                                                pass
                                        
                                        if firm != 'N/A' and firm and rating != 'N/A' and rating:
                                            recommendations.append({
                                                'date': date_str,
                                                'firm': firm,
                                                'to_grade': rating,
                                                'from_grade': rating_change.split('→')[0].strip() if '→' in rating_change else 'N/A',
                                                'target_price': target_price
                                            })
                                    except Exception as e:
                                        logger.exception(f"Error parsing Finviz analyst row: {str(e)}")
                                        continue
                                
                                if recommendations:
                                    break
        
        return recommendations if recommendations else None
        
    except Exception as e:
        logger.exception(f"Error scraping Finviz analyst ratings for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_finviz_insider_trading(ticker: str) -> Optional[List[Dict]]:
    """
    Scrape insider trading data from Finviz
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of insider transactions or None
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.warning(f"Finviz returned status {response.status_code} for {ticker}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        transactions = []
        
        all_tables = soup.find_all('table')
        insider_table = None
        
        for table in all_tables:
            rows = table.find_all('tr')
            if len(rows) > 1:
                header_row = rows[0]
                header_cells = header_row.find_all(['td', 'th'])
                header_text = header_row.get_text().lower()
                
                if 'insider trading' in header_text and 'relationship' in header_text and 'date' in header_text and 'transaction' in header_text:
                    if 7 <= len(header_cells) <= 10:
                        insider_table = table
                        break
        
        if insider_table:
            rows = insider_table.find_all('tr')
            for row_idx, row in enumerate(rows[1:31]):
                try:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 6:
                        continue
                    
                    insider_name = cells[0].get_text(strip=True) if len(cells) > 0 else 'N/A'
                    position = cells[1].get_text(strip=True) if len(cells) > 1 else 'N/A'
                    date_str = cells[2].get_text(strip=True) if len(cells) > 2 else 'N/A'
                    transaction_text = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                    cost_per_share_text = cells[4].get_text(strip=True) if len(cells) > 4 else ''
                    shares_text = cells[5].get_text(strip=True) if len(cells) > 5 else ''
                    value_text = cells[6].get_text(strip=True) if len(cells) > 6 else ''
                    
                    transaction_type = None
                    trans_lower = transaction_text.lower()
                    if 'sale' in trans_lower or 'sell' in trans_lower:
                        transaction_type = 'sell'
                    elif ('purchase' in trans_lower or 'buy' in trans_lower or 'acquisition' in trans_lower or 
                          'option exercise' in trans_lower or 'exercise' in trans_lower or
                          'grant' in trans_lower or 'award' in trans_lower or
                          'conversion' in trans_lower or 'convert' in trans_lower):
                        transaction_type = 'buy'
                    
                    shares = None
                    if shares_text:
                        try:
                            clean_shares = shares_text.replace(',', '').replace(' ', '')
                            if clean_shares:
                                shares = int(float(clean_shares))
                        except:
                            pass
                    
                    value = None
                    if value_text:
                        try:
                            clean_value = value_text.replace('$', '').replace(',', '').replace(' ', '')
                            if clean_value:
                                value = float(clean_value)
                        except:
                            pass
                    
                    if value is None and cost_per_share_text and shares:
                        try:
                            cost_per_share = float(cost_per_share_text.replace(',', '').replace(' ', ''))
                            value = cost_per_share * shares
                        except:
                            pass
                    
                    if transaction_type and value and value > 0:
                        transactions.append({
                            'date': date_str,
                            'transaction_type': transaction_type,
                            'value': value,
                            'shares': shares,
                            'insider': insider_name,
                            'position': position,
                            'text': transaction_text
                        })
                except Exception as e:
                    logger.exception(f"Error parsing Finviz row: {str(e)}")
                    continue
        
        return transactions if transactions else None
        
    except Exception as e:
        logger.exception(f"Error scraping Finviz for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

