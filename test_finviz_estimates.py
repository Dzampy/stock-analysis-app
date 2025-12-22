#!/usr/bin/env python3
"""Test script to compare Finviz estimates with our results"""

import requests
from bs4 import BeautifulSoup
import re
import yfinance as yf
import pandas as pd

def get_finviz_estimates_scraping(ticker):
    """Scrape estimates directly from Finviz HTML"""
    estimates = {'revenue': {}, 'eps': {}}
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Finviz returned status {response.status_code}")
            return estimates
        
        # Save HTML for inspection
        with open(f'finviz_{ticker}.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved Finviz HTML to finviz_{ticker}.html")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Print all table structures
        all_tables = soup.find_all('table')
        print(f"\n=== Found {len(all_tables)} tables ===")
        
        for table_idx, table in enumerate(all_tables):
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            print(f"\n--- Table {table_idx} ({len(rows)} rows) ---")
            
            # Print first few rows to see structure
            for row_idx, row in enumerate(rows[:10]):
                cells = row.find_all(['td', 'th'])
                cell_texts = [c.get_text(strip=True) for c in cells]
                if len(cell_texts) > 0:
                    print(f"Row {row_idx}: {cell_texts[:8]}")  # First 8 cells
            
            # Look for earnings/estimates keywords
            table_text = table.get_text().upper()
            if 'EARNINGS' in table_text or 'ESTIMATE' in table_text or 'EPS' in table_text or 'REVENUE' in table_text:
                print(f"*** Table {table_idx} contains earnings/estimate keywords ***")
                
                # Try to find quarter headers
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) < 2:
                        continue
                    row_text = ' '.join([c.get_text(strip=True) for c in cells]).upper()
                    if 'Q' in row_text and any(c.isdigit() for c in row_text):
                        print(f"  Found quarter row: {[c.get_text(strip=True) for c in cells[:8]]}")
                
                # Try to find estimate rows
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    first_cell = cells[0].get_text(strip=True).upper()
                    if 'ESTIMATE' in first_cell or 'EPS' in first_cell:
                        print(f"  Found estimate row: {first_cell} -> {[c.get_text(strip=True) for c in cells[1:6]]}")
        
        # Also check snapshot table for EPS next Q
        snapshot_table = soup.find('table', class_='snapshot-table2')
        if snapshot_table:
            print("\n=== Snapshot Table ===")
            rows = snapshot_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if 'EPS' in label.upper() or 'ESTIMATE' in label.upper() or 'EARNINGS' in label.upper():
                        print(f"  {label}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return estimates

def get_yfinance_estimates(ticker):
    """Get estimates from yfinance"""
    estimates = {'revenue': {}, 'eps': {}}
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        
        print(f"\n=== yfinance calendar ===")
        if calendar is not None and not calendar.empty:
            print(f"Calendar shape: {calendar.shape}")
            print(f"Calendar columns: {list(calendar.columns)}")
            print(f"Calendar index: {list(calendar.index)}")
            print(f"\nCalendar data:\n{calendar}")
        else:
            print("Calendar is empty or None")
        
        earnings_dates = stock.earnings_dates
        print(f"\n=== yfinance earnings_dates ===")
        if earnings_dates is not None and not earnings_dates.empty:
            print(f"Earnings dates shape: {earnings_dates.shape}")
            print(f"Earnings dates columns: {list(earnings_dates.columns)}")
            print(f"Earnings dates index: {list(earnings_dates.index)}")
            print(f"\nEarnings dates data:\n{earnings_dates}")
        else:
            print("Earnings dates is empty or None")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return estimates

if __name__ == '__main__':
    ticker = 'CIFR'
    print(f"Testing estimates for {ticker}\n")
    print("=" * 60)
    
    print("\n1. Finviz scraping:")
    finviz_estimates = get_finviz_estimates_scraping(ticker)
    
    print("\n2. yfinance:")
    yfinance_estimates = get_yfinance_estimates(ticker)
    
    print("\n" + "=" * 60)
    print("Comparison complete. Check finviz_CIFR.html for full HTML structure.")











