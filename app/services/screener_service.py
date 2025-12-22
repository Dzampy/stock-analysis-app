"""
Stock screener service - filters stocks based on various criteria
"""
import yfinance as yf
import time
from ta.momentum import RSIIndicator
from app.services.finviz_service import get_short_interest_from_finviz


def get_popular_tickers():
    """Get list of popular tickers for screener"""
    popular_tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
        'CRM', 'ORCL', 'ADBE', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'COF',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD',
        # Consumer
        'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'ROST', 'DG',
        # Industrial
        'BA', 'CAT', 'GE', 'HON', 'RTX', 'LMT', 'NOC', 'GD', 'TDG', 'PH',
        # Energy
        'XOM', 'CVX', 'SLB', 'EOG', 'COP', 'MPC', 'VLO', 'PSX', 'HAL', 'OXY',
        # Materials
        'LIN', 'APD', 'ECL', 'SHW', 'DD', 'PPG', 'FCX', 'NEM', 'VALE', 'RIO',
        # Utilities
        'NEE', 'DUK', 'SO', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ES', 'PEG',
        # Real Estate
        'AMT', 'PLD', 'EQIX', 'PSA', 'WELL', 'SPG', 'O', 'DLR', 'VICI', 'EXPI',
        # Communication
        'VZ', 'T', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'TMUS', 'FOXA', 'LBRDK', 'DISH',
        # Small/Mid Cap Growth
        'ROKU', 'ZM', 'DOCU', 'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR', 'RBLX', 'HOOD',
        'COIN', 'SQ', 'PYPL', 'SHOP', 'ETSY', 'MELI', 'SE', 'GRAB', 'U', 'AFRM',
        # Biotech
        'MRNA', 'BNTX', 'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'ALNY', 'FOLD', 'IONS',
        # Semiconductors
        'NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'MRVL', 'ON', 'SWKS', 'CRUS',
        # Cloud/SaaS
        'CRM', 'NOW', 'WDAY', 'ZS', 'OKTA', 'FTNT', 'PANW', 'QLYS', 'TENB', 'VRRM',
        # E-commerce
        'AMZN', 'SHOP', 'ETSY', 'MELI', 'SE', 'JD', 'PDD', 'BABA', 'VIPS', 'DADA',
        # Fintech
        'SQ', 'PYPL', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'NU', 'PAG',
        # EV/Auto
        'TSLA', 'RIVN', 'LCID', 'F', 'GM', 'FORD', 'HMC', 'TM', 'NIO', 'XPEV',
        # Energy Transition
        'ENPH', 'SEDG', 'RUN', 'ARRY', 'NOVA', 'CSIQ', 'SPWR', 'FSLR', 'PLUG', 'BE',
        # Data Centers
        'CIFR', 'IREN', 'NBIS', 'CORZ', 'BITF', 'HUT', 'MARA', 'RIOT', 'ARBK', 'WULF'
    ]
    return popular_tickers


def run_stock_screener(filters):
    """Run stock screener with given filters"""
    try:
        tickers = get_popular_tickers()
        results = []
        
        # Extract filters
        market_cap_min = filters.get('market_cap_min')
        market_cap_max = filters.get('market_cap_max')
        pe_min = filters.get('pe_min')
        pe_max = filters.get('pe_max')
        pb_min = filters.get('pb_min')
        pb_max = filters.get('pb_max')
        ps_min = filters.get('ps_min')
        ps_max = filters.get('ps_max')
        dividend_yield_min = filters.get('dividend_yield_min')
        dividend_yield_max = filters.get('dividend_yield_max')
        revenue_growth_min = filters.get('revenue_growth_min')
        revenue_growth_max = filters.get('revenue_growth_max')
        eps_growth_min = filters.get('eps_growth_min')
        eps_growth_max = filters.get('eps_growth_max')
        roe_min = filters.get('roe_min')
        roe_max = filters.get('roe_max')
        debt_equity_min = filters.get('debt_equity_min')
        debt_equity_max = filters.get('debt_equity_max')
        beta_min = filters.get('beta_min')
        beta_max = filters.get('beta_max')
        rsi_min = filters.get('rsi_min')
        rsi_max = filters.get('rsi_max')
        short_float_min = filters.get('short_float_min')
        short_float_max = filters.get('short_float_max')
        volume_min = filters.get('volume_min')
        volume_max = filters.get('volume_max')
        price_min = filters.get('price_min')
        price_max = filters.get('price_max')
        sector = filters.get('sector')
        industry = filters.get('industry')
        
        print(f"[SCREENER] Screening {len(tickers)} tickers with filters: {filters}")
        
        # Process tickers in batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            
            for ticker in batch:
                try:
                    time.sleep(0.2)  # Rate limiting
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    if not info or 'symbol' not in info:
                        continue
                    
                    # Get historical data for RSI calculation
                    hist = stock.history(period='6mo')
                    if hist.empty:
                        continue
                    
                    # Calculate RSI
                    rsi_value = None
                    try:
                        rsi_indicator = RSIIndicator(hist['Close'], window=14)
                        rsi_series = rsi_indicator.rsi()
                        if not rsi_series.empty:
                            rsi_value = float(rsi_series.iloc[-1])
                    except:
                        pass
                    
                    # Extract metrics
                    market_cap = info.get('marketCap')
                    pe_ratio = info.get('trailingPE')
                    pb_ratio = info.get('priceToBook')
                    ps_ratio = info.get('priceToSalesTrailing12Months')
                    dividend_yield = info.get('dividendYield')
                    if dividend_yield:
                        dividend_yield = dividend_yield * 100  # Convert to percentage
                    revenue_growth = info.get('revenueGrowth')
                    if revenue_growth:
                        revenue_growth = revenue_growth * 100  # Convert to percentage
                    eps_growth = info.get('earningsGrowth')
                    if eps_growth:
                        eps_growth = eps_growth * 100  # Convert to percentage
                    roe = info.get('returnOnEquity')
                    if roe:
                        roe = roe * 100  # Convert to percentage
                    debt_equity = info.get('debtToEquity')
                    beta = info.get('beta')
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    avg_volume = info.get('averageVolume')
                    sector_info = info.get('sector')
                    industry_info = info.get('industry')
                    
                    # Apply filters
                    if market_cap_min and market_cap and market_cap < market_cap_min:
                        continue
                    if market_cap_max and market_cap and market_cap > market_cap_max:
                        continue
                    if pe_min and pe_ratio and pe_ratio < pe_min:
                        continue
                    if pe_max and pe_ratio and pe_ratio > pe_max:
                        continue
                    if pb_min and pb_ratio and pb_ratio < pb_min:
                        continue
                    if pb_max and pb_ratio and pb_ratio > pb_max:
                        continue
                    if ps_min and ps_ratio and ps_ratio < ps_min:
                        continue
                    if ps_max and ps_ratio and ps_ratio > ps_max:
                        continue
                    if dividend_yield_min and dividend_yield and dividend_yield < dividend_yield_min:
                        continue
                    if dividend_yield_max and dividend_yield and dividend_yield > dividend_yield_max:
                        continue
                    if revenue_growth_min and revenue_growth and revenue_growth < revenue_growth_min:
                        continue
                    if revenue_growth_max and revenue_growth and revenue_growth > revenue_growth_max:
                        continue
                    if eps_growth_min and eps_growth and eps_growth < eps_growth_min:
                        continue
                    if eps_growth_max and eps_growth and eps_growth > eps_growth_max:
                        continue
                    if roe_min and roe and roe < roe_min:
                        continue
                    if roe_max and roe and roe > roe_max:
                        continue
                    if debt_equity_min and debt_equity and debt_equity < debt_equity_min:
                        continue
                    if debt_equity_max and debt_equity and debt_equity > debt_equity_max:
                        continue
                    if beta_min and beta and beta < beta_min:
                        continue
                    if beta_max and beta and beta > beta_max:
                        continue
                    if rsi_min and rsi_value and rsi_value < rsi_min:
                        continue
                    if rsi_max and rsi_value and rsi_value > rsi_max:
                        continue
                    if volume_min and avg_volume and avg_volume < volume_min:
                        continue
                    if volume_max and avg_volume and avg_volume > volume_max:
                        continue
                    if price_min and current_price and current_price < price_min:
                        continue
                    if price_max and current_price and current_price > price_max:
                        continue
                    if sector and sector_info and sector.lower() not in sector_info.lower():
                        continue
                    if industry and industry_info and industry.lower() not in industry_info.lower():
                        continue
                    
                    # Get short float if available
                    short_float_pct = None
                    try:
                        short_data = get_short_interest_from_finviz(ticker)
                        if short_data and short_data.get('short_float_pct'):
                            short_float_pct = short_data['short_float_pct']
                    except:
                        pass
                    
                    # Apply short float filter
                    if short_float_min and short_float_pct and short_float_pct < short_float_min:
                        continue
                    if short_float_max and short_float_pct and short_float_pct > short_float_max:
                        continue
                    
                    # Stock passes all filters
                    results.append({
                        'ticker': ticker,
                        'name': info.get('longName') or info.get('shortName', 'N/A'),
                        'sector': sector_info or 'N/A',
                        'industry': industry_info or 'N/A',
                        'market_cap': market_cap,
                        'price': current_price,
                        'pe_ratio': pe_ratio,
                        'pb_ratio': pb_ratio,
                        'ps_ratio': ps_ratio,
                        'dividend_yield': dividend_yield,
                        'revenue_growth': revenue_growth,
                        'eps_growth': eps_growth,
                        'roe': roe,
                        'debt_equity': debt_equity,
                        'beta': beta,
                        'rsi': rsi_value,
                        'short_float_pct': short_float_pct,
                        'volume': avg_volume,
                        'volume_24h': info.get('volume') or avg_volume
                    })
                    
                except Exception as e:
                    print(f"[SCREENER] Error processing {ticker}: {str(e)}")
                    continue
            
            # Small delay between batches
            if i + batch_size < len(tickers):
                time.sleep(0.5)
        
        print(f"[SCREENER] Found {len(results)} stocks matching filters")
        return results
        
    except Exception as e:
        print(f"Error in stock screener: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

