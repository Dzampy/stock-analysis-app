"""
News service - News aggregation, analyst ratings, insider trading from various sources
"""
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import time
from app.utils.constants import RATE_LIMIT_DELAY
from app.services.sentiment_service import analyze_sentiment
from app.utils.logger import logger


def normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize date string to YYYY-MM-DD HH:MM format
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Normalized date string or None
    """
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return None
    
    try:
        # Try parsing RFC 2822 format (from RSS feeds) FIRST
        try:
            import email.utils
            parsed_time = email.utils.parsedate_tz(date_str)
            if parsed_time:
                timestamp = email.utils.mktime_tz(parsed_time)
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime('%Y-%m-%d %H:%M')
        except Exception as e:
            pass
        
        # Try parsing ISO format
        if 'T' in date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        # Try common date formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%d.%m.%Y %H:%M', '%d.%m.%Y', '%B %d, %Y', '%b %d, %Y']:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d %H:%M')
            except:
                continue
        
        return None
    except Exception as e:
        logger.warning(f"Error normalizing date '{date_str}': {str(e)}")
        return None


def classify_news_type(title: str, summary: str) -> str:
    """
    Classify news type based on keywords
    
    Args:
        title: News title
        summary: News summary
        
    Returns:
        News type string
    """
    text = f"{title} {summary}".lower()
    
    # Earnings related
    if any(keyword in text for keyword in ['earnings', 'eps', 'revenue', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'guidance', 'beat', 'miss']):
        return 'earnings'
    
    # Product launch
    if any(keyword in text for keyword in ['launch', 'release', 'unveil', 'introduce', 'announce', 'new product', 'debut']):
        return 'product_launch'
    
    # Regulatory
    if any(keyword in text for keyword in ['fda', 'approval', 'regulatory', 'sec', 'investigation', 'lawsuit', 'settlement', 'compliance']):
        return 'regulatory'
    
    # M&A
    if any(keyword in text for keyword in ['acquisition', 'merger', 'takeover', 'buyout', 'deal', 'purchase']):
        return 'm_a'
    
    # Management
    if any(keyword in text for keyword in ['ceo', 'cfo', 'executive', 'resign', 'appoint', 'hire', 'departure']):
        return 'management'
    
    # Financial
    if any(keyword in text for keyword in ['dividend', 'buyback', 'split', 'offering', 'debt', 'credit', 'rating']):
        return 'financial'
    
    # Partnership
    if any(keyword in text for keyword in ['partnership', 'collaboration', 'joint venture', 'alliance', 'agreement']):
        return 'partnership'
    
    # Guidance/Outlook
    if any(keyword in text for keyword in ['outlook', 'forecast', 'guidance', 'expect', 'projection']):
        return 'guidance'
    
    return 'other'


def calculate_news_price_impact(ticker: str, news_date_str: str, news_title: str, news_summary: str) -> Optional[Dict]:
    """
    Calculate actual price movement after news (1h, 1d, 1w)
    
    Args:
        ticker: Stock ticker
        news_date_str: News date string
        news_title: News title
        news_summary: News summary
        
    Returns:
        Dict with price impact percentages or None
    """
    try:
        # Parse news date
        try:
            if 'T' in news_date_str:
                news_date = datetime.fromisoformat(news_date_str.replace('Z', '+00:00'))
            else:
                news_date = datetime.strptime(news_date_str, '%Y-%m-%d %H:%M')
        except:
            try:
                news_date = datetime.strptime(news_date_str, '%Y-%m-%d')
            except:
                return None
        
        # Get historical data
        stock = yf.Ticker(ticker.upper())
        time.sleep(0.2)
        
        price_1h = None
        price_1d = None
        price_1w = None
        
        try:
            # Try to get intraday data for 1h impact
            days_since_news = (datetime.now() - news_date).days
            if days_since_news <= 60:
                try:
                    hist_intraday = stock.history(period='1mo', interval='1h', auto_adjust=True, prepost=False)
                    
                    if not hist_intraday.empty and len(hist_intraday) > 0:
                        news_date_tz = news_date.replace(tzinfo=None) if news_date.tzinfo else news_date
                        hist_intraday_sorted = hist_intraday.sort_index()
                        
                        # Find closest time to news
                        time_diffs = []
                        for idx in hist_intraday_sorted.index:
                            idx_naive = idx.replace(tzinfo=None) if idx.tzinfo else idx
                            diff = abs((idx_naive - news_date_tz).total_seconds())
                            if diff <= 86400:  # Within 24 hours
                                time_diffs.append((diff, idx))
                        
                        if time_diffs:
                            time_diffs.sort(key=lambda x: x[0])
                            closest_idx = time_diffs[0][1]
                            price_at_news = hist_intraday_sorted.loc[closest_idx, 'Close']
                            
                            # Find 1h after
                            one_hour_later = news_date_tz + timedelta(hours=1)
                            time_diffs_1h = []
                            for idx in hist_intraday_sorted.index:
                                idx_naive = idx.replace(tzinfo=None) if idx.tzinfo else idx
                                if idx_naive > news_date_tz:
                                    diff = abs((idx_naive - one_hour_later).total_seconds())
                                    if diff <= 7200:  # Within 2 hours
                                        time_diffs_1h.append((diff, idx))
                            
                            if time_diffs_1h:
                                time_diffs_1h.sort(key=lambda x: x[0])
                                closest_1h_idx = time_diffs_1h[0][1]
                                price_1h_after = hist_intraday_sorted.loc[closest_1h_idx, 'Close']
                                
                                if price_at_news > 0:
                                    price_1h = ((price_1h_after / price_at_news) - 1) * 100
                except Exception as e:
                    logger.warning(f"Error getting 1h intraday data for {ticker}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error in 1h impact calculation for {ticker}: {str(e)}")
        
        # Get daily data for 1d and 1w impact
        hist_daily = stock.history(period='1mo', interval='1d', auto_adjust=True)
        if hist_daily.empty:
            return None
        
        # Find price at news date
        news_date_only = news_date.date()
        hist_daily['date'] = hist_daily.index.date
        
        closest_daily_idx = None
        min_diff = float('inf')
        for idx, row_date in enumerate(hist_daily['date']):
            diff = abs((row_date - news_date_only).days)
            if diff < min_diff:
                min_diff = diff
                closest_daily_idx = hist_daily.index[idx]
        
        if closest_daily_idx is None or min_diff > 5:
            return None
        
        price_at_news = hist_daily.loc[closest_daily_idx, 'Close']
        
        # Find 1 day after
        one_day_after = news_date_only + timedelta(days=1)
        closest_1d_idx = None
        min_diff_1d = float('inf')
        for idx, row_date in enumerate(hist_daily['date']):
            if row_date >= one_day_after:
                diff = abs((row_date - one_day_after).days)
                if diff < min_diff_1d:
                    min_diff_1d = diff
                    closest_1d_idx = hist_daily.index[idx]
        
        if closest_1d_idx and min_diff_1d <= 2:
            price_1d_after = hist_daily.loc[closest_1d_idx, 'Close']
            price_1d = ((price_1d_after / price_at_news) - 1) * 100
        
        # Find 1 week after
        one_week_after = news_date_only + timedelta(days=7)
        closest_1w_idx = None
        min_diff_1w = float('inf')
        for idx, row_date in enumerate(hist_daily['date']):
            if row_date >= one_week_after:
                diff = abs((row_date - one_week_after).days)
                if diff < min_diff_1w:
                    min_diff_1w = diff
                    closest_1w_idx = hist_daily.index[idx]
        
        if closest_1w_idx and min_diff_1w <= 3:
            price_1w_after = hist_daily.loc[closest_1w_idx, 'Close']
            price_1w = ((price_1w_after / price_at_news) - 1) * 100
        
        return {
            'price_1h_pct': round(price_1h, 2) if price_1h is not None else None,
            'price_1d_pct': round(price_1d, 2) if price_1d is not None else None,
            'price_1w_pct': round(price_1w, 2) if price_1w is not None else None,
            'price_at_news': round(price_at_news, 2) if price_at_news else None
        }
        
    except Exception as e:
        logger.exception(f"Error calculating price impact for {ticker}")
        import traceback
        traceback.print_exc()
        return None


def calculate_news_impact_score(price_movement: Optional[Dict], sentiment_score: float, news_type: str, source_weight: float = 1.0) -> float:
    """
    Calculate overall news impact score (0-100) based on price movement and sentiment
    
    Args:
        price_movement: Dict with price impact percentages
        sentiment_score: Sentiment score (-1 to 1)
        news_type: News type classification
        source_weight: Source weight multiplier
        
    Returns:
        Impact score (0-100)
    """
    try:
        # Base score from price movement (0-50 points)
        price_score = 0
        if price_movement:
            movement = price_movement.get('price_1d_pct') or price_movement.get('price_1w_pct')
            if movement is not None:
                price_score = min(50, abs(movement) * 5)
        
        # Sentiment score (0-30 points)
        sentiment_magnitude = abs(sentiment_score) if sentiment_score else 0
        sentiment_points = sentiment_magnitude * 30
        
        # News type weight (0-20 points)
        type_weights = {
            'earnings': 20,
            'm_a': 18,
            'regulatory': 16,
            'product_launch': 14,
            'financial': 12,
            'guidance': 12,
            'management': 10,
            'partnership': 8,
            'other': 5
        }
        type_points = type_weights.get(news_type, 5)
        
        # Source weight multiplier
        total_score = (price_score + sentiment_points + type_points) * source_weight
        
        # Normalize to 0-100
        impact_score = min(100, max(0, total_score))
        
        return round(impact_score, 1)
    except:
        return 0


def get_stock_news(ticker: str, max_news: int = 10) -> List[Dict]:
    """
    Get latest Press Releases for a stock from Yahoo Finance and analyze sentiment
    
    Args:
        ticker: Stock ticker symbol
        max_news: Maximum number of news items to return
        
    Returns:
        List of analyzed news items
    """
    try:
        analyzed_news = []
        
        try:
            # Get Press Releases from yfinance
            stock = yf.Ticker(ticker.upper())
            news = stock.get_news(tab='press releases')
            
            if not news:
                logger.info(f"No Press Releases found for {ticker}")
                return []
            
            # Process Press Releases
            for item in news:
                try:
                    content = item.get('content', {})
                    if not content:
                        continue
                    
                    provider = content.get('provider', {})
                    provider_name = provider.get('displayName', '') if provider else 'Press Release'
                    
                    # Extract news data
                    title = content.get('title', '')
                    summary = content.get('summary', '') or content.get('description', '')
                    pub_date = content.get('pubDate', '') or content.get('displayTime', '')
                    
                    # Get URL
                    canonical_url = content.get('canonicalUrl', {})
                    link = canonical_url.get('url', '') if canonical_url else ''
                    if not link:
                        click_url = content.get('clickThroughUrl', {})
                        link = click_url.get('url', '') if click_url else ''
                    
                    # Extract thumbnail/image URL
                    thumbnail_url = None
                    if content.get('thumbnail'):
                        thumb = content.get('thumbnail')
                        thumbnail_url = thumb.get('url') if isinstance(thumb, dict) else thumb
                    elif content.get('thumbnailUrl'):
                        thumbnail_url = content.get('thumbnailUrl')
                    elif content.get('image'):
                        img = content.get('image')
                        thumbnail_url = img.get('url') if isinstance(img, dict) else img
                    elif content.get('imageUrl'):
                        thumbnail_url = content.get('imageUrl')
                    elif content.get('relatedImages'):
                        related_images = content.get('relatedImages', [])
                        if related_images and len(related_images) > 0:
                            thumbnail_url = related_images[0].get('url') if isinstance(related_images[0], dict) else related_images[0]
                    
                    # Analyze sentiment
                    text_for_analysis = f"{title}. {summary}".strip()
                    if not text_for_analysis:
                        continue
                    
                    sentiment_data = analyze_sentiment(text_for_analysis)
                    
                    # Determine source weight
                    major_financial_sources = ['Bloomberg', 'Reuters', 'Wall Street Journal', 'Financial Times', 'CNBC', 'MarketWatch', 'Yahoo Finance', 'Seeking Alpha']
                    source_weight = 1.0
                    if provider_name in major_financial_sources:
                        source_weight = 1.5
                    elif 'Press Release' in provider_name or provider_name == 'Press Release':
                        source_weight = 1.2
                    
                    # Classify news type
                    news_type = classify_news_type(title, summary)
                    
                    # Normalize date
                    if not pub_date or pub_date.strip() == '':
                        pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
                    
                    normalized_date = normalize_date(pub_date) if pub_date else None
                    published_str = normalized_date if normalized_date else datetime.now().strftime('%Y-%m-%d %H:%M')
                    
                    # Calculate price impact (1h, 1d, 1w)
                    price_impact = calculate_news_price_impact(ticker, published_str, title, summary)
                    
                    # Calculate comprehensive impact score
                    impact_score = calculate_news_impact_score(
                        price_impact, 
                        sentiment_data['score'], 
                        news_type, 
                        source_weight
                    )
                    
                    analyzed_news.append({
                        'title': title,
                        'summary': summary[:500] if summary else 'No summary available',
                        'publisher': provider_name if provider_name else 'Press Release',
                        'link': link,
                        'published': published_str,
                        'thumbnail': thumbnail_url,
                        'sentiment': sentiment_data['sentiment'],
                        'sentiment_label': sentiment_data['label'],
                        'sentiment_score': sentiment_data['score'],
                        'impact_score': impact_score,
                        'news_type': news_type,
                        'price_impact': price_impact,
                        'sentiment_details': {
                            'positive': sentiment_data['positive'],
                            'negative': sentiment_data['negative'],
                            'neutral': sentiment_data['neutral']
                        }
                    })
                    
                    if len(analyzed_news) >= max_news:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing news item for {ticker}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(analyzed_news)} Press Releases for {ticker}")
            return analyzed_news
            
        except Exception as e:
            logger.exception(f"Error fetching news from yfinance for {ticker}")
            import traceback
            traceback.print_exc()
            return []
        
    except Exception as e:
        logger.exception(f"Error fetching news for {ticker}")
        import traceback
        traceback.print_exc()
        return []


def calculate_historical_news_impact_patterns(ticker: str, news_list: List[Dict]) -> Optional[Dict]:
    """Calculate historical patterns of news impact"""
    try:
        if not news_list or len(news_list) < 3:
            return None
        
        # Group by news type
        type_impacts = {}
        sentiment_impacts = {'positive': [], 'negative': [], 'neutral': []}
        
        for news in news_list:
            news_type = news.get('news_type', 'other')
            sentiment = news.get('sentiment', 'neutral')
            price_impact = news.get('price_impact', {})
            
            # Collect price movements by type
            if news_type not in type_impacts:
                type_impacts[news_type] = {'1d': [], '1w': []}
            
            if price_impact:
                if price_impact.get('price_1d_pct') is not None:
                    type_impacts[news_type]['1d'].append(price_impact['price_1d_pct'])
                if price_impact.get('price_1w_pct') is not None:
                    type_impacts[news_type]['1w'].append(price_impact['price_1w_pct'])
            
            # Collect by sentiment
            if sentiment in sentiment_impacts and price_impact:
                if price_impact.get('price_1d_pct') is not None:
                    sentiment_impacts[sentiment].append(price_impact['price_1d_pct'])
        
        # Calculate averages
        type_averages = {}
        for news_type, impacts in type_impacts.items():
            if impacts['1d'] or impacts['1w']:
                type_averages[news_type] = {
                    'avg_1d_pct': round(sum(impacts['1d']) / len(impacts['1d']), 2) if impacts['1d'] else None,
                    'avg_1w_pct': round(sum(impacts['1w']) / len(impacts['1w']), 2) if impacts['1w'] else None,
                    'count': len(impacts['1d']) + len(impacts['1w'])
                }
        
        sentiment_averages = {}
        for sentiment, impacts in sentiment_impacts.items():
            if impacts:
                sentiment_averages[sentiment] = {
                    'avg_1d_pct': round(sum(impacts) / len(impacts), 2),
                    'count': len(impacts)
                }
        
        return {
            'by_type': type_averages,
            'by_sentiment': sentiment_averages,
            'total_news_analyzed': len(news_list)
        }
        
    except Exception as e:
        logger.exception(f"Error calculating historical patterns for {ticker}")
        return None
# - get_marketbeat_insider_trading()
# - get_tipranks_insider_trading()
# - get_economic_calendar()

