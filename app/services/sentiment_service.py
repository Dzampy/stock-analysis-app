"""
Sentiment analysis service - VADER, Reddit, Twitter, StockTwits sentiment
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from app.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_AVAILABLE
from app.utils.constants import RATE_LIMIT_DELAY
from app.utils.logger import logger

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> Dict:
    """
    Analyze sentiment of text using VADER
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with sentiment scores and labels
    """
    if not text or len(text.strip()) == 0:
        return {'sentiment': 'neutral', 'score': 0.0, 'label': 'Neutrální'}
    
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = 'positive'
        label = 'Pozitivní'
    elif compound <= -0.05:
        sentiment = 'negative'
        label = 'Negativní'
    else:
        sentiment = 'neutral'
        label = 'Neutrální'
    
    return {
        'sentiment': sentiment,
        'score': round(compound, 3),
        'label': label,
        'positive': round(scores['pos'], 3),
        'negative': round(scores['neg'], 3),
        'neutral': round(scores['neu'], 3)
    }


import time
from app.config import GEMINI_API_KEY, GEMINI_AVAILABLE

# Try to import Gemini
try:
    import google.generativeai as genai
    if GEMINI_AVAILABLE:
        genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    pass


def get_reddit_sentiment(ticker, days=7):
    """Get Reddit sentiment for a ticker"""
    try:
        # This is a placeholder - implement if needed
        return None
    except Exception as e:
        logger.exception(f"Error getting Reddit sentiment")
        return None


def get_social_sentiment_fallback(ticker, days=7):
    """Fallback to web scraping if PRAW not available or failed"""
    try:
        # Placeholder implementation
        return {
            'posts': [],
            'sentiment_score': 50.0,
            'mention_count': 0,
            'trending': False,
            'total_upvotes': 0,
            'total_comments': 0
        }
    except Exception as e:
        logger.exception(f"Error in get_social_sentiment_fallback")
        return {
            'posts': [],
            'sentiment_score': 50.0,
            'mention_count': 0,
            'trending': False,
            'total_upvotes': 0,
            'total_comments': 0,
            'error': str(e)
        }


def get_twitter_sentiment(ticker, days=7):
    """Get sentiment from Twitter/X about a stock ticker"""
    try:
        tweets = []
        sentiment_scores = []
        total_likes = 0
        total_retweets = 0
        
        # Web scraping approach (Twitter API requires paid access)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try to scrape Twitter search results
        # Note: Twitter has strong anti-scraping, so this is a simplified approach
        try:
            # Search for tweets with the ticker
            search_query = f"${ticker} OR {ticker} stock"
            # Using nitter.net as a Twitter frontend (if available)
            # Or use Twitter's search page (may be blocked)
            
            # For now, return sample data structure
            # In production, you'd use Twitter API v2 (paid) or a service like snscrape
            
            # Simulated data structure - replace with actual scraping
            
        except Exception as e:
            logger.warning(f"Twitter scraping not available: {str(e)}")
        
        # Calculate sentiment from collected tweets
        if sentiment_scores:
            weighted_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            sentiment_score = ((weighted_sentiment + 1) / 2) * 100
        else:
            # Return neutral if no data available
            sentiment_score = 50.0
        
        return {
            'tweets': tweets[:20],
            'sentiment_score': round(sentiment_score, 2),
            'mention_count': len(tweets),
            'total_likes': total_likes,
            'total_retweets': total_retweets,
            'influencers': []
        }
        
    except Exception as e:
        logger.exception(f"Error in get_twitter_sentiment")
        return {
            'tweets': [],
            'sentiment_score': 50.0,
            'mention_count': 0,
            'total_likes': 0,
            'total_retweets': 0,
            'influencers': [],
            'error': str(e)
        }


def get_stocktwits_sentiment(ticker, days=7):
    """Get sentiment from StockTwits about a stock ticker"""
    try:
        messages = []
        sentiment_scores = []
        bullish_count = 0
        bearish_count = 0
        
        # StockTwits API (free but rate limited)
        response = None
        data = None
        try:
            api_url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'messages' in data and len(data['messages']) > 0:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    
                    for msg in data['messages'][:30]:  # Limit to 30 messages
                        try:
                            # Handle different date formats
                            created_at_str = msg.get('created_at', '')
                            if not created_at_str:
                                # If no date, include message anyway
                                created_at = datetime.now()
                            else:
                                try:
                                    created_at = datetime.strptime(created_at_str, '%Y-%m-%dT%H:%M:%SZ')
                                except:
                                    try:
                                        created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                                    except:
                                        # If date parsing fails, include the message anyway
                                        created_at = datetime.now()
                            
                            if created_at < cutoff_date:
                                continue
                            
                            body = msg.get('body', '')
                            sentiment_flag = msg.get('entities', {}).get('sentiment', {})
                            sentiment_basic = sentiment_flag.get('basic', 'neutral') if sentiment_flag else 'neutral'
                            
                            # Analyze sentiment
                            sentiment = sentiment_analyzer.polarity_scores(body)
                            
                            # Use StockTwits sentiment if available, otherwise use VADER
                            if sentiment_basic == 'bullish':
                                sentiment_score = 0.5 + (sentiment['compound'] * 0.5)
                                bullish_count += 1
                            elif sentiment_basic == 'bearish':
                                sentiment_score = -0.5 + (sentiment['compound'] * 0.5)
                                bearish_count += 1
                            else:
                                sentiment_score = sentiment['compound']
                            
                            messages.append({
                                'body': body[:500],
                                'sentiment': sentiment_basic,
                                'sentiment_score': sentiment_score,
                                'user': msg.get('user', {}).get('username', 'Unknown'),
                                'date': created_at.isoformat(),
                                'id': msg.get('id', 0)
                            })
                            
                            sentiment_scores.append(sentiment_score)
                            
                        except Exception as e:
                            logger.warning(f"Error processing StockTwits message: {str(e)}")
                            continue
                            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.warning(f"Error accessing StockTwits API: {str(e)}")
        
        # Calculate sentiment score
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_score = ((avg_sentiment + 1) / 2) * 100
        else:
            sentiment_score = 50.0
        
        # Calculate bullish percentage
        total_sentiment_flags = bullish_count + bearish_count
        bullish_pct = (bullish_count / total_sentiment_flags * 100) if total_sentiment_flags > 0 else 50.0
        
        # Get watchlist count (if available in API response)
        watchlist_count = 0
        try:
            if 'response' in locals() and response.status_code == 200:
                if 'symbol' in data:
                    watchlist_count = data['symbol'].get('watchlist_count', 0)
        except Exception:
            pass
        
        return {
            'messages': messages[:20],
            'sentiment_score': round(sentiment_score, 2),
            'watchlist_count': watchlist_count,
            'bullish_pct': round(bullish_pct, 2),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }
        
    except Exception as e:
        logger.exception(f"Error in get_stocktwits_sentiment")
        return {
            'messages': [],
            'sentiment_score': 50.0,
            'watchlist_count': 0,
            'bullish_pct': 50.0,
            'bullish_count': 0,
            'bearish_count': 0,
            'error': str(e)
        }


def aggregate_social_sentiment(ticker, days=7):
    """Aggregate sentiment from Reddit only"""
    try:
        # Get data from Reddit only
        reddit_data = get_reddit_sentiment(ticker, days)
        
        # Use Reddit sentiment as overall sentiment
        reddit_score = reddit_data.get('sentiment_score', 50.0) if reddit_data else 50.0
        overall_score = reddit_score
        
        # Determine overall sentiment
        if overall_score >= 60:
            overall_sentiment = 'positive'
        elif overall_score <= 40:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate trends (simple comparison - in production, compare with historical data)
        trends = {
            'reddit': 'stable'  # Would compare with previous period
        }
        
        # Extract key topics from Reddit posts (simple keyword extraction)
        all_texts = []
        if reddit_data:
            for post in reddit_data.get('posts', [])[:10]:
                all_texts.append(f"{post.get('title', '')} {post.get('text', '')}")
        
        # Simple keyword extraction (common financial terms)
        key_topics = []
        if all_texts:
            combined_text = ' '.join(all_texts).lower()
            # Look for common topics
            topic_keywords = {
                'earnings': ['earnings', 'eps', 'revenue', 'profit'],
                'guidance': ['guidance', 'outlook', 'forecast'],
                'product': ['product', 'launch', 'release'],
                'partnership': ['partnership', 'deal', 'agreement'],
                'regulation': ['regulation', 'fda', 'approval'],
                'competition': ['competitor', 'competition', 'market share']
            }
            
            for topic, keywords in topic_keywords.items():
                count = sum(1 for keyword in keywords if keyword in combined_text)
                if count > 0:
                    key_topics.append({
                        'topic': topic,
                        'mentions': count
                    })
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': round(overall_score, 2),
            'trends': trends,
            'key_topics': key_topics[:10],
            'platform_breakdown': {
                'reddit': {
                    'sentiment_score': reddit_score,
                    'mention_count': reddit_data.get('mention_count', 0) if reddit_data else 0,
                    'trending': reddit_data.get('trending', False) if reddit_data else False
                }
            },
            'raw_data': {
                'reddit': reddit_data
            }
        }
        
    except Exception as e:
        logger.exception(f"Error in aggregate_social_sentiment")
        import traceback
        traceback.print_exc()
        return {
            'overall_sentiment': 'neutral',
            'sentiment_score': 50.0,
            'trends': {},
            'key_topics': [],
            'platform_breakdown': {},
            'error': str(e)
        }


def analyze_social_topics_with_ai(posts_data, ticker):
    """Analyze social media posts and extract key topics using AI"""
    if not GEMINI_AVAILABLE:
        return {
            'success': False,
            'error': 'Google Gemini API key not configured'
        }
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Find available model
        model = None
        model_name_used = 'unknown'
        
        preferred_models = ["gemini-1.5-flash", "gemini-pro"]
        available_models_list = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models_list.append(m.name)
        except:
            available_models_list = ["gemini-1.5-flash", "gemini-pro"]
        
        for preferred in preferred_models:
            if preferred in available_models_list:
                model = genai.GenerativeModel(preferred)
                model_name_used = preferred
                break
        
        if model is None:
            if available_models_list:
                model = genai.GenerativeModel(available_models_list[0])
                model_name_used = available_models_list[0]
            else:
                model = genai.GenerativeModel("gemini-1.5-flash")
                model_name_used = "gemini-1.5-flash (fallback)"
        
        # Prepare text data
        all_texts = []
        for post in posts_data.get('reddit', {}).get('posts', [])[:15]:
            all_texts.append(f"Reddit: {post.get('title', '')} {post.get('text', '')}")
        for tweet in posts_data.get('twitter', {}).get('tweets', [])[:15]:
            all_texts.append(f"Twitter: {tweet.get('text', '')}")
        for msg in posts_data.get('stocktwits', {}).get('messages', [])[:15]:
            all_texts.append(f"StockTwits: {msg.get('body', '')}")
        
        combined_text = '\n---\n'.join(all_texts)
        if len(combined_text) > 20000:
            combined_text = combined_text[:20000] + "\n... (truncated)"
        
        prompt = f"""Jsi expertní analytik sociálních médií. Analyzuj následující diskuse o akcii {ticker} z Redditu a identifikuj klíčová témata, motivy a varování.

Formátuj odpověď PŘESNĚ takto:

=== Key Topics ===
Uveď 5-8 nejčastěji zmiňovaných témat. Každé téma na samostatný řádek s odrážkou.

=== Themes ===
Uveď 3-5 hlavních motivů nebo trendů v diskusích. Každý motiv na samostatný řádek s odrážkou.

=== Warnings ===
Identifikuj varování (FOMO, FUD, pump & dump signály, manipulace). Uveď 2-4 varování na samostatný řádek s odrážkou.

Diskuse:
{combined_text}
"""
        
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 2048,
            }
        )
        
        ai_analysis = response.text
        
        # Parse AI response
        key_topics = []
        themes = []
        warnings = []
        
        current_section = None
        for line in ai_analysis.split('\n'):
            line = line.strip()
            if '=== Key Topics ===' in line:
                current_section = 'topics'
            elif '=== Themes ===' in line:
                current_section = 'themes'
            elif '=== Warnings ===' in line:
                current_section = 'warnings'
            elif line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                item = line.lstrip('-•* ').strip()
                if current_section == 'topics' and item:
                    key_topics.append(item)
                elif current_section == 'themes' and item:
                    themes.append(item)
                elif current_section == 'warnings' and item:
                    warnings.append(item)
        
        return {
            'success': True,
            'key_topics': key_topics,
            'themes': themes,
            'warnings': warnings,
            'full_analysis': ai_analysis,
            'model_used': model_name_used
        }
        
    except Exception as e:
        logger.exception(f"Error in analyze_social_topics_with_ai")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'key_topics': [],
            'themes': [],
            'warnings': []
        }

