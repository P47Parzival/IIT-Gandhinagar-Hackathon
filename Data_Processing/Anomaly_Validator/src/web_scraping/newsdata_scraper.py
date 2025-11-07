"""
NewsData.io API scraper for news articles.
Primary source for BalanceGuard AI news gathering.
https://newsdata.io/
"""

import os
from typing import Dict, Any, List
from datetime import datetime, timedelta
from .scraper_utils import ScraperCache, RateLimiter, ScraperSession


class NewsDataScraper:
    """
    NewsData.io API client for financial news.
    Free tier: 200 requests/day
    """
    
    def __init__(
        self,
        cache_ttl_hours: int = 24,
        rate_limit_rpm: int = 60
    ):
        """
        Initialize NewsData.io scraper.
        
        Args:
            cache_ttl_hours: Cache time-to-live
            rate_limit_rpm: Rate limit requests per minute
        """
        self.cache = ScraperCache(ttl_hours=cache_ttl_hours)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
        self.session = ScraperSession()
        self.api_key = os.getenv('NEWSDATA_API_KEY')
    
    def search(
        self,
        entity_name: str,
        keywords: List[str],
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Search NewsData.io for entity and keywords.
        
        Args:
            entity_name: Company/entity name
            keywords: Search keywords
            days_back: Days to look back
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            print("⚠️  NewsData.io API key not found (NEWSDATA_API_KEY)")
            return []
        
        # Build simple query (NewsData.io doesn't like complex queries)
        query = entity_name  # Just use entity name for reliability
        
        # Check cache
        cache_key = f"{query}_{days_back}"
        cached = self.cache.get(cache_key, 'newsdata_search')
        if cached:
            print(f"✓ Using cached NewsData.io results")
            return cached
        
        articles = []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            print(f"🔍 Searching NewsData.io: {query}")
            
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # NewsData.io API endpoint
            url = "https://newsdata.io/api/1/news"
            params = {
                'apikey': self.api_key,
                'q': query,
                'language': 'en',
                'category': 'business',
                'from_date': from_date,
                'size': 10  # Max results
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'success':
                results = data.get('results', [])
                
                for article in results:
                    articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('link', ''),
                        'snippet': article.get('description', ''),
                        'source': article.get('source_id', 'NewsData.io'),
                        'published_date': article.get('pubDate', ''),
                        'content': article.get('content', ''),
                        'keywords': article.get('keywords', []),
                        'scraped_at': datetime.now().isoformat()
                    })
                
                print(f"✓ Found {len(articles)} articles from NewsData.io")
            else:
                error_msg = data.get('results', {}).get('message', 'Unknown error')
                print(f"⚠️  NewsData.io error: {error_msg}")
        
        except Exception as e:
            print(f"⚠️  NewsData.io request error: {e}")
        
        # Cache results
        if articles:
            self.cache.set(cache_key, 'newsdata_search', articles)
        
        return articles
    
    def get_latest_headlines(
        self,
        entity_name: str,
        country: str = 'in'  # India for Adani
    ) -> List[Dict[str, Any]]:
        """
        Get latest headlines for entity.
        
        Args:
            entity_name: Company name
            country: Country code (in=India)
            
        Returns:
            List of articles
        """
        if not self.api_key:
            return []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            url = "https://newsdata.io/api/1/news"
            params = {
                'apikey': self.api_key,
                'q': entity_name,
                'language': 'en',
                'country': country,
                'category': 'business',
                'size': 5
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'success':
                articles = []
                for article in data.get('results', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('link', ''),
                        'snippet': article.get('description', ''),
                        'source': article.get('source_id', 'NewsData.io'),
                        'published_date': article.get('pubDate', ''),
                        'scraped_at': datetime.now().isoformat()
                    })
                return articles
        
        except Exception as e:
            print(f"⚠️  NewsData.io headlines error: {e}")
        
        return []

