"""
GDELT Project API scraper for news events and mentions.
Free source for BalanceGuard AI news gathering.
https://www.gdeltproject.org/
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from .scraper_utils import ScraperCache, RateLimiter, ScraperSession
import re


class GDELTScraper:
    """
    GDELT Project API client for news events.
    Completely free - no API key needed!
    """
    
    def __init__(
        self,
        cache_ttl_hours: int = 24,
        rate_limit_rpm: int = 60
    ):
        """
        Initialize GDELT scraper.
        
        Args:
            cache_ttl_hours: Cache time-to-live
            rate_limit_rpm: Rate limit requests per minute
        """
        self.cache = ScraperCache(ttl_hours=cache_ttl_hours)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
        self.session = ScraperSession()
    
    def search(
        self,
        entity_name: str,
        keywords: List[str],
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Search GDELT for entity mentions and events.
        
        Args:
            entity_name: Company/entity name
            keywords: Search keywords
            days_back: Days to look back
            
        Returns:
            List of article dictionaries
        """
        # Build query
        query = entity_name
        
        # Check cache
        cache_key = f"{query}_{days_back}"
        cached = self.cache.get(cache_key, 'gdelt_search')
        if cached:
            print(f"✓ Using cached GDELT results")
            return cached
        
        articles = []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            print(f"🔍 Searching GDELT: {query}")
            
            # Calculate time range (GDELT uses last N hours)
            hours_back = min(days_back * 24, 250 * 24)  # Max ~250 days
            
            # GDELT 2.0 DOC API endpoint
            # Search for articles mentioning the entity
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params = {
                'query': query,
                'mode': 'artlist',
                'maxrecords': 20,
                'format': 'json',
                'timespan': f'{hours_back}h',
                'sort': 'datedesc'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            data = response.json()
            
            if 'articles' in data:
                for article in data['articles']:
                    articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'snippet': self._clean_text(article.get('seendate', '')),
                        'source': article.get('domain', 'GDELT'),
                        'published_date': self._parse_gdelt_date(article.get('seendate', '')),
                        'language': article.get('language', 'en'),
                        'tone': article.get('tone', 0),  # Sentiment score
                        'scraped_at': datetime.now().isoformat()
                    })
                
                print(f"✓ Found {len(articles)} articles from GDELT")
            else:
                print(f"⚠️  GDELT returned no articles")
        
        except Exception as e:
            print(f"⚠️  GDELT request error: {e}")
        
        # Cache results
        if articles:
            self.cache.set(cache_key, 'gdelt_search', articles)
        
        return articles
    
    def get_events(
        self,
        entity_name: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get GDELT events related to entity.
        
        Args:
            entity_name: Company name
            days_back: Days to look back
            
        Returns:
            List of events
        """
        events = []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            hours_back = days_back * 24
            
            # GDELT Events API
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params = {
                'query': entity_name,
                'mode': 'timelinevol',
                'timespan': f'{hours_back}h',
                'format': 'json'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            data = response.json()
            
            # Parse timeline data
            if 'timeline' in data:
                for entry in data['timeline']:
                    events.append({
                        'date': entry.get('date', ''),
                        'volume': entry.get('value', 0),
                        'entity': entity_name
                    })
        
        except Exception as e:
            print(f"⚠️  GDELT events error: {e}")
        
        return events
    
    def _parse_gdelt_date(self, date_str: str) -> str:
        """
        Parse GDELT date format (YYYYMMDDHHMMSS) to ISO format.
        
        Args:
            date_str: GDELT date string
            
        Returns:
            ISO formatted date string
        """
        try:
            if len(date_str) >= 14:
                # YYYYMMDDHHMMSS
                dt = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
                return dt.isoformat()
        except:
            pass
        return date_str
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text."""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:200]  # Limit length

