"""
Utilities for web scraping: caching, rate limiting, error handling.
"""

import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class ScraperCache:
    """
    Cache for web scraping results to avoid duplicate requests.
    """
    
    def __init__(self, cache_dir: str = "data/web_cache", ttl_hours: int = 24):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, query: str, source: str) -> str:
        """Generate cache key from query and source."""
        key_string = f"{source}:{query}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, query: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available and not expired.
        
        Args:
            query: Search query
            source: Data source identifier
            
        Returns:
            Cached data or None
        """
        cache_key = self._get_cache_key(query, source)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check if expired
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                cache_path.unlink()  # Delete expired cache
                return None
            
            return cached['data']
        except:
            return None
    
    def set(self, query: str, source: str, data: Dict[str, Any]):
        """
        Cache scraping results.
        
        Args:
            query: Search query
            source: Data source identifier
            data: Data to cache
        """
        cache_key = self._get_cache_key(query, source)
        cache_path = self._get_cache_path(cache_key)
        
        cached = {
            'query': query,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached, f, indent=2)
        except Exception as e:
            print(f"⚠️  Cache write error: {e}")
    
    def clear_old_cache(self):
        """Delete all expired cache files."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                cached_time = datetime.fromisoformat(cached['timestamp'])
                if datetime.now() - cached_time > self.ttl:
                    cache_file.unlink()
                    count += 1
            except:
                pass
        
        if count > 0:
            print(f"✓ Cleared {count} expired cache files")


class RateLimiter:
    """
    Rate limiter to avoid overwhelming web services.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        requests_per_hour: int = 100
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
        """
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour
        
        self.minute_requests = []
        self.hour_requests = []
    
    def wait_if_needed(self):
        """Wait if rate limits would be exceeded."""
        now = time.time()
        
        # Clean old timestamps
        minute_ago = now - 60
        hour_ago = now - 3600
        
        self.minute_requests = [t for t in self.minute_requests if t > minute_ago]
        self.hour_requests = [t for t in self.hour_requests if t > hour_ago]
        
        # Check minute limit
        if len(self.minute_requests) >= self.rpm_limit:
            wait_time = 60 - (now - self.minute_requests[0])
            if wait_time > 0:
                print(f"⏳ Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.minute_requests = []
        
        # Check hour limit
        if len(self.hour_requests) >= self.rph_limit:
            wait_time = 3600 - (now - self.hour_requests[0])
            if wait_time > 0:
                print(f"⏳ Hourly limit reached: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.hour_requests = []
        
        # Record this request
        now = time.time()
        self.minute_requests.append(now)
        self.hour_requests.append(now)


class ScraperSession:
    """
    HTTP session with retry logic and error handling for web scraping.
    """
    
    def __init__(
        self,
        timeout: int = 15,
        user_agent: str = None
    ):
        """
        Initialize scraper session.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set user agent
        if user_agent is None:
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60)
    )
    def get(self, url: str, **kwargs) -> requests.Response:
        """
        GET request with retry logic.
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments for requests.get
            
        Returns:
            Response object
        """
        try:
            response = self.session.get(
                url,
                timeout=kwargs.pop('timeout', self.timeout),
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request error for {url}: {e}")
            raise
    
    def close(self):
        """Close session."""
        self.session.close()


def extract_keywords_from_anomaly(anomaly_data: Dict[str, Any]) -> list:
    """
    Extract search keywords from anomaly data.
    
    Args:
        anomaly_data: Anomaly details
        
    Returns:
        List of search keywords
    """
    keywords = []
    
    # Entity name
    entity = anomaly_data.get('entity_id') or anomaly_data.get('entity_name')
    if entity:
        keywords.append(entity)
    
    # Period
    period = anomaly_data.get('period')
    if period:
        keywords.append(period)
    
    # GL account name
    gl_name = anomaly_data.get('gl_name')
    if gl_name:
        keywords.append(gl_name)
    
    # Transaction type keywords
    amount = anomaly_data.get('amount', 0)
    if amount > 0:
        keywords.extend(['acquisition', 'purchase', 'investment'])
    else:
        keywords.extend(['sale', 'divestment', 'disposal'])
    
    # Add general financial keywords
    keywords.extend(['transaction', 'announcement', 'filing'])
    
    return keywords


def clean_html_text(html: str) -> str:
    """
    Clean HTML and extract text.
    
    Args:
        html: HTML content
        
    Returns:
        Cleaned text
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except:
        return html


if __name__ == "__main__":
    print("Testing Scraper Utils...")
    
    # Test cache
    cache = ScraperCache(cache_dir="test_cache")
    
    cache.set("test query", "test_source", {"result": "test data"})
    result = cache.get("test query", "test_source")
    print(f"✓ Cache test: {result}")
    
    # Test rate limiter
    limiter = RateLimiter(requests_per_minute=5)
    for i in range(3):
        limiter.wait_if_needed()
        print(f"  Request {i+1}")
    
    # Cleanup
    import shutil
    if Path("test_cache").exists():
        shutil.rmtree("test_cache")
    
    print("\n✓ Scraper utils tests passed!")

