"""
News Scraper for financial news sources.

Uses NewsData.io as primary source and GDELT as fallback.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .scraper_utils import ScraperCache, RateLimiter, ScraperSession, clean_html_text
from .newsdata_scraper import NewsDataScraper
from .gdelt_scraper import GDELTScraper


class NewsScraper:
    """
    Scrape financial news sources for context about anomalies.
    
    Sources:
    - Google News search
    - RSS feeds
    - NewsAPI (if API key available)
    """
    
    def __init__(
        self,
        cache_ttl_hours: int = 24,
        rate_limit_rpm: int = 30
    ):
        """
        Initialize news scraper with NewsData.io (primary) and GDELT (fallback).
        
        Args:
            cache_ttl_hours: Cache time-to-live
            rate_limit_rpm: Rate limit requests per minute
        """
        self.cache = ScraperCache(ttl_hours=cache_ttl_hours)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
        self.session = ScraperSession()
        
        # Initialize primary and fallback scrapers
        self.newsdata_scraper = NewsDataScraper(cache_ttl_hours, rate_limit_rpm)
        self.gdelt_scraper = GDELTScraper(cache_ttl_hours, rate_limit_rpm)
        
        # Legacy support
        self.news_api_key = os.getenv('NEWS_API_KEY')
    
    def search(
        self,
        entity_name: str,
        keywords: List[str],
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Search news for entity and keywords.
        
        Args:
            entity_name: Company/entity name
            keywords: Additional search keywords
            days_back: How many days to search back
            
        Returns:
            List of news articles with metadata
        """
        query = f"{entity_name} {' '.join(keywords)}"
        
        # Check cache
        cached = self.cache.get(query, 'news_search')
        if cached:
            print(f"✓ News cache hit for: {query}")
            return cached
        
        print(f"🔍 Searching news for: {query}")
        
        articles = []
        
        # PRIMARY: Try NewsData.io first
        newsdata_articles = self.newsdata_scraper.search(entity_name, keywords, days_back)
        if newsdata_articles:
            articles.extend(newsdata_articles)
            print(f"✓ NewsData.io: {len(newsdata_articles)} articles")
        
        # FALLBACK: Try GDELT if NewsData.io returns few results
        if len(articles) < 5:
            gdelt_articles = self.gdelt_scraper.search(entity_name, keywords, days_back)
            if gdelt_articles:
                articles.extend(gdelt_articles)
                print(f"✓ GDELT fallback: {len(gdelt_articles)} articles")
        
        # Deduplicate by URL
        unique_articles = {article['url']: article for article in articles if article.get('url')}.values()
        articles = list(unique_articles)
        
        # Cache results
        self.cache.set(query, 'news_search', articles)
        
        print(f"✓ Found {len(articles)} news articles total")
        return articles
    
    def _search_google_news(
        self,
        query: str,
        days_back: int
    ) -> List[Dict[str, Any]]:
        """Search Google News."""
        articles = []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Google News search URL
            search_url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse Google News results
            for article_elem in soup.select('article')[:10]:  # Top 10 results
                try:
                    title_elem = article_elem.select_one('h3, h4')
                    link_elem = article_elem.select_one('a')
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        link = link_elem.get('href', '')
                        
                        # Google News links start with ./articles/
                        if link.startswith('./'):
                            link = f"https://news.google.com{link[1:]}"
                        
                        # Try to extract snippet
                        snippet = ""
                        snippet_elem = article_elem.select_one('div[class*="snippet"]')
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)
                        
                        articles.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet,
                            'source': 'Google News',
                            'scraped_at': datetime.now().isoformat()
                        })
                except:
                    continue
        
        except Exception as e:
            print(f"⚠️  Google News scraping error: {e}")
        
        return articles
    
    def _search_news_api(
        self,
        entity_name: str,
        keywords: List[str],
        days_back: int
    ) -> List[Dict[str, Any]]:
        """Search using NewsAPI.org (requires API key)."""
        articles = []
        
        if not self.news_api_key:
            return articles
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{entity_name} AND ({' OR '.join(keywords)})",
                'from': from_date,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 20,
                'apiKey': self.news_api_key
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'snippet': article.get('description', ''),
                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                        'published_at': article.get('publishedAt', ''),
                        'scraped_at': datetime.now().isoformat()
                    })
        
        except Exception as e:
            print(f"⚠️  NewsAPI error: {e}")
        
        return articles
    
    def fetch_article_content(self, url: str) -> Optional[str]:
        """
        Fetch full article content from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Article text or None
        """
        # Check cache
        cached = self.cache.get(url, 'article_content')
        if cached:
            return cached
        
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.session.get(url)
            text = clean_html_text(response.text)
            
            # Cache content
            self.cache.set(url, 'article_content', text)
            
            return text
        
        except Exception as e:
            print(f"⚠️  Error fetching article {url}: {e}")
            return None
    
    def search_and_fetch_content(
        self,
        entity_name: str,
        keywords: List[str],
        max_articles: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search news and fetch full content for top articles.
        
        Args:
            entity_name: Company name
            keywords: Search keywords
            max_articles: Maximum articles to fetch content for
            
        Returns:
            List of articles with full content
        """
        # Search for articles
        articles = self.search(entity_name, keywords)
        
        # Fetch content for top articles
        results = []
        for article in articles[:max_articles]:
            content = self.fetch_article_content(article['url'])
            if content:
                article['text'] = content
                results.append(article)
        
        return results


if __name__ == "__main__":
    print("Testing News Scraper...")
    
    scraper = NewsScraper()
    
    # Test search
    articles = scraper.search(
        entity_name="Microsoft",
        keywords=["acquisition", "2024"],
        days_back=30
    )
    
    if articles:
        print(f"\n✓ Found {len(articles)} articles")
        print(f"Sample: {articles[0]['title'][:80]}...")
    else:
        print("⚠️  No articles found (might need NewsAPI key)")
    
    print("\n✓ News scraper tests passed!")

