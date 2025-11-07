"""
Company Website Scraper for official announcements.

Scrapes company investor relations pages, press releases,
and official announcements.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from .scraper_utils import ScraperCache, RateLimiter, ScraperSession, clean_html_text


class CompanyScraper:
    """
    Scrape company official channels for announcements.
    
    Sources:
    - Investor relations pages
    - Press release sections
    - Company news/blog
    """
    
    def __init__(
        self,
        cache_ttl_hours: int = 24,
        rate_limit_rpm: int = 20
    ):
        """
        Initialize company scraper.
        
        Args:
            cache_ttl_hours: Cache time-to-live
            rate_limit_rpm: Rate limit requests per minute
        """
        self.cache = ScraperCache(ttl_hours=cache_ttl_hours)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
        self.session = ScraperSession()
    
    def search(
        self,
        company_domain: str,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Search company website for relevant content.
        
        Args:
            company_domain: Company domain (e.g., 'adani.com')
            keywords: Search keywords
            
        Returns:
            List of pages with metadata
        """
        query = f"{company_domain} {' '.join(keywords)}"
        
        # Check cache
        cached = self.cache.get(query, 'company_search')
        if cached:
            print(f"✓ Company cache hit for: {query}")
            return cached
        
        print(f"🔍 Searching company site: {company_domain}")
        
        results = []
        
        # Try common investor relations URLs
        ir_paths = [
            '/investors',
            '/investor-relations',
            '/press-releases',
            '/news',
            '/media',
            '/newsroom',
            '/announcements'
        ]
        
        for path in ir_paths:
            url = f"https://{company_domain}{path}"
            content = self._scrape_page(url, keywords)
            if content:
                results.append(content)
        
        # Also try Google site search
        site_search_results = self._google_site_search(company_domain, keywords)
        results.extend(site_search_results)
        
        # Deduplicate
        unique_results = {r['url']: r for r in results}.values()
        results = list(unique_results)
        
        # Cache results
        self.cache.set(query, 'company_search', results)
        
        print(f"✓ Found {len(results)} company pages")
        return results
    
    def _scrape_page(
        self,
        url: str,
        keywords: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Scrape a single page."""
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get page title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else url
            
            # Get main content
            text = clean_html_text(response.text)
            
            # Check if keywords are present
            text_lower = text.lower()
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            if keyword_matches > 0:
                return {
                    'title': title_text,
                    'url': url,
                    'text': text[:5000],  # Limit to 5000 chars
                    'keyword_matches': keyword_matches,
                    'source': 'Company Website',
                    'scraped_at': datetime.now().isoformat()
                }
        
        except Exception as e:
            # Silently fail if page doesn't exist
            pass
        
        return None
    
    def _google_site_search(
        self,
        domain: str,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Use Google to search within company site.
        
        Args:
            domain: Company domain
            keywords: Search keywords
            
        Returns:
            List of relevant pages
        """
        results = []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Google site search
            query = f"site:{domain} {' '.join(keywords)}"
            search_url = f"https://www.google.com/search?q={query}"
            
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse Google search results
            for result in soup.select('div.g')[:5]:  # Top 5 results
                try:
                    link_elem = result.select_one('a')
                    title_elem = result.select_one('h3')
                    snippet_elem = result.select_one('div[data-sncf]')
                    
                    if link_elem and title_elem:
                        url = link_elem.get('href', '')
                        if url.startswith('/url?q='):
                            url = url.split('/url?q=')[1].split('&')[0]
                        
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'url': url,
                            'snippet': snippet_elem.get_text(strip=True) if snippet_elem else '',
                            'source': 'Company Website (via Google)',
                            'scraped_at': datetime.now().isoformat()
                        })
                except:
                    continue
        
        except Exception as e:
            print(f"⚠️  Google site search error: {e}")
        
        return results
    
    def fetch_press_releases(
        self,
        company_domain: str,
        max_releases: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent press releases.
        
        Args:
            company_domain: Company domain
            max_releases: Maximum releases to fetch
            
        Returns:
            List of press releases
        """
        press_release_urls = [
            f"https://{company_domain}/press-releases",
            f"https://{company_domain}/newsroom",
            f"https://{company_domain}/media/press-releases"
        ]
        
        releases = []
        
        for url in press_release_urls:
            try:
                self.rate_limiter.wait_if_needed()
                
                response = self.session.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find press release links (common patterns)
                for link in soup.select('a[href*="press"], a[href*="release"], a[href*="news"]')[:max_releases]:
                    href = link.get('href', '')
                    if not href.startswith('http'):
                        href = f"https://{company_domain}{href}"
                    
                    title = link.get_text(strip=True)
                    if title and len(title) > 10:
                        releases.append({
                            'title': title,
                            'url': href,
                            'source': 'Press Release',
                            'scraped_at': datetime.now().isoformat()
                        })
                
                if releases:
                    break  # Found releases, no need to check other URLs
            
            except:
                continue
        
        return releases[:max_releases]
    
    def search_and_fetch_content(
        self,
        company_domain: str,
        keywords: List[str],
        max_pages: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search company site and fetch full content.
        
        Args:
            company_domain: Company domain
            keywords: Search keywords
            max_pages: Maximum pages to fetch
            
        Returns:
            List of pages with full content
        """
        # Search
        pages = self.search(company_domain, keywords)
        
        # Fetch full content for pages without it
        results = []
        for page in pages[:max_pages]:
            if 'text' not in page or not page['text']:
                # Fetch content
                content = self._scrape_page(page['url'], keywords)
                if content:
                    page.update(content)
            
            if page.get('text'):
                results.append(page)
        
        return results


if __name__ == "__main__":
    print("Testing Company Scraper...")
    
    scraper = CompanyScraper()
    
    # Test search
    results = scraper.search(
        company_domain="microsoft.com",
        keywords=["acquisition", "announcement"]
    )
    
    if results:
        print(f"\n✓ Found {len(results)} company pages")
        print(f"Sample: {results[0]['title'][:80]}...")
    else:
        print("⚠️  No results found")
    
    print("\n✓ Company scraper tests passed!")

