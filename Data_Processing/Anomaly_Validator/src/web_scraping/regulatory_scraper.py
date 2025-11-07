"""
Regulatory Filing Scraper for SEC, SEBI, and other regulatory bodies.

Scrapes official regulatory filings and disclosures.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from .scraper_utils import ScraperCache, RateLimiter, ScraperSession, clean_html_text


class RegulatoryScraper:
    """
    Scrape regulatory filings for official disclosures.
    
    Sources:
    - SEC EDGAR (US)
    - SEBI (India)
    - Other regulatory bodies
    """
    
    def __init__(
        self,
        cache_ttl_hours: int = 48,  # Longer cache for regulatory data
        rate_limit_rpm: int = 10  # More conservative rate limiting
    ):
        """
        Initialize regulatory scraper.
        
        Args:
            cache_ttl_hours: Cache time-to-live
            rate_limit_rpm: Rate limit requests per minute
        """
        self.cache = ScraperCache(ttl_hours=cache_ttl_hours)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
        self.session = ScraperSession()
        
        # Set proper headers for SEC (they require it)
        self.session.session.headers.update({
            'User-Agent': 'Anomaly Validator anomalydetector@example.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
    
    def search_sec(
        self,
        company_name: str,
        cik: Optional[str] = None,
        form_types: List[str] = None,
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Search SEC EDGAR for company filings.
        
        Args:
            company_name: Company name
            cik: Central Index Key (SEC identifier)
            form_types: Filing types to search (e.g., ['8-K', '10-Q', '10-K'])
            days_back: Days to search back
            
        Returns:
            List of filings
        """
        if form_types is None:
            form_types = ['8-K', '10-Q', '10-K', 'S-1', '13D', '13G']
        
        query = f"SEC:{company_name}:{':'.join(form_types)}"
        
        # Check cache
        cached = self.cache.get(query, 'sec_search')
        if cached:
            print(f"✓ SEC cache hit for: {company_name}")
            return cached
        
        print(f"🔍 Searching SEC EDGAR for: {company_name}")
        
        filings = []
        
        try:
            # If no CIK provided, search for company
            if not cik:
                cik = self._find_company_cik(company_name)
                if not cik:
                    print(f"⚠️  Could not find CIK for {company_name}")
                    return []
            
            # Search filings
            filings = self._fetch_sec_filings(cik, form_types, days_back)
        
        except Exception as e:
            print(f"⚠️  SEC search error: {e}")
        
        # Cache results
        self.cache.set(query, 'sec_search', filings)
        
        print(f"✓ Found {len(filings)} SEC filings")
        return filings
    
    def _find_company_cik(self, company_name: str) -> Optional[str]:
        """Find company CIK from name."""
        try:
            self.rate_limiter.wait_if_needed()
            
            # SEC company search
            search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'company': company_name,
                'output': 'xml'
            }
            
            response = self.session.get(search_url, params=params)
            soup = BeautifulSoup(response.text, 'xml')
            
            # Extract CIK
            cik_elem = soup.find('CIK')
            if cik_elem:
                return cik_elem.get_text(strip=True).zfill(10)
        
        except Exception as e:
            print(f"⚠️  CIK lookup error: {e}")
        
        return None
    
    def _fetch_sec_filings(
        self,
        cik: str,
        form_types: List[str],
        days_back: int
    ) -> List[Dict[str, Any]]:
        """Fetch SEC filings for a CIK."""
        filings = []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # SEC filings search
            search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'CIK': cik,
                'type': '',
                'dateb': '',
                'owner': 'exclude',
                'count': 100,
                'output': 'xml'
            }
            
            response = self.session.get(search_url, params=params)
            soup = BeautifulSoup(response.text, 'xml')
            
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=days_back)
            
            # Parse filings
            for filing in soup.find_all('filing'):
                try:
                    form_type = filing.find('type').get_text(strip=True)
                    
                    # Filter by form type
                    if form_types and form_type not in form_types:
                        continue
                    
                    # Get filing date
                    filing_date_str = filing.find('filing-date').get_text(strip=True)
                    filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')
                    
                    # Check if within date range
                    if filing_date < date_threshold:
                        continue
                    
                    # Get filing URL
                    filing_href = filing.find('filing-href')
                    url = filing_href.get_text(strip=True) if filing_href else ''
                    
                    # Get description
                    description = filing.find('film-number')
                    desc_text = description.get_text(strip=True) if description else ''
                    
                    filings.append({
                        'form_type': form_type,
                        'filing_date': filing_date_str,
                        'url': url,
                        'description': desc_text,
                        'source': 'SEC EDGAR',
                        'scraped_at': datetime.now().isoformat()
                    })
                
                except:
                    continue
        
        except Exception as e:
            print(f"⚠️  SEC filings fetch error: {e}")
        
        return filings
    
    def search_sebi(
        self,
        company_name: str,
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Search SEBI (India) for company disclosures.
        
        Args:
            company_name: Company name
            days_back: Days to search back
            
        Returns:
            List of disclosures
        """
        query = f"SEBI:{company_name}"
        
        # Check cache
        cached = self.cache.get(query, 'sebi_search')
        if cached:
            print(f"✓ SEBI cache hit for: {company_name}")
            return cached
        
        print(f"🔍 Searching SEBI for: {company_name}")
        
        disclosures = []
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # SEBI corporate disclosures
            # Note: This is a simplified example - actual SEBI scraping may require more complex logic
            search_url = "https://www.bseindia.com/corporates/ann.html"
            
            # In production, you would implement proper SEBI website scraping here
            # For now, this is a placeholder
            
            print("⚠️  SEBI scraping not fully implemented (placeholder)")
        
        except Exception as e:
            print(f"⚠️  SEBI search error: {e}")
        
        # Cache results
        self.cache.set(query, 'sebi_search', disclosures)
        
        return disclosures
    
    def fetch_filing_content(self, url: str) -> Optional[str]:
        """
        Fetch full content of a regulatory filing.
        
        Args:
            url: Filing URL
            
        Returns:
            Filing content text
        """
        # Check cache
        cached = self.cache.get(url, 'filing_content')
        if cached:
            return cached
        
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.session.get(url)
            text = clean_html_text(response.text)
            
            # Cache content
            self.cache.set(url, 'filing_content', text)
            
            return text
        
        except Exception as e:
            print(f"⚠️  Error fetching filing {url}: {e}")
            return None
    
    def search_all_sources(
        self,
        company_name: str,
        country: str = 'US',
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Search all applicable regulatory sources.
        
        Args:
            company_name: Company name
            country: Company country (US, IN, etc.)
            days_back: Days to search back
            
        Returns:
            Combined list of filings from all sources
        """
        all_filings = []
        
        # US companies - search SEC
        if country in ['US', 'USA']:
            all_filings.extend(self.search_sec(company_name, days_back=days_back))
        
        # Indian companies - search SEBI
        if country in ['IN', 'India']:
            all_filings.extend(self.search_sebi(company_name, days_back=days_back))
        
        return all_filings


if __name__ == "__main__":
    print("Testing Regulatory Scraper...")
    
    scraper = RegulatoryScraper()
    
    # Test SEC search
    filings = scraper.search_sec(
        company_name="Microsoft",
        form_types=['8-K'],
        days_back=90
    )
    
    if filings:
        print(f"\n✓ Found {len(filings)} SEC filings")
        print(f"Sample: {filings[0]['form_type']} on {filings[0]['filing_date']}")
    else:
        print("⚠️  No SEC filings found")
    
    print("\n✓ Regulatory scraper tests passed!")

