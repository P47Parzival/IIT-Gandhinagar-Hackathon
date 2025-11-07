"""
Research Agent - Performs web scraping and external context gathering.

Based on Paper 2's Web Research Agent that verifies anomalies through 
external sources (news, press releases, regulatory filings).
"""

from typing import Dict, Any, List
import sys
import os

# Handle both relative and absolute imports
try:
    from ..agents.base_agent import BaseAgent
    from ..web_scraping import NewsScraper, CompanyScraper, RegulatoryScraper
    from ..llm import GeminiClient, RAGEngine
    from ..web_scraping.scraper_utils import extract_keywords_from_anomaly
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.base_agent import BaseAgent
    from web_scraping import NewsScraper, CompanyScraper, RegulatoryScraper
    from llm import GeminiClient, RAGEngine
    from web_scraping.scraper_utils import extract_keywords_from_anomaly


class ResearchAgent(BaseAgent):
    """
    Agent responsible for:
    1. Web scraping (news, company site, regulatory)
    2. RAG processing of scraped content
    3. Extracting relevant context using embeddings
    4. Providing focused external validation
    
    This implements Paper 2's Web Research Agent with RAG optimization.
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Dict[str, Any] = None
    ):
        """
        Initialize research agent.
        
        Args:
            gemini_client: Gemini client for LLM + RAG
            config: Configuration options
        """
        super().__init__(
            agent_name="ResearchAgent",
            agent_type="research",
            gemini_client=gemini_client,
            config=config or {}
        )
        
        # Initialize scrapers
        self.news_scraper = NewsScraper()
        self.company_scraper = CompanyScraper()
        self.regulatory_scraper = RegulatoryScraper()
        
        # Initialize RAG engine with Gemini PARALLEL embedding (6 API keys)
        self.rag = RAGEngine(
            api_key=gemini_client.api_key,
            chunk_size=self.config.get('rag_chunk_size', 1000),  # 1000 chars per chunk
            top_k=self.config.get('rag_top_k', 5),
            use_a4f=False  # Use Gemini API with parallel processing
        )
        
        print("✓ Research Agent initialized with RAG engine")
    
    def _process(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str, float]:
        """
        Process web research task.
        
        Args:
            task: Contains anomaly_data
            context: Additional context
            
        Returns:
            Tuple of (result, reasoning, confidence)
        """
        anomaly_data = task.get('anomaly_data', {})
        
        # Extract search parameters
        entity_name = anomaly_data.get('entity_name') or anomaly_data.get('entity_id')
        company_domain = anomaly_data.get('company_domain')
        country = anomaly_data.get('country', 'US')
        period = anomaly_data.get('period')
        
        keywords = extract_keywords_from_anomaly(anomaly_data)
        
        print(f"  Researching: {entity_name} with keywords {keywords}")
        
        # Step 1: Parallel web scraping
        scraping_results = self._scrape_all_sources(
            entity_name,
            company_domain,
            country,
            keywords,
            period
        )
        
        if not scraping_results['articles']:
            return (
                {
                    'sources_searched': scraping_results['sources_searched'],
                    'total_articles': 0,
                    'rag_context': '',
                    'web_validation': 'NO_INFORMATION',
                    'external_events': []
                },
                "No external information found through web research",
                0.0
            )
        
        # Step 2: LLM Summarization (NEW PRE-RAG LAYER)
        # Send all scraped content to Gemini Flash 2.5 for comprehensive summarization
        print(f"  📝 Stage 1: LLM Summarization (Gemini Flash 2.5)")
        llm_summary = self._summarize_with_llm(scraping_results['articles'], anomaly_data)
        
        # Step 3: Load LLM summary into RAG (NOT raw content)
        print(f"  📝 Stage 2: RAG Processing")
        self._load_summary_into_rag(llm_summary, scraping_results['articles'])
        
        # Step 4: Query RAG for relevant context from the summary
        rag_context = self._query_rag_for_anomaly(anomaly_data)
        
        # Step 5: Analyze if web context explains anomaly
        analysis = self._analyze_web_context(anomaly_data, rag_context)
        
        result = {
            'sources_searched': scraping_results['sources_searched'],
            'total_articles': len(scraping_results['articles']),
            'news_count': len(scraping_results.get('news', [])),
            'company_count': len(scraping_results.get('company', [])),
            'regulatory_count': len(scraping_results.get('regulatory', [])),
            'llm_summary_length': len(llm_summary),
            'rag_context': rag_context,
            'rag_chunks_retrieved': self.config.get('rag_top_k', 5),
            'web_validation': analysis['validation'],
            'explains_anomaly': analysis['explains'],
            'external_events': analysis['events'],
            'supporting_evidence': analysis['evidence']
        }
        
        reasoning = self._build_reasoning(result, analysis)
        confidence = analysis['confidence']
        
        return (result, reasoning, confidence)
    
    def _scrape_all_sources(
        self,
        entity_name: str,
        company_domain: str,
        country: str,
        keywords: List[str],
        period: str
    ) -> Dict[str, Any]:
        """Scrape all web sources in parallel."""
        all_articles = []
        sources_searched = []
        results = {'articles': []}
        
        # News scraping
        try:
            news_articles = self.news_scraper.search(entity_name, keywords)
            
            # Fetch full content for top articles (for better RAG embeddings)
            print(f"  📥 Fetching full content for top {min(5, len(news_articles))} articles...")
            for i, article in enumerate(news_articles[:5]):  # Top 5 articles
                url = article.get('url')
                if url and not article.get('text'):  # Only if no content yet
                    try:
                        content = self.news_scraper.fetch_article_content(url)
                        if content:
                            article['text'] = content
                            print(f"     ✓ {i+1}. Fetched {len(content)} chars")
                    except Exception as fetch_error:
                        print(f"     ⚠️  {i+1}. Failed: {str(fetch_error)[:50]}")
            
            all_articles.extend(news_articles)
            sources_searched.append('news')
            results['news'] = news_articles
            print(f"  ✓ News: {len(news_articles)} articles (content fetched for {sum(1 for a in news_articles[:5] if a.get('text'))} articles)")
        except Exception as e:
            print(f"  ⚠️  News scraping failed: {e}")
        
        # Company website scraping
        if company_domain:
            try:
                company_pages = self.company_scraper.search(company_domain, keywords)
                all_articles.extend(company_pages)
                sources_searched.append('company_website')
                results['company'] = company_pages
                print(f"  ✓ Company: {len(company_pages)} pages")
            except Exception as e:
                print(f"  ⚠️  Company scraping failed: {e}")
        
        # Regulatory filings
        try:
            filings = self.regulatory_scraper.search_all_sources(entity_name, country)
            all_articles.extend(filings)
            sources_searched.append('regulatory')
            results['regulatory'] = filings
            print(f"  ✓ Regulatory: {len(filings)} filings")
        except Exception as e:
            print(f"  ⚠️  Regulatory scraping failed: {e}")
        
        results['articles'] = all_articles
        results['sources_searched'] = sources_searched
        
        return results
    
    def _summarize_with_llm(
        self,
        articles: List[Dict[str, Any]],
        anomaly_data: Dict[str, Any]
    ) -> str:
        """
        NEW: Use Gemini Flash 2.5 to summarize all web content before RAG.
        
        This creates a comprehensive summary that preserves important details
        while reducing the content size for more efficient RAG processing.
        
        Flow: Raw scraped content (50k+ chars) → LLM summary (10-20k chars) → RAG (2.5k chars)
        """
        try:
            summary = self.gemini_client.summarize_web_content(
                articles=articles,
                anomaly_context=anomaly_data,
                max_summary_length=15000  # Target 15k chars for summary
            )
            return summary
        except Exception as e:
            print(f"  ⚠️  LLM summarization failed: {e}")
            print(f"  ℹ️  Falling back to direct RAG loading")
            # Fallback: concatenate article snippets
            fallback = "\n\n".join([
                f"{a.get('title', 'Article')}: {a.get('text', a.get('snippet', ''))[:1000]}"
                for a in articles[:10]  # Top 10 articles
            ])
            return fallback
    
    def _load_summary_into_rag(
        self,
        llm_summary: str,
        original_articles: List[Dict[str, Any]]
    ):
        """
        Load LLM-generated summary into RAG engine (instead of raw articles).
        
        The summary is treated as a single comprehensive document that RAG
        will chunk and retrieve from.
        """
        # Clear previous content
        self.rag.clear()
        
        # Load the summary as a single document
        if llm_summary and llm_summary != "No web content available to summarize.":
            # Prepare sources list for reference
            sources = [
                article.get('url') or article.get('source', 'Unknown')
                for article in original_articles
            ]
            sources_text = ", ".join(sources[:5])  # Top 5 sources
            if len(sources) > 5:
                sources_text += f" (and {len(sources)-5} more)"
            
            rag_doc = {
                'text': llm_summary,
                'source': f"LLM Summary of {len(original_articles)} sources: {sources_text}",
                'title': 'Comprehensive Web Research Summary',
                'date': 'Summarized from multiple sources'
            }
            
            self.rag.add_documents([rag_doc], text_key='text', source_key='source')
            print(f"  ✓ Loaded LLM summary ({len(llm_summary):,} chars) into RAG")
            print(f"  📊 RAG will chunk this into {self.rag.get_stats().get('total_chunks', 0)} retrievable chunks")
        else:
            print(f"  ⚠️  No valid summary to load into RAG")
    
    def _query_rag_for_anomaly(self, anomaly_data: Dict[str, Any]) -> str:
        """Query RAG for relevant context about the anomaly."""
        # Build query
        entity = anomaly_data.get('entity_name') or anomaly_data.get('entity_id')
        amount = anomaly_data.get('amount', 0)
        gl_name = anomaly_data.get('gl_name', '')
        period = anomaly_data.get('period', '')
        
        query = (
            f"Why did {entity} have unusual {gl_name} transaction of "
            f"${abs(amount):,.0f} in {period}? "
            f"What external events, announcements, or transactions explain this?"
        )
        
        # Get relevant context from RAG
        context = self.rag.get_context_for_query(
            query,
            max_context_length=3000,
            include_sources=True
        )
        
        return context
    
    def _analyze_web_context(
        self,
        anomaly_data: Dict[str, Any],
        rag_context: str
    ) -> Dict[str, Any]:
        """Analyze if web context explains the anomaly."""
        if not rag_context or rag_context == "No relevant information found.":
            return {
                'validation': 'NO_INFORMATION',
                'explains': False,
                'confidence': 0.0,
                'events': [],
                'evidence': []
            }
        
        # Use Gemini to analyze
        system_instruction = (
            "You are analyzing web research results to determine if external "
            "sources explain a detected financial anomaly."
        )
        
        prompt = f"""Analyze this web research context to determine if it explains the detected anomaly.

**Anomaly:**
- Entity: {anomaly_data.get('entity_name') or anomaly_data.get('entity_id')}
- GL Account: {anomaly_data.get('gl_account')} ({anomaly_data.get('gl_name')})
- Amount: ${anomaly_data.get('amount', 0):,.2f}
- Period: {anomaly_data.get('period')}

**Web Research Context (from {self.rag.get_stats()['total_chunks']} sources via RAG):**
{rag_context}

**Analysis Task:**
1. Does the web context mention this transaction or explain why it occurred?
2. Are there external events (acquisitions, sales, announcements) that validate this anomaly?
3. What evidence from web sources supports or contradicts the transaction?

**Output Format (JSON):**
{{
    "validation": "EXPLAINS/PARTIAL/NO_EXPLANATION/CONTRADICTS",
    "explains": true/false,
    "confidence": <0.0-1.0>,
    "events": ["<event1>", "<event2>", ...],
    "evidence": ["<evidence1>", "<evidence2>", ...],
    "summary": "<brief summary of findings>"
}}"""

        try:
            response = self.gemini_client.generate(prompt, system_instruction)
            
            # Check if response is an error JSON
            import json
            if response.startswith('{"error":'):
                error_data = json.loads(response)
                print(f"  ⚠️  Web context analysis failed: {error_data.get('message', 'Unknown error')}")
                return {
                    'validation': 'ANALYSIS_ERROR',
                    'explains': False,
                    'confidence': 0.3,
                    'events': [],
                    'evidence': [],
                    'summary': f"Analysis failed: {error_data.get('error', 'unknown')}"
                }
            
            # Parse JSON
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(json_text)
            return analysis
        
        except Exception as e:
            print(f"  ⚠️  Web context parsing failed: {str(e)[:100]}")
            return {
                'validation': 'ANALYSIS_ERROR',
                'explains': False,
                'confidence': 0.3,
                'events': [],
                'evidence': [],
                'summary': 'Failed to parse analysis results'
            }
    
    def _build_reasoning(
        self,
        result: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Build reasoning summary."""
        parts = [
            f"Searched {len(result['sources_searched'])} sources, "
            f"found {result['total_articles']} articles."
        ]
        
        validation = analysis.get('validation', 'UNKNOWN')
        
        if validation == 'EXPLAINS':
            parts.append(
                f"External sources EXPLAIN the anomaly. "
                f"Found {len(analysis.get('events', []))} relevant events."
            )
        elif validation == 'PARTIAL':
            parts.append(
                "External sources provide PARTIAL context but not complete explanation."
            )
        elif validation == 'CONTRADICTS':
            parts.append(
                "⚠️  External sources CONTRADICT the transaction!"
            )
        else:
            parts.append(
                "No external explanation found in web research."
            )
        
        return " ".join(parts)


if __name__ == "__main__":
    print("Research Agent initialized")
    print("✓ Ready to perform web research with RAG")

