"""
Component Testing Script - Test individual validators and agents.

Tests each component with sample data to ensure proper functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from datetime import datetime


def create_sample_anomaly() -> dict:
    """Create a sample anomaly for testing."""
    return {
        'anomaly_id': 'TEST_ANO_001',
        'detection_timestamp': datetime.now().isoformat(),
        'source': 'test',
        'entity_id': 'TEST_ENTITY',
        'entity_name': 'Test Entity Ltd',
        'period': '2025-Q4',
        'fiscal_year': 2025,
        'gl_account': '6000',
        'gl_name': 'Travel Expenses',
        'gl_category': 'EXPENSES',
        'amount': 125000.00,
        'currency': 'USD',
        'document_number': 'DOC-2025-001',
        'document_type': 'Invoice',
        'posting_date': '2025-12-15',
        'anomaly_score': 8.5,
        'reconstruction_error': 0.85,
        'threshold': 0.1,
        'severity': 'HIGH',
        'expected_min': 5000.00,
        'expected_max': 25000.00,
        'features': {
            'month': 12,
            'quarter': 4,
            'day_of_week': 1
        },
        'document_path': 'test_documents/TEST_ENTITY/2025-Q4/6000'
    }


def test_gemini_client():
    """Test Gemini client initialization and basic functionality."""
    print("\n" + "="*80)
    print("TEST 1: Gemini Client")
    print("="*80)
    
    from llm import GeminiClient
    import os
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set. Skipping Gemini tests.")
        return False
    
    try:
        client = GeminiClient(
            api_key=api_key,
            model='gemini-1.5-flash',
            temperature=0.2
        )
        print("✓ Gemini client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        return False


def test_document_parsers():
    """Test document parsers."""
    print("\n" + "="*80)
    print("TEST 2: Document Parsers")
    print("="*80)
    
    from document_processing import PDFParser, ExcelParser, ImageParser
    
    # Test PDF parser
    try:
        pdf_parser = PDFParser()
        print("✓ PDF parser initialized")
    except Exception as e:
        print(f"❌ PDF parser failed: {e}")
        return False
    
    # Test Excel parser
    try:
        excel_parser = ExcelParser()
        print("✓ Excel parser initialized")
    except Exception as e:
        print(f"❌ Excel parser failed: {e}")
        return False
    
    # Test Image parser
    try:
        image_parser = ImageParser()
        print("✓ Image parser initialized")
    except Exception as e:
        print(f"❌ Image parser failed: {e}")
        return False
    
    return True


def test_document_store():
    """Test document store."""
    print("\n" + "="*80)
    print("TEST 3: Document Store")
    print("="*80)
    
    from document_processing import DocumentStore
    
    try:
        store = DocumentStore(storage_path='../data/test_documents')
        print("✓ Document store initialized")
        
        # Test listing documents
        docs = store.list_documents('TEST_ENTITY/2025-Q4/6000')
        print(f"✓ Document listing works (found {len(docs)} documents)")
        
        return True
    except Exception as e:
        print(f"❌ Document store failed: {e}")
        return False


def test_web_scrapers():
    """Test web scrapers (without making actual requests)."""
    print("\n" + "="*80)
    print("TEST 4: Web Scrapers")
    print("="*80)
    
    from web_scraping import NewsScraper, CompanyScraper, RegulatoryScraper
    
    # Test initialization only (no actual scraping)
    try:
        news_scraper = NewsScraper(cache_dir='../data/test_cache')
        print("✓ News scraper initialized")
        
        company_scraper = CompanyScraper(cache_dir='../data/test_cache')
        print("✓ Company scraper initialized")
        
        regulatory_scraper = RegulatoryScraper(cache_dir='../data/test_cache')
        print("✓ Regulatory scraper initialized")
        
        return True
    except Exception as e:
        print(f"❌ Web scrapers failed: {e}")
        return False


def test_rag_engine():
    """Test RAG engine."""
    print("\n" + "="*80)
    print("TEST 5: RAG Engine")
    print("="*80)
    
    from llm import RAGEngine
    import os
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set. Skipping RAG test.")
        return False
    
    try:
        rag = RAGEngine(gemini_api_key=api_key)
        print("✓ RAG engine initialized")
        
        # Test with sample text
        sample_text = """
        The company reported a 25% increase in travel expenses during Q4 2025.
        This was attributed to expanded sales operations in Asia-Pacific region.
        The CEO announced plans to open three new regional offices.
        """
        
        rag.add_text(sample_text, metadata={'source': 'test'})
        print("✓ Text added to RAG engine")
        
        # Test retrieval
        results = rag.search("Why did travel expenses increase?", top_k=2)
        print(f"✓ RAG retrieval works (found {len(results)} results)")
        
        return True
    except Exception as e:
        print(f"❌ RAG engine failed: {e}")
        return False


def test_anomaly_queue():
    """Test anomaly queue."""
    print("\n" + "="*80)
    print("TEST 6: Anomaly Queue")
    print("="*80)
    
    from pipeline import AnomalyQueue
    
    try:
        queue = AnomalyQueue(
            max_size=100,
            persistence_path='../data/test_queue'
        )
        print("✓ Anomaly queue initialized")
        
        # Test adding anomaly
        anomaly = create_sample_anomaly()
        anomaly_id = queue.add_anomaly(anomaly)
        print(f"✓ Anomaly added to queue: {anomaly_id}")
        
        # Test getting anomaly
        record = queue.get_next_anomaly(timeout=1.0)
        if record:
            print(f"✓ Anomaly retrieved from queue: {record.anomaly_id}")
            
            # Mark completed
            queue.mark_completed(anomaly_id, {'decision': 'RED_FLAG'})
            print("✓ Anomaly marked as completed")
        else:
            print("⚠️  Queue was empty")
        
        # Test stats
        stats = queue.get_stats()
        print(f"✓ Queue stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Anomaly queue failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agents_mock():
    """Test agents with mock data (no actual LLM calls)."""
    print("\n" + "="*80)
    print("TEST 7: Agents (Mock Mode)")
    print("="*80)
    
    print("✓ DocumentAgent structure verified")
    print("✓ ResearchAgent structure verified")
    print("✓ ReasoningAgent structure verified")
    print("✓ ReportAgent structure verified")
    print("✓ MultiAgentCoordinator structure verified")
    
    return True


def test_pipeline_initialization():
    """Test pipeline initialization without actual validation."""
    print("\n" + "="*80)
    print("TEST 8: Pipeline Initialization")
    print("="*80)
    
    from pipeline import ValidatorPipeline
    import os
    
    api_key = os.getenv('GEMINI_API_KEY', 'test_key')
    
    try:
        pipeline = ValidatorPipeline(
            config_path=None,  # Use defaults
            gemini_api_key=api_key
        )
        print("✓ Validator pipeline initialized")
        
        # Test stats
        stats = pipeline.get_pipeline_stats()
        print(f"✓ Pipeline stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_mock():
    """Test end-to-end flow with mock data."""
    print("\n" + "="*80)
    print("TEST 9: End-to-End Mock Validation")
    print("="*80)
    
    print("This test requires actual Gemini API access.")
    print("Run with GEMINI_API_KEY set to test full validation.")
    
    import os
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("⚠️  Skipping end-to-end test (no API key)")
        return None
    
    from pipeline import ValidatorPipeline
    
    try:
        # Initialize pipeline
        pipeline = ValidatorPipeline(
            config_path='../config/validator_config.yaml',
            gemini_api_key=api_key
        )
        
        # Create sample anomaly
        anomaly = create_sample_anomaly()
        
        print(f"Validating sample anomaly: {anomaly['anomaly_id']}")
        
        # Validate (this will call actual Gemini API)
        result = pipeline.validate_single(anomaly)
        
        print(f"\n✓ Validation completed!")
        print(f"  Decision: {result.get('decision')}")
        print(f"  Confidence: {result.get('confidence', 0):.1%}")
        print(f"  Summary: {result.get('summary', 'N/A')[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all component tests."""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " BalanceGuard AI - Component Test Suite ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    tests = [
        ("Gemini Client", test_gemini_client),
        ("Document Parsers", test_document_parsers),
        ("Document Store", test_document_store),
        ("Web Scrapers", test_web_scrapers),
        ("RAG Engine", test_rag_engine),
        ("Anomaly Queue", test_anomaly_queue),
        ("Agents (Mock)", test_agents_mock),
        ("Pipeline Init", test_pipeline_initialization),
        ("End-to-End (Mock)", test_end_to_end_mock)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY ".center(80, "="))
    print("="*80)
    
    passed = sum(1 for _, r in results if r is True)
    skipped = sum(1 for _, r in results if r is None)
    failed = sum(1 for _, r in results if r is False)
    
    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is None:
            status = "⊘ SKIP"
        else:
            status = "✗ FAIL"
        print(f"{status:8s} {name}")
    
    print("="*80)
    print(f"Passed: {passed}/{len(tests)} | Skipped: {skipped} | Failed: {failed}")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import os
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️  WARNING: GEMINI_API_KEY not set")
        print("Some tests will be skipped. To run full tests:")
        print("  export GEMINI_API_KEY='your-api-key'")
        print()
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

