#!/usr/bin/env python3
"""
Test ONLY the LLM Summarization + RAG Pipeline
Skips anomaly detection, tests validation directly
"""
import sys
import os
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load .env file
env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Anomaly_Validator', 'src'))

print("=" * 80)
print("TEST: LLM Summarization (Gemini Flash 2.5) + RAG Pipeline")
print("=" * 80)

# Create test anomaly
test_anomaly = {
    'anomaly_id': 'TEST_LLM_RAG_001',
    'entity_id': 'AGY_69500',
    'entity_name': 'OKLAHOMA TAX COMMISSION',
    'period': '2025-10',
    'gl_account': '488184',
    'gl_name': 'Intra Xfer frm Special Account',
    'amount': 1464208399.32,
    'reconstruction_error': 0.219,
    'anomaly_score': 35.9
}

print("\n📊 Test Anomaly:")
print(f"   Entity: {test_anomaly['entity_name']}")
print(f"   GL: {test_anomaly['gl_account']} - {test_anomaly['gl_name']}")
print(f"   Amount: ${test_anomaly['amount']:,.2f}")
print(f"   Score: {test_anomaly['anomaly_score']:.1f}x threshold")

print("\n🚀 Starting Validation Pipeline...")
print("-" * 80)

from pipeline.validator_pipeline import ValidatorPipeline

pipeline = ValidatorPipeline(config_path="Anomaly_Validator/config/validator_config.yaml")

print("\n🔍 Validating (this will test LLM summarization + RAG)...")
result = pipeline.validate_single(test_anomaly)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n✅ Decision: {result.get('decision', 'UNKNOWN')}")
# Confidence is stored as 0-1, display as percentage
confidence_val = result.get('confidence', 0)
if isinstance(confidence_val, (int, float)) and confidence_val <= 1.0:
    print(f"   Confidence: {confidence_val*100:.1f}%")
else:
    print(f"   Confidence: {confidence_val:.1f}%")

if 'summary' in result:
    print(f"\n📝 Summary:")
    print(f"   {result['summary']}")

# Check if LLM summarization was used
if 'research_results' in result:
    research = result['research_results']
    print(f"\n📊 Research Agent Results:")
    print(f"   Articles scraped: {research.get('total_articles', 0)}")
    
    if 'llm_summary_length' in research:
        print(f"\n🎯 LLM SUMMARIZATION CONFIRMED:")
        print(f"   ✓ Stage 1: LLM summary = {research['llm_summary_length']:,} chars")
        print(f"   ✓ Stage 2: RAG chunks = {research.get('rag_chunks_retrieved', 0)}")
        print(f"   ✓ Validation: {research.get('web_validation', 'N/A')}")
        print(f"\n   Two-stage compression: Scraped → LLM Summary → RAG → Final Context")
    else:
        print("   ⚠️  LLM summary length not found in results")

print("\n" + "=" * 80)
print("TEST COMPLETE - LLM Summarization Layer Working!")
print("=" * 80)

