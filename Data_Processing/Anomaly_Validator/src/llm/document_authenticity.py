"""
Document Authenticity Checker

Uses LLM to detect signs of fake, tampered, or fraudulent documents.
"""

from typing import Dict, Any, List
from .gemini_client import GeminiClient


class DocumentAuthenticityChecker:
    """
    Check if supporting documents show signs of being fake or tampered.
    
    Uses Gemini to analyze document characteristics and detect fraud indicators.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize authenticity checker.
        
        Args:
            gemini_client: GeminiClient instance
        """
        self.gemini_client = gemini_client
    
    def check_authenticity(
        self,
        document_text: str,
        document_metadata: Dict[str, Any],
        anomaly_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if document is authentic or shows signs of fraud.
        
        Args:
            document_text: Extracted document text
            document_metadata: Document metadata (file info, dates, etc.)
            anomaly_context: Context about the anomaly
            
        Returns:
            Dictionary with:
                - authenticity: "AUTHENTIC", "SUSPICIOUS", or "FAKE"
                - confidence: 0-1 confidence score
                - fraud_indicators: List of detected fraud signs
                - reasoning: Explanation
        """
        system_instruction = (
            "You are a forensic document examiner and fraud investigator specializing in "
            "detecting fake or tampered financial documents. Analyze documents for signs "
            "of fraud, forgery, or manipulation."
        )
        
        metadata_str = "\n".join([f"- {k}: {v}" for k, v in document_metadata.items()])
        
        prompt = f"""Analyze this document for authenticity. Detect signs of fraud, tampering, or forgery.

**Document Content:**
{document_text[:2000]}{'...[truncated]' if len(document_text) > 2000 else ''}

**Document Metadata:**
{metadata_str}

**Anomaly Context:**
- GL Account: {anomaly_context.get('gl_account')}
- Amount: ${anomaly_context.get('amount', 0):,.2f}
- Period: {anomaly_context.get('period')}

**Fraud Detection Checklist:**

1. **Formatting Issues:**
   - Inconsistent fonts, sizes, or styles
   - Poor quality or pixelated elements
   - Misaligned text or numbers
   - Unusual spacing or kerning

2. **Content Inconsistencies:**
   - Dates that don't make sense (future dates, wrong order)
   - Amounts that are suspiciously round or oddly specific
   - Missing standard document elements (letterhead, signatures, reference numbers)
   - Contradictory information within the document

3. **Metadata Red Flags:**
   - Creation/modification dates don't match document date
   - File created recently but claims to be old
   - Multiple rapid edits suggesting fabrication

4. **Business Logic Issues:**
   - Transaction doesn't match typical patterns
   - Vendor/party not found in public records
   - Terms or conditions that are unusual
   - Amount doesn't align with business scale

5. **Language/Style Issues:**
   - Unusual wording or grammar for official documents
   - Too perfect or too sloppy
   - Generic language lacking specific details
   - AI-generated appearance

6. **Supporting Evidence Missing:**
   - No reference to related documents
   - Can't be verified externally
   - No audit trail or approval workflow
   - Isolated transaction with no context

**Output Format (JSON):**
{{
    "authenticity": "AUTHENTIC/SUSPICIOUS/FAKE",
    "confidence": <0.0-1.0>,
    "fraud_indicators": ["<indicator1>", "<indicator2>", ...] or [],
    "reasoning": "<detailed explanation>",
    "red_flags_found": <number>,
    "recommendation": "ACCEPT/INVESTIGATE/REJECT"
}}

**Assessment Guidelines:**
- AUTHENTIC: No concerning signs, passes all checks (confidence > 0.8)
- SUSPICIOUS: Some red flags but not conclusive (confidence 0.4-0.8)
- FAKE: Multiple serious red flags indicating forgery (confidence > 0.8)"""

        response = self.gemini_client.generate(prompt, system_instruction)
        
        # Parse response
        try:
            import json
            json_text = response
            
            # Check if response contains an error
            if response.startswith('{"error":'):
                result = json.loads(response)
                error_msg = result.get("message", "Unknown error")
                
                return {
                    "authenticity": "ANALYSIS_ERROR",
                    "confidence": 0.3,
                    "fraud_indicators": [],
                    "reasoning": f"Could not analyze document authenticity: {error_msg}",
                    "red_flags_found": 0,
                    "recommendation": "INVESTIGATE"
                }
            
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_text)
            return result
        except:
            return {
                "authenticity": "SUSPICIOUS",
                "confidence": 0.5,
                "fraud_indicators": [],
                "reasoning": response if len(response) < 500 else response[:500] + "...",
                "red_flags_found": 0,
                "recommendation": "INVESTIGATE"
            }
    
    def batch_check_documents(
        self,
        documents: List[Dict[str, Any]],
        anomaly_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check multiple documents and provide consolidated assessment.
        
        Args:
            documents: List of parsed documents
            anomaly_context: Anomaly context
            
        Returns:
            Consolidated authenticity assessment
        """
        if not documents:
            return {
                "overall_authenticity": "NO_DOCUMENTS",
                "confidence": 0.0,
                "document_count": 0,
                "authentic_count": 0,
                "suspicious_count": 0,
                "fake_count": 0,
                "all_fraud_indicators": [],
                "recommendation": "RED_FLAG"
            }
        
        results = []
        for doc in documents:
            doc_text = doc.get('text', '')
            doc_metadata = doc.get('metadata', {})
            
            result = self.check_authenticity(doc_text, doc_metadata, anomaly_context)
            results.append(result)
        
        # Aggregate results
        authentic_count = sum(1 for r in results if r['authenticity'] == 'AUTHENTIC')
        suspicious_count = sum(1 for r in results if r['authenticity'] == 'SUSPICIOUS')
        fake_count = sum(1 for r in results if r['authenticity'] == 'FAKE')
        
        all_fraud_indicators = []
        for r in results:
            all_fraud_indicators.extend(r.get('fraud_indicators', []))
        
        # Overall assessment
        if fake_count > 0:
            overall = "FAKE"
            recommendation = "RED_FLAG"
        elif suspicious_count > authentic_count:
            overall = "SUSPICIOUS"
            recommendation = "RED_FLAG"
        elif authentic_count == len(documents):
            overall = "AUTHENTIC"
            recommendation = "YELLOW_FLAG"  # Can be yellow if all docs are authentic
        else:
            overall = "MIXED"
            recommendation = "RED_FLAG"
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        return {
            "overall_authenticity": overall,
            "confidence": avg_confidence,
            "document_count": len(documents),
            "authentic_count": authentic_count,
            "suspicious_count": suspicious_count,
            "fake_count": fake_count,
            "all_fraud_indicators": list(set(all_fraud_indicators)),
            "individual_results": results,
            "recommendation": recommendation
        }


if __name__ == "__main__":
    print("Document Authenticity Checker initialized")
    print("✓ Ready to detect fake and tampered documents")

