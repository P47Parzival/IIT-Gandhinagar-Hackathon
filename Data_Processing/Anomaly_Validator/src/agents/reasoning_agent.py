"""
Reasoning Agent - Validates anomalies and assigns priority flags.

Combines evidence from DocumentAgent and ResearchAgent to make the
final decision: RED_FLAG (urgent) or YELLOW_FLAG (explained).

CRITICAL: Never unflags anomalies - all require human review.
"""

from typing import Dict, Any, List
import sys
import os

# Handle both relative and absolute imports
try:
    from ..agents.base_agent import BaseAgent
    from ..llm import GeminiClient
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.base_agent import BaseAgent
    from llm import GeminiClient


class ReasoningAgent(BaseAgent):
    """
    Agent responsible for:
    1. Analyzing all gathered evidence (documents + web)
    2. Checking document authenticity and fraud indicators
    3. Assessing explanation completeness
    4. Assigning RED_FLAG or YELLOW_FLAG (never unflags)
    
    This is the critical decision-making agent.
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Dict[str, Any] = None
    ):
        """
        Initialize reasoning agent.
        
        Args:
            gemini_client: Gemini client for analysis
            config: Configuration options
        """
        super().__init__(
            agent_name="ReasoningAgent",
            agent_type="reasoning",
            gemini_client=gemini_client,
            config=config or {}
        )
        
        # Load validation thresholds from config
        self.relevance_threshold = self.config.get('relevance_threshold', 7.0)
        self.authenticity_threshold = self.config.get('authenticity_confidence_threshold', 0.8)
        self.explanation_threshold = self.config.get('explanation_quality_threshold', 0.9)
        self.yellow_flag_confidence = self.config.get('yellow_flag_confidence_threshold', 0.8)
        self.max_fraud_indicators = self.config.get('max_fraud_indicators_for_yellow', 0)
        
        print("✓ Reasoning Agent initialized (NEVER unflags anomalies)")
    
    def _process(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str, float]:
        """
        Process reasoning task - the core validation logic.
        
        Args:
            task: Contains anomaly_data, document_results, research_results
            context: Additional context
            
        Returns:
            Tuple of (result, reasoning, confidence)
        """
        anomaly_data = task.get('anomaly_data', {})
        document_results = task.get('document_results', {})
        research_results = task.get('research_results', {})
        
        # Step 1: Check document authenticity (fraud detection first)
        auth_assessment = document_results.get('authenticity_assessment', {})
        auth_status = auth_assessment.get('overall_authenticity', 'UNKNOWN')
        fraud_indicators = auth_assessment.get('all_fraud_indicators', [])
        
        # Automatic RED_FLAG if documents are fake or suspicious
        if auth_status in ['FAKE', 'SUSPICIOUS'] or len(fraud_indicators) > self.max_fraud_indicators:
            return self._create_red_flag_result(
                "Fraudulent or suspicious documents detected",
                anomaly_data,
                document_results,
                research_results,
                auth_status,
                fraud_indicators
            )
        
        # Step 2: Check if documents are present and relevant
        docs_found = document_results.get('documents_found', 0)
        highly_relevant = document_results.get('highly_relevant_count', 0)
        avg_relevance = document_results.get('average_relevance', 0.0)
        
        if docs_found == 0:
            return self._create_red_flag_result(
                "No supporting documents provided",
                anomaly_data,
                document_results,
                research_results,
                'NO_DOCUMENTS',
                []
            )
        
        if highly_relevant == 0 or avg_relevance < self.relevance_threshold:
            return self._create_red_flag_result(
                f"Documents not sufficiently relevant (avg {avg_relevance:.1f}/10)",
                anomaly_data,
                document_results,
                research_results,
                auth_status,
                fraud_indicators
            )
        
        # Step 3: Use Gemini to analyze all evidence
        gemini_analysis = self.gemini_client.analyze_anomaly(
            anomaly_data=anomaly_data,
            document_analysis=document_results.get('relevance_assessments', []),
            web_context=research_results,
            institutional_knowledge=context.get('institutional_knowledge')
        )
        
        # Step 4: Validate Gemini's decision (enforce RED/YELLOW only)
        decision = gemini_analysis.get('decision', 'RED_FLAG')
        if decision not in ['RED_FLAG', 'YELLOW_FLAG']:
            decision = 'RED_FLAG'  # Safety fallback
        
        # Step 5: Verify yellow flag requirements if Gemini suggested it
        if decision == 'YELLOW_FLAG':
            yellow_flag_valid, violation_reason = self._verify_yellow_flag_requirements(
                gemini_analysis,
                auth_assessment,
                avg_relevance,
                fraud_indicators
            )
            
            if not yellow_flag_valid:
                # Override to RED_FLAG if requirements not met
                decision = 'RED_FLAG'
                gemini_analysis['decision'] = 'RED_FLAG'
                gemini_analysis['override_reason'] = violation_reason
        
        # Build final result
        result = {
            'decision': decision,
            'confidence': gemini_analysis.get('confidence', 0.5),
            'document_authenticity': auth_status,
            'fraud_indicators': fraud_indicators,
            'explanation_quality': gemini_analysis.get('explanation_quality', 'NONE'),
            'requires_urgent_review': decision == 'RED_FLAG',
            'gemini_reasoning': gemini_analysis.get('reasoning', ''),
            'evidence': gemini_analysis.get('evidence', []),
            'document_count': docs_found,
            'relevant_document_count': highly_relevant,
            'web_sources_count': research_results.get('total_articles', 0),
            'web_validation': research_results.get('web_validation', 'NO_INFORMATION')
        }
        
        reasoning = self._build_reasoning(result, gemini_analysis)
        confidence = result['confidence']
        
        return (result, reasoning, confidence)
    
    def _verify_yellow_flag_requirements(
        self,
        gemini_analysis: Dict[str, Any],
        auth_assessment: Dict[str, Any],
        avg_relevance: float,
        fraud_indicators: List[str]
    ) -> tuple[bool, str]:
        """
        Verify ALL yellow flag requirements are met.
        
        Returns:
            Tuple of (is_valid, violation_reason)
        """
        # Requirement 1: Document authenticity
        auth_confidence = auth_assessment.get('confidence', 0.0)
        if auth_confidence < self.authenticity_threshold:
            return False, f"Document authenticity confidence too low ({auth_confidence:.2f} < {self.authenticity_threshold})"
        
        # Requirement 2: No fraud indicators
        if len(fraud_indicators) > self.max_fraud_indicators:
            return False, f"Fraud indicators detected: {len(fraud_indicators)}"
        
        # Requirement 3: Document relevance
        if avg_relevance < self.relevance_threshold:
            return False, f"Document relevance too low ({avg_relevance:.1f} < {self.relevance_threshold})"
        
        # Requirement 4: Explanation quality
        explanation_quality = gemini_analysis.get('explanation_quality', 'NONE')
        if explanation_quality != 'COMPLETE':
            return False, f"Explanation not complete ({explanation_quality})"
        
        # Requirement 5: Overall confidence
        confidence = gemini_analysis.get('confidence', 0.0)
        if confidence < self.yellow_flag_confidence:
            return False, f"Overall confidence too low ({confidence:.2f} < {self.yellow_flag_confidence})"
        
        return True, ""
    
    def _create_red_flag_result(
        self,
        reason: str,
        anomaly_data: Dict[str, Any],
        document_results: Dict[str, Any],
        research_results: Dict[str, Any],
        auth_status: str,
        fraud_indicators: List[str]
    ) -> tuple[Dict[str, Any], str, float]:
        """Create RED_FLAG result when automatic conditions trigger."""
        result = {
            'decision': 'RED_FLAG',
            'confidence': 0.9,  # High confidence in RED_FLAG
            'document_authenticity': auth_status,
            'fraud_indicators': fraud_indicators,
            'explanation_quality': 'NONE',
            'requires_urgent_review': True,
            'automatic_red_flag_reason': reason,
            'document_count': document_results.get('documents_found', 0),
            'relevant_document_count': document_results.get('highly_relevant_count', 0),
            'web_sources_count': research_results.get('total_articles', 0),
            'web_validation': research_results.get('web_validation', 'NO_INFORMATION')
        }
        
        reasoning = f"RED_FLAG (Automatic): {reason}"
        
        return (result, reasoning, 0.9)
    
    def _build_reasoning(
        self,
        result: Dict[str, Any],
        gemini_analysis: Dict[str, Any]
    ) -> str:
        """Build comprehensive reasoning summary."""
        parts = []
        
        decision = result['decision']
        parts.append(f"**DECISION: {decision}**")
        
        # Document assessment
        doc_count = result['document_count']
        relevant_count = result['relevant_document_count']
        auth_status = result['document_authenticity']
        
        parts.append(
            f"Documents: {doc_count} found, {relevant_count} highly relevant, "
            f"authenticity: {auth_status}"
        )
        
        # Fraud indicators
        fraud_count = len(result.get('fraud_indicators', []))
        if fraud_count > 0:
            parts.append(f"⚠️  {fraud_count} fraud indicators detected")
        
        # Web validation
        web_validation = result.get('web_validation', 'NO_INFORMATION')
        web_count = result.get('web_sources_count', 0)
        parts.append(f"Web research: {web_count} sources, validation: {web_validation}")
        
        # Explanation quality
        explanation_quality = result.get('explanation_quality', 'NONE')
        parts.append(f"Explanation quality: {explanation_quality}")
        
        # Gemini reasoning (abbreviated)
        gemini_reasoning = gemini_analysis.get('reasoning', '')
        if gemini_reasoning:
            parts.append(f"Analysis: {gemini_reasoning[:200]}...")
        
        # Override reason if present
        if 'override_reason' in gemini_analysis:
            parts.append(f"⚠️  Overridden to RED_FLAG: {gemini_analysis['override_reason']}")
        
        return " | ".join(parts)


if __name__ == "__main__":
    print("Reasoning Agent initialized")
    print("✓ Ready to validate anomalies (RED_FLAG or YELLOW_FLAG only)")

