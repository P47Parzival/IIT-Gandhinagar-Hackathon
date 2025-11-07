"""
Report Agent - Generates natural language explanations and reports.

Based on Paper 2's Consolidation and Reporting Agent that synthesizes
all expert findings into human-readable reports.
"""

from typing import Dict, Any, List
from datetime import datetime
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


class ReportAgent(BaseAgent):
    """
    Agent responsible for:
    1. Generating one-line summaries
    2. Creating detailed NLP explanations
    3. Summarizing supporting documents
    4. Providing actionable recommendations
    
    This is the final agent that packages everything for human review.
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Dict[str, Any] = None
    ):
        """
        Initialize report agent.
        
        Args:
            gemini_client: Gemini client for explanation generation
            config: Configuration options
        """
        super().__init__(
            agent_name="ReportAgent",
            agent_type="report",
            gemini_client=gemini_client,
            config=config or {}
        )
        
        print("✓ Report Agent initialized")
    
    def _process(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str, float]:
        """
        Process report generation task.
        
        Args:
            task: Contains anomaly_data, validation_result, all agent results
            context: Additional context
            
        Returns:
            Tuple of (result, reasoning, confidence)
        """
        anomaly_data = task.get('anomaly_data', {})
        validation_result = task.get('validation_result', {})
        document_results = task.get('document_results', {})
        research_results = task.get('research_results', {})
        
        # Generate document summaries
        document_summaries = self._generate_document_summaries(document_results)
        
        # Generate main explanation using Gemini
        try:
        explanation = self.gemini_client.generate_explanation(
            anomaly_data=anomaly_data,
            validation_result=validation_result,
            document_summaries=document_summaries
        )
        except Exception as e:
            print(f"  ⚠️  Explanation generation failed: {str(e)[:100]}")
            # Fallback to basic explanation
            explanation = {
                'summary': f"{validation_result.get('decision', 'UNKNOWN')} - Manual review required",
                'full_explanation': validation_result.get('reasoning', 'No explanation available due to API error.'),
                'document_summary': f"{len(document_summaries)} documents analyzed",
                'recommendation': 'Manual review required due to report generation issues.'
            }
        
        # Build comprehensive report
        try:
        report = self._build_comprehensive_report(
            anomaly_data,
            validation_result,
            document_results,
            research_results,
            explanation
        )
        except Exception as e:
            print(f"  ⚠️  Report building failed: {str(e)[:100]}")
            report = f"# Report Generation Error\n\nFailed to build comprehensive report: {str(e)[:200]}"
        
        # Generate recommendations
        try:
        recommendations = self._generate_recommendations(
            validation_result,
            document_results,
            research_results
        )
        except Exception as e:
            print(f"  ⚠️  Recommendation generation failed: {str(e)[:100]}")
            recommendations = [
                "🔴 Manual review required",
                "⚠️  Report generation encountered errors"
            ]
        
        result = {
            'summary': explanation.get('summary', 'Validation complete'),
            'full_explanation': explanation.get('full_explanation', ''),
            'document_summary': explanation.get('document_summary', ''),
            'recommendation': explanation.get('recommendation', ''),
            'comprehensive_report': report,
            'action_items': recommendations,
            'report_generated_at': datetime.now().isoformat()
        }
        
        reasoning = f"Generated comprehensive report with {len(document_summaries)} document summaries"
        confidence = 1.0  # Report generation doesn't have uncertainty
        
        return (result, reasoning, confidence)
    
    def _generate_document_summaries(
        self,
        document_results: Dict[str, Any]
    ) -> List[str]:
        """Generate brief summaries for each document."""
        summaries = []
        
        relevance_assessments = document_results.get('relevance_assessments', [])
        
        for assessment in relevance_assessments:
            file_name = assessment.get('file_name', 'Unknown')
            relevance_score = assessment.get('relevance_score', 0)
            is_relevant = assessment.get('is_relevant', False)
            reasoning = assessment.get('reasoning', '')
            
            # Extract key findings
            key_findings = assessment.get('key_findings', [])
            findings_str = ', '.join(key_findings[:3]) if key_findings else 'No key findings'
            
            summary = (
                f"{file_name} ({'✓ Relevant' if is_relevant else '✗ Not relevant'}, "
                f"score: {relevance_score}/10): {findings_str}"
            )
            summaries.append(summary)
        
        return summaries
    
    def _build_comprehensive_report(
        self,
        anomaly_data: Dict[str, Any],
        validation_result: Dict[str, Any],
        document_results: Dict[str, Any],
        research_results: Dict[str, Any],
        explanation: Dict[str, Any]
    ) -> str:
        """Build comprehensive HTML/Markdown report."""
        decision = validation_result.get('decision', 'UNKNOWN')
        flag_emoji = "🔴" if decision == "RED_FLAG" else "🟡"
        
        report_parts = []
        
        # Header
        report_parts.append(f"# {flag_emoji} Anomaly Validation Report")
        report_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append(f"**Decision:** {decision}")
        report_parts.append("")
        
        # Anomaly Details
        report_parts.append("## 📊 Anomaly Details")
        report_parts.append(f"- **Entity:** {anomaly_data.get('entity_name') or anomaly_data.get('entity_id')}")
        report_parts.append(f"- **GL Account:** {anomaly_data.get('gl_account')} - {anomaly_data.get('gl_name')}")
        report_parts.append(f"- **Period:** {anomaly_data.get('period')}")
        report_parts.append(f"- **Amount:** ${anomaly_data.get('amount', 0):,.2f}")
        report_parts.append(f"- **Expected Range:** ${anomaly_data.get('expected_min', 0):,.2f} - ${anomaly_data.get('expected_max', 0):,.2f}")
        report_parts.append(f"- **Anomaly Score:** {anomaly_data.get('anomaly_score', 0):.2f}x threshold")
        report_parts.append("")
        
        # Validation Summary
        report_parts.append("## 🎯 Validation Summary")
        report_parts.append(f"**{explanation.get('summary', 'Validation complete')}**")
        report_parts.append("")
        report_parts.append(explanation.get('full_explanation', 'No detailed explanation available.'))
        report_parts.append("")
        
        # Document Analysis
        report_parts.append("## 📄 Document Analysis")
        doc_count = document_results.get('documents_found', 0)
        relevant_count = document_results.get('highly_relevant_count', 0)
        auth_status = validation_result.get('document_authenticity', 'UNKNOWN')
        
        report_parts.append(f"- **Documents Found:** {doc_count}")
        report_parts.append(f"- **Highly Relevant:** {relevant_count}")
        report_parts.append(f"- **Authenticity:** {auth_status}")
        
        fraud_indicators = validation_result.get('fraud_indicators', [])
        if fraud_indicators:
            report_parts.append(f"- **⚠️  Fraud Indicators:** {', '.join(fraud_indicators)}")
        
        report_parts.append("")
        report_parts.append(explanation.get('document_summary', 'No document summary available.'))
        report_parts.append("")
        
        # Web Research
        report_parts.append("## 🌐 Web Research")
        web_count = research_results.get('total_articles', 0)
        web_validation = validation_result.get('web_validation', 'NO_INFORMATION')
        
        report_parts.append(f"- **Sources Searched:** {', '.join(research_results.get('sources_searched', []))}")
        report_parts.append(f"- **Articles Found:** {web_count}")
        report_parts.append(f"- **Validation Status:** {web_validation}")
        
        if research_results.get('explains_anomaly'):
            external_events = research_results.get('external_events', [])
            if external_events:
                report_parts.append(f"- **External Events:** {', '.join(external_events[:3])}")
        
        report_parts.append("")
        
        # Explanation Quality
        report_parts.append("## ✅ Explanation Quality")
        explanation_quality = validation_result.get('explanation_quality', 'NONE')
        confidence = validation_result.get('confidence', 0.0)
        
        report_parts.append(f"- **Quality:** {explanation_quality}")
        report_parts.append(f"- **Confidence:** {confidence:.1%}")
        report_parts.append("")
        
        # Recommendation
        report_parts.append("## 💡 Recommendation")
        report_parts.append(explanation.get('recommendation', 'Review with senior auditor.'))
        report_parts.append("")
        
        # Next Steps
        if decision == "RED_FLAG":
            report_parts.append("## ⚠️  Urgent Action Required")
            report_parts.append("- **Timeline:** Review within 24-48 hours")
            report_parts.append("- **Assigned To:** Senior Auditor")
            report_parts.append("- **Priority:** HIGH")
        else:
            report_parts.append("## 📋 Standard Review")
            report_parts.append("- **Timeline:** Review within 5-10 business days")
            report_parts.append("- **Assigned To:** Standard audit queue")
            report_parts.append("- **Priority:** MEDIUM")
        
        return "\n".join(report_parts)
    
    def _generate_recommendations(
        self,
        validation_result: Dict[str, Any],
        document_results: Dict[str, Any],
        research_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        decision = validation_result.get('decision')
        
        if decision == 'RED_FLAG':
            recommendations.append("🔴 Escalate to senior auditor for immediate review")
            
            # Specific recommendations based on findings
            if validation_result.get('document_authenticity') in ['FAKE', 'SUSPICIOUS']:
                recommendations.append("⚠️  Conduct forensic document analysis")
                recommendations.append("⚠️  Verify document chain of custody")
            
            if document_results.get('documents_found', 0) == 0:
                recommendations.append("📄 Request supporting documentation from entity")
            
            if len(validation_result.get('fraud_indicators', [])) > 0:
                recommendations.append("🚨 Escalate to fraud investigation team")
                recommendations.append("🚨 Preserve all evidence and audit trail")
            
            if validation_result.get('web_validation') == 'CONTRADICTS':
                recommendations.append("🔍 Cross-reference with original sources")
                recommendations.append("🔍 Interview entity stakeholders")
        
        else:  # YELLOW_FLAG
            recommendations.append("✅ Schedule standard review in audit queue")
            recommendations.append("✅ Verify all documentation is properly filed")
            recommendations.append("✅ Obtain final sign-off from reviewing auditor")
            
            # Additional verification
            if research_results.get('web_validation') == 'EXPLAINS':
                recommendations.append("📰 Archive web research evidence for audit trail")
        
        # Common recommendations
        recommendations.append("📝 Update GL account notes with validation findings")
        recommendations.append("💾 Save all supporting documents to permanent storage")
        
        return recommendations


if __name__ == "__main__":
    print("Report Agent initialized")
    print("✓ Ready to generate comprehensive reports")

