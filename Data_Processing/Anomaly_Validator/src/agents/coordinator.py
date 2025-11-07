"""
Multi-Agent Coordinator - Orchestrates the validation workflow.

Based on Paper 2's multi-agent framework architecture where specialized
agents work in sequence to validate anomalies.

Workflow:
1. DocumentAgent → Parse and validate documents
2. ResearchAgent → Web scraping and external context  
3. ReasoningAgent → Validate and assign flag
4. ReportAgent → Generate comprehensive report
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path
import sys
import os

# Handle both relative and absolute imports
try:
    from .document_agent import DocumentAgent
    from .research_agent import ResearchAgent
    from .reasoning_agent import ReasoningAgent
    from .report_agent import ReportAgent
    from ..llm import GeminiClient
    from ..document_processing import DocumentStore
except ImportError:
    # Fallback for when imported from outside package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.document_agent import DocumentAgent
    from agents.research_agent import ResearchAgent
    from agents.reasoning_agent import ReasoningAgent
    from agents.report_agent import ReportAgent
    from llm import GeminiClient
    from document_processing import DocumentStore


class MultiAgentCoordinator:
    """
    Coordinates the multi-agent validation workflow.
    
    Implements Paper 2's sequential agent execution with
    error handling, logging, and performance monitoring.
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        document_store: Optional[DocumentStore] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize coordinator with all agents.
        
        Args:
            gemini_api_key: Gemini API key
            document_store: Document storage system
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize Gemini client (shared across agents)
        self.gemini_client = GeminiClient(
            api_key=gemini_api_key,
            model=self.config.get('llm', {}).get('model', 'gemini-1.5-flash'),
            temperature=self.config.get('llm', {}).get('temperature', 0.2),
            max_tokens=self.config.get('llm', {}).get('max_tokens', 1024)
        )
        
        # Initialize all agents
        self.document_agent = DocumentAgent(
            gemini_client=self.gemini_client,
            document_store=document_store,
            config=self.config.get('document_processing', {})
        )
        
        self.research_agent = ResearchAgent(
            gemini_client=self.gemini_client,
            config=self.config.get('web_scraping', {})
        )
        
        self.reasoning_agent = ReasoningAgent(
            gemini_client=self.gemini_client,
            config=self.config.get('validation', {})
        )
        
        self.report_agent = ReportAgent(
            gemini_client=self.gemini_client,
            config=self.config.get('output', {})
        )
        
        # Performance tracking
        self.validation_count = 0
        self.total_processing_time = 0.0
        
        print("✓ Multi-Agent Coordinator initialized")
        print(f"  ├─ DocumentAgent: Ready")
        print(f"  ├─ ResearchAgent: Ready")
        print(f"  ├─ ReasoningAgent: Ready")
        print(f"  └─ ReportAgent: Ready")
    
    def validate_anomaly(
        self,
        anomaly_data: Dict[str, Any],
        institutional_knowledge: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Validate an anomaly through the complete multi-agent pipeline.
        
        Args:
            anomaly_data: Anomaly details from Anomaly_Detector
            institutional_knowledge: Optional domain knowledge
            save_results: Whether to save validation results
            
        Returns:
            Complete validation result with decision and explanation
        """
        start_time = datetime.now()
        anomaly_id = anomaly_data.get('anomaly_id') or f"ANO_{start_time.strftime('%Y%m%d%H%M%S')}"
        
        print(f"\n{'='*80}")
        print(f"🔍 Validating Anomaly: {anomaly_id}")
        print(f"{'='*80}")
        
        result = {
            'anomaly_id': anomaly_id,
            'anomaly_data': anomaly_data,
            'validation_timestamp': start_time.isoformat(),
            'agent_results': {},
            'errors': []
        }
        
        try:
            # Step 1: Document Analysis
            print("\n[1/4] 📄 DocumentAgent: Analyzing supporting documents...")
            document_response = self.document_agent.execute(
                task={'anomaly_data': anomaly_data},
                context={}
            )
            
            result['agent_results']['document'] = document_response.to_dict()
            
            if not document_response.is_successful():
                result['errors'].append(f"DocumentAgent failed: {document_response.errors}")
            
            print(f"  ✓ Completed in {document_response.processing_time:.2f}s")
            print(f"  ├─ Documents: {document_response.result.get('documents_found', 0)}")
            print(f"  ├─ Relevant: {document_response.result.get('highly_relevant_count', 0)}")
            print(f"  └─ Authenticity: {document_response.result.get('authenticity_assessment', {}).get('overall_authenticity')}")
            
            # Step 2: Web Research
            print("\n[2/4] 🌐 ResearchAgent: Performing web research...")
            research_response = self.research_agent.execute(
                task={'anomaly_data': anomaly_data},
                context={}
            )
            
            result['agent_results']['research'] = research_response.to_dict()
            
            if not research_response.is_successful():
                result['errors'].append(f"ResearchAgent failed: {research_response.errors}")
            
            print(f"  ✓ Completed in {research_response.processing_time:.2f}s")
            print(f"  ├─ Sources: {', '.join(research_response.result.get('sources_searched', []))}")
            print(f"  ├─ Articles: {research_response.result.get('total_articles', 0)}")
            print(f"  └─ Validation: {research_response.result.get('web_validation', 'UNKNOWN')}")
            
            # Step 3: Reasoning & Validation
            print("\n[3/4] 🧠 ReasoningAgent: Analyzing evidence and assigning flag...")
            reasoning_response = self.reasoning_agent.execute(
                task={
                    'anomaly_data': anomaly_data,
                    'document_results': document_response.result,
                    'research_results': research_response.result
                },
                context={'institutional_knowledge': institutional_knowledge}
            )
            
            result['agent_results']['reasoning'] = reasoning_response.to_dict()
            
            if not reasoning_response.is_successful():
                result['errors'].append(f"ReasoningAgent failed: {reasoning_response.errors}")
            
            decision = reasoning_response.result.get('decision', 'UNKNOWN')
            confidence = reasoning_response.result.get('confidence', 0.0)
            
            print(f"  ✓ Completed in {reasoning_response.processing_time:.2f}s")
            print(f"  ├─ Decision: {decision}")
            print(f"  ├─ Confidence: {confidence:.1%}")
            print(f"  ├─ Document Auth: {reasoning_response.result.get('document_authenticity')}")
            print(f"  └─ Fraud Indicators: {len(reasoning_response.result.get('fraud_indicators', []))}")
            
            # Step 4: Report Generation
            print("\n[4/4] 📝 ReportAgent: Generating comprehensive report...")
            report_response = self.report_agent.execute(
                task={
                    'anomaly_data': anomaly_data,
                    'validation_result': reasoning_response.result,
                    'document_results': document_response.result,
                    'research_results': research_response.result
                },
                context={}
            )
            
            result['agent_results']['report'] = report_response.to_dict()
            
            if not report_response.is_successful():
                result['errors'].append(f"ReportAgent failed: {report_response.errors}")
            
            print(f"  ✓ Completed in {report_response.processing_time:.2f}s")
            
            # Build final result
            result.update({
                'decision': decision,
                'confidence': confidence,
                'requires_urgent_review': reasoning_response.result.get('requires_urgent_review', True),
                'summary': report_response.result.get('summary', ''),
                'full_explanation': report_response.result.get('full_explanation', ''),
                'recommendation': report_response.result.get('recommendation', ''),
                'comprehensive_report': report_response.result.get('comprehensive_report', ''),
                'action_items': report_response.result.get('action_items', [])
            })
            
            # Calculate total time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            result['processing_time_seconds'] = processing_time
            
            # Update statistics
            self.validation_count += 1
            self.total_processing_time += processing_time
            
            print(f"\n{'='*80}")
            print(f"✅ Validation Complete: {decision}")
            print(f"⏱️  Total time: {processing_time:.2f}s")
            print(f"{'='*80}\n")
            
            # Save results if requested
            if save_results:
                self._save_results(result)
            
        except Exception as e:
            result['errors'].append(f"Coordinator error: {str(e)}")
            result['decision'] = 'RED_FLAG'  # Default to RED_FLAG on error
            result['requires_urgent_review'] = True
            print(f"\n❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def validate_batch(
        self,
        anomalies: List[Dict[str, Any]],
        institutional_knowledge: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple anomalies in batch.
        
        Args:
            anomalies: List of anomaly records
            institutional_knowledge: Optional domain knowledge
            
        Returns:
            List of validation results
        """
        print(f"\n🔄 Batch validation: Processing {len(anomalies)} anomalies...")
        
        results = []
        for i, anomaly in enumerate(anomalies, 1):
            print(f"\n[{i}/{len(anomalies)}] Processing anomaly...")
            result = self.validate_anomaly(anomaly, institutional_knowledge)
            results.append(result)
        
        # Summary statistics
        red_flags = sum(1 for r in results if r.get('decision') == 'RED_FLAG')
        yellow_flags = sum(1 for r in results if r.get('decision') == 'YELLOW_FLAG')
        avg_time = sum(r.get('processing_time_seconds', 0) for r in results) / len(results)
        
        print(f"\n{'='*80}")
        print(f"📊 Batch Validation Summary")
        print(f"{'='*80}")
        print(f"Total: {len(anomalies)}")
        print(f"🔴 RED FLAGS: {red_flags} ({red_flags/len(anomalies)*100:.1f}%)")
        print(f"🟡 YELLOW FLAGS: {yellow_flags} ({yellow_flags/len(anomalies)*100:.1f}%)")
        print(f"⏱️  Average time: {avg_time:.2f}s per anomaly")
        print(f"{'='*80}\n")
        
        return results
    
    def _save_results(self, result: Dict[str, Any]):
        """Save validation results to disk."""
        try:
            output_dir = Path('data/validation_results')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            anomaly_id = result['anomaly_id']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{anomaly_id}_{timestamp}.json"
            
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"💾 Results saved: {filepath}")
            
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator performance statistics."""
        avg_time = (
            self.total_processing_time / self.validation_count
            if self.validation_count > 0 else 0
        )
        
        return {
            'validations_completed': self.validation_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'agent_stats': {
                'document': self.document_agent.get_stats(),
                'research': self.research_agent.get_stats(),
                'reasoning': self.reasoning_agent.get_stats(),
                'report': self.report_agent.get_stats()
            }
        }


if __name__ == "__main__":
    print("Multi-Agent Coordinator")
    print("✓ Ready to orchestrate validation workflow")

