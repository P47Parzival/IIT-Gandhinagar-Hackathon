"""
Document Agent - Retrieves and analyzes supporting documents.

Based on Paper 2's approach of verifying anomalies through documentation.
"""

from typing import Dict, Any, List
from pathlib import Path
import sys
import os

# Handle both relative and absolute imports
try:
    from ..agents.base_agent import BaseAgent, AgentResponse
    from ..document_processing import PDFParser, ExcelParser, ImageParser, DocumentStore
    from ..llm import GeminiClient, DocumentAuthenticityChecker
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.base_agent import BaseAgent, AgentResponse
    from document_processing import PDFParser, ExcelParser, ImageParser, DocumentStore
    from llm import GeminiClient, DocumentAuthenticityChecker


class DocumentAgent(BaseAgent):
    """
    Agent responsible for:
    1. Retrieving supporting documents for GL accounts
    2. Parsing documents (PDF, Excel, images)
    3. Assessing document relevance using Gemini
    4. Checking document authenticity (fraud detection)
    
    This implements Paper 2's document verification component.
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        document_store: DocumentStore = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize document agent.
        
        Args:
            gemini_client: Gemini client for LLM analysis
            document_store: Document storage system
            config: Configuration options
        """
        super().__init__(
            agent_name="DocumentAgent",
            agent_type="document",
            gemini_client=gemini_client,
            config=config
        )
        
        self.document_store = document_store or DocumentStore()
        
        # Initialize parsers
        self.pdf_parser = PDFParser(extract_tables=True)
        self.excel_parser = ExcelParser()
        self.image_parser = ImageParser()
        
        # Initialize authenticity checker
        self.auth_checker = DocumentAuthenticityChecker(gemini_client)
        
        print("✓ Document Agent initialized with fraud detection")
    
    def _process(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str, float]:
        """
        Process document analysis task.
        
        Args:
            task: Contains anomaly_data with GL account info
            context: Additional context
            
        Returns:
            Tuple of (result, reasoning, confidence)
        """
        anomaly_data = task.get('anomaly_data', {})
        
        # Step 1: Retrieve documents
        documents = self._retrieve_documents(anomaly_data)
        
        if not documents:
            return (
                {
                    'documents_found': 0,
                    'parsed_documents': [],
                    'relevance_assessments': [],
                    'authenticity_assessment': {
                        'overall_authenticity': 'NO_DOCUMENTS',
                        'recommendation': 'RED_FLAG'
                    }
                },
                "No supporting documents found for this GL account",
                0.0
            )
        
        # Step 2: Parse documents
        parsed_docs = self._parse_documents(documents)
        
        # Step 3: Assess relevance using Gemini
        relevance_assessments = self._assess_relevance(parsed_docs, anomaly_data)
        
        # Step 4: Check authenticity (fraud detection)
        authenticity = self.auth_checker.batch_check_documents(parsed_docs, anomaly_data)
        
        # Calculate overall confidence
        avg_relevance = (
            sum(a['relevance_score'] for a in relevance_assessments) / len(relevance_assessments)
            if relevance_assessments else 0.0
        )
        confidence = (avg_relevance / 10.0 + authenticity['confidence']) / 2.0
        
        # Build reasoning
        reasoning = self._build_reasoning(
            len(documents),
            relevance_assessments,
            authenticity
        )
        
        result = {
            'documents_found': len(documents),
            'parsed_documents': parsed_docs,
            'relevance_assessments': relevance_assessments,
            'authenticity_assessment': authenticity,
            'average_relevance': avg_relevance,
            'highly_relevant_count': sum(1 for a in relevance_assessments if a['is_relevant'])
        }
        
        return (result, reasoning, confidence)
    
    def _retrieve_documents(
        self,
        anomaly_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve documents from document store."""
        gl_account = anomaly_data.get('gl_account')
        entity_id = anomaly_data.get('entity_id')
        period = anomaly_data.get('period')
        
        if not all([gl_account, entity_id, period]):
            return []
        
        documents = self.document_store.get_documents_for_gl(
            gl_account,
            entity_id,
            period
        )
        
        print(f"  Retrieved {len(documents)} documents from store")
        return documents
    
    def _parse_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse documents based on file type."""
        parsed = []
        
        for doc in documents:
            file_path = doc.get('file_path')
            if not file_path or not Path(file_path).exists():
                continue
            
            file_ext = Path(file_path).suffix.lower()
            
            try:
                if file_ext == '.pdf':
                    parse_result = self.pdf_parser.parse(file_path)
                elif file_ext in ['.xlsx', '.xls', '.csv']:
                    parse_result = self.excel_parser.parse(file_path)
                    parse_result['text'] = self.excel_parser.to_text(parse_result)
                elif file_ext in ['.png', '.jpg', '.jpeg']:
                    parse_result = self.image_parser.parse(file_path)
                else:
                    print(f"⚠️  Unsupported file type: {file_ext}")
                    continue
                
                # Add to parsed list
                parsed.append({
                    **doc,
                    **parse_result
                })
                
            except Exception as e:
                print(f"⚠️  Error parsing {file_path}: {e}")
        
        print(f"  Parsed {len(parsed)} documents successfully")
        return parsed
    
    def _assess_relevance(
        self,
        parsed_docs: List[Dict[str, Any]],
        anomaly_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assess document relevance using Gemini."""
        assessments = []
        
        for doc in parsed_docs:
            doc_text = doc.get('text', '')
            if not doc_text:
                continue
            
            # Assess relevance
            assessment = self.gemini_client.assess_document_relevance(
                document_text=doc_text,
                anomaly_context=anomaly_data
            )
            
            assessment['document_id'] = doc.get('doc_id')
            assessment['file_name'] = doc.get('file_name')
            
            assessments.append(assessment)
        
        return assessments
    
    def _build_reasoning(
        self,
        doc_count: int,
        relevance_assessments: List[Dict[str, Any]],
        authenticity: Dict[str, Any]
    ) -> str:
        """Build reasoning summary."""
        relevant_count = sum(1 for a in relevance_assessments if a.get('is_relevant', False))
        auth_status = authenticity.get('overall_authenticity', 'UNKNOWN')
        
        reasoning_parts = [
            f"Found {doc_count} supporting documents."
        ]
        
        if relevant_count > 0:
            reasoning_parts.append(
                f"{relevant_count} documents are highly relevant (score ≥ 7/10)."
            )
        else:
            reasoning_parts.append(
                "No highly relevant documents found."
            )
        
        reasoning_parts.append(
            f"Document authenticity: {auth_status}."
        )
        
        if authenticity.get('fake_count', 0) > 0:
            reasoning_parts.append(
                f"⚠️  {authenticity['fake_count']} documents flagged as potentially FAKE!"
            )
        
        return " ".join(reasoning_parts)


if __name__ == "__main__":
    print("Document Agent initialized")
    print("✓ Ready to process supporting documents")

