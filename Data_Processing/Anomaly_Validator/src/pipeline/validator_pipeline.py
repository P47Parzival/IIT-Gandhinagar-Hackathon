"""
Validator Pipeline - End-to-end anomaly validation workflow.

Integrates Paper 1 (anomaly detection) with Paper 2 (validation framework).
Manages the complete flow from detection → validation → reporting.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import threading
import time
import yaml
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
if __name__ != '__main__':
    try:
        from .anomaly_queue import AnomalyQueue, AnomalyStatus
        from ..agents import MultiAgentCoordinator
        from ..document_processing import DocumentStore
    except ImportError:
        # Fallback for when imported from outside package
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from pipeline.anomaly_queue import AnomalyQueue, AnomalyStatus
        from agents import MultiAgentCoordinator
        from document_processing import DocumentStore
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from pipeline.anomaly_queue import AnomalyQueue, AnomalyStatus
    from agents import MultiAgentCoordinator
    from document_processing import DocumentStore


class ValidatorPipeline:
    """
    Complete validation pipeline that:
    1. Receives anomalies from detector
    2. Queues them for processing
    3. Validates through multi-agent system
    4. Produces final reports
    
    Supports both real-time and batch processing modes.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        """
        Initialize validation pipeline.
        
        Args:
            config_path: Path to configuration file
            gemini_api_key: Gemini API key (overrides config)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Override API key if provided
        if gemini_api_key:
            self.config['llm']['api_key'] = gemini_api_key
        
        # Initialize components
        self.document_store = DocumentStore(
            base_path=self.config.get('document_storage_path', 'data/documents')
        )
        
        self.queue = AnomalyQueue(
            max_size=self.config.get('queue_max_size', 1000),
            persistence_path=self.config.get('queue_persistence_path', 'data/queue_state')
        )
        
        # Get API key from environment variable or config
        api_key_env_name = self.config['llm'].get('api_key_env', 'GEMINI_API_KEY')
        gemini_api_key = os.environ.get(api_key_env_name) or self.config['llm'].get('api_key')
        
        if not gemini_api_key:
            raise ValueError(f"Gemini API key not found in environment variable '{api_key_env_name}' or config")
        
        self.coordinator = MultiAgentCoordinator(
            gemini_api_key=gemini_api_key,
            document_store=self.document_store,
            config=self.config
        )
        
        # Processing thread
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.start_time = datetime.now()
        self.total_processed = 0
        self.total_red_flags = 0
        self.total_yellow_flags = 0
        self.total_failures = 0
        
        print("✓ Validator Pipeline initialized")
        print(f"  ├─ Config loaded: {config_path or 'default'}")
        print(f"  ├─ Document store: {self.document_store}")
        print(f"  ├─ Queue: {self.queue}")
        print(f"  └─ Coordinator: {self.coordinator}")
    
    def start_realtime_processing(self):
        """
        Start real-time processing thread.
        
        This continuously monitors the queue and validates anomalies
        as they arrive.
        """
        if self.is_running:
            print("⚠️  Pipeline already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        print("✓ Real-time processing started")
    
    def stop_realtime_processing(self):
        """Stop real-time processing thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        print("✓ Real-time processing stopped")
    
    def submit_anomaly(
        self,
        anomaly_data: Dict[str, Any],
        priority: Optional[int] = None
    ) -> str:
        """
        Submit an anomaly for validation.
        
        Args:
            anomaly_data: Anomaly details from detector
            priority: Optional priority override
            
        Returns:
            Anomaly ID
        """
        anomaly_id = self.queue.add_anomaly(
            anomaly_data=anomaly_data,
            priority=priority,
            source="detector"
        )
        
        print(f"✓ Submitted anomaly {anomaly_id} for validation")
        return anomaly_id
    
    def submit_batch(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Submit multiple anomalies for validation.
        
        Args:
            anomalies: List of anomaly records
            
        Returns:
            List of anomaly IDs
        """
        print(f"📦 Submitting batch of {len(anomalies)} anomalies...")
        
        anomaly_ids = []
        for anomaly_data in anomalies:
            anomaly_id = self.submit_anomaly(anomaly_data)
            anomaly_ids.append(anomaly_id)
        
        print(f"✓ Batch submitted: {len(anomaly_ids)} anomalies queued")
        return anomaly_ids
    
    def validate_single(
        self,
        anomaly_data: Dict[str, Any],
        institutional_knowledge: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a single anomaly synchronously (bypass queue).
        
        Useful for testing or high-priority validations.
        
        Args:
            anomaly_data: Anomaly details
            institutional_knowledge: Optional domain knowledge
            
        Returns:
            Validation result
        """
        print(f"🔍 Validating anomaly (synchronous)...")
        
        result = self.coordinator.validate_anomaly(
            anomaly_data=anomaly_data,
            institutional_knowledge=institutional_knowledge,
            save_results=True
        )
        
        # Update statistics
        self._update_stats(result)
        
        return result
    
    def validate_batch_sync(
        self,
        anomalies: List[Dict[str, Any]],
        institutional_knowledge: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple anomalies synchronously (bypass queue).
        
        Args:
            anomalies: List of anomaly records
            institutional_knowledge: Optional domain knowledge
            
        Returns:
            List of validation results
        """
        print(f"📦 Validating batch of {len(anomalies)} anomalies (synchronous)...")
        
        results = self.coordinator.validate_batch(
            anomalies=anomalies,
            institutional_knowledge=institutional_knowledge
        )
        
        # Update statistics
        for result in results:
            self._update_stats(result)
        
        return results
    
    def get_status(self, anomaly_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of anomaly validation.
        
        Args:
            anomaly_id: Anomaly ID
            
        Returns:
            Status dictionary
        """
        status = self.queue.get_status(anomaly_id)
        
        if status is None:
            return None
        
        result = {
            'anomaly_id': anomaly_id,
            'status': status.value
        }
        
        if status == AnomalyStatus.VALIDATED:
            result['validation_result'] = self.queue.get_validation_result(anomaly_id)
        
        return result
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        queue_stats = self.queue.get_stats()
        coordinator_stats = self.coordinator.get_stats()
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'pipeline': {
                'uptime_seconds': uptime,
                'is_running': self.is_running,
                'total_processed': self.total_processed,
                'red_flags': self.total_red_flags,
                'yellow_flags': self.total_yellow_flags,
                'failures': self.total_failures,
                'throughput': self.total_processed / uptime if uptime > 0 else 0
            },
            'queue': queue_stats,
            'coordinator': coordinator_stats
        }
    
    def _processing_loop(self):
        """
        Main processing loop for real-time validation.
        
        Runs in background thread.
        """
        print("🔄 Processing loop started")
        
        while self.is_running:
            try:
                # Get next anomaly (with timeout to allow checking is_running)
                record = self.queue.get_next_anomaly(timeout=1.0)
                
                if record is None:
                    continue  # Queue empty, continue waiting
                
                print(f"\n{'='*80}")
                print(f"🔍 Processing anomaly: {record.anomaly_id}")
                print(f"{'='*80}")
                
                # Validate through coordinator
                try:
                    result = self.coordinator.validate_anomaly(
                        anomaly_data=record.anomaly_data,
                        institutional_knowledge=None,
                        save_results=True
                    )
                    
                    # Mark as completed
                    self.queue.mark_completed(record.anomaly_id, result)
                    
                    # Update statistics
                    self._update_stats(result)
                    
                    print(f"✅ Anomaly {record.anomaly_id} validated: {result.get('decision')}")
                    
                except Exception as e:
                    print(f"❌ Validation failed for {record.anomaly_id}: {e}")
                    self.queue.mark_failed(
                        record.anomaly_id,
                        str(e),
                        retry=True,
                        max_retries=3
                    )
                    self.total_failures += 1
            
            except Exception as e:
                print(f"❌ Processing loop error: {e}")
                time.sleep(1)  # Prevent tight loop on repeated errors
        
        print("🔄 Processing loop stopped")
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update pipeline statistics."""
        self.total_processed += 1
        
        decision = result.get('decision')
        if decision == 'RED_FLAG':
            self.total_red_flags += 1
        elif decision == 'YELLOW_FLAG':
            self.total_yellow_flags += 1
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[OK] Configuration loaded from {config_path}")
        else:
            # Default configuration
            config = {
                'llm': {
                    'api_key': '',  # Must be provided
                    'model': 'gemini-1.5-flash',
                    'temperature': 0.2,
                    'max_tokens': 1024
                },
                'document_storage_path': 'data/documents',
                'queue_max_size': 1000,
                'queue_persistence_path': 'data/queue_state',
                'document_processing': {
                    'require_documents': True,
                    'min_documents_for_yellow': 1
                },
                'web_scraping': {
                    'enable_news': True,
                    'enable_company': True,
                    'enable_regulatory': True,
                    'max_articles_per_source': 5,
                    'cache_ttl_hours': 24
                },
                'validation': {
                    'can_unflag': False,  # NEVER change this
                    'relevance_threshold': 7.0,
                    'authenticity_confidence_threshold': 0.8,
                    'explanation_quality_threshold': 0.9,
                    'yellow_flag_confidence_threshold': 0.8,
                    'max_fraud_indicators_for_yellow': 0,
                    'require_web_context': True
                },
                'output': {
                    'save_reports': True,
                    'report_format': 'markdown'
                }
            }
            print("[OK] Using default configuration")
        
        return config
    
    def export_results(
        self,
        output_path: str,
        format: str = 'json',
        include_archived: bool = False
    ):
        """
        Export validation results.
        
        Args:
            output_path: Path to save results
            format: Export format ('json', 'csv', 'excel')
            include_archived: Include archived results
        """
        # Implementation would export results in various formats
        print(f"💾 Exporting results to {output_path} (format: {format})")
        # TODO: Implement export logic


if __name__ == "__main__":
    print("Validator Pipeline")
    print("[OK] Ready to process anomalies end-to-end")

