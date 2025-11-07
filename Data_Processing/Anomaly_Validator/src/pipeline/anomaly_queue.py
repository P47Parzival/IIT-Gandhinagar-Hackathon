"""
Anomaly Queue - Real-time processing queue for detected anomalies.

Implements a priority queue system where RED_FLAG anomalies are
processed with higher priority than YELLOW_FLAG anomalies.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
from queue import PriorityQueue, Empty
import threading
import json
from pathlib import Path


class AnomalyStatus(Enum):
    """Status of anomaly in processing pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    FAILED = "failed"
    ARCHIVED = "archived"


class AnomalyRecord:
    """Record for tracking an anomaly through the validation pipeline."""
    
    def __init__(
        self,
        anomaly_id: str,
        anomaly_data: Dict[str, Any],
        priority: int = 0,
        source: str = "detector"
    ):
        """
        Initialize anomaly record.
        
        Args:
            anomaly_id: Unique identifier
            anomaly_data: Anomaly details from detector
            priority: Priority (lower = higher priority)
            source: Source system
        """
        self.anomaly_id = anomaly_id
        self.anomaly_data = anomaly_data
        self.priority = priority
        self.source = source
        
        self.status = AnomalyStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        
        self.validation_result = None
        self.error_message = None
        self.retry_count = 0
    
    def __lt__(self, other):
        """Compare by priority for queue ordering."""
        return self.priority < other.priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'anomaly_id': self.anomaly_id,
            'anomaly_data': self.anomaly_data,
            'priority': self.priority,
            'source': self.source,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'validation_result': self.validation_result,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


class AnomalyQueue:
    """
    Priority queue for processing anomalies in real-time.
    
    Features:
    - Priority-based processing (RED_FLAG candidates get higher priority)
    - Thread-safe operations
    - Persistence to disk
    - Retry mechanism for failed validations
    - Status tracking
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize anomaly queue.
        
        Args:
            max_size: Maximum queue size
            persistence_path: Path to save queue state
        """
        self.queue = PriorityQueue(maxsize=max_size)
        self.max_size = max_size
        
        # Tracking dictionaries
        self.pending = {}  # anomaly_id -> AnomalyRecord
        self.processing = {}  # anomaly_id -> AnomalyRecord
        self.completed = {}  # anomaly_id -> AnomalyRecord
        self.failed = {}  # anomaly_id -> AnomalyRecord
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Persistence
        self.persistence_path = Path(persistence_path) if persistence_path else None
        if self.persistence_path:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self._load_state()
        
        print("✓ Anomaly Queue initialized")
        print(f"  ├─ Max size: {max_size}")
        print(f"  └─ Persistence: {'Enabled' if persistence_path else 'Disabled'}")
    
    def add_anomaly(
        self,
        anomaly_data: Dict[str, Any],
        priority: Optional[int] = None,
        source: str = "detector"
    ) -> str:
        """
        Add anomaly to queue.
        
        Args:
            anomaly_data: Anomaly details
            priority: Priority (lower = higher priority), auto-determined if None
            source: Source system
            
        Returns:
            Anomaly ID
        """
        # Generate anomaly ID if not present
        anomaly_id = anomaly_data.get('anomaly_id')
        if not anomaly_id:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            anomaly_id = f"ANO_{timestamp}"
            anomaly_data['anomaly_id'] = anomaly_id
        
        # Auto-determine priority based on anomaly severity
        if priority is None:
            priority = self._calculate_priority(anomaly_data)
        
        # Create record
        record = AnomalyRecord(
            anomaly_id=anomaly_id,
            anomaly_data=anomaly_data,
            priority=priority,
            source=source
        )
        
        with self.lock:
            # Check if already exists
            if anomaly_id in self.pending or anomaly_id in self.processing:
                print(f"⚠️  Anomaly {anomaly_id} already in queue")
                return anomaly_id
            
            # Add to queue
            self.queue.put((priority, record))
            self.pending[anomaly_id] = record
            
            print(f"✓ Added anomaly {anomaly_id} to queue (priority: {priority})")
            
            # Persist if enabled
            if self.persistence_path:
                self._save_state()
        
        return anomaly_id
    
    def get_next_anomaly(self, timeout: Optional[float] = None) -> Optional[AnomalyRecord]:
        """
        Get next anomaly from queue.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            AnomalyRecord or None if queue empty
        """
        try:
            priority, record = self.queue.get(timeout=timeout)
            
            with self.lock:
                # Move from pending to processing
                if record.anomaly_id in self.pending:
                    del self.pending[record.anomaly_id]
                
                record.status = AnomalyStatus.PROCESSING
                record.started_at = datetime.now()
                self.processing[record.anomaly_id] = record
            
            return record
            
        except Empty:
            return None
    
    def mark_completed(
        self,
        anomaly_id: str,
        validation_result: Dict[str, Any]
    ):
        """
        Mark anomaly as successfully validated.
        
        Args:
            anomaly_id: Anomaly ID
            validation_result: Validation result from coordinator
        """
        with self.lock:
            if anomaly_id not in self.processing:
                print(f"⚠️  Anomaly {anomaly_id} not in processing")
                return
            
            record = self.processing[anomaly_id]
            del self.processing[anomaly_id]
            
            record.status = AnomalyStatus.VALIDATED
            record.completed_at = datetime.now()
            record.validation_result = validation_result
            
            self.completed[anomaly_id] = record
            
            print(f"✓ Anomaly {anomaly_id} validation completed")
            
            # Persist if enabled
            if self.persistence_path:
                self._save_state()
    
    def mark_failed(
        self,
        anomaly_id: str,
        error_message: str,
        retry: bool = True,
        max_retries: int = 3
    ):
        """
        Mark anomaly as failed.
        
        Args:
            anomaly_id: Anomaly ID
            error_message: Error message
            retry: Whether to retry
            max_retries: Maximum retry attempts
        """
        with self.lock:
            if anomaly_id not in self.processing:
                print(f"⚠️  Anomaly {anomaly_id} not in processing")
                return
            
            record = self.processing[anomaly_id]
            del self.processing[anomaly_id]
            
            record.retry_count += 1
            record.error_message = error_message
            
            # Retry if under limit
            if retry and record.retry_count < max_retries:
                print(f"⚠️  Anomaly {anomaly_id} failed (retry {record.retry_count}/{max_retries})")
                
                # Re-add to queue with lower priority
                record.status = AnomalyStatus.PENDING
                record.priority += 10  # Decrease priority
                record.started_at = None
                
                self.queue.put((record.priority, record))
                self.pending[anomaly_id] = record
            else:
                # Mark as permanently failed
                print(f"❌ Anomaly {anomaly_id} permanently failed")
                record.status = AnomalyStatus.FAILED
                record.completed_at = datetime.now()
                self.failed[anomaly_id] = record
            
            # Persist if enabled
            if self.persistence_path:
                self._save_state()
    
    def get_status(self, anomaly_id: str) -> Optional[AnomalyStatus]:
        """Get status of anomaly."""
        with self.lock:
            if anomaly_id in self.pending:
                return AnomalyStatus.PENDING
            elif anomaly_id in self.processing:
                return AnomalyStatus.PROCESSING
            elif anomaly_id in self.completed:
                return AnomalyStatus.VALIDATED
            elif anomaly_id in self.failed:
                return AnomalyStatus.FAILED
            else:
                return None
    
    def get_validation_result(self, anomaly_id: str) -> Optional[Dict[str, Any]]:
        """Get validation result for completed anomaly."""
        with self.lock:
            if anomaly_id in self.completed:
                return self.completed[anomaly_id].validation_result
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                'pending': len(self.pending),
                'processing': len(self.processing),
                'completed': len(self.completed),
                'failed': len(self.failed),
                'total_processed': len(self.completed) + len(self.failed),
                'queue_size': self.queue.qsize(),
                'max_size': self.max_size
            }
    
    def _calculate_priority(self, anomaly_data: Dict[str, Any]) -> int:
        """
        Calculate priority based on anomaly characteristics.
        
        Lower number = higher priority.
        """
        # Base priority
        priority = 50
        
        # Adjust based on anomaly score
        anomaly_score = anomaly_data.get('anomaly_score', 0)
        if anomaly_score > 10:
            priority -= 30  # Very high anomaly
        elif anomaly_score > 5:
            priority -= 20  # High anomaly
        elif anomaly_score > 2:
            priority -= 10  # Medium anomaly
        
        # Adjust based on amount
        amount = abs(anomaly_data.get('amount', 0))
        if amount > 1_000_000:
            priority -= 20  # Large amount
        elif amount > 100_000:
            priority -= 10  # Medium amount
        
        # Adjust based on entity importance (if available)
        entity_priority = anomaly_data.get('entity_priority', 0)
        priority -= entity_priority
        
        return max(0, priority)  # Ensure non-negative
    
    def _save_state(self):
        """Save queue state to disk."""
        if not self.persistence_path:
            return
        
        try:
            state = {
                'pending': [r.to_dict() for r in self.pending.values()],
                'processing': [r.to_dict() for r in self.processing.values()],
                'completed': [r.to_dict() for r in self.completed.values()],
                'failed': [r.to_dict() for r in self.failed.values()],
                'saved_at': datetime.now().isoformat()
            }
            
            filepath = self.persistence_path / 'queue_state.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️  Failed to save queue state: {e}")
    
    def _load_state(self):
        """Load queue state from disk."""
        if not self.persistence_path:
            return
        
        filepath = self.persistence_path / 'queue_state.json'
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore pending items to queue
            for record_dict in state.get('pending', []):
                record = self._record_from_dict(record_dict)
                self.queue.put((record.priority, record))
                self.pending[record.anomaly_id] = record
            
            # Restore other states (for tracking only)
            for record_dict in state.get('processing', []):
                record = self._record_from_dict(record_dict)
                self.processing[record.anomaly_id] = record
            
            for record_dict in state.get('completed', []):
                record = self._record_from_dict(record_dict)
                self.completed[record.anomaly_id] = record
            
            for record_dict in state.get('failed', []):
                record = self._record_from_dict(record_dict)
                self.failed[record.anomaly_id] = record
            
            print(f"✓ Loaded queue state: {len(self.pending)} pending, {len(self.completed)} completed")
            
        except Exception as e:
            print(f"⚠️  Failed to load queue state: {e}")
    
    def _record_from_dict(self, data: Dict[str, Any]) -> AnomalyRecord:
        """Reconstruct AnomalyRecord from dictionary."""
        record = AnomalyRecord(
            anomaly_id=data['anomaly_id'],
            anomaly_data=data['anomaly_data'],
            priority=data['priority'],
            source=data['source']
        )
        record.status = AnomalyStatus(data['status'])
        record.created_at = datetime.fromisoformat(data['created_at'])
        record.started_at = datetime.fromisoformat(data['started_at']) if data['started_at'] else None
        record.completed_at = datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None
        record.validation_result = data.get('validation_result')
        record.error_message = data.get('error_message')
        record.retry_count = data.get('retry_count', 0)
        return record


if __name__ == "__main__":
    print("Anomaly Queue initialized")
    print("✓ Ready to manage real-time anomaly processing")

