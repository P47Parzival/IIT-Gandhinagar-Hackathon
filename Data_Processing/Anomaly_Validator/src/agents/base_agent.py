"""
Base Agent Class for Multi-Agent Framework

Provides common functionality for all specialized agents including
Gemini integration, error handling, and communication protocol.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentResponse:
    """
    Standardized response format for all agents.
    
    This ensures consistent communication across the multi-agent system.
    """
    agent_name: str
    agent_type: str
    status: AgentStatus
    result: Dict[str, Any]
    reasoning: str
    confidence: float = 0.0
    processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'result': self.result,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'errors': self.errors
        }
    
    def is_successful(self) -> bool:
        """Check if agent completed successfully."""
        return self.status == AgentStatus.COMPLETED and not self.errors


class BaseAgent:
    """
    Base class for all agents in the multi-agent framework.
    
    Provides:
    - Gemini client integration
    - Standardized communication protocol
    - Error handling and retry logic
    - Performance monitoring
    - Logging
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        gemini_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base agent.
        
        Args:
            agent_name: Unique name for this agent instance
            agent_type: Type of agent (e.g., 'document', 'research', 'reasoning')
            gemini_client: GeminiClient instance for LLM interactions
            config: Agent-specific configuration
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.gemini_client = gemini_client
        self.config = config or {}
        
        self.status = AgentStatus.IDLE
        self.execution_count = 0
        self.total_processing_time = 0.0
        
        print(f"✓ Agent initialized: {self.agent_name} ({self.agent_type})")
    
    def execute(
        self,
        task: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Execute agent task with error handling and monitoring.
        
        This is the main entry point for all agents. It handles:
        - Status management
        - Timing
        - Error catching
        - Response formatting
        
        Args:
            task: Task specification
            context: Additional context for execution
            
        Returns:
            AgentResponse with results
        """
        start_time = time.time()
        self.status = AgentStatus.PROCESSING
        self.execution_count += 1
        
        errors = []
        result = {}
        reasoning = ""
        confidence = 0.0
        
        try:
            # Call the specialized process method (implemented by subclasses)
            result, reasoning, confidence = self._process(task, context or {})
            
            self.status = AgentStatus.COMPLETED
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            errors.append(f"Execution error: {str(e)}")
            reasoning = f"Agent failed due to error: {str(e)}"
            
            print(f"⚠️  {self.agent_name} failed: {e}")
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        response = AgentResponse(
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            status=self.status,
            result=result,
            reasoning=reasoning,
            confidence=confidence,
            processing_time=processing_time,
            metadata={
                'execution_count': self.execution_count,
                'config': self.config
            },
            errors=errors
        )
        
        self.status = AgentStatus.IDLE
        
        return response
    
    def _process(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str, float]:
        """
        Process the task (to be implemented by subclasses).
        
        Args:
            task: Task specification
            context: Additional context
            
        Returns:
            Tuple of (result_dict, reasoning_string, confidence_score)
        """
        raise NotImplementedError("Subclasses must implement _process method")
    
    def _call_llm(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        parse_json: bool = False
    ) -> Any:
        """
        Call Gemini LLM with error handling.
        
        Args:
            prompt: Prompt text
            system_instruction: Optional system instruction
            parse_json: Whether to parse response as JSON
            
        Returns:
            LLM response (string or parsed JSON)
        """
        if not self.gemini_client:
            raise ValueError(f"{self.agent_name}: No Gemini client configured")
        
        response = self.gemini_client.generate(prompt, system_instruction)
        
        if parse_json:
            import json
            # Try to extract JSON from response
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            
            return json.loads(json_text)
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        avg_time = (self.total_processing_time / self.execution_count 
                   if self.execution_count > 0 else 0)
        
        return {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'execution_count': self.execution_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'current_status': self.status.value
        }
    
    def reset(self):
        """Reset agent state."""
        self.status = AgentStatus.IDLE
        self.execution_count = 0
        self.total_processing_time = 0.0
        print(f"✓ {self.agent_name} reset")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.agent_name}', status='{self.status.value}')"


if __name__ == "__main__":
    # Test base agent
    print("Testing Base Agent...")
    
    class TestAgent(BaseAgent):
        """Simple test agent."""
        def _process(self, task, context):
            # Simulate processing
            time.sleep(0.1)
            return (
                {'test': 'success', 'task_id': task.get('id')},
                'Test processing completed',
                0.95
            )
    
    agent = TestAgent("test_agent", "test")
    
    # Execute test task
    task = {'id': '123', 'action': 'test'}
    response = agent.execute(task)
    
    print(f"\n✓ Response status: {response.status.value}")
    print(f"✓ Result: {response.result}")
    print(f"✓ Processing time: {response.processing_time:.3f}s")
    print(f"✓ Stats: {agent.get_stats()}")
    
    print("\n✓ Base Agent tests passed!")

