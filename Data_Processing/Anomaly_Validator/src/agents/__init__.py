"""
Multi-Agent System for Anomaly Validation

Based on Park (2024): "Enhancing Anomaly Detection in Financial Markets 
with an LLM-based Multi-Agent Framework"
"""

import sys
import os

# Handle both relative and absolute imports
try:
    from .base_agent import BaseAgent, AgentResponse, AgentStatus
    from .document_agent import DocumentAgent
    from .research_agent import ResearchAgent
    from .reasoning_agent import ReasoningAgent
    from .report_agent import ReportAgent
    from .coordinator import MultiAgentCoordinator
except ImportError:
    # Fallback for when imported from outside package
    sys.path.insert(0, os.path.dirname(__file__))
    from base_agent import BaseAgent, AgentResponse, AgentStatus
    from document_agent import DocumentAgent
    from research_agent import ResearchAgent
    from reasoning_agent import ReasoningAgent
    from report_agent import ReportAgent
    from coordinator import MultiAgentCoordinator

__all__ = [
    'BaseAgent',
    'AgentResponse',
    'AgentStatus',
    'DocumentAgent',
    'ResearchAgent',
    'ReasoningAgent',
    'ReportAgent',
    'MultiAgentCoordinator'
]
