"""
Continual Learning Module

Implements continual learning strategies and evaluation metrics for
preventing catastrophic forgetting in sequential learning scenarios.
"""

from .metrics import (
    ContinualLearningMetrics,
    evaluate_anomaly_detection,
    compute_stability_plasticity_tradeoff
)

__all__ = [
    'ContinualLearningMetrics',
    'evaluate_anomaly_detection',
    'compute_stability_plasticity_tradeoff'
]
