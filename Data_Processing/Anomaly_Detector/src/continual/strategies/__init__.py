"""
Continual Learning Strategies

Implements various strategies to prevent catastrophic forgetting:
- EWC: Elastic Weight Consolidation
- Replay: Experience Replay
- LwF: Learning without Forgetting
"""

from .ewc import EWC, OnlineEWC
from .replay import ExperienceReplay, ReplayBuffer, GenerativeReplay
from .lwf import LearningWithoutForgetting, LwFWithAttention, AdaptiveLwF

__all__ = [
    'EWC',
    'OnlineEWC',
    'ExperienceReplay',
    'ReplayBuffer',
    'GenerativeReplay',
    'LearningWithoutForgetting',
    'LwFWithAttention',
    'AdaptiveLwF'
]
