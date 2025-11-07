"""
Federated Learning Module

Implements federated learning infrastructure for distributed model training.
"""

from .client import FCLClient
from .server import FCLServer
from .strategies import (
    FederatedAveraging,
    FederatedProximal,
    Scaffold,
    get_federated_strategy
)

__all__ = [
    'FCLClient',
    'FCLServer',
    'FederatedAveraging',
    'FederatedProximal',
    'Scaffold',
    'get_federated_strategy'
]
