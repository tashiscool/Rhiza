"""
Graceful Degradation Systems
============================

Ensures system resilience through graceful degradation when components
fail or become unavailable within The Mesh network.

Components:
- DegradationManager: Manages graceful system degradation
- ServiceFallback: Provides fallback services when primary systems fail
- ResourcePrioritizer: Prioritizes resources during degradation
- RecoveryOrchestrator: Orchestrates system recovery processes
"""

from .degradation_manager import DegradationManager
from .service_fallback import ServiceFallback
from .resource_prioritizer import ResourcePrioritizer
from .recovery_orchestrator import RecoveryOrchestrator

__all__ = [
    'DegradationManager',
    'ServiceFallback',
    'ResourcePrioritizer',
    'RecoveryOrchestrator'
]

__version__ = "1.0.0"