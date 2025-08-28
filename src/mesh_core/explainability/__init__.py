"""
Explainability Layer
===================

Provides comprehensive explanations for AI decisions and behaviors
within The Mesh network, ensuring transparency and accountability.

Components:
- ExplanationGenerator: Generates explanations for decisions
- DecisionTracker: Tracks decision-making processes
- TransparencyManager: Manages transparency requirements
- AuditTrailManager: Maintains audit trails
"""

from .explanation_generator import ExplanationGenerator
from .decision_tracker import DecisionTracker
from .transparency_manager import TransparencyManager
from .audit_trail_manager import AuditTrailManager

__all__ = [
    'ExplanationGenerator',
    'DecisionTracker',
    'TransparencyManager',
    'AuditTrailManager'
]

__version__ = "1.0.0"