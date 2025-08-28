"""
Value Alignment System
=====================

Ensures AI models remain aligned with human values and mesh community standards.
Provides continuous alignment monitoring and correction mechanisms.

Components:
- AlignmentMonitor: Monitors value alignment over time
- ValueAlignmentTracker: Tracks alignment metrics
- AlignmentCorrector: Applies alignment corrections
- EthicsEnforcer: Enforces ethical guidelines
"""

from .alignment_monitor import AlignmentMonitor
from .value_alignment_tracker import ValueAlignmentTracker
from .alignment_corrector import AlignmentCorrector
from .ethics_enforcer import EthicsEnforcer

__all__ = [
    'AlignmentMonitor',
    'ValueAlignmentTracker', 
    'AlignmentCorrector',
    'EthicsEnforcer'
]

__version__ = "1.0.0"