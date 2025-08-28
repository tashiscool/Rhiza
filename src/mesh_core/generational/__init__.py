"""
Generational Ethics Drift Protection
====================================

Protects against generational drift in ethical standards and ensures
long-term value preservation across generations within The Mesh network.

Components:
- EthicsDriftDetector: Detects drift in ethical standards over time
- ValuePreservationSystem: Preserves core values across generations
- GenerationalBridging: Facilitates understanding between generations
- LongTermEthicsMonitor: Monitors ethics over extended time periods
"""

from .ethics_drift_detector import EthicsDriftDetector
from .value_preservation_system import ValuePreservationSystem
from .generational_bridging import GenerationalBridging
from .longterm_ethics_monitor import LongTermEthicsMonitor

__all__ = [
    'EthicsDriftDetector',
    'ValuePreservationSystem',
    'GenerationalBridging',
    'LongTermEthicsMonitor'
]

__version__ = "1.0.0"