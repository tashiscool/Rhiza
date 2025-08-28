"""
Mesh Immunity System - Protection against attacks and corruption

The immune system monitors network health, detects malicious behavior,
and automatically responds to threats while maintaining system integrity.

Key Components:
- Corruption detection algorithms
- Trust divergence monitoring  
- Immune response mechanisms
- Node isolation protocols
- Attack pattern recognition
"""

try:
    from mesh_core.immunity.corruption_detector import CorruptionDetector
    from mesh_core.immunity.trust_divergence import TrustDivergenceMonitor
    from mesh_core.immunity.immune_response import ImmuneResponse
    from mesh_core.immunity.node_isolation import NodeIsolation
    from mesh_core.immunity.attack_recognition import AttackRecognition
except ImportError:
    # Fallback to relative imports
    from .corruption_detector import CorruptionDetector
    from .trust_divergence import TrustDivergenceMonitor
    from .immune_response import ImmuneResponse
    from .node_isolation import NodeIsolation
    from .attack_recognition import AttackRecognition

__version__ = "0.1.0"
__all__ = [
    "CorruptionDetector",
    "TrustDivergenceMonitor", 
    "ImmuneResponse",
    "NodeIsolation",
    "AttackRecognition"
]
