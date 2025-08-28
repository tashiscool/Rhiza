"""
Advanced Authentication & Authorization for The Mesh
===================================================

Phase 4.2: Enhanced security infrastructure providing:
- Triple-signed peer approval system
- Distributed identity verification
- Coercion-resistant protocols
- Hardware security integration
- Zero-knowledge proof systems

Key Components:
- TripleSignAuth: Three-party verification protocols
- DistributedIdentity: Network-wide identity management
- CoercionResistance: Anti-coercion detection and prevention
- HardwareSecurity: Hardware-level security integration
- ZeroKnowledge: Privacy-preserving authentication proofs
"""

from .triple_sign_auth import TripleSignAuth, VerificationRequest, AuthenticationResult
from .distributed_identity import DistributedIdentity, Identity, IdentityProof, VerificationLevel
from .coercion_resistance import CoercionResistance, CoercionSignal, ResistanceLevel, ThreatAssessment
from .hardware_security import HardwareSecurity, SecurityDevice, HardwareToken, SecurityLevel
from .zero_knowledge import ZeroKnowledge, ZKProof, ZKChallenge, ProofType

__version__ = "1.0.0"
__author__ = "The Mesh Development Team"

__all__ = [
    # Core Security Components
    'TripleSignAuth',
    'DistributedIdentity',
    'CoercionResistance', 
    'HardwareSecurity',
    'ZeroKnowledge',
    
    # Authentication Data Types
    'VerificationRequest',
    'AuthenticationResult',
    'Identity',
    'IdentityProof',
    'VerificationLevel',
    
    # Coercion Protection
    'CoercionSignal',
    'ResistanceLevel',
    'ThreatAssessment',
    
    # Hardware Security
    'SecurityDevice',
    'HardwareToken', 
    'SecurityLevel',
    
    # Zero-Knowledge Proofs
    'ZKProof',
    'ZKChallenge',
    'ProofType'
]