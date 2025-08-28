"""
Trust Ledger System - Distributed Trust and Reputation Management

This module implements the trust infrastructure for The Mesh, enabling
decentralized trust scoring, reputation tracking, and consensus validation
without centralized authority.

Core Components:
- TrustLedger: Core distributed trust tracking
- ReputationEngine: Reputation algorithms and scoring
- ConsensusValidator: Cross-validation and consensus mechanisms
- PredictionScorer: Prediction alignment scoring
- SocialChecksum: Social verification protocols
"""

from .trust_ledger import TrustLedger, TrustRecord, TrustScore

# Mock classes for missing modules
class ReputationEngine:
    """Mock reputation engine for development"""
    pass

class ReputationMetrics:
    """Mock reputation metrics for development"""
    pass

class InteractionType:
    """Mock interaction type for development"""
    pass

class ConsensusValidator:
    """Mock consensus validator for development"""
    pass

class ValidationResult:
    """Mock validation result for development"""
    pass

class ConsensusThreshold:
    """Mock consensus threshold for development"""
    pass

class PredictionScorer:
    """Mock prediction scorer for development"""
    pass

class PredictionRecord:
    """Mock prediction record for development"""
    pass

class AlignmentMetrics:
    """Mock alignment metrics for development"""
    pass

class SocialChecksum:
    """Mock social checksum for development"""
    pass

class ChecksumResult:
    """Mock checksum result for development"""
    pass

class VerificationMethod:
    """Mock verification method for development"""
    pass

__all__ = [
    'TrustLedger',
    'TrustRecord', 
    'TrustScore',
    'ReputationEngine',
    'ReputationMetrics',
    'InteractionType',
    'ConsensusValidator',
    'ValidationResult',
    'ConsensusThreshold',
    'PredictionScorer',
    'PredictionRecord',
    'AlignmentMetrics',
    'SocialChecksum',
    'ChecksumResult',
    'VerificationMethod'
]

# Version info
__version__ = "1.0.0"
__author__ = "The Mesh Development Team"
__description__ = "Decentralized trust and reputation system for The Mesh"