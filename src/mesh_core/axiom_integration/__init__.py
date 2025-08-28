"""
AxiomEngine Integration Module

This module integrates AxiomEngine with The Mesh network to provide
advanced truth verification, fact-checking, and knowledge validation
capabilities through both distributed consensus and centralized analysis.

Core Components:
- AxiomProcessor: Core integration between AxiomEngine and Mesh
- TruthValidator: Enhanced truth verification using AxiomEngine
- KnowledgeAnalyzer: Knowledge claim analysis and validation
- ConfidenceScorer: Truth confidence scoring with multiple sources
- AxiomMeshBridge: Bridge between AxiomEngine and Mesh systems

Usage Example:
    from mesh_core.axiom_integration import AxiomProcessor
    
    processor = AxiomProcessor()
    result = await processor.verify_claim("The Earth is round")
    print(f"Verification result: {result['confidence']} confidence")
"""

from .axiom_processor import AxiomProcessor, AxiomVerificationResult, AxiomFactSubmission
from .truth_validator import TruthValidator, TruthValidationResult, TruthClaim, ValidationEvidence, ConsensusResult
from .knowledge_validator import KnowledgeValidator, ValidationResult, KnowledgeClaim
from .confidence_scorer import ConfidenceScorer, ConfidenceScore, ScoreComponents, ConfidenceFactor
from .axiom_mesh_bridge import AxiomMeshBridge, HybridVerificationResult, VerificationRequest

__all__ = [
    # Core Components
    'AxiomProcessor',
    'TruthValidator', 
    'KnowledgeValidator',
    'ConfidenceScorer',
    'AxiomMeshBridge',
    
    # Result Classes
    'AxiomVerificationResult',
    'TruthValidationResult',
    'ValidationResult',
    'ConfidenceScore',
    'HybridVerificationResult',
    
    # Data Classes
    'AxiomFactSubmission',
    'TruthClaim',
    'KnowledgeClaim',
    'VerificationRequest',
    'ValidationEvidence',
    'ConsensusResult',
    'ScoreComponents',
    'ConfidenceFactor'
]

# Version information
__version__ = "1.0.0"
__author__ = "The Mesh Development Team"
__description__ = "AxiomEngine integration for advanced truth verification"