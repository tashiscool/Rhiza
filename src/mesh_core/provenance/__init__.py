"""
Mesh Provenance System - Track information sources and build audit trails

The provenance system provides comprehensive tracking of information flow,
source references, confidence scores, and contextual metadata throughout
the Mesh network.

Key Components:
- Source reference tracking (anonymized but traceable)
- Confidence score histories and evolution
- Use-context framing and cultural origins
- Information flow tracking and audit trails
- Provenance verification and validation
"""

from .provenance_tracker import ProvenanceTracker
from .source_reference import SourceReference, SourceReferenceManager, SourceType, ConfidenceLevel
from .confidence_history import ConfidenceHistory, ConfidenceEntry, ConfidenceMetric, ChangeType
from .context_framing import ContextFraming, ContextFrame, BiasIndicator, ContextType, BiasType
from .information_flow_tracker import InformationFlowTracker, FlowEvent, InformationTrail, FlowEventType, TransformationType
from .provenance_validator import ProvenanceValidator, ValidationResult, ValidationStatus, ValidationCheck

__version__ = "1.0.0"
__author__ = "The Mesh Development Team"
__description__ = "Comprehensive provenance tracking system for The Mesh"

__all__ = [
    # Core Components
    "ProvenanceTracker",
    "SourceReferenceManager", 
    "ConfidenceHistory",
    "ContextFraming",
    "InformationFlowTracker",
    "ProvenanceValidator",
    
    # Data Classes
    "SourceReference",
    "ConfidenceEntry",
    "ContextFrame",
    "BiasIndicator",
    "FlowEvent",
    "InformationTrail",
    "ValidationResult",
    
    # Enums
    "SourceType",
    "ConfidenceLevel",
    "ConfidenceMetric",
    "ChangeType",
    "ContextType",
    "BiasType",
    "FlowEventType",
    "TransformationType",
    "ValidationStatus",
    "ValidationCheck"
]
