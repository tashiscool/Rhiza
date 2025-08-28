"""
Model Mutation Tracking Module
==============================

Tracks mutations and changes in AI models within The Mesh network.
Monitors model evolution, performance drift, and behavioral changes.

Components:
- MutationTracker: Tracks model mutations over time
- ModelVersionControl: Manages model versions and changes
- DriftDetector: Detects performance and behavioral drift
- EvolutionAnalyzer: Analyzes model evolution patterns
"""

from .mutation_tracker import (
    MutationTracker,
    MutationRecord,
    MutationType,
    MutationImpact,
    Mutation,
    ModelMutationTracker
)

# Create alias for compatibility
MutationSeverity = MutationImpact

from .model_version_control import (
    ModelVersionControl,
    ModelVersion,
    VersionMetadata,
    VersioningStrategy
)

from .drift_detector import (
    DriftDetector,
    DriftType,
    DriftSeverity,
    DriftAlert
)

from .evolution_analyzer import (
    EvolutionAnalyzer,
    EvolutionPattern,
    EvolutionTrend,
    EvolutionInsight
)

__all__ = [
    'MutationTracker',
    'MutationRecord',
    'MutationType', 
    'MutationSeverity',
    'ModelVersionControl',
    'ModelVersion',
    'VersionMetadata',
    'VersioningStrategy',
    'DriftDetector',
    'DriftType',
    'DriftSeverity',
    'DriftAlert',
    'EvolutionAnalyzer',
    'EvolutionPattern',
    'EvolutionTrend',
    'EvolutionInsight'
]

__version__ = "1.0.0"