"""
Mesh Learning Module
===================

Phase 8: Autonomous Evolution & Learning
Enable nodes to learn and evolve while maintaining alignment
"""

from .continual_learner import ContinualLearner
from .adapter_manager import AdapterManager
from .interaction_learner import InteractionLearner
from .knowledge_distiller import KnowledgeDistiller
from .quality_assurer import QualityAssurer
from .value_alignment_system import ValueAlignmentSystem, ValueVector, AlignmentScore

# Create LearningCoordinator alias to ContinualLearner for Phase 8 validation
LearningCoordinator = ContinualLearner

__all__ = [
    "ContinualLearner",
    "LearningCoordinator",  # Phase 8 alias
    "AdapterManager", 
    "InteractionLearner",
    "KnowledgeDistiller",
    "QualityAssurer",
    "ValueAlignmentSystem",
    "ValueVector",
    "AlignmentScore"
]

