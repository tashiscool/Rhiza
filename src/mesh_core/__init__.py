"""
The Mesh - A Decentralized, Self-Correcting Knowledge and Consensus Network

Core module providing the foundation for decentralized AI systems.
"""

# Core components
from .mesh_orchestrator import MeshOrchestrator
from .truth_layer import TruthLayer
from .intent_monitor import IntentMonitor
from .empathy_engine import EmpathyEngine
from .personal_agent import PersonalAgent, create_personal_agent
from .palm_slab import PalmSlabInterface

# Axiom integration
from .axiom_integration import (
    AxiomProcessor, AxiomVerificationResult, AxiomFactSubmission,
    TruthValidator, TruthValidationResult, TruthClaim, ValidationEvidence, ConsensusResult,
    ConfidenceScorer, ConfidenceScore, ScoreComponents, ConfidenceFactor,
    KnowledgeValidator, ValidationResult, KnowledgeClaim,
    AxiomMeshBridge, HybridVerificationResult, VerificationRequest
)

# Leon integration (Voice & Audio)
from .leon_integration import (
    LocalVoiceProcessor, VoiceProcessingConfig, TranscriptionResult, SynthesisResult, create_voice_processor,
    VoiceActivityDetector, VADConfig, VADResult, create_vad_detector,
    VoiceOptimizer, OptimizationConfig, OptimizedAudio, create_voice_optimizer
)

# Memory systems
from .memory import (
    FactExtractor, FactExtractionConfig, ExtractedFact, FactExtractionResult, create_fact_extractor,
    RelevanceScorer, RelevanceConfig, RelevanceScore, RelevanceResult, create_relevance_scorer,
    KnowledgeGraph, KnowledgeGraphConfig, KnowledgeNode, KnowledgeRelationship, KnowledgeGraphResult, create_knowledge_graph
)

# Task management
from .tasks import (
    TaskParser, TaskParserConfig, ParsedTask, TaskSchedule, TaskParsingResult, create_task_parser,
    WorkflowEngine, WorkflowEngineConfig, WorkflowStep, WorkflowExecution, WorkflowResult, TaskStatus, TaskStepStatus, create_workflow_engine,
    TaskScheduler, SchedulerConfig, ScheduledTask, ExecutionContext, SchedulingResult, ScheduleType, TaskPriority, create_task_scheduler
)

# Personal AI components (Phase 4)
from .proactive_manager import (
    ProactiveManager, ProactiveConfig, ProactiveSuggestion, ContextSnapshot, ProactiveResult,
    SuggestionType, SuggestionStatus, ProactivityLevel, create_proactive_manager
)

from .context_learner import (
    ContextLearner, ContextLearnerConfig, LearningPattern, LearningInsight, AdaptationRecommendation,
    LearningPatternType, LearningQuality, AdaptationType, LearningResult, create_context_learner
)

# Mesh Integration Bridge - Connects Sentient modules with Mesh systems
from .sentient_mesh_bridge import (
    SentientMeshBridge, PalmSlabConfig, MeshValidationResult, create_palm_slab_node
)

# Complete Palm Slab Interface - True palm slab node implementation
from .palm_slab_interface import (
    PalmSlabInterface, PalmSlabUser, PalmSlabInteraction, PrivacyRing, create_palm_slab_node as create_complete_palm_slab
)

# Governance systems
from .governance import (
    ConstitutionEngine, ProtocolEnforcer, AmendmentSystem, LocalCustomizer, RightsFramework, GovernanceIntegration
)

# Simulation systems (optional - create mock if not available)
try:
    from .simulation import (
        ScenarioGenerator, ChoiceRehearser, EmpathyTrainer, ConsequencePredictor, ScenarioSharer
    )
except ImportError:
    # Mock implementations for missing simulation module
    ScenarioGenerator = ChoiceRehearser = EmpathyTrainer = ConsequencePredictor = ScenarioSharer = None

# Learning systems (optional - create mock if not available) 
try:
    from .learning import (
        ContinualLearner, LearningCoordinator, AdapterManager, InteractionLearner, KnowledgeDistiller, QualityAssurer, ValueAlignmentSystem, ValueVector, AlignmentScore
    )
except ImportError:
    # Mock implementations for missing learning module
    ContinualLearner = LearningCoordinator = AdapterManager = InteractionLearner = KnowledgeDistiller = QualityAssurer = ValueAlignmentSystem = ValueVector = AlignmentScore = None

# Interface systems (optional - create mock if not available)
try:
    from .interface import (
        IntentModeler, ContextUnderstander, MultimodalProcessor, ClarificationEngine, NaturalProcessor
    )
except ImportError:
    # Mock implementations for missing interface module
    IntentModeler = ContextUnderstander = MultimodalProcessor = ClarificationEngine = NaturalProcessor = None

# Watchdog systems
from .watchdogs import (
    EntropyMonitor, ManipulationDetector, FeedbackAnalyzer, DriftWarner, HealthMonitor
)

# Version information
__version__ = "5.0.0"
__description__ = "Complete Palm Slab implementation with Mesh-integrated Sentient capabilities"
__author__ = "The Mesh Development Team"

# Export all components
__all__ = [
    # Core components
    'MeshOrchestrator', 'TruthLayer', 'IntentMonitor', 'EmpathyEngine', 'PersonalAgent', 'PalmSlabInterface',
    
    # Axiom integration
    'AxiomProcessor', 'AxiomVerificationResult', 'AxiomFactSubmission',
    'TruthValidator', 'TruthValidationResult', 'TruthClaim', 'ValidationEvidence', 'ConsensusResult',
    'ConfidenceScorer', 'ConfidenceScore', 'ScoreComponents', 'ConfidenceFactor',
    'KnowledgeValidator', 'ValidationResult', 'KnowledgeClaim',
    'AxiomMeshBridge', 'HybridVerificationResult', 'VerificationRequest',
    
    # Leon integration (Voice & Audio)
    'LocalVoiceProcessor', 'VoiceProcessingConfig', 'TranscriptionResult', 'SynthesisResult', 'create_voice_processor',
    'VoiceActivityDetector', 'VADConfig', 'VADResult', 'create_vad_detector',
    'VoiceOptimizer', 'OptimizationConfig', 'OptimizedAudio', 'create_voice_optimizer',
    
    # Memory systems
    'FactExtractor', 'FactExtractionConfig', 'ExtractedFact', 'FactExtractionResult', 'create_fact_extractor',
    'RelevanceScorer', 'RelevanceConfig', 'RelevanceScore', 'RelevanceResult', 'create_relevance_scorer',
    'KnowledgeGraph', 'KnowledgeGraphConfig', 'KnowledgeNode', 'KnowledgeRelationship', 'KnowledgeGraphResult', 'create_knowledge_graph',
    
    # Task management
    'TaskParser', 'TaskParserConfig', 'ParsedTask', 'TaskSchedule', 'TaskParsingResult', 'create_task_parser',
    'WorkflowEngine', 'WorkflowEngineConfig', 'WorkflowStep', 'WorkflowExecution', 'WorkflowResult', 'TaskStatus', 'TaskStepStatus', 'create_workflow_engine',
    'TaskScheduler', 'SchedulerConfig', 'ScheduledTask', 'ExecutionContext', 'SchedulingResult', 'ScheduleType', 'TaskPriority', 'create_task_scheduler',
    
    # Personal AI components (Phase 4)
    'ProactiveManager', 'ProactiveConfig', 'ProactiveSuggestion', 'ContextSnapshot', 'ProactiveResult',
    'SuggestionType', 'SuggestionStatus', 'ProactivityLevel', 'create_proactive_manager',
    
    'ContextLearner', 'ContextLearnerConfig', 'LearningPattern', 'LearningInsight', 'AdaptationRecommendation',
    'LearningPatternType', 'LearningQuality', 'AdaptationType', 'LearningResult', 'create_context_learner',
    
    # Mesh Integration Bridge
    'SentientMeshBridge', 'PalmSlabConfig', 'MeshValidationResult', 'create_palm_slab_node',
    
    # Complete Palm Slab Interface
    'PalmSlabInterface', 'PalmSlabUser', 'PalmSlabInteraction', 'PrivacyRing', 'create_complete_palm_slab',
    
    # Governance systems
    'ConstitutionEngine', 'ProtocolEnforcer', 'AmendmentSystem', 'LocalCustomizer', 'RightsFramework', 'GovernanceIntegration',
    
    # Simulation systems
    'ScenarioGenerator', 'ChoiceRehearser', 'EmpathyTrainer', 'ConsequencePredictor', 'ScenarioSharer',
    
    # Learning systems
    'ContinualLearner', 'LearningCoordinator', 'AdapterManager', 'InteractionLearner', 'KnowledgeDistiller', 'QualityAssurer', 'ValueAlignmentSystem', 'ValueVector', 'AlignmentScore',
    
    # Interface systems
    'IntentModeler', 'ContextUnderstander', 'MultimodalProcessor', 'ClarificationEngine', 'NaturalProcessor',
    
    # Watchdog systems
    'EntropyMonitor', 'ManipulationDetector', 'FeedbackAnalyzer', 'DriftWarner', 'HealthMonitor',
    
    # Factory functions
    'create_personal_agent', 'create_proactive_manager', 'create_context_learner', 'create_palm_slab_node', 'create_complete_palm_slab'
]