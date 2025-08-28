"""
Multi-Agent Coordination Module
==============================

Advanced multi-agent coordination engine for The Mesh simulation and rehearsal system.
Manages coordination between multiple agents in complex scenarios.

Components:
- AgentCoordinator: Central coordination engine
- TaskAllocator: Distributes tasks across agent networks
- ConflictResolver: Resolves conflicts between agents
- EmergentBehaviorAnalyzer: Analyzes emergent behaviors
"""

from .agent_coordinator import (
    AgentCoordinator,
    CoordinationStrategy,
    AgentRole,
    CoordinationState
)

from .task_allocator import (
    TaskAllocator,
    TaskAllocation,
    AllocationStrategy,
    TaskPriority
)

from .conflict_resolver import (
    ConflictResolver,
    AgentConflict,
    ConflictType,
    ResolutionStrategy
)

from .emergent_behavior_analyzer import (
    EmergentBehaviorAnalyzer,
    BehaviorPattern,
    EmergenceLevel,
    BehaviorInsight
)

# Alias for Phase 7 validation
MultiAgentCoordinationEngine = AgentCoordinator

__all__ = [
    'AgentCoordinator',
    'MultiAgentCoordinationEngine',  # Phase 7 alias
    'CoordinationStrategy',
    'AgentRole',
    'CoordinationState',
    'TaskAllocator',
    'TaskAllocation',
    'AllocationStrategy', 
    'TaskPriority',
    'ConflictResolver',
    'AgentConflict',
    'ConflictType',
    'ResolutionStrategy',
    'EmergentBehaviorAnalyzer',
    'BehaviorPattern',
    'EmergenceLevel',
    'BehaviorInsight'
]

__version__ = "1.0.0"