"""
Multi-User Scenario Interface
============================

Enables multiple users to participate in shared scenarios and simulations
within The Mesh network with proper coordination and conflict resolution.

Components:
- ScenarioCoordinator: Coordinates multi-user scenarios
- UserSessionManager: Manages user sessions and interactions
- CollaborationEngine: Enables collaborative decision-making
- ConflictMediator: Mediates conflicts between users
"""

from .scenario_coordinator import ScenarioCoordinator
from .user_session_manager import UserSessionManager
from .collaboration_engine import CollaborationEngine
from .conflict_mediator import ConflictMediator

__all__ = [
    'ScenarioCoordinator',
    'UserSessionManager',
    'CollaborationEngine',
    'ConflictMediator'
]

__version__ = "1.0.0"