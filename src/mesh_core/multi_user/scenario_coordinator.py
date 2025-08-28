"""
Scenario Coordinator
===================

Coordinates multi-user scenarios and simulations with proper
user interaction management and conflict resolution.
"""

import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ScenarioStatus(Enum):
    """Status of multi-user scenario"""
    INITIALIZING = "initializing"
    WAITING_FOR_USERS = "waiting_for_users"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class MultiUserScenario:
    """Multi-user scenario configuration"""
    scenario_id: str
    scenario_name: str
    max_participants: int
    current_participants: Set[str]
    scenario_data: Dict
    status: ScenarioStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class ScenarioCoordinator:
    """Coordinates multi-user scenarios"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.active_scenarios: Dict[str, MultiUserScenario] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of scenario_ids
        
        logger.info(f"ScenarioCoordinator initialized for node {node_id}")

    async def create_scenario(
        self,
        scenario_name: str,
        scenario_data: Dict,
        max_participants: int = 10,
        creator_id: Optional[str] = None
    ) -> str:
        """Create a new multi-user scenario"""
        scenario_id = f"scenario_{int(time.time())}_{len(self.active_scenarios)}"
        
        scenario = MultiUserScenario(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            max_participants=max_participants,
            current_participants=set(),
            scenario_data=scenario_data,
            status=ScenarioStatus.INITIALIZING,
            created_at=time.time()
        )
        
        # Add creator as first participant if provided
        if creator_id:
            scenario.current_participants.add(creator_id)
            if creator_id not in self.user_sessions:
                self.user_sessions[creator_id] = set()
            self.user_sessions[creator_id].add(scenario_id)
        
        self.active_scenarios[scenario_id] = scenario
        scenario.status = ScenarioStatus.WAITING_FOR_USERS
        
        logger.info(f"Multi-user scenario created: {scenario_id}")
        return scenario_id

    async def join_scenario(self, scenario_id: str, user_id: str) -> bool:
        """Add user to scenario"""
        if scenario_id not in self.active_scenarios:
            return False
        
        scenario = self.active_scenarios[scenario_id]
        
        # Check if scenario is accepting participants
        if scenario.status not in [ScenarioStatus.WAITING_FOR_USERS, ScenarioStatus.ACTIVE]:
            return False
        
        # Check if already at max capacity
        if len(scenario.current_participants) >= scenario.max_participants:
            return False
        
        # Add user to scenario
        scenario.current_participants.add(user_id)
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(scenario_id)
        
        logger.info(f"User {user_id} joined scenario {scenario_id}")
        
        # Auto-start if minimum participants met (2+ users)
        if len(scenario.current_participants) >= 2 and scenario.status == ScenarioStatus.WAITING_FOR_USERS:
            await self._start_scenario(scenario_id)
        
        return True

    async def leave_scenario(self, scenario_id: str, user_id: str) -> bool:
        """Remove user from scenario"""
        if scenario_id not in self.active_scenarios:
            return False
        
        scenario = self.active_scenarios[scenario_id]
        
        if user_id not in scenario.current_participants:
            return False
        
        # Remove user from scenario
        scenario.current_participants.remove(user_id)
        
        # Update user sessions
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(scenario_id)
        
        logger.info(f"User {user_id} left scenario {scenario_id}")
        
        # Handle scenario state based on remaining participants
        if len(scenario.current_participants) == 0:
            scenario.status = ScenarioStatus.CANCELLED
        elif len(scenario.current_participants) == 1 and scenario.status == ScenarioStatus.ACTIVE:
            scenario.status = ScenarioStatus.PAUSED
        
        return True

    async def coordinate_user_actions(
        self,
        scenario_id: str,
        user_actions: Dict[str, Dict]
    ) -> Dict[str, any]:
        """Coordinate actions from multiple users in scenario"""
        if scenario_id not in self.active_scenarios:
            return {"error": "Scenario not found"}
        
        scenario = self.active_scenarios[scenario_id]
        
        if scenario.status != ScenarioStatus.ACTIVE:
            return {"error": "Scenario not active"}
        
        # Process each user's action
        coordination_results = {}
        conflicts = []
        
        for user_id, action in user_actions.items():
            if user_id not in scenario.current_participants:
                coordination_results[user_id] = {"status": "rejected", "reason": "not_participant"}
                continue
            
            # Validate action against scenario rules
            validation_result = await self._validate_user_action(scenario, user_id, action)
            coordination_results[user_id] = validation_result
            
            # Check for conflicts with other actions
            conflict_check = await self._check_action_conflicts(scenario, user_id, action, user_actions)
            if conflict_check:
                conflicts.extend(conflict_check)
        
        # Resolve conflicts if any
        if conflicts:
            resolution_results = await self._resolve_action_conflicts(scenario, conflicts)
            coordination_results["conflict_resolutions"] = resolution_results
        
        # Update scenario state based on actions
        await self._update_scenario_state(scenario, user_actions, coordination_results)
        
        return {
            "scenario_id": scenario_id,
            "coordination_results": coordination_results,
            "conflicts": conflicts,
            "scenario_state": scenario.scenario_data.get("current_state", {})
        }

    async def get_scenario_status(self, scenario_id: str) -> Optional[Dict]:
        """Get current status of scenario"""
        if scenario_id not in self.active_scenarios:
            return None
        
        scenario = self.active_scenarios[scenario_id]
        
        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario.scenario_name,
            "status": scenario.status.value,
            "participants": list(scenario.current_participants),
            "participant_count": len(scenario.current_participants),
            "max_participants": scenario.max_participants,
            "created_at": scenario.created_at,
            "started_at": scenario.started_at,
            "duration": (time.time() - scenario.started_at) if scenario.started_at else 0
        }

    async def _start_scenario(self, scenario_id: str) -> bool:
        """Start the scenario"""
        scenario = self.active_scenarios[scenario_id]
        scenario.status = ScenarioStatus.ACTIVE
        scenario.started_at = time.time()
        
        # Initialize scenario state
        scenario.scenario_data["current_state"] = {
            "started": True,
            "turn": 0,
            "phase": "initial"
        }
        
        logger.info(f"Scenario started: {scenario_id}")
        return True

    async def _validate_user_action(
        self,
        scenario: MultiUserScenario,
        user_id: str,
        action: Dict
    ) -> Dict[str, any]:
        """Validate a user action against scenario rules"""
        # Mock validation - in real implementation would check scenario-specific rules
        return {
            "status": "approved",
            "action": action,
            "timestamp": time.time()
        }

    async def _check_action_conflicts(
        self,
        scenario: MultiUserScenario,
        user_id: str,
        action: Dict,
        all_actions: Dict[str, Dict]
    ) -> List[Dict]:
        """Check for conflicts between user actions"""
        conflicts = []
        
        # Mock conflict detection - in real implementation would detect actual conflicts
        action_type = action.get("type", "")
        for other_user_id, other_action in all_actions.items():
            if (other_user_id != user_id and 
                other_action.get("type") == action_type and
                action_type in ["claim_resource", "move_to_location"]):
                conflicts.append({
                    "type": "resource_conflict",
                    "users": [user_id, other_user_id],
                    "conflicting_action": action_type,
                    "details": action
                })
        
        return conflicts

    async def _resolve_action_conflicts(
        self,
        scenario: MultiUserScenario,
        conflicts: List[Dict]
    ) -> List[Dict]:
        """Resolve conflicts between user actions"""
        resolutions = []
        
        for conflict in conflicts:
            # Simple resolution: first user wins
            winner = conflict["users"][0]
            loser = conflict["users"][1]
            
            resolutions.append({
                "conflict_type": conflict["type"],
                "resolution": "first_user_priority",
                "winner": winner,
                "loser": loser,
                "compensation": "next_turn_priority"
            })
        
        return resolutions

    async def _update_scenario_state(
        self,
        scenario: MultiUserScenario,
        user_actions: Dict[str, Dict],
        results: Dict[str, any]
    ) -> None:
        """Update scenario state based on user actions"""
        if "current_state" not in scenario.scenario_data:
            scenario.scenario_data["current_state"] = {}
        
        state = scenario.scenario_data["current_state"]
        
        # Increment turn counter
        state["turn"] = state.get("turn", 0) + 1
        
        # Track successful actions
        successful_actions = [
            action for user_id, result in results.items()
            if isinstance(result, dict) and result.get("status") == "approved"
        ]
        
        state["last_actions"] = len(successful_actions)
        state["last_update"] = time.time()

    async def complete_scenario(self, scenario_id: str) -> bool:
        """Complete and finalize scenario"""
        if scenario_id not in self.active_scenarios:
            return False
        
        scenario = self.active_scenarios[scenario_id]
        scenario.status = ScenarioStatus.COMPLETED
        scenario.completed_at = time.time()
        
        # Remove from user sessions
        for user_id in scenario.current_participants:
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(scenario_id)
        
        logger.info(f"Scenario completed: {scenario_id}")
        return True

    async def get_user_scenarios(self, user_id: str) -> List[str]:
        """Get all scenarios a user is participating in"""
        return list(self.user_sessions.get(user_id, set()))