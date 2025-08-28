"""
Multi-Agent Conflict Resolver
=============================

Sophisticated conflict resolution system for managing disputes and conflicts
that arise during multi-agent coordination in The Mesh network.
"""

import asyncio
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of conflicts that can occur in multi-agent systems"""
    RESOURCE_COMPETITION = "resource_competition"
    GOAL_CONTRADICTION = "goal_contradiction"
    COMMUNICATION_BREAKDOWN = "communication_breakdown"
    ROLE_AMBIGUITY = "role_ambiguity"
    COORDINATION_FAILURE = "coordination_failure"
    TRUST_EROSION = "trust_erosion"
    PERFORMANCE_DISPARITY = "performance_disparity"
    AUTHORITY_DISPUTE = "authority_dispute"
    ETHICAL_DISAGREEMENT = "ethical_disagreement"

class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts"""
    MEDIATION = "mediation"
    ARBITRATION = "arbitration"
    NEGOTIATION = "negotiation"
    RESTRUCTURING = "restructuring"
    SEPARATION = "separation"
    ESCALATION = "escalation"
    AUTOMATIC_RESOLUTION = "automatic_resolution"

class ConflictStatus(Enum):
    """Status of conflict resolution process"""
    DETECTED = "detected"
    ANALYZING = "analyzing"
    MEDIATING = "mediating"
    NEGOTIATING = "negotiating"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"

@dataclass
class ConflictParticipant:
    """Participant in a conflict"""
    agent_id: str
    role: str
    position: Dict[str, Any]
    evidence: List[Dict]
    credibility_score: float
    compromise_willingness: float

@dataclass
class ConflictEvidence:
    """Evidence related to a conflict"""
    evidence_id: str
    source: str
    type: str  # "performance", "communication", "behavior", "resource"
    content: Dict[str, Any]
    reliability: float
    timestamp: float
    supporting_agents: List[str]
    disputing_agents: List[str]

@dataclass
class ConflictRecord:
    """Complete record of a conflict and its resolution"""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    participants: List[ConflictParticipant]
    description: str
    evidence: List[ConflictEvidence]
    resolution_strategy: ResolutionStrategy
    status: ConflictStatus
    start_time: float
    resolution_time: Optional[float]
    resolution_outcome: Optional[Dict]
    mediator_notes: List[str]
    lessons_learned: List[str]

# Alias for compatibility
AgentConflict = ConflictRecord

class ConflictResolver:
    """Advanced conflict resolution system for multi-agent coordination"""
    
    def __init__(self, node_id: str, resolution_config: Optional[Dict] = None):
        self.node_id = node_id
        self.config = resolution_config or {}
        
        # Conflict tracking
        self.active_conflicts: Dict[str, ConflictRecord] = {}
        self.resolved_conflicts: Dict[str, ConflictRecord] = {}
        self.conflict_patterns: Dict[str, List[Dict]] = {}
        
        # Resolution resources
        self.mediator_agents: Set[str] = set()
        self.arbitrator_agents: Set[str] = set()
        self.resolution_protocols: Dict[ConflictType, Dict] = {}
        
        # Performance tracking
        self.resolution_success_rates: Dict[ResolutionStrategy, float] = {}
        self.agent_conflict_history: Dict[str, List[str]] = {}
        
        # Initialize default protocols
        self._initialize_resolution_protocols()
        
        logger.info(f"ConflictResolver initialized for node {node_id}")

    def _initialize_resolution_protocols(self):
        """Initialize default conflict resolution protocols"""
        self.resolution_protocols = {
            ConflictType.RESOURCE_COMPETITION: {
                "strategy": ResolutionStrategy.MEDIATION,
                "timeout": 1800,  # 30 minutes
                "escalation_threshold": 3,
                "auto_resolution": True
            },
            ConflictType.GOAL_CONTRADICTION: {
                "strategy": ResolutionStrategy.NEGOTIATION,
                "timeout": 2400,  # 40 minutes
                "escalation_threshold": 2,
                "auto_resolution": False
            },
            ConflictType.COMMUNICATION_BREAKDOWN: {
                "strategy": ResolutionStrategy.MEDIATION,
                "timeout": 900,   # 15 minutes
                "escalation_threshold": 2,
                "auto_resolution": True
            },
            ConflictType.TRUST_EROSION: {
                "strategy": ResolutionStrategy.ARBITRATION,
                "timeout": 3600,  # 1 hour
                "escalation_threshold": 1,
                "auto_resolution": False
            },
            ConflictType.AUTHORITY_DISPUTE: {
                "strategy": ResolutionStrategy.ARBITRATION,
                "timeout": 2400,  # 40 minutes
                "escalation_threshold": 1,
                "auto_resolution": False
            }
        }

    async def detect_conflict(
        self,
        session_id: str,
        agents: List[str],
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Detect potential conflicts in multi-agent coordination"""
        try:
            # Analyze coordination context for conflict indicators
            conflict_indicators = await self._analyze_conflict_indicators(
                agents, context
            )
            
            if not conflict_indicators:
                return None
            
            # Classify conflict type
            conflict_type = await self._classify_conflict(conflict_indicators)
            
            # Assess severity
            severity = await self._assess_conflict_severity(
                conflict_indicators, conflict_type
            )
            
            # Create conflict record
            conflict_id = self._generate_conflict_id(session_id)
            
            participants = [
                ConflictParticipant(
                    agent_id=agent_id,
                    role=context.get("agent_roles", {}).get(agent_id, "participant"),
                    position={},
                    evidence=[],
                    credibility_score=0.8,
                    compromise_willingness=0.5
                )
                for agent_id in agents
            ]
            
            conflict_record = ConflictRecord(
                conflict_id=conflict_id,
                conflict_type=conflict_type,
                severity=severity,
                participants=participants,
                description=conflict_indicators.get("description", "Multi-agent conflict detected"),
                evidence=[],
                resolution_strategy=self._select_resolution_strategy(conflict_type, severity),
                status=ConflictStatus.DETECTED,
                start_time=time.time(),
                resolution_time=None,
                resolution_outcome=None,
                mediator_notes=[],
                lessons_learned=[]
            )
            
            self.active_conflicts[conflict_id] = conflict_record
            
            logger.warning(f"Conflict detected: {conflict_id} ({conflict_type.value}, {severity.value})")
            return conflict_id
            
        except Exception as e:
            logger.error(f"Failed to detect conflict: {e}")
            return None

    async def resolve_conflict(self, conflict_id: str) -> bool:
        """Resolve an active conflict using appropriate strategy"""
        try:
            if conflict_id not in self.active_conflicts:
                return False
            
            conflict = self.active_conflicts[conflict_id]
            conflict.status = ConflictStatus.ANALYZING
            
            # Gather additional evidence
            await self._gather_conflict_evidence(conflict)
            
            # Update participant positions
            await self._update_participant_positions(conflict)
            
            # Execute resolution strategy
            resolution_success = False
            
            if conflict.resolution_strategy == ResolutionStrategy.MEDIATION:
                resolution_success = await self._mediate_conflict(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.NEGOTIATION:
                resolution_success = await self._negotiate_conflict(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.ARBITRATION:
                resolution_success = await self._arbitrate_conflict(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.RESTRUCTURING:
                resolution_success = await self._restructure_for_conflict(conflict)
            elif conflict.resolution_strategy == ResolutionStrategy.AUTOMATIC_RESOLUTION:
                resolution_success = await self._auto_resolve_conflict(conflict)
            
            # Update conflict status and record outcome
            if resolution_success:
                conflict.status = ConflictStatus.RESOLVED
                conflict.resolution_time = time.time()
                
                # Move to resolved conflicts
                self.resolved_conflicts[conflict_id] = conflict
                del self.active_conflicts[conflict_id]
                
                # Update agent conflict history
                for participant in conflict.participants:
                    if participant.agent_id not in self.agent_conflict_history:
                        self.agent_conflict_history[participant.agent_id] = []
                    self.agent_conflict_history[participant.agent_id].append(conflict_id)
                
                logger.info(f"Conflict resolved: {conflict_id}")
            else:
                # Consider escalation
                if conflict.severity.value < 4:  # Not already critical
                    conflict.severity = ConflictSeverity(conflict.severity.value + 1)
                    conflict.status = ConflictStatus.ESCALATED
                    logger.warning(f"Conflict escalated: {conflict_id}")
                else:
                    conflict.status = ConflictStatus.UNRESOLVED
                    logger.error(f"Conflict unresolved: {conflict_id}")
            
            return resolution_success
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return False

    async def _analyze_conflict_indicators(
        self,
        agents: List[str],
        context: Dict[str, Any]
    ) -> Optional[Dict]:
        """Analyze context for conflict indicators"""
        indicators = {}
        
        # Check performance disparities
        performance_data = context.get("agent_performance", {})
        if performance_data:
            performance_values = list(performance_data.values())
            if performance_values:
                max_perf = max(performance_values)
                min_perf = min(performance_values)
                if max_perf - min_perf > 0.3:  # 30% disparity
                    indicators["performance_disparity"] = {
                        "severity": (max_perf - min_perf),
                        "agents": agents
                    }
        
        # Check communication patterns
        comm_data = context.get("communication_log", [])
        if len(comm_data) < len(agents) * 2:  # Low communication
            indicators["communication_breakdown"] = {
                "severity": 0.6,
                "low_communication": True
            }
        
        # Check resource competition
        resource_usage = context.get("resource_usage", {})
        if resource_usage:
            high_usage_agents = [
                agent for agent, usage in resource_usage.items()
                if usage > 0.8
            ]
            if len(high_usage_agents) > 1:
                indicators["resource_competition"] = {
                    "severity": 0.7,
                    "competing_agents": high_usage_agents
                }
        
        # Check goal alignment
        agent_goals = context.get("agent_goals", {})
        if len(set(agent_goals.values())) > len(agent_goals) * 0.7:
            indicators["goal_contradiction"] = {
                "severity": 0.5,
                "conflicting_goals": True
            }
        
        return indicators if indicators else None

    async def _classify_conflict(self, indicators: Dict) -> ConflictType:
        """Classify the type of conflict based on indicators"""
        # Priority-based classification
        if "resource_competition" in indicators:
            return ConflictType.RESOURCE_COMPETITION
        elif "goal_contradiction" in indicators:
            return ConflictType.GOAL_CONTRADICTION
        elif "communication_breakdown" in indicators:
            return ConflictType.COMMUNICATION_BREAKDOWN
        elif "performance_disparity" in indicators:
            return ConflictType.PERFORMANCE_DISPARITY
        else:
            return ConflictType.COORDINATION_FAILURE

    async def _assess_conflict_severity(
        self,
        indicators: Dict,
        conflict_type: ConflictType
    ) -> ConflictSeverity:
        """Assess the severity of a conflict"""
        max_severity = 0
        
        for indicator_type, data in indicators.items():
            severity = data.get("severity", 0.5)
            max_severity = max(max_severity, severity)
        
        if max_severity >= 0.8:
            return ConflictSeverity.CRITICAL
        elif max_severity >= 0.6:
            return ConflictSeverity.HIGH
        elif max_severity >= 0.4:
            return ConflictSeverity.MODERATE
        else:
            return ConflictSeverity.LOW

    def _select_resolution_strategy(
        self,
        conflict_type: ConflictType,
        severity: ConflictSeverity
    ) -> ResolutionStrategy:
        """Select appropriate resolution strategy"""
        protocol = self.resolution_protocols.get(conflict_type)
        
        if not protocol:
            # Default based on severity
            if severity == ConflictSeverity.CRITICAL:
                return ResolutionStrategy.ARBITRATION
            elif severity == ConflictSeverity.HIGH:
                return ResolutionStrategy.MEDIATION
            else:
                return ResolutionStrategy.NEGOTIATION
        
        return protocol["strategy"]

    async def _gather_conflict_evidence(self, conflict: ConflictRecord) -> bool:
        """Gather evidence related to the conflict"""
        try:
            # Simulate evidence gathering
            evidence_items = []
            
            for i, participant in enumerate(conflict.participants):
                evidence = ConflictEvidence(
                    evidence_id=f"{conflict.conflict_id}_evidence_{i}",
                    source=participant.agent_id,
                    type="behavioral",
                    content={"performance_metrics": {"task_success": 0.7 + (i * 0.1)}},
                    reliability=0.8,
                    timestamp=time.time(),
                    supporting_agents=[],
                    disputing_agents=[]
                )
                evidence_items.append(evidence)
            
            conflict.evidence = evidence_items
            return True
            
        except Exception as e:
            logger.error(f"Failed to gather conflict evidence: {e}")
            return False

    async def _update_participant_positions(self, conflict: ConflictRecord) -> bool:
        """Update positions of conflict participants"""
        try:
            for participant in conflict.participants:
                # Simulate position updates based on evidence
                participant.position = {
                    "stance": "cooperative" if participant.compromise_willingness > 0.6 else "defensive",
                    "demands": ["resource_allocation", "role_clarification"],
                    "concessions": ["shared_resources"] if participant.compromise_willingness > 0.7 else []
                }
                
                # Update credibility based on evidence
                participant.credibility_score = 0.8  # Simplified
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update participant positions: {e}")
            return False

    async def _mediate_conflict(self, conflict: ConflictRecord) -> bool:
        """Mediate a conflict through guided discussion"""
        try:
            conflict.status = ConflictStatus.MEDIATING
            
            # Simulate mediation process
            mediation_rounds = 3
            agreement_probability = 0.7
            
            for round_num in range(mediation_rounds):
                conflict.mediator_notes.append(
                    f"Mediation round {round_num + 1}: Facilitated discussion between agents"
                )
                
                # Check for agreement
                compromise_scores = [p.compromise_willingness for p in conflict.participants]
                avg_compromise = sum(compromise_scores) / len(compromise_scores)
                
                if avg_compromise > agreement_probability:
                    conflict.resolution_outcome = {
                        "type": "mediated_agreement",
                        "terms": ["shared_resources", "role_clarification", "performance_goals"],
                        "satisfaction_level": avg_compromise
                    }
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to mediate conflict: {e}")
            return False

    async def _negotiate_conflict(self, conflict: ConflictRecord) -> bool:
        """Resolve conflict through negotiation"""
        try:
            conflict.status = ConflictStatus.NEGOTIATING
            
            # Simulate negotiation process
            negotiation_success_rate = 0.6
            
            conflict.resolution_outcome = {
                "type": "negotiated_settlement",
                "terms": ["resource_sharing", "goal_alignment"],
                "satisfaction_level": negotiation_success_rate
            }
            
            return negotiation_success_rate > 0.5
            
        except Exception as e:
            logger.error(f"Failed to negotiate conflict: {e}")
            return False

    async def _arbitrate_conflict(self, conflict: ConflictRecord) -> bool:
        """Resolve conflict through arbitration"""
        try:
            # Simulate arbitration decision
            arbitration_decision = {
                "type": "arbitrated_decision",
                "decision": "resource_reallocation",
                "binding": True,
                "enforcement": "automatic"
            }
            
            conflict.resolution_outcome = arbitration_decision
            return True
            
        except Exception as e:
            logger.error(f"Failed to arbitrate conflict: {e}")
            return False

    async def _restructure_for_conflict(self, conflict: ConflictRecord) -> bool:
        """Resolve conflict through system restructuring"""
        try:
            restructure_plan = {
                "type": "system_restructure",
                "changes": ["role_redefinition", "resource_reallocation", "communication_protocols"],
                "timeline": "immediate"
            }
            
            conflict.resolution_outcome = restructure_plan
            return True
            
        except Exception as e:
            logger.error(f"Failed to restructure for conflict: {e}")
            return False

    async def _auto_resolve_conflict(self, conflict: ConflictRecord) -> bool:
        """Automatically resolve conflict using predefined rules"""
        try:
            if conflict.conflict_type == ConflictType.RESOURCE_COMPETITION:
                # Implement resource allocation algorithm
                auto_resolution = {
                    "type": "automatic_resolution",
                    "action": "equal_resource_distribution",
                    "implementation": "immediate"
                }
            elif conflict.conflict_type == ConflictType.COMMUNICATION_BREAKDOWN:
                auto_resolution = {
                    "type": "automatic_resolution", 
                    "action": "communication_protocol_reset",
                    "implementation": "immediate"
                }
            else:
                return False
            
            conflict.resolution_outcome = auto_resolution
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-resolve conflict: {e}")
            return False

    def _generate_conflict_id(self, session_id: str) -> str:
        """Generate unique conflict ID"""
        content = f"conflict|{session_id}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get statistics about conflict resolution performance"""
        try:
            total_conflicts = len(self.resolved_conflicts) + len(self.active_conflicts)
            resolved_count = len(self.resolved_conflicts)
            
            if total_conflicts == 0:
                return {"total_conflicts": 0, "resolution_rate": 0}
            
            resolution_rate = resolved_count / total_conflicts
            
            # Calculate average resolution time
            resolution_times = [
                (conflict.resolution_time - conflict.start_time) 
                for conflict in self.resolved_conflicts.values()
                if conflict.resolution_time
            ]
            
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            # Conflict type distribution
            conflict_types = {}
            for conflict in list(self.active_conflicts.values()) + list(self.resolved_conflicts.values()):
                conflict_type = conflict.conflict_type.value
                conflict_types[conflict_type] = conflict_types.get(conflict_type, 0) + 1
            
            return {
                "total_conflicts": total_conflicts,
                "active_conflicts": len(self.active_conflicts),
                "resolved_conflicts": resolved_count,
                "resolution_rate": resolution_rate,
                "average_resolution_time_seconds": avg_resolution_time,
                "conflict_type_distribution": conflict_types
            }
            
        except Exception as e:
            logger.error(f"Failed to get conflict statistics: {e}")
            return {"error": str(e)}