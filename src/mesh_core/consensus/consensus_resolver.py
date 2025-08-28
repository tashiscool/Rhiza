"""
Consensus Resolver
=================

Resolves consensus from voting results and manages implementation
of democratic decisions in The Mesh governance system.
"""

import asyncio
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConsensusThreshold(Enum):
    """Different consensus thresholds for different decision types"""
    SIMPLE_MAJORITY = 0.51
    SUPERMAJORITY_60 = 0.60
    SUPERMAJORITY_66 = 0.67
    SUPERMAJORITY_75 = 0.75
    UNANIMOUS = 1.0

class DecisionStatus(Enum):
    """Status of consensus decisions"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTING = "implementing"
    IMPLEMENTED = "implemented"
    FAILED = "failed"
    APPEALED = "appealed"

@dataclass
class ConsensusDecision:
    """Consensus decision record"""
    decision_id: str
    proposal_id: str
    session_id: str
    decision_type: str
    threshold_required: float
    threshold_achieved: float
    status: DecisionStatus
    implementation_plan: Dict
    implementation_start: Optional[float] = None
    implementation_deadline: Optional[float] = None
    implementation_progress: float = 0.0
    stakeholder_approvals: List[str] = None
    rollback_plan: Optional[Dict] = None
    audit_requirements: List[str] = None
    
    def __post_init__(self):
        if self.stakeholder_approvals is None:
            self.stakeholder_approvals = []
        if self.audit_requirements is None:
            self.audit_requirements = []

@dataclass
class DecisionImplementor:
    """Implementation handler for consensus decisions"""
    implementor_id: str
    capabilities: Set[str]
    max_concurrent_implementations: int
    current_implementations: List[str]
    success_rate: float
    average_implementation_time: float

class ConsensusResolver:
    """Resolves consensus and manages decision implementation"""
    
    def __init__(self, node_id: str, governance_config: Optional[Dict] = None):
        self.node_id = node_id
        self.config = governance_config or {}
        self.decisions: Dict[str, ConsensusDecision] = {}
        self.implementors: Dict[str, DecisionImplementor] = {}
        self.implementation_callbacks: Dict[str, Callable] = {}
        self.appeal_window_hours = 72  # 3 days to appeal
        self.implementation_queue: List[str] = []
        
        logger.info(f"ConsensusResolver initialized for node {node_id}")

    async def resolve_consensus(
        self,
        proposal_id: str,
        session_id: str,
        voting_results: Dict,
        threshold_type: ConsensusThreshold = ConsensusThreshold.SIMPLE_MAJORITY
    ) -> str:
        """Resolve consensus from voting results"""
        try:
            decision_id = self._generate_decision_id(proposal_id, session_id)
            
            # Extract voting metrics
            yes_percentage = voting_results.get("yes_percentage", 0.0)
            participation_rate = voting_results.get("participation_rate", 0.0)
            statistical_confidence = voting_results.get("statistical_confidence", 0.0)
            
            # Determine if threshold is met
            required_threshold = threshold_type.value
            threshold_achieved = yes_percentage
            threshold_met = threshold_achieved >= required_threshold
            
            # Additional validation criteria
            consensus_valid = (
                threshold_met and
                participation_rate >= self.config.get("minimum_participation", 0.25) and
                statistical_confidence >= self.config.get("minimum_confidence", 0.75)
            )
            
            # Create decision record
            decision = ConsensusDecision(
                decision_id=decision_id,
                proposal_id=proposal_id,
                session_id=session_id,
                decision_type=voting_results.get("decision_type", "general"),
                threshold_required=required_threshold,
                threshold_achieved=threshold_achieved,
                status=DecisionStatus.APPROVED if consensus_valid else DecisionStatus.REJECTED,
                implementation_plan=voting_results.get("implementation_plan", {}),
                audit_requirements=self._determine_audit_requirements(voting_results)
            )
            
            self.decisions[decision_id] = decision
            
            if consensus_valid:
                # Begin implementation planning
                await self._plan_implementation(decision)
                logger.info(f"Consensus achieved: {decision_id} (threshold: {threshold_achieved:.2f})")
            else:
                logger.info(f"Consensus failed: {decision_id} (threshold: {threshold_achieved:.2f} < {required_threshold})")
            
            return decision_id
            
        except Exception as e:
            logger.error(f"Failed to resolve consensus: {e}")
            raise

    async def implement_decision(self, decision_id: str) -> bool:
        """Begin implementation of an approved decision"""
        try:
            if decision_id not in self.decisions:
                return False
            
            decision = self.decisions[decision_id]
            
            # Verify decision is approved and not already implementing
            if decision.status != DecisionStatus.APPROVED:
                return False
            
            # Check if appeal window has passed
            if not await self._appeal_window_expired(decision_id):
                logger.info(f"Decision {decision_id} still in appeal window")
                return False
            
            # Find suitable implementor
            implementor = await self._assign_implementor(decision)
            if not implementor:
                logger.warning(f"No suitable implementor found for decision {decision_id}")
                return False
            
            # Update decision status
            decision.status = DecisionStatus.IMPLEMENTING
            decision.implementation_start = time.time()
            
            # Calculate implementation deadline
            estimated_duration = self._estimate_implementation_duration(decision)
            decision.implementation_deadline = decision.implementation_start + estimated_duration
            
            # Add to implementation queue
            self.implementation_queue.append(decision_id)
            
            # Begin actual implementation
            await self._execute_implementation(decision, implementor)
            
            logger.info(f"Implementation started: {decision_id} by {implementor.implementor_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to implement decision {decision_id}: {e}")
            return False

    async def track_implementation_progress(self, decision_id: str) -> Optional[Dict]:
        """Track progress of decision implementation"""
        try:
            if decision_id not in self.decisions:
                return None
            
            decision = self.decisions[decision_id]
            
            # Calculate progress based on implementation plan
            progress = await self._calculate_implementation_progress(decision)
            decision.implementation_progress = progress
            
            # Check for completion
            if progress >= 1.0:
                await self._complete_implementation(decision)
            
            now = time.time()
            time_elapsed = (now - decision.implementation_start) if decision.implementation_start else 0
            time_remaining = (decision.implementation_deadline - now) if decision.implementation_deadline else 0
            
            return {
                "decision_id": decision_id,
                "status": decision.status.value,
                "progress": progress,
                "time_elapsed_hours": time_elapsed / 3600,
                "time_remaining_hours": max(0, time_remaining / 3600),
                "on_schedule": time_remaining > 0 if decision.implementation_deadline else True,
                "implementation_steps_completed": self._get_completed_steps(decision),
                "next_milestones": self._get_next_milestones(decision)
            }
            
        except Exception as e:
            logger.error(f"Failed to track implementation progress: {e}")
            return None

    async def appeal_decision(
        self,
        decision_id: str,
        appellant_id: str,
        appeal_grounds: str,
        supporting_evidence: Optional[List[Dict]] = None
    ) -> bool:
        """Submit an appeal against a consensus decision"""
        try:
            if decision_id not in self.decisions:
                return False
            
            decision = self.decisions[decision_id]
            
            # Check if appeal window is still open
            if not await self._appeal_window_open(decision_id):
                return False
            
            # Validate appeal grounds
            valid_grounds = [
                "procedural_violation",
                "insufficient_participation",
                "technical_error",
                "new_evidence",
                "conflict_of_interest"
            ]
            
            if appeal_grounds not in valid_grounds:
                return False
            
            # Create appeal record
            appeal_id = self._generate_appeal_id(decision_id, appellant_id)
            appeal_record = {
                "appeal_id": appeal_id,
                "decision_id": decision_id,
                "appellant_id": appellant_id,
                "appeal_grounds": appeal_grounds,
                "supporting_evidence": supporting_evidence or [],
                "submitted_at": time.time(),
                "status": "pending_review"
            }
            
            # Store appeal (would be in separate appeals system)
            if not hasattr(self, 'appeals'):
                self.appeals = {}
            self.appeals[appeal_id] = appeal_record
            
            # Update decision status
            decision.status = DecisionStatus.APPEALED
            
            logger.info(f"Appeal submitted: {appeal_id} for decision {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit appeal: {e}")
            return False

    async def rollback_implementation(
        self,
        decision_id: str,
        rollback_reason: str,
        authorized_by: str
    ) -> bool:
        """Rollback a decision implementation"""
        try:
            if decision_id not in self.decisions:
                return False
            
            decision = self.decisions[decision_id]
            
            # Verify rollback is possible
            if decision.status not in [DecisionStatus.IMPLEMENTING, DecisionStatus.IMPLEMENTED]:
                return False
            
            if not decision.rollback_plan:
                logger.error(f"No rollback plan available for decision {decision_id}")
                return False
            
            # Execute rollback plan
            rollback_success = await self._execute_rollback(decision, rollback_reason, authorized_by)
            
            if rollback_success:
                decision.status = DecisionStatus.FAILED
                decision.implementation_progress = 0.0
                
                # Create rollback audit record
                rollback_record = {
                    "decision_id": decision_id,
                    "rollback_reason": rollback_reason,
                    "authorized_by": authorized_by,
                    "rollback_timestamp": time.time(),
                    "rollback_success": True
                }
                
                if not hasattr(decision, 'rollback_history'):
                    decision.rollback_history = []
                decision.rollback_history.append(rollback_record)
                
                logger.info(f"Implementation rolled back: {decision_id}")
                return True
            else:
                logger.error(f"Rollback failed for decision {decision_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rollback implementation: {e}")
            return False

    def register_implementor(
        self,
        implementor_id: str,
        capabilities: Set[str],
        max_concurrent: int = 3,
        success_rate: float = 0.95
    ) -> bool:
        """Register a decision implementor"""
        try:
            implementor = DecisionImplementor(
                implementor_id=implementor_id,
                capabilities=capabilities,
                max_concurrent_implementations=max_concurrent,
                current_implementations=[],
                success_rate=success_rate,
                average_implementation_time=24 * 3600  # 24 hours default
            )
            
            self.implementors[implementor_id] = implementor
            logger.info(f"Implementor registered: {implementor_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register implementor: {e}")
            return False

    def register_implementation_callback(
        self,
        decision_type: str,
        callback_function: Callable
    ) -> bool:
        """Register callback for specific decision types"""
        try:
            self.implementation_callbacks[decision_type] = callback_function
            logger.info(f"Implementation callback registered for: {decision_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to register callback: {e}")
            return False

    async def get_implementation_queue(self) -> List[Dict]:
        """Get current implementation queue status"""
        queue_status = []
        
        for decision_id in self.implementation_queue:
            if decision_id in self.decisions:
                decision = self.decisions[decision_id]
                progress = await self.track_implementation_progress(decision_id)
                
                queue_status.append({
                    "decision_id": decision_id,
                    "proposal_id": decision.proposal_id,
                    "status": decision.status.value,
                    "progress": progress,
                    "priority": self._calculate_implementation_priority(decision)
                })
        
        # Sort by priority
        queue_status.sort(key=lambda x: x["priority"], reverse=True)
        return queue_status

    def _generate_decision_id(self, proposal_id: str, session_id: str) -> str:
        """Generate unique decision ID"""
        content = f"{proposal_id}|{session_id}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_appeal_id(self, decision_id: str, appellant_id: str) -> str:
        """Generate unique appeal ID"""
        content = f"{decision_id}|{appellant_id}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _determine_audit_requirements(self, voting_results: Dict) -> List[str]:
        """Determine audit requirements based on decision type"""
        requirements = ["implementation_log", "progress_reports"]
        
        # Add specific requirements based on decision characteristics
        if voting_results.get("high_impact", False):
            requirements.extend(["stakeholder_signoff", "external_review"])
        
        if voting_results.get("technical_change", False):
            requirements.extend(["technical_review", "rollback_testing"])
        
        if voting_results.get("financial_impact", 0) > 1000:
            requirements.extend(["financial_audit", "cost_tracking"])
        
        return requirements

    async def _plan_implementation(self, decision: ConsensusDecision) -> bool:
        """Plan the implementation of a decision"""
        try:
            # Create default implementation plan if none provided
            if not decision.implementation_plan:
                decision.implementation_plan = {
                    "phases": [
                        {"name": "preparation", "duration_hours": 24, "dependencies": []},
                        {"name": "execution", "duration_hours": 48, "dependencies": ["preparation"]},
                        {"name": "validation", "duration_hours": 24, "dependencies": ["execution"]}
                    ],
                    "resources_required": [],
                    "success_criteria": []
                }
            
            # Generate rollback plan
            decision.rollback_plan = await self._generate_rollback_plan(decision)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to plan implementation: {e}")
            return False

    async def _appeal_window_expired(self, decision_id: str) -> bool:
        """Check if the appeal window has expired"""
        if decision_id not in self.decisions:
            return True
        
        decision = self.decisions[decision_id]
        if decision.status == DecisionStatus.APPEALED:
            return False  # Appeal is active
        
        # Appeal window starts when decision is approved
        decision_time = decision.implementation_start or time.time()
        appeal_deadline = decision_time + (self.appeal_window_hours * 3600)
        
        return time.time() > appeal_deadline

    async def _appeal_window_open(self, decision_id: str) -> bool:
        """Check if the appeal window is still open"""
        return not await self._appeal_window_expired(decision_id)

    async def _assign_implementor(self, decision: ConsensusDecision) -> Optional[DecisionImplementor]:
        """Find and assign a suitable implementor"""
        try:
            required_capabilities = set(decision.implementation_plan.get("required_capabilities", []))
            
            # Find implementors with required capabilities and availability
            suitable_implementors = []
            for impl_id, implementor in self.implementors.items():
                if (len(implementor.current_implementations) < implementor.max_concurrent_implementations and
                    (not required_capabilities or required_capabilities.issubset(implementor.capabilities))):
                    suitable_implementors.append(implementor)
            
            if not suitable_implementors:
                return None
            
            # Select best implementor based on success rate and availability
            best_implementor = max(suitable_implementors, key=lambda x: x.success_rate)
            best_implementor.current_implementations.append(decision.decision_id)
            
            return best_implementor
            
        except Exception as e:
            logger.error(f"Failed to assign implementor: {e}")
            return None

    def _estimate_implementation_duration(self, decision: ConsensusDecision) -> float:
        """Estimate implementation duration in seconds"""
        phases = decision.implementation_plan.get("phases", [])
        total_hours = sum(phase.get("duration_hours", 24) for phase in phases)
        return total_hours * 3600  # Convert to seconds

    async def _execute_implementation(
        self,
        decision: ConsensusDecision,
        implementor: DecisionImplementor
    ) -> bool:
        """Execute the actual implementation"""
        try:
            # Check if there's a registered callback for this decision type
            callback = self.implementation_callbacks.get(decision.decision_type)
            
            if callback:
                # Use registered callback
                implementation_result = await callback(decision, implementor)
            else:
                # Default implementation process
                implementation_result = await self._default_implementation_process(decision)
            
            return implementation_result
            
        except Exception as e:
            logger.error(f"Failed to execute implementation: {e}")
            return False

    async def _default_implementation_process(self, decision: ConsensusDecision) -> bool:
        """Default implementation process"""
        try:
            phases = decision.implementation_plan.get("phases", [])
            
            for phase in phases:
                phase_name = phase.get("name", "unknown")
                duration = phase.get("duration_hours", 24) * 3600
                
                logger.info(f"Executing implementation phase: {phase_name}")
                
                # Simulate phase execution
                await asyncio.sleep(min(1.0, duration / 3600))  # Max 1 second simulation
                
                # Update progress
                decision.implementation_progress += 1.0 / len(phases)
            
            return True
            
        except Exception as e:
            logger.error(f"Default implementation process failed: {e}")
            return False

    async def _calculate_implementation_progress(self, decision: ConsensusDecision) -> float:
        """Calculate current implementation progress"""
        if decision.status != DecisionStatus.IMPLEMENTING:
            return 0.0 if decision.status == DecisionStatus.APPROVED else 1.0
        
        if not decision.implementation_start:
            return 0.0
        
        # Calculate progress based on elapsed time and planned duration
        elapsed = time.time() - decision.implementation_start
        planned_duration = self._estimate_implementation_duration(decision)
        
        time_progress = min(1.0, elapsed / planned_duration)
        
        # Combine with actual progress tracking
        return max(decision.implementation_progress, time_progress)

    async def _complete_implementation(self, decision: ConsensusDecision) -> bool:
        """Mark implementation as completed"""
        try:
            decision.status = DecisionStatus.IMPLEMENTED
            decision.implementation_progress = 1.0
            
            # Remove from implementation queue
            if decision.decision_id in self.implementation_queue:
                self.implementation_queue.remove(decision.decision_id)
            
            # Update implementor status
            for implementor in self.implementors.values():
                if decision.decision_id in implementor.current_implementations:
                    implementor.current_implementations.remove(decision.decision_id)
                    break
            
            logger.info(f"Implementation completed: {decision.decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete implementation: {e}")
            return False

    def _get_completed_steps(self, decision: ConsensusDecision) -> List[str]:
        """Get list of completed implementation steps"""
        phases = decision.implementation_plan.get("phases", [])
        completed_count = int(decision.implementation_progress * len(phases))
        return [phase["name"] for phase in phases[:completed_count]]

    def _get_next_milestones(self, decision: ConsensusDecision) -> List[str]:
        """Get next implementation milestones"""
        phases = decision.implementation_plan.get("phases", [])
        completed_count = int(decision.implementation_progress * len(phases))
        return [phase["name"] for phase in phases[completed_count:completed_count + 2]]

    async def _generate_rollback_plan(self, decision: ConsensusDecision) -> Dict:
        """Generate rollback plan for a decision"""
        return {
            "rollback_steps": [
                "halt_current_implementation",
                "assess_rollback_impact", 
                "execute_reversal_steps",
                "validate_rollback_success"
            ],
            "rollback_dependencies": [],
            "rollback_risks": ["temporary_service_interruption"],
            "rollback_duration_hours": 12
        }

    async def _execute_rollback(
        self,
        decision: ConsensusDecision,
        reason: str,
        authorized_by: str
    ) -> bool:
        """Execute rollback of a decision"""
        try:
            rollback_steps = decision.rollback_plan.get("rollback_steps", [])
            
            for step in rollback_steps:
                logger.info(f"Executing rollback step: {step}")
                await asyncio.sleep(0.1)  # Simulate rollback execution
            
            logger.info(f"Rollback completed for decision {decision.decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            return False

    def _calculate_implementation_priority(self, decision: ConsensusDecision) -> float:
        """Calculate implementation priority"""
        priority = decision.threshold_achieved  # Base on consensus strength
        
        # Boost priority for high-impact decisions
        if decision.decision_type in ["constitutional_amendment", "emergency_action"]:
            priority += 0.5
        
        # Boost priority for decisions with approaching deadlines
        if decision.implementation_deadline:
            time_pressure = (decision.implementation_deadline - time.time()) / (24 * 3600)
            if time_pressure < 7:  # Less than 7 days
                priority += (7 - time_pressure) * 0.1
        
        return min(2.0, priority)

    async def export_consensus_data(self) -> Dict:
        """Export consensus resolution data"""
        return {
            "node_id": self.node_id,
            "decisions": {did: asdict(d) for did, d in self.decisions.items()},
            "implementors": {iid: asdict(i) for iid, i in self.implementors.items()},
            "implementation_queue": self.implementation_queue,
            "appeals": getattr(self, 'appeals', {}),
            "export_timestamp": time.time()
        }