"""
Mesh Governance Integration Layer
=================================

Integrates governance systems with existing mesh infrastructure including
immunity, trust, and network layers for comprehensive constitutional enforcement.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .constitution_engine import ConstitutionEngine, ConstitutionalRule, RuleViolation
from .protocol_enforcer import ProtocolEnforcer, EnforcementAction, EnforcementEvent
from .rights_framework import RightsFramework, UserRight, RightType

# Import existing mesh systems for integration
try:
    from ..immunity.immune_response import ImmuneResponseSystem
    from ..immunity.node_isolation import NodeIsolation, IsolationLevel, IsolationReason
    from ..trust.trust_ledger import TrustLedger
    from ..network.network_health import NetworkHealth
except ImportError:
    # Mock classes for standalone operation
    class ImmuneResponseSystem:
        def __init__(self):
            pass
    
    class NodeIsolation:
        def __init__(self):
            pass
    
    class TrustLedger:
        def __init__(self, node_id):
            pass
    
    class NetworkHealth:
        def __init__(self, node_id):
            pass
    
    class IsolationLevel:
        QUARANTINE = "quarantine"
        ISOLATION = "isolation"
    
    class IsolationReason:
        CONSTITUTIONAL_VIOLATION = "constitutional_violation"

logger = logging.getLogger(__name__)


class GovernanceIntegrationLevel(Enum):
    """Levels of governance integration"""
    ADVISORY = "advisory"          # Governance provides recommendations only
    ENFORCEMENT = "enforcement"    # Governance can enforce rules
    CONSTITUTIONAL = "constitutional"  # Full constitutional governance


@dataclass
class GovernanceAction:
    """An action taken by the governance system"""
    action_id: str
    node_id: str
    rule_id: str
    action_type: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class GovernanceIntegration:
    """
    Integrates governance systems with existing mesh infrastructure
    """
    
    def __init__(self,
                 node_id: str,
                 constitution_engine: ConstitutionEngine,
                 protocol_enforcer: ProtocolEnforcer,
                 rights_framework: RightsFramework,
                 trust_ledger: Optional[TrustLedger] = None,
                 network_health: Optional[NetworkHealth] = None,
                 immune_response: Optional[ImmuneResponseSystem] = None,
                 node_isolation: Optional[NodeIsolation] = None):
        
        self.node_id = node_id
        self.constitution = constitution_engine
        self.enforcer = protocol_enforcer
        self.rights = rights_framework
        
        # Optional integrations with existing systems
        self.trust_ledger = trust_ledger
        self.network_health = network_health
        self.immune_response = immune_response
        self.node_isolation = node_isolation
        
        # Integration state
        self.integration_level = GovernanceIntegrationLevel.ADVISORY
        self.governance_actions: List[GovernanceAction] = []
        self.enforcement_active = False
        
        # Setup enforcement callbacks if systems are available
        self._setup_enforcement_integration()
        
        logger.info(f"Governance integration initialized for node {node_id}")
    
    def _setup_enforcement_integration(self):
        """Setup enforcement integration with existing systems"""
        if not (self.trust_ledger and self.node_isolation):
            logger.warning("Missing required systems for full governance integration")
            return
        
        # Register enforcement callbacks with protocol enforcer
        self.enforcer.register_enforcement_callback(
            EnforcementAction.QUARANTINE, 
            self._handle_quarantine_enforcement
        )
        
        self.enforcer.register_enforcement_callback(
            EnforcementAction.BAN, 
            self._handle_ban_enforcement
        )
        
        self.enforcer.register_enforcement_callback(
            EnforcementAction.THROTTLE, 
            self._handle_throttle_enforcement
        )
        
        self.integration_level = GovernanceIntegrationLevel.ENFORCEMENT
        self.enforcement_active = True
        
        logger.info("Governance enforcement integration activated")
    
    async def _handle_quarantine_enforcement(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Handle quarantine enforcement action"""
        try:
            if not self.node_isolation:
                return False
            
            # Quarantine the node using isolation system
            success = await self.node_isolation.quarantine_node(
                node_id=node_id,
                reason=IsolationReason.CONSTITUTIONAL_VIOLATION,
                reason_details=f"Constitutional rule violation: {rule_id}",
                initiated_by="governance_system"
            )
            
            # Record governance action
            action = GovernanceAction(
                action_id=f"gov_quarantine_{int(datetime.utcnow().timestamp())}",
                node_id=node_id,
                rule_id=rule_id,
                action_type="quarantine",
                description=f"Quarantined node for rule violation: {rule_id}",
                timestamp=datetime.utcnow(),
                metadata=metadata,
                success=success
            )
            self.governance_actions.append(action)
            
            # Update trust score if available
            if self.trust_ledger and success:
                await self._apply_trust_penalty(node_id, "constitutional_violation", 0.3)
            
            logger.info(f"Governance quarantine {'successful' if success else 'failed'} for node {node_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error handling quarantine enforcement: {e}")
            return False
    
    async def _handle_ban_enforcement(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Handle ban enforcement action"""
        try:
            if not self.node_isolation:
                return False
            
            # Isolate the node permanently
            success = await self.node_isolation.emergency_isolate_node(
                node_id=node_id,
                reason=IsolationReason.CONSTITUTIONAL_VIOLATION,
                reason_details=f"Serious constitutional violation: {rule_id}",
                initiated_by="governance_system"
            )
            
            # Record governance action
            action = GovernanceAction(
                action_id=f"gov_ban_{int(datetime.utcnow().timestamp())}",
                node_id=node_id,
                rule_id=rule_id,
                action_type="ban",
                description=f"Banned node for serious rule violation: {rule_id}",
                timestamp=datetime.utcnow(),
                metadata=metadata,
                success=success
            )
            self.governance_actions.append(action)
            
            # Apply severe trust penalty
            if self.trust_ledger and success:
                await self._apply_trust_penalty(node_id, "constitutional_ban", 0.8)
            
            logger.warning(f"Governance ban {'successful' if success else 'failed'} for node {node_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error handling ban enforcement: {e}")
            return False
    
    async def _handle_throttle_enforcement(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Handle throttle enforcement action"""
        try:
            # Apply trust penalty to reduce node's influence
            if self.trust_ledger:
                success = await self._apply_trust_penalty(node_id, "constitutional_throttle", 0.1)
            else:
                success = True  # Mock success if no trust system
            
            # Record governance action
            action = GovernanceAction(
                action_id=f"gov_throttle_{int(datetime.utcnow().timestamp())}",
                node_id=node_id,
                rule_id=rule_id,
                action_type="throttle",
                description=f"Throttled node for rule violation: {rule_id}",
                timestamp=datetime.utcnow(),
                metadata=metadata,
                success=success
            )
            self.governance_actions.append(action)
            
            logger.info(f"Governance throttle {'successful' if success else 'failed'} for node {node_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error handling throttle enforcement: {e}")
            return False
    
    async def _apply_trust_penalty(self, node_id: str, violation_type: str, penalty: float) -> bool:
        """Apply trust penalty for governance violations"""
        try:
            if not self.trust_ledger:
                return False
            
            # Apply trust penalty (this would need to be implemented in TrustLedger)
            # For now, we'll log the intended action
            logger.info(f"Would apply trust penalty of {penalty} to node {node_id} for {violation_type}")
            
            # In a real implementation, this might call:
            # await self.trust_ledger.apply_penalty(node_id, penalty, violation_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying trust penalty: {e}")
            return False
    
    async def check_action_compliance(self, node_id: str, action: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if an action complies with constitutional rules"""
        try:
            # Use constitution engine to check compliance
            is_compliant, violations = self.constitution.check_compliance(node_id, action, context)
            
            # If not compliant and enforcement is active, take action
            if not is_compliant and self.enforcement_active:
                await self._handle_violations(node_id, violations, context)
            
            return is_compliant, violations
            
        except Exception as e:
            logger.error(f"Error checking action compliance: {e}")
            return False, []
    
    async def _handle_violations(self, node_id: str, violations: List[str], context: Dict[str, Any]):
        """Handle rule violations through enforcement system"""
        try:
            for rule_id in violations:
                if rule_id in self.constitution.rules:
                    rule = self.constitution.rules[rule_id]
                    
                    # Create mock violation detection for enforcer
                    violation_data = {
                        "rule_id": rule_id,
                        "node_id": node_id,
                        "priority": rule.priority,
                        "context": context
                    }
                    
                    # Let enforcer handle the violation
                    await self.enforcer.handle_violation(node_id, rule_id, violation_data)
            
        except Exception as e:
            logger.error(f"Error handling violations: {e}")
    
    async def verify_user_rights(self, user_id: str, requested_action: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify user rights for a requested action"""
        try:
            # Check user rights through rights framework
            user_rights = self.rights.get_user_rights(user_id)
            violations = []
            allowed = True
            
            # Simple rights checking logic (can be extended)
            if requested_action == "participate_in_governance":
                if not any(r.right_type == RightType.PARTICIPATION and r.is_active() for r in user_rights):
                    violations.append("participation_right_required")
                    allowed = False
            
            elif requested_action == "access_private_data":
                if not any(r.right_type == RightType.PRIVACY and r.is_active() for r in user_rights):
                    violations.append("privacy_right_violation")
                    allowed = False
            
            return allowed, violations
            
        except Exception as e:
            logger.error(f"Error verifying user rights: {e}")
            return False, ["rights_verification_error"]
    
    def get_governance_status(self) -> Dict[str, Any]:
        """Get current governance integration status"""
        try:
            constitution_summary = self.constitution.get_constitution_summary()
            
            return {
                "integration_level": self.integration_level.value,
                "enforcement_active": self.enforcement_active,
                "total_governance_actions": len(self.governance_actions),
                "recent_actions": len([a for a in self.governance_actions 
                                    if (datetime.utcnow() - a.timestamp).days < 1]),
                "constitution_summary": constitution_summary,
                "connected_systems": {
                    "trust_ledger": self.trust_ledger is not None,
                    "network_health": self.network_health is not None,
                    "immune_response": self.immune_response is not None,
                    "node_isolation": self.node_isolation is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting governance status: {e}")
            return {"error": str(e)}
    
    def get_governance_actions(self, node_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent governance actions"""
        try:
            actions = self.governance_actions
            
            if node_id:
                actions = [a for a in actions if a.node_id == node_id]
            
            # Sort by timestamp (most recent first) and limit
            actions = sorted(actions, key=lambda a: a.timestamp, reverse=True)[:limit]
            
            return [
                {
                    "action_id": a.action_id,
                    "node_id": a.node_id,
                    "rule_id": a.rule_id,
                    "action_type": a.action_type,
                    "description": a.description,
                    "timestamp": a.timestamp.isoformat(),
                    "success": a.success,
                    "error_message": a.error_message
                }
                for a in actions
            ]
            
        except Exception as e:
            logger.error(f"Error getting governance actions: {e}")
            return []