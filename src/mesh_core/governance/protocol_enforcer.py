"""
Mesh Protocol Enforcer
======================

Automatically enforces constitutional rules and behavioral protocols
across The Mesh network, ensuring compliance and taking corrective actions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

from .constitution_engine import ConstitutionEngine, ConstitutionalRule, RuleViolation, RuleType, RulePriority

logger = logging.getLogger(__name__)


class EnforcementAction(Enum):
    """Types of enforcement actions"""
    WARN = "warn"                    # Send warning to node
    THROTTLE = "throttle"            # Reduce node's bandwidth/priority
    QUARANTINE = "quarantine"        # Isolate node temporarily
    BAN = "ban"                      # Permanently ban node
    EDUCATE = "educate"              # Send educational content
    COMPENSATE = "compensate"        # Require compensation for violations


@dataclass
class EnforcementEvent:
    """Record of an enforcement action taken"""
    event_id: str
    node_id: str
    rule_id: str
    action: EnforcementAction
    reason: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        content = f"{self.node_id}{self.rule_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ProtocolEnforcer:
    """
    Enforces constitutional rules and protocols across the network
    """
    
    def __init__(self, constitution_engine: ConstitutionEngine, node_id: str):
        self.constitution = constitution_engine
        self.node_id = node_id
        self.enforcement_history: Dict[str, List[EnforcementEvent]] = {}  # node_id -> events
        self.active_enforcements: Dict[str, EnforcementEvent] = {}  # node_id -> current enforcement
        self.enforcement_callbacks: Dict[EnforcementAction, List[Callable]] = {
            action: [] for action in EnforcementAction
        }
        
        # Enforcement thresholds and policies
        self.enforcement_policies = {
            RulePriority.CRITICAL: {
                "first_violation": EnforcementAction.QUARANTINE,
                "repeated_violation": EnforcementAction.BAN,
                "threshold": 1
            },
            RulePriority.HIGH: {
                "first_violation": EnforcementAction.WARN,
                "repeated_violation": EnforcementAction.THROTTLE,
                "threshold": 2
            },
            RulePriority.MEDIUM: {
                "first_violation": EnforcementAction.WARN,
                "repeated_violation": EnforcementAction.EDUCATE,
                "threshold": 3
            },
            RulePriority.LOW: {
                "first_violation": EnforcementAction.EDUCATE,
                "repeated_violation": EnforcementAction.WARN,
                "threshold": 5
            }
        }
        
        # Initialize enforcement callbacks
        self._setup_default_callbacks()
    
    def _setup_default_callbacks(self):
        """Setup default enforcement action callbacks"""
        self.register_enforcement_callback(EnforcementAction.WARN, self._send_warning)
        self.register_enforcement_callback(EnforcementAction.THROTTLE, self._throttle_node)
        self.register_enforcement_callback(EnforcementAction.QUARANTINE, self._quarantine_node)
        self.register_enforcement_callback(EnforcementAction.BAN, self._ban_node)
        self.register_enforcement_callback(EnforcementAction.EDUCATE, self._educate_node)
        self.register_enforcement_callback(EnforcementAction.COMPENSATE, self._require_compensation)
    
    def register_enforcement_callback(self, action: EnforcementAction, callback: Callable):
        """Register a callback for a specific enforcement action"""
        self.enforcement_callbacks[action].append(callback)
        logger.info(f"Registered callback for enforcement action: {action.value}")
    
    async def enforce_compliance(self, node_id: str, action: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check compliance and enforce rules if violations are found
        
        Returns:
            Tuple of (is_compliant, list_of_enforcement_actions_taken)
        """
        # Check compliance first
        is_compliant, violations = self.constitution.check_compliance(node_id, action, context)
        
        if is_compliant:
            return True, []
        
        # Take enforcement actions for violations
        enforcement_actions = []
        for violation_id in violations:
            action_taken = await self._enforce_violation(node_id, violation_id, context)
            if action_taken:
                enforcement_actions.append(action_taken)
        
        return False, enforcement_actions
    
    async def _enforce_violation(self, node_id: str, violation_id: str, context: Dict[str, Any]) -> Optional[str]:
        """Enforce a specific violation"""
        try:
            violation = self.constitution.violations.get(violation_id)
            if not violation:
                logger.warning(f"Violation {violation_id} not found")
                return None
            
            rule = self.constitution.rules.get(violation.rule_id)
            if not rule:
                logger.warning(f"Rule {violation.rule_id} not found")
                return None
            
            # Determine enforcement action based on violation history and rule priority
            enforcement_action = self._determine_enforcement_action(node_id, rule, violation)
            
            # Execute enforcement action
            if await self._execute_enforcement(node_id, enforcement_action, rule, violation):
                # Record enforcement event
                event = EnforcementEvent(
                    event_id="",
                    node_id=node_id,
                    rule_id=violation.rule_id,
                    action=enforcement_action,
                    reason=f"Violated rule: {rule.title}",
                    timestamp=datetime.utcnow(),
                    metadata={"violation_id": violation_id, "context": context}
                )
                
                # Store enforcement history
                if node_id not in self.enforcement_history:
                    self.enforcement_history[node_id] = []
                self.enforcement_history[node_id].append(event)
                
                # Update active enforcements
                self.active_enforcements[node_id] = event
                
                logger.info(f"Enforced {enforcement_action.value} on node {node_id} for rule {rule.title}")
                return enforcement_action.value
            
        except Exception as e:
            logger.error(f"Error enforcing violation {violation_id}: {e}")
        
        return None
    
    def _determine_enforcement_action(self, node_id: str, rule: ConstitutionalRule, violation: RuleViolation) -> EnforcementAction:
        """Determine appropriate enforcement action based on violation history and rule priority"""
        # Get violation count for this node and rule
        violation_count = self._get_violation_count(node_id, rule.rule_id)
        
        # Get enforcement policy for rule priority
        policy = self.enforcement_policies.get(rule.priority, self.enforcement_policies[RulePriority.MEDIUM])
        
        # Determine action based on violation count
        if violation_count >= policy["threshold"]:
            return policy["repeated_violation"]
        else:
            return policy["first_violation"]
    
    def _get_violation_count(self, node_id: str, rule_id: str) -> int:
        """Get count of violations for a specific node and rule"""
        count = 0
        for violation in self.constitution.violations.values():
            if violation.node_id == node_id and violation.rule_id == rule_id:
                count += 1
        return count
    
    async def _execute_enforcement(self, node_id: str, action: EnforcementAction, rule: ConstitutionalRule, violation: RuleViolation) -> bool:
        """Execute a specific enforcement action"""
        try:
            # Get callbacks for this action
            callbacks = self.enforcement_callbacks.get(action, [])
            
            # Execute all callbacks
            results = []
            for callback in callbacks:
                try:
                    result = await callback(node_id, rule, violation)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Enforcement callback failed: {e}")
                    results.append(False)
            
            # Return True if at least one callback succeeded
            return any(results)
            
        except Exception as e:
            logger.error(f"Failed to execute enforcement action {action.value}: {e}")
            return False
    
    # Default enforcement action implementations
    async def _send_warning(self, node_id: str, rule: ConstitutionalRule, violation: RuleViolation) -> bool:
        """Send warning to violating node"""
        warning_message = {
            "type": "constitutional_warning",
            "rule_title": rule.title,
            "rule_description": rule.description,
            "violation_description": violation.description,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": rule.priority.value
        }
        
        # TODO: Implement actual warning delivery mechanism
        logger.info(f"Warning sent to node {node_id}: {warning_message}")
        return True
    
    async def _throttle_node(self, node_id: str, rule: ConstitutionalRule, violation: RuleViolation) -> bool:
        """Throttle node's network access"""
        # TODO: Implement actual throttling mechanism
        logger.info(f"Node {node_id} throttled for rule violation: {rule.title}")
        return True
    
    async def _quarantine_node(self, node_id: str, rule: ConstitutionalRule, violation: RuleViolation) -> bool:
        """Quarantine node from network"""
        # TODO: Implement actual quarantine mechanism
        logger.info(f"Node {node_id} quarantined for rule violation: {rule.title}")
        return True
    
    async def _ban_node(self, node_id: str, rule: ConstitutionalRule, violation: RuleViolation) -> bool:
        """Permanently ban node from network"""
        # TODO: Implement actual ban mechanism
        logger.info(f"Node {node_id} banned for rule violation: {rule.title}")
        return True
    
    async def _educate_node(self, node_id: str, rule: ConstitutionalRule, violation: RuleViolation) -> bool:
        """Send educational content to node"""
        education_content = {
            "type": "constitutional_education",
            "rule_title": rule.title,
            "rule_description": rule.description,
            "best_practices": self._get_best_practices(rule),
            "examples": self._get_rule_examples(rule)
        }
        
        # TODO: Implement actual education delivery mechanism
        logger.info(f"Education content sent to node {node_id}: {education_content}")
        return True
    
    async def _require_compensation(self, node_id: str, rule: ConstitutionalRule, violation: RuleViolation) -> bool:
        """Require compensation for rule violation"""
        compensation_request = {
            "type": "compensation_request",
            "rule_title": rule.title,
            "violation_description": violation.description,
            "compensation_type": "reputation_restoration",
            "requirements": self._get_compensation_requirements(rule, violation)
        }
        
        # TODO: Implement actual compensation mechanism
        logger.info(f"Compensation requested from node {node_id}: {compensation_request}")
        return True
    
    def _get_best_practices(self, rule: ConstitutionalRule) -> List[str]:
        """Get best practices for a specific rule"""
        # TODO: Implement rule-specific best practices
        return [
            "Always verify information before sharing",
            "Respect user privacy and data protection",
            "Contribute positively to network health",
            "Report suspicious behavior promptly"
        ]
    
    def _get_rule_examples(self, rule: ConstitutionalRule) -> List[Dict[str, Any]]:
        """Get examples of compliant behavior for a rule"""
        # TODO: Implement rule-specific examples
        return [
            {
                "scenario": "Information sharing",
                "compliant_action": "Verify source and check confidence score",
                "non_compliant_action": "Share unverified information"
            }
        ]
    
    def _get_compensation_requirements(self, rule: ConstitutionalRule, violation: RuleViolation) -> Dict[str, Any]:
        """Get compensation requirements for a violation"""
        # TODO: Implement violation-specific compensation requirements
        return {
            "reputation_restoration": "Demonstrate compliant behavior for 24 hours",
            "community_service": "Help other nodes understand the rule",
            "verification_required": "Future actions require additional verification"
        }
    
    def get_enforcement_summary(self) -> Dict[str, Any]:
        """Get summary of enforcement activities"""
        total_enforcements = sum(len(events) for events in self.enforcement_history.values())
        active_enforcements = len(self.active_enforcements)
        
        enforcement_by_type = {}
        for events in self.enforcement_history.values():
            for event in events:
                action_type = event.action.value
                enforcement_by_type[action_type] = enforcement_by_type.get(action_type, 0) + 1
        
        return {
            "total_enforcements": total_enforcements,
            "active_enforcements": active_enforcements,
            "enforcement_by_type": enforcement_by_type,
            "nodes_under_enforcement": list(self.active_enforcements.keys()),
            "recent_enforcements": self._get_recent_enforcements(10)
        }
    
    def _get_recent_enforcements(self, count: int) -> List[Dict[str, Any]]:
        """Get recent enforcement events"""
        all_events = []
        for node_events in self.enforcement_history.values():
            all_events.extend(node_events)
        
        # Sort by timestamp and return most recent
        sorted_events = sorted(all_events, key=lambda e: e.timestamp, reverse=True)
        recent_events = sorted_events[:count]
        
        return [
            {
                "node_id": event.node_id,
                "rule_id": event.rule_id,
                "action": event.action.value,
                "reason": event.reason,
                "timestamp": event.timestamp.isoformat()
            }
            for event in recent_events
        ]
    
    def is_node_enforced(self, node_id: str) -> bool:
        """Check if a node is currently under enforcement"""
        return node_id in self.active_enforcements
    
    def get_node_enforcement_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get enforcement status for a specific node"""
        if node_id not in self.active_enforcements:
            return None
        
        event = self.active_enforcements[node_id]
        return {
            "action": event.action.value,
            "reason": event.reason,
            "timestamp": event.timestamp.isoformat(),
            "duration": str(event.duration) if event.duration else None,
            "metadata": event.metadata
        }
    
    def clear_enforcement(self, node_id: str, reason: str = "Manual clearance") -> bool:
        """Clear enforcement for a specific node"""
        if node_id not in self.active_enforcements:
            return False
        
        # Mark enforcement as resolved
        event = self.active_enforcements[node_id]
        event.duration = datetime.utcnow() - event.timestamp
        
        # Remove from active enforcements
        del self.active_enforcements[node_id]
        
        # Update violation as resolved
        for violation in self.constitution.violations.values():
            if violation.node_id == node_id and violation.rule_id == event.rule_id:
                violation.resolved = True
                violation.resolution_notes = reason
                break
        
        # Update enforcement stats
        self.constitution.enforcement_stats["resolved_violations"] += 1
        self.constitution.enforcement_stats["active_violations"] -= 1
        
        logger.info(f"Enforcement cleared for node {node_id}: {reason}")
        return True
    
    # Default enforcement callback implementations
    async def _send_warning(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Send warning to a node"""
        try:
            logger.warning(f"Sending warning to node {node_id} for rule {rule_id}")
            # In a real implementation, this would send a message to the node
            # For demo, we just log it
            return True
        except Exception as e:
            logger.error(f"Error sending warning: {e}")
            return False
    
    async def _throttle_node(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Throttle a node's activities"""
        try:
            logger.info(f"Throttling node {node_id} for rule {rule_id}")
            # In a real implementation, this would reduce the node's priority/bandwidth
            # Could integrate with network layer to limit message rates
            return True
        except Exception as e:
            logger.error(f"Error throttling node: {e}")
            return False
    
    async def _quarantine_node(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Quarantine a node"""
        try:
            logger.warning(f"Quarantining node {node_id} for rule {rule_id}")
            # In a real implementation, this would isolate the node
            # Could integrate with NodeIsolation system
            return True
        except Exception as e:
            logger.error(f"Error quarantining node: {e}")
            return False
    
    async def _ban_node(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Ban a node from the network"""
        try:
            logger.error(f"Banning node {node_id} for rule {rule_id}")
            # In a real implementation, this would permanently exclude the node
            # Could integrate with NodeIsolation system for permanent ban
            return True
        except Exception as e:
            logger.error(f"Error banning node: {e}")
            return False
    
    async def _educate_node(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Send educational content to a node"""
        try:
            logger.info(f"Sending educational content to node {node_id} for rule {rule_id}")
            # In a real implementation, this would send educational materials
            # about the violated rule and how to comply
            return True
        except Exception as e:
            logger.error(f"Error sending educational content: {e}")
            return False
    
    async def _require_compensation(self, node_id: str, rule_id: str, metadata: Dict[str, Any]) -> bool:
        """Require compensation from a node"""
        try:
            logger.info(f"Requiring compensation from node {node_id} for rule {rule_id}")
            # In a real implementation, this would create compensation requirements
            # such as community service, reputation restoration, etc.
            return True
        except Exception as e:
            logger.error(f"Error requiring compensation: {e}")
            return False

