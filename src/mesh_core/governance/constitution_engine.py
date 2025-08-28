"""
Mesh Constitution Engine
========================

Core governance system that defines and enforces constitutional rules,
constraints, and behavioral protocols for The Mesh network.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of constitutional rules"""
    BEHAVIORAL = "behavioral"      # How nodes should behave
    STRUCTURAL = "structural"       # Network structure constraints
    RIGHTS = "rights"              # User rights protection
    RESPONSIBILITY = "responsibility"  # Node responsibilities
    EMERGENCY = "emergency"        # Emergency protocols


class RulePriority(Enum):
    """Priority levels for rule enforcement"""
    CRITICAL = "critical"          # Must be enforced immediately
    HIGH = "high"                  # High priority enforcement
    MEDIUM = "medium"              # Standard enforcement
    LOW = "low"                    # Low priority, advisory


@dataclass
class ConstitutionalRule:
    """A single constitutional rule"""
    rule_id: str
    rule_type: RuleType
    priority: RulePriority
    title: str
    description: str
    constraints: Dict[str, Any]
    enforcement_mechanism: str
    created_at: datetime
    created_by: str
    version: int = 1
    is_active: bool = True
    last_modified: Optional[datetime] = None
    modified_by: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = self._generate_rule_id()
    
    def _generate_rule_id(self) -> str:
        """Generate unique rule ID from content"""
        content = f"{self.title}{self.description}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary for storage/transmission"""
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "constraints": self.constraints,
            "enforcement_mechanism": self.enforcement_mechanism,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
            "is_active": self.is_active,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "modified_by": self.modified_by,
            "dependencies": self.dependencies,
            "exceptions": self.exceptions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstitutionalRule':
        """Create rule from dictionary"""
        data = data.copy()
        data['rule_type'] = RuleType(data['rule_type'])
        data['priority'] = RulePriority(data['priority'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_modified'):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


@dataclass
class RuleViolation:
    """Record of a rule violation"""
    violation_id: str
    rule_id: str
    node_id: str
    violation_type: str
    description: str
    severity: str
    timestamp: datetime
    evidence: Dict[str, Any]
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def __post_init__(self):
        if not self.violation_id:
            self.violation_id = self._generate_violation_id()
    
    def _generate_violation_id(self) -> str:
        """Generate unique violation ID"""
        content = f"{self.rule_id}{self.node_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ConstitutionEngine:
    """
    Core engine for managing and enforcing constitutional rules
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.rules: Dict[str, ConstitutionalRule] = {}
        self.violations: Dict[str, RuleViolation] = {}
        self.rule_history: List[Tuple[datetime, str, str]] = []  # timestamp, action, rule_id
        self.enforcement_stats = {
            "total_violations": 0,
            "resolved_violations": 0,
            "active_violations": 0
        }
        
        # Initialize with default constitutional rules
        self._initialize_default_constitution()
    
    def _initialize_default_constitution(self):
        """Initialize with default constitutional rules"""
        default_rules = [
            ConstitutionalRule(
                rule_id="",
                rule_type=RuleType.BEHAVIORAL,
                priority=RulePriority.CRITICAL,
                title="Truth Verification Mandate",
                description="All nodes must verify information before propagation",
                constraints={"verification_required": True, "confidence_threshold": 0.8},
                enforcement_mechanism="automatic_blocking",
                created_at=datetime.utcnow(),
                created_by="system"
            ),
            ConstitutionalRule(
                rule_id="",
                rule_type=RuleType.RIGHTS,
                priority=RulePriority.HIGH,
                title="Privacy Protection",
                description="User privacy must be protected at all times",
                constraints={"privacy_level": "maximum", "data_retention": "minimal"},
                enforcement_mechanism="encryption_enforcement",
                created_at=datetime.utcnow(),
                created_by="system"
            ),
            ConstitutionalRule(
                rule_id="",
                rule_type=RuleType.RESPONSIBILITY,
                priority=RulePriority.HIGH,
                title="Network Health Responsibility",
                description="Nodes must contribute to network health monitoring",
                constraints={"health_reporting": True, "response_time": "immediate"},
                enforcement_mechanism="reputation_tracking",
                created_at=datetime.utcnow(),
                created_by="system"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: ConstitutionalRule) -> bool:
        """Add a new constitutional rule"""
        try:
            # Validate rule dependencies
            if not self._validate_dependencies(rule):
                logger.error(f"Rule {rule.title} has unmet dependencies")
                return False
            
            # Check for conflicts with existing rules
            conflicts = self._check_rule_conflicts(rule)
            if conflicts:
                logger.warning(f"Rule {rule.title} conflicts with: {conflicts}")
                # For now, allow conflicts but log them
            
            self.rules[rule.rule_id] = rule
            self.rule_history.append((datetime.utcnow(), "added", rule.rule_id))
            logger.info(f"Added constitutional rule: {rule.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add rule {rule.title}: {e}")
            return False
    
    def remove_rule(self, rule_id: str, reason: str = "Administrative removal") -> bool:
        """Remove a constitutional rule"""
        if rule_id not in self.rules:
            logger.warning(f"Attempted to remove non-existent rule: {rule_id}")
            return False
        
        rule = self.rules[rule_id]
        rule.is_active = False
        rule.last_modified = datetime.utcnow()
        rule.modified_by = self.node_id
        
        self.rule_history.append((datetime.utcnow(), "deactivated", rule_id))
        logger.info(f"Deactivated rule: {rule.title} - Reason: {reason}")
        return True
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any], modified_by: str) -> bool:
        """Update an existing constitutional rule"""
        if rule_id not in self.rules:
            logger.error(f"Cannot update non-existent rule: {rule_id}")
            return False
        
        rule = self.rules[rule_id]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(rule, field) and field not in ['rule_id', 'created_at', 'created_by']:
                setattr(rule, field, value)
        
        rule.version += 1
        rule.last_modified = datetime.utcnow()
        rule.modified_by = modified_by
        
        self.rule_history.append((datetime.utcnow(), "updated", rule_id))
        logger.info(f"Updated rule: {rule.title} to version {rule.version}")
        return True
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[ConstitutionalRule]:
        """Get all rules of a specific type"""
        return [rule for rule in self.rules.values() if rule.rule_type == rule_type and rule.is_active]
    
    def get_rules_by_priority(self, priority: RulePriority) -> List[ConstitutionalRule]:
        """Get all rules of a specific priority"""
        return [rule for rule in self.rules.values() if rule.priority == priority and rule.is_active]
    
    def get_active_rules(self) -> List[ConstitutionalRule]:
        """Get all currently active rules"""
        return [rule for rule in self.rules.values() if rule.is_active]
    
    def check_compliance(self, node_id: str, action: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if an action complies with constitutional rules
        
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        violations = []
        is_compliant = True
        
        for rule in self.get_active_rules():
            if not self._evaluate_rule_compliance(rule, action, context):
                violations.append(rule.rule_id)
                is_compliant = False
                
                # Record violation
                violation = RuleViolation(
                    violation_id="",
                    rule_id=rule.rule_id,
                    node_id=node_id,
                    violation_type="constitutional_violation",
                    description=f"Violated rule: {rule.title}",
                    severity=rule.priority.value,
                    timestamp=datetime.utcnow(),
                    evidence={"action": action, "context": context, "rule": rule.to_dict()}
                )
                
                self.violations[violation.violation_id] = violation
                self.enforcement_stats["total_violations"] += 1
                self.enforcement_stats["active_violations"] += 1
        
        return is_compliant, violations
    
    def _evaluate_rule_compliance(self, rule: ConstitutionalRule, action: str, context: Dict[str, Any]) -> bool:
        """Evaluate if an action complies with a specific rule"""
        try:
            # Simple rule evaluation - can be extended with more sophisticated logic
            if rule.rule_type == RuleType.BEHAVIORAL:
                return self._evaluate_behavioral_rule(rule, action, context)
            elif rule.rule_type == RuleType.STRUCTURAL:
                return self._evaluate_structural_rule(rule, action, context)
            elif rule.rule_type == RuleType.RIGHTS:
                return self._evaluate_rights_rule(rule, action, context)
            elif rule.rule_type == RuleType.RESPONSIBILITY:
                return self._evaluate_responsibility_rule(rule, action, context)
            else:
                return True  # Default to compliant for unknown rule types
                
        except Exception as e:
            logger.error(f"Error evaluating rule compliance: {e}")
            return False  # Fail safe - treat as non-compliant
    
    def _evaluate_behavioral_rule(self, rule: ConstitutionalRule, action: str, context: Dict[str, Any]) -> bool:
        """Evaluate behavioral rule compliance"""
        constraints = rule.constraints
        
        if "verification_required" in constraints:
            if constraints["verification_required"] and not context.get("verified", False):
                return False
        
        if "confidence_threshold" in constraints:
            confidence = context.get("confidence", 0.0)
            if confidence < constraints["confidence_threshold"]:
                return False
        
        return True
    
    def _evaluate_structural_rule(self, rule: ConstitutionalRule, action: str, context: Dict[str, Any]) -> bool:
        """Evaluate structural rule compliance"""
        # Implement structural constraint checking
        return True
    
    def _evaluate_rights_rule(self, rule: ConstitutionalRule, action: str, context: Dict[str, Any]) -> bool:
        """Evaluate rights protection rule compliance"""
        # Implement rights protection checking
        return True
    
    def _evaluate_responsibility_rule(self, rule: ConstitutionalRule, action: str, context: Dict[str, Any]) -> bool:
        """Evaluate responsibility rule compliance"""
        # Implement responsibility checking
        return True
    
    def _validate_dependencies(self, rule: ConstitutionalRule) -> bool:
        """Validate that rule dependencies are met"""
        for dep_id in rule.dependencies:
            if dep_id not in self.rules or not self.rules[dep_id].is_active:
                return False
        return True
    
    def _check_rule_conflicts(self, rule: ConstitutionalRule) -> List[str]:
        """Check for conflicts with existing rules"""
        conflicts = []
        # Implement conflict detection logic
        return conflicts
    
    def get_constitution_summary(self) -> Dict[str, Any]:
        """Get summary of current constitution state"""
        active_rules = self.get_active_rules()
        
        return {
            "total_rules": len(self.rules),
            "active_rules": len(active_rules),
            "rule_types": {rt.value: len(self.get_rules_by_type(rt)) for rt in RuleType},
            "priorities": {rp.value: len(self.get_rules_by_priority(rp)) for rp in RulePriority},
            "enforcement_stats": self.enforcement_stats.copy(),
            "recent_changes": self.rule_history[-10:] if self.rule_history else [],
            "constitution_hash": self._calculate_constitution_hash()
        }
    
    def _calculate_constitution_hash(self) -> str:
        """Calculate hash of current constitution state"""
        active_rules = sorted(self.get_active_rules(), key=lambda r: r.rule_id)
        content = "".join([rule.rule_id + str(rule.version) for rule in active_rules])
        return hashlib.sha256(content.encode()).hexdigest()
    
    def export_constitution(self) -> str:
        """Export constitution as JSON string"""
        constitution_data = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "exported_by": self.node_id,
                "version": "1.0"
            },
            "rules": [rule.to_dict() for rule in self.rules.values()],
            "summary": self.get_constitution_summary()
        }
        return json.dumps(constitution_data, indent=2)
    
    def import_constitution(self, constitution_json: str) -> bool:
        """Import constitution from JSON string"""
        try:
            data = json.loads(constitution_json)
            
            # Clear existing rules
            self.rules.clear()
            
            # Import rules
            for rule_data in data.get("rules", []):
                rule = ConstitutionalRule.from_dict(rule_data)
                self.rules[rule.rule_id] = rule
            
            logger.info(f"Imported constitution with {len(self.rules)} rules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import constitution: {e}")
            return False

