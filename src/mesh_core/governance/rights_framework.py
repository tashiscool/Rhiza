"""
Mesh Rights Framework
====================

Protects user rights and ensures constitutional constraints are enforced
through cryptographic guarantees and automated monitoring.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import secrets

from .constitution_engine import ConstitutionEngine, ConstitutionalRule, RuleType, RulePriority

logger = logging.getLogger(__name__)


class RightType(Enum):
    """Types of user rights"""
    PRIVACY = "privacy"                    # Right to privacy
    TRUTH = "truth"                        # Right to accurate information
    PARTICIPATION = "participation"        # Right to participate in governance
    EXPRESSION = "expression"              # Right to express views
    ASSOCIATION = "association"            # Right to associate with others
    DUE_PROCESS = "due_process"            # Right to fair treatment
    APPEAL = "appeal"                      # Right to appeal decisions
    TRANSPARENCY = "transparency"          # Right to transparency


class RightStatus(Enum):
    """Status of a right"""
    ACTIVE = "active"                      # Right is active and protected
    SUSPENDED = "suspended"                # Right is temporarily suspended
    REVOKED = "revoked"                    # Right has been revoked
    RESTRICTED = "restricted"              # Right is restricted under certain conditions


@dataclass
class UserRight:
    """A specific user right"""
    right_id: str
    right_type: RightType
    user_id: str
    status: RightStatus
    granted_at: datetime
    granted_by: str
    expires_at: Optional[datetime] = None
    
    # Rights metadata
    scope: str = "full"                    # full, limited, conditional
    conditions: Dict[str, Any] = field(default_factory=dict)
    restrictions: List[str] = field(default_factory=list)
    
    # Audit trail
    last_verified: Optional[datetime] = None
    verification_count: int = 0
    violation_count: int = 0
    
    def __post_init__(self):
        if not self.right_id:
            self.right_id = self._generate_right_id()
    
    def _generate_right_id(self) -> str:
        """Generate unique right ID"""
        content = f"{self.right_type.value}{self.user_id}{self.granted_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_active(self) -> bool:
        """Check if right is currently active"""
        if self.status != RightStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert right to dictionary"""
        return {
            "right_id": self.right_id,
            "right_type": self.right_type.value,
            "user_id": self.user_id,
            "status": self.status.value,
            "granted_at": self.granted_at.isoformat(),
            "granted_by": self.granted_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scope": self.scope,
            "conditions": self.conditions,
            "restrictions": self.restrictions,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "verification_count": self.verification_count,
            "violation_count": self.violation_count
        }


@dataclass
class RightsViolation:
    """Record of a rights violation"""
    violation_id: str
    right_id: str
    user_id: str
    violation_type: str
    description: str
    severity: str
    timestamp: datetime
    evidence: Dict[str, Any]
    reported_by: str
    resolved: bool = False
    resolution_notes: Optional[str] = None
    compensation_required: bool = False
    
    def __post_init__(self):
        if not self.violation_id:
            self.violation_id = self._generate_violation_id()
    
    def _generate_violation_id(self) -> str:
        """Generate unique violation ID"""
        content = f"{self.right_id}{self.user_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class RightsClaim:
    """A claim to exercise a right"""
    claim_id: str
    right_id: str
    user_id: str
    claim_type: str
    description: str
    timestamp: datetime
    evidence: Dict[str, Any]
    status: str = "pending"  # pending, approved, denied, under_review
    
    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = self._generate_claim_id()
    
    def _generate_claim_id(self) -> str:
        """Generate unique claim ID"""
        content = f"{self.right_id}{self.user_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class RightsFramework:
    """
    Protects user rights and ensures constitutional constraints
    """
    
    def __init__(self, constitution_engine: ConstitutionEngine, node_id: str):
        self.constitution = constitution_engine
        self.node_id = node_id
        self.user_rights: Dict[str, List[UserRight]] = {}  # user_id -> rights
        self.rights_violations: Dict[str, RightsViolation] = {}
        self.rights_claims: Dict[str, RightsClaim] = {}
        self.rights_history: List[Tuple[datetime, str, str]] = []  # timestamp, action, right_id
        
        # Rights protection mechanisms
        self.protection_mechanisms = {
            RightType.PRIVACY: ["encryption", "anonymization", "access_control"],
            RightType.TRUTH: ["verification", "source_tracking", "confidence_scoring"],
            RightType.PARTICIPATION: ["voting_protection", "equal_access", "representation"],
            RightType.EXPRESSION: ["content_moderation", "hate_speech_detection", "quality_standards"],
            RightType.ASSOCIATION: ["group_formation", "communication_protection", "privacy_guards"],
            RightType.DUE_PROCESS: ["appeal_mechanisms", "transparency", "fair_treatment"],
            RightType.APPEAL: ["review_processes", "evidence_collection", "decision_reversal"],
            RightType.TRANSPARENCY: ["audit_logs", "public_records", "disclosure_mechanisms"]
        }
        
        # Initialize default rights for this node
        self._initialize_default_rights()
    
    def _initialize_default_rights(self):
        """Initialize default rights for this node"""
        default_rights = [
            UserRight(
                right_id="",
                right_type=RightType.PRIVACY,
                user_id=self.node_id,
                status=RightStatus.ACTIVE,
                granted_at=datetime.utcnow(),
                granted_by="system",
                scope="full"
            ),
            UserRight(
                right_id="",
                right_type=RightType.TRUTH,
                user_id=self.node_id,
                status=RightStatus.ACTIVE,
                granted_at=datetime.utcnow(),
                granted_by="system",
                scope="full"
            ),
            UserRight(
                right_id="",
                right_type=RightType.PARTICIPATION,
                user_id=self.node_id,
                status=RightStatus.ACTIVE,
                granted_at=datetime.utcnow(),
                granted_by="system",
                scope="full"
            )
        ]
        
        for right in default_rights:
            self.grant_right(right)
    
    def grant_right(self, right: UserRight) -> bool:
        """Grant a right to a user"""
        try:
            # Validate right
            if not self._validate_right(right):
                logger.error(f"Invalid right: {right.right_type.value}")
                return False
            
            # Check for existing rights
            if self._has_right(right.user_id, right.right_type):
                logger.warning(f"User {right.user_id} already has right {right.right_type.value}")
                return False
            
            # Add right to user's rights
            if right.user_id not in self.user_rights:
                self.user_rights[right.user_id] = []
            
            self.user_rights[right.user_id].append(right)
            self.rights_history.append((datetime.utcnow(), "granted", right.right_id))
            
            logger.info(f"Granted {right.right_type.value} right to user {right.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grant right: {e}")
            return False
    
    def revoke_right(self, user_id: str, right_type: RightType, reason: str = "Administrative action") -> bool:
        """Revoke a right from a user"""
        try:
            if user_id not in self.user_rights:
                logger.warning(f"User {user_id} has no rights")
                return False
            
            # Find and revoke the right
            for right in self.user_rights[user_id]:
                if right.right_type == right_type and right.status == RightStatus.ACTIVE:
                    right.status = RightStatus.REVOKED
                    self.rights_history.append((datetime.utcnow(), "revoked", right.right_id))
                    logger.info(f"Revoked {right_type.value} right from user {user_id}: {reason}")
                    return True
            
            logger.warning(f"User {user_id} does not have active {right_type.value} right")
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke right: {e}")
            return False
    
    def suspend_right(self, user_id: str, right_type: RightType, duration: timedelta, reason: str) -> bool:
        """Temporarily suspend a right"""
        try:
            if user_id not in self.user_rights:
                return False
            
            # Find and suspend the right
            for right in self.user_rights[user_id]:
                if right.right_type == right_type and right.status == RightStatus.ACTIVE:
                    right.status = RightStatus.SUSPENDED
                    right.expires_at = datetime.utcnow() + duration
                    self.rights_history.append((datetime.utcnow(), "suspended", right.right_id))
                    logger.info(f"Suspended {right_type.value} right for user {user_id} for {duration}: {reason}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to suspend right: {e}")
            return False
    
    def verify_right(self, user_id: str, right_type: RightType, context: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify if a user can exercise a right
        
        Returns:
            Tuple of (can_exercise, reason_if_denied)
        """
        try:
            if user_id not in self.user_rights:
                return False, "User has no rights"
            
            # Find the right
            for right in self.user_rights[user_id]:
                if right.right_type == right_type:
                    if not right.is_active():
                        return False, f"Right {right_type.value} is not active (status: {right.status.value})"
                    
                    # Check conditions
                    if not self._check_right_conditions(right, context):
                        return False, f"Right {right_type.value} conditions not met"
                    
                    # Update verification stats
                    right.last_verified = datetime.utcnow()
                    right.verification_count += 1
                    
                    return True, None
            
            return False, f"User does not have {right_type.value} right"
            
        except Exception as e:
            logger.error(f"Error verifying right: {e}")
            return False, f"Error verifying right: {e}"
    
    def _check_right_conditions(self, right: UserRight, context: Dict[str, Any]) -> bool:
        """Check if right conditions are met"""
        if not context:
            return True
        
        conditions = right.conditions
        
        # Check time-based conditions
        if "time_restrictions" in conditions:
            current_time = datetime.utcnow().time()
            allowed_times = conditions["time_restrictions"]
            if not self._is_time_allowed(current_time, allowed_times):
                return False
        
        # Check location-based conditions
        if "location_restrictions" in conditions:
            user_location = context.get("location")
            allowed_locations = conditions["location_restrictions"]
            if user_location and user_location not in allowed_locations:
                return False
        
        # Check behavior-based conditions
        if "behavior_requirements" in conditions:
            user_behavior = context.get("behavior_score", 0.0)
            required_behavior = conditions["behavior_requirements"].get("minimum_score", 0.0)
            if user_behavior < required_behavior:
                return False
        
        return True
    
    def _is_time_allowed(self, current_time: datetime.time, allowed_times: List[Dict[str, Any]]) -> bool:
        """Check if current time is within allowed time windows"""
        for time_window in allowed_times:
            start_time = datetime.strptime(time_window["start"], "%H:%M").time()
            end_time = datetime.strptime(time_window["end"], "%H:%M").time()
            
            if start_time <= current_time <= end_time:
                return True
        
        return False
    
    def _has_right(self, user_id: str, right_type: RightType) -> bool:
        """Check if user has a specific right"""
        if user_id not in self.user_rights:
            return False
        
        for right in self.user_rights[user_id]:
            if right.right_type == right_type and right.is_active():
                return True
        
        return False
    
    def _validate_right(self, right: UserRight) -> bool:
        """Validate a right before granting"""
        # Check if right type is valid
        if not isinstance(right.right_type, RightType):
            return False
        
        # Check if user ID is valid
        if not right.user_id or len(right.user_id) < 1:
            return False
        
        # Check if granted by is valid
        if not right.granted_by or len(right.granted_by) < 1:
            return False
        
        return True
    
    def report_violation(self, right_id: str, user_id: str, violation_type: str, 
                        description: str, evidence: Dict[str, Any], reported_by: str) -> str:
        """Report a rights violation"""
        try:
            violation = RightsViolation(
                violation_id="",
                right_id=right_id,
                user_id=user_id,
                violation_type=violation_type,
                description=description,
                severity="medium",  # Can be enhanced with severity calculation
                timestamp=datetime.utcnow(),
                evidence=evidence,
                reported_by=reported_by
            )
            
            self.rights_violations[violation.violation_id] = violation
            
            # Update violation count for the right
            self._update_violation_count(user_id, right_id)
            
            # Check if violation triggers automatic actions
            self._handle_violation_automatically(violation)
            
            logger.info(f"Reported rights violation: {violation_type} for user {user_id}")
            return violation.violation_id
            
        except Exception as e:
            logger.error(f"Failed to report violation: {e}")
            return ""
    
    def _update_violation_count(self, user_id: str, right_id: str):
        """Update violation count for a right"""
        if user_id in self.user_rights:
            for right in self.user_rights[user_id]:
                if right.right_id == right_id:
                    right.violation_count += 1
                    break
    
    def _handle_violation_automatically(self, violation: RightsViolation):
        """Handle violation automatically based on severity and history"""
        try:
            # Get the violated right
            user_rights = self.user_rights.get(violation.user_id, [])
            violated_right = None
            
            for right in user_rights:
                if right.right_id == violation.right_id:
                    violated_right = right
                    break
            
            if not violated_right:
                return
            
            # Automatic actions based on violation count
            if violated_right.violation_count >= 3:
                # Suspend right temporarily
                self.suspend_right(
                    violation.user_id,
                    violated_right.right_type,
                    timedelta(hours=24),
                    f"Automatic suspension due to {violated_right.violation_count} violations"
                )
            
            if violated_right.violation_count >= 5:
                # Revoke right
                self.revoke_right(
                    violation.user_id,
                    violated_right.right_type,
                    f"Automatic revocation due to {violated_right.violation_count} violations"
                )
            
        except Exception as e:
            logger.error(f"Error handling violation automatically: {e}")
    
    def submit_rights_claim(self, right_id: str, user_id: str, claim_type: str,
                           description: str, evidence: Dict[str, Any]) -> str:
        """Submit a claim to exercise a right"""
        try:
            claim = RightsClaim(
                claim_id="",
                right_id=right_id,
                user_id=user_id,
                claim_type=claim_type,
                description=description,
                timestamp=datetime.utcnow(),
                evidence=evidence
            )
            
            self.rights_claims[claim.claim_id] = claim
            self.rights_history.append((datetime.utcnow(), "claim_submitted", claim.claim_id))
            
            logger.info(f"Submitted rights claim: {claim_type} for user {user_id}")
            return claim.claim_id
            
        except Exception as e:
            logger.error(f"Failed to submit rights claim: {e}")
            return ""
    
    def review_rights_claim(self, claim_id: str, decision: str, reviewer_id: str, 
                           notes: Optional[str] = None) -> bool:
        """Review and decide on a rights claim"""
        try:
            if claim_id not in self.rights_claims:
                logger.error(f"Claim {claim_id} not found")
                return False
            
            claim = self.rights_claims[claim_id]
            claim.status = decision
            
            if decision == "approved":
                # Grant the right if it doesn't exist
                if not self._has_right(claim.user_id, self._get_right_type_from_id(claim.right_id)):
                    right_type = self._get_right_type_from_id(claim.right_id)
                    if right_type:
                        new_right = UserRight(
                            right_id="",
                            right_type=right_type,
                            user_id=claim.user_id,
                            status=RightStatus.ACTIVE,
                            granted_at=datetime.utcnow(),
                            granted_by=reviewer_id,
                            scope="conditional"
                        )
                        self.grant_right(new_right)
            
            self.rights_history.append((datetime.utcnow(), f"claim_{decision}", claim_id))
            logger.info(f"Claim {claim_id} {decision} by {reviewer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to review claim: {e}")
            return False
    
    def _get_right_type_from_id(self, right_id: str) -> Optional[RightType]:
        """Get right type from right ID"""
        for user_rights in self.user_rights.values():
            for right in user_rights:
                if right.right_id == right_id:
                    return right.right_type
        return None
    
    def get_user_rights_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of rights for a specific user"""
        if user_id not in self.user_rights:
            return {"user_id": user_id, "rights": [], "total_rights": 0, "active_rights": 0}
        
        user_rights = self.user_rights[user_id]
        active_rights = [r for r in user_rights if r.is_active()]
        
        rights_by_type = {}
        for right in user_rights:
            right_type = right.right_type.value
            if right_type not in rights_by_type:
                rights_by_type[right_type] = []
            rights_by_type[right_type].append(right.to_dict())
        
        return {
            "user_id": user_id,
            "total_rights": len(user_rights),
            "active_rights": len(active_rights),
            "rights_by_type": rights_by_type,
            "recent_rights_activity": self._get_recent_user_activity(user_id, 10)
        }
    
    def _get_recent_user_activity(self, user_id: str, count: int) -> List[Dict[str, Any]]:
        """Get recent rights activity for a user"""
        user_activity = []
        
        for timestamp, action, right_id in self.rights_history:
            # Check if this activity involves the user
            if self._activity_involves_user(action, right_id, user_id):
                user_activity.append({
                    "timestamp": timestamp.isoformat(),
                    "action": action,
                    "right_id": right_id
                })
        
        # Sort by timestamp and return most recent
        sorted_activity = sorted(user_activity, key=lambda x: x["timestamp"], reverse=True)
        return sorted_activity[:count]
    
    def _activity_involves_user(self, action: str, right_id: str, user_id: str) -> bool:
        """Check if an activity involves a specific user"""
        # Check if the right belongs to the user
        if user_id in self.user_rights:
            for right in self.user_rights[user_id]:
                if right.right_id == right_id:
                    return True
        
        # Check if it's a claim by the user
        if right_id in self.rights_claims:
            claim = self.rights_claims[right_id]
            if claim.user_id == user_id:
                return True
        
        return False
    
    def get_rights_framework_summary(self) -> Dict[str, Any]:
        """Get summary of the entire rights framework"""
        total_users = len(self.user_rights)
        total_rights = sum(len(rights) for rights in self.user_rights.values())
        total_violations = len(self.rights_violations)
        total_claims = len(self.rights_claims)
        
        rights_by_type = {}
        for user_rights in self.user_rights.values():
            for right in user_rights:
                right_type = right.right_type.value
                rights_by_type[right_type] = rights_by_type.get(right_type, 0) + 1
        
        return {
            "total_users": total_users,
            "total_rights": total_rights,
            "total_violations": total_violations,
            "total_claims": total_claims,
            "rights_by_type": rights_by_type,
            "recent_activity": self.rights_history[-10:] if self.rights_history else []
        }
    
    def export_rights_data(self, user_id: Optional[str] = None) -> str:
        """Export rights data as JSON string"""
        try:
            export_data = {
                "metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "exported_by": self.node_id,
                    "version": "1.0"
                },
                "user_rights": {},
                "rights_violations": {},
                "rights_claims": {}
            }
            
            # Export user rights
            users_to_export = [user_id] if user_id else self.user_rights.keys()
            for uid in users_to_export:
                if uid in self.user_rights:
                    export_data["user_rights"][uid] = [r.to_dict() for r in self.user_rights[uid]]
            
            # Export violations
            violations_to_export = self.rights_violations.values()
            if user_id:
                violations_to_export = [v for v in violations_to_export if v.user_id == user_id]
            
            for violation in violations_to_export:
                export_data["rights_violations"][violation.violation_id] = {
                    "right_id": violation.right_id,
                    "user_id": violation.user_id,
                    "violation_type": violation.violation_type,
                    "description": violation.description,
                    "severity": violation.severity,
                    "timestamp": violation.timestamp.isoformat(),
                    "evidence": violation.evidence,
                    "reported_by": violation.reported_by,
                    "resolved": violation.resolved
                }
            
            # Export claims
            claims_to_export = self.rights_claims.values()
            if user_id:
                claims_to_export = [c for c in claims_to_export if c.user_id == user_id]
            
            for claim in claims_to_export:
                export_data["rights_claims"][claim.claim_id] = {
                    "right_id": claim.right_id,
                    "user_id": claim.user_id,
                    "claim_type": claim.claim_type,
                    "description": claim.description,
                    "timestamp": claim.timestamp.isoformat(),
                    "evidence": claim.evidence,
                    "status": claim.status
                }
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to export rights data: {e}")
            return ""

