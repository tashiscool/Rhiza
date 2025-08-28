"""
Mesh Local Customizer
=====================

Allows communities to customize governance rules and protocols
while maintaining compatibility with the broader Mesh network.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from .constitution_engine import ConstitutionEngine, ConstitutionalRule, RuleType, RulePriority

logger = logging.getLogger(__name__)


class CustomizationScope(Enum):
    """Scope of local customizations"""
    NODE = "node"                    # Single node customization
    COMMUNITY = "community"          # Community-wide customization
    REGIONAL = "regional"            # Regional customization
    FUNCTIONAL = "functional"        # Function-specific customization


class CustomizationType(Enum):
    """Types of local customizations"""
    RULE_ADDITION = "rule_addition"      # Add local rules
    RULE_MODIFICATION = "rule_modification"  # Modify existing rules
    RULE_OVERRIDE = "rule_override"      # Override global rules
    PROTOCOL_CUSTOMIZATION = "protocol_customization"  # Customize protocols
    ENFORCEMENT_CUSTOMIZATION = "enforcement_customization"  # Customize enforcement


@dataclass
class LocalCustomization:
    """A local customization to the mesh governance"""
    customization_id: str
    scope: CustomizationScope
    customization_type: CustomizationType
    title: str
    description: str
    original_rule_id: Optional[str]  # If modifying/overriding existing rule
    custom_content: Dict[str, Any]
    created_by: str
    created_at: datetime
    community_id: str
    priority: int = 1  # Higher priority customizations take precedence
    
    # Compatibility and validation
    mesh_compatibility: bool = True
    validation_status: str = "pending"
    validation_notes: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    last_modified: Optional[datetime] = None
    modified_by: Optional[str] = None
    
    def __post_init__(self):
        if not self.customization_id:
            self.customization_id = self._generate_customization_id()
    
    def _generate_customization_id(self) -> str:
        """Generate unique customization ID"""
        content = f"{self.title}{self.community_id}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert customization to dictionary"""
        return {
            "customization_id": self.customization_id,
            "scope": self.scope.value,
            "customization_type": self.customization_type.value,
            "title": self.title,
            "description": self.description,
            "original_rule_id": self.original_rule_id,
            "custom_content": self.custom_content,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "community_id": self.community_id,
            "priority": self.priority,
            "mesh_compatibility": self.mesh_compatibility,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "conflicts": self.conflicts,
            "is_active": self.is_active,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "modified_by": self.modified_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocalCustomization':
        """Create customization from dictionary"""
        data = data.copy()
        data['scope'] = CustomizationScope(data['scope'])
        data['customization_type'] = CustomizationType(data['customization_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('last_modified'):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


@dataclass
class CommunityProfile:
    """Profile for a community's governance preferences"""
    community_id: str
    name: str
    description: str
    governance_style: str  # e.g., "consensus", "democratic", "hierarchical"
    customizations: List[str] = field(default_factory=list)  # customization IDs
    compatibility_level: str = "full"  # full, partial, minimal
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            "community_id": self.community_id,
            "name": self.name,
            "description": self.description,
            "governance_style": self.governance_style,
            "customizations": self.customizations,
            "compatibility_level": self.compatibility_level,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


class LocalCustomizer:
    """
    Manages local customizations while maintaining mesh compatibility
    """
    
    def __init__(self, constitution_engine: ConstitutionEngine, node_id: str):
        self.constitution = constitution_engine
        self.node_id = node_id
        self.customizations: Dict[str, LocalCustomization] = {}
        self.community_profiles: Dict[str, CommunityProfile] = {}
        self.customization_history: List[Tuple[datetime, str, str]] = []  # timestamp, action, customization_id
        
        # Compatibility rules and constraints
        self.compatibility_rules = {
            "core_rights": ["privacy_protection", "truth_verification"],
            "network_integrity": ["corruption_detection", "trust_validation"],
            "communication_protocols": ["p2p_messaging", "data_sync"]
        }
        
        # Initialize default community profile for this node
        self._initialize_default_profile()
    
    def _initialize_default_profile(self):
        """Initialize default community profile for this node"""
        default_profile = CommunityProfile(
            community_id=f"node_{self.node_id}",
            name=f"Node {self.node_id}",
            description="Default governance profile for individual node",
            governance_style="consensus",
            compatibility_level="full"
        )
        self.community_profiles[default_profile.community_id] = default_profile
    
    def create_customization(self, scope: CustomizationScope, customization_type: CustomizationType,
                           title: str, description: str, custom_content: Dict[str, Any],
                           community_id: str, original_rule_id: Optional[str] = None,
                           priority: int = 1, tags: List[str] = None) -> str:
        """Create a new local customization"""
        try:
            # Validate customization content
            if not self._validate_customization_content(customization_type, custom_content):
                logger.error(f"Invalid customization content for {title}")
                return ""
            
            # Check for conflicts with existing customizations
            conflicts = self._check_customization_conflicts(customization_type, custom_content, community_id)
            if conflicts:
                logger.warning(f"Customization {title} has conflicts: {conflicts}")
            
            customization = LocalCustomization(
                customization_id="",
                scope=scope,
                customization_type=customization_type,
                title=title,
                description=description,
                original_rule_id=original_rule_id,
                custom_content=custom_content,
                created_by=self.node_id,
                created_at=datetime.utcnow(),
                community_id=community_id,
                priority=priority,
                tags=tags or []
            )
            
            # Validate mesh compatibility
            customization.mesh_compatibility = self._validate_mesh_compatibility(customization)
            if customization.mesh_compatibility:
                customization.validation_status = "validated"
            else:
                customization.validation_status = "incompatible"
                customization.validation_notes = "Customization conflicts with core mesh principles"
            
            self.customizations[customization.customization_id] = customization
            self.customization_history.append((datetime.utcnow(), "created", customization.customization_id))
            
            # Add to community profile
            if community_id in self.community_profiles:
                self.community_profiles[community_id].customizations.append(customization.customization_id)
                self.community_profiles[community_id].last_updated = datetime.utcnow()
            
            logger.info(f"Created local customization: {title} for community {community_id}")
            return customization.customization_id
            
        except Exception as e:
            logger.error(f"Failed to create customization {title}: {e}")
            return ""
    
    def _validate_customization_content(self, customization_type: CustomizationType, content: Dict[str, Any]) -> bool:
        """Validate customization content format"""
        try:
            if customization_type == CustomizationType.RULE_ADDITION:
                required_fields = ["rule_type", "priority", "title", "description", "constraints", "enforcement_mechanism"]
                return all(field in content for field in required_fields)
            
            elif customization_type == CustomizationType.RULE_MODIFICATION:
                required_fields = ["modifications", "reasoning"]
                return all(field in content for field in required_fields)
            
            elif customization_type == CustomizationType.RULE_OVERRIDE:
                required_fields = ["override_conditions", "local_implementation"]
                return all(field in content for field in required_fields)
            
            elif customization_type == CustomizationType.PROTOCOL_CUSTOMIZATION:
                required_fields = ["protocol_name", "custom_behavior", "compatibility_guarantees"]
                return all(field in content for field in required_fields)
            
            elif customization_type == CustomizationType.ENFORCEMENT_CUSTOMIZATION:
                required_fields = ["enforcement_rule", "local_policy", "escalation_path"]
                return all(field in content for field in required_fields)
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating customization content: {e}")
            return False
    
    def _check_customization_conflicts(self, customization_type: CustomizationType, 
                                     content: Dict[str, Any], community_id: str) -> List[str]:
        """Check for conflicts with existing customizations"""
        conflicts = []
        
        for existing_id, existing_customization in self.customizations.items():
            if existing_customization.community_id == community_id and existing_customization.is_active:
                if self._detect_conflict(customization_type, content, existing_customization):
                    conflicts.append(existing_id)
        
        return conflicts
    
    def _detect_conflict(self, new_type: CustomizationType, new_content: Dict[str, Any], 
                        existing: LocalCustomization) -> bool:
        """Detect conflict between new and existing customizations"""
        # Simple conflict detection - can be enhanced
        if new_type == CustomizationType.RULE_OVERRIDE and existing.customization_type == CustomizationType.RULE_OVERRIDE:
            if new_content.get("override_conditions") == existing.custom_content.get("override_conditions"):
                return True
        
        return False
    
    def _validate_mesh_compatibility(self, customization: LocalCustomization) -> bool:
        """Validate that customization maintains mesh compatibility"""
        try:
            if customization.customization_type == CustomizationType.RULE_OVERRIDE:
                # Check if overriding core mesh principles
                original_rule = self.constitution.rules.get(customization.original_rule_id)
                if original_rule and original_rule.rule_type in [RuleType.RIGHTS, RuleType.STRUCTURAL]:
                    return False
            
            # Check compatibility with core protocols
            if "communication_protocols" in customization.tags:
                if not self._validate_protocol_compatibility(customization):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating mesh compatibility: {e}")
            return False
    
    def _validate_protocol_compatibility(self, customization: LocalCustomization) -> bool:
        """Validate protocol customization compatibility"""
        # Ensure core P2P and sync protocols remain functional
        core_protocols = ["p2p_messaging", "data_sync", "trust_validation"]
        
        if "protocol_name" in customization.custom_content:
            protocol_name = customization.custom_content["protocol_name"]
            if protocol_name in core_protocols:
                # Check if customization maintains core functionality
                if "compatibility_guarantees" not in customization.custom_content:
                    return False
        
        return True
    
    def apply_customization(self, customization_id: str) -> bool:
        """Apply a local customization to the constitution"""
        if customization_id not in self.customizations:
            logger.error(f"Customization {customization_id} not found")
            return False
        
        customization = self.customizations[customization_id]
        
        if not customization.is_active:
            logger.warning(f"Customization {customization_id} is not active")
            return False
        
        if not customization.mesh_compatibility:
            logger.error(f"Cannot apply incompatible customization {customization_id}")
            return False
        
        try:
            if customization.customization_type == CustomizationType.RULE_ADDITION:
                # Add new rule
                rule_data = customization.custom_content.copy()
                # Ensure required fields are present
                rule_data['created_at'] = datetime.utcnow().isoformat()
                rule_data['created_by'] = self.node_id
                rule_data['rule_id'] = ""
                
                rule = ConstitutionalRule.from_dict(rule_data)
                rule.rule_id = f"local_{customization.community_id}_{rule.rule_id}"
                self.constitution.add_rule(rule)
                logger.info(f"Added local rule: {rule.title}")
            
            elif customization.customization_type == CustomizationType.RULE_MODIFICATION:
                # Modify existing rule
                if customization.original_rule_id:
                    modifications = customization.custom_content["modifications"]
                    self.constitution.update_rule(customization.original_rule_id, modifications, self.node_id)
                    logger.info(f"Modified rule: {customization.original_rule_id}")
            
            elif customization.customization_type == CustomizationType.RULE_OVERRIDE:
                # Create override rule
                override_rule = self._create_override_rule(customization)
                if override_rule:
                    self.constitution.add_rule(override_rule)
                    logger.info(f"Added override rule: {override_rule.title}")
            
            self.customization_history.append((datetime.utcnow(), "applied", customization_id))
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply customization {customization_id}: {e}")
            return False
    
    def _create_override_rule(self, customization: LocalCustomization) -> Optional[ConstitutionalRule]:
        """Create an override rule for local customization"""
        try:
            original_rule = self.constitution.rules.get(customization.original_rule_id)
            if not original_rule:
                return None
            
            override_conditions = customization.custom_content.get("override_conditions", {})
            local_implementation = customization.custom_content.get("local_implementation", {})
            
            # Create override rule with higher priority
            override_rule = ConstitutionalRule(
                rule_id=f"override_{customization.customization_id}",
                rule_type=original_rule.rule_type,
                priority=RulePriority.CRITICAL,  # Override rules get highest priority
                title=f"Local Override: {original_rule.title}",
                description=f"Local override of {original_rule.title} for community {customization.community_id}",
                constraints={**original_rule.constraints, **local_implementation},
                enforcement_mechanism=original_rule.enforcement_mechanism,
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                dependencies=[customization.original_rule_id]
            )
            
            return override_rule
            
        except Exception as e:
            logger.error(f"Failed to create override rule: {e}")
            return None
    
    def remove_customization(self, customization_id: str, reason: str = "Administrative removal") -> bool:
        """Remove a local customization"""
        if customization_id not in self.customizations:
            return False
        
        customization = self.customizations[customization_id]
        customization.is_active = False
        customization.last_modified = datetime.utcnow()
        customization.modified_by = self.node_id
        
        # Remove from community profile
        if customization.community_id in self.community_profiles:
            if customization_id in self.community_profiles[customization.community_id].customizations:
                self.community_profiles[customization.community_id].customizations.remove(customization_id)
                self.community_profiles[customization.community_id].last_updated = datetime.utcnow()
        
        self.customization_history.append((datetime.utcnow(), "removed", customization_id))
        logger.info(f"Removed customization: {customization.title} - Reason: {reason}")
        return True
    
    def get_customizations_by_community(self, community_id: str) -> List[LocalCustomization]:
        """Get all customizations for a specific community"""
        return [c for c in self.customizations.values() if c.community_id == community_id and c.is_active]
    
    def get_customizations_by_type(self, customization_type: CustomizationType) -> List[LocalCustomization]:
        """Get all customizations of a specific type"""
        return [c for c in self.customizations.values() if c.customization_type == customization_type and c.is_active]
    
    def get_compatible_customizations(self) -> List[LocalCustomization]:
        """Get all mesh-compatible customizations"""
        return [c for c in self.customizations.values() if c.mesh_compatibility and c.is_active]
    
    def create_community_profile(self, name: str, description: str, governance_style: str,
                               compatibility_level: str = "full") -> str:
        """Create a new community profile"""
        try:
            community_id = f"community_{len(self.community_profiles) + 1}"
            
            profile = CommunityProfile(
                community_id=community_id,
                name=name,
                description=description,
                governance_style=governance_style,
                compatibility_level=compatibility_level
            )
            
            self.community_profiles[community_id] = profile
            logger.info(f"Created community profile: {name}")
            return community_id
            
        except Exception as e:
            logger.error(f"Failed to create community profile: {e}")
            return ""
    
    def get_customization_summary(self) -> Dict[str, Any]:
        """Get summary of all customizations"""
        total_customizations = len(self.customizations)
        active_customizations = len([c for c in self.customizations.values() if c.is_active])
        compatible_customizations = len([c for c in self.customizations.values() if c.mesh_compatibility])
        
        customizations_by_type = {}
        for customization in self.customizations.values():
            if customization.is_active:
                ctype = customization.customization_type.value
                customizations_by_type[ctype] = customizations_by_type.get(ctype, 0) + 1
        
        customizations_by_scope = {}
        for customization in self.customizations.values():
            if customization.is_active:
                scope = customization.scope.value
                customizations_by_scope[scope] = customizations_by_scope.get(scope, 0) + 1
        
        return {
            "total_customizations": total_customizations,
            "active_customizations": active_customizations,
            "compatible_customizations": compatible_customizations,
            "customizations_by_type": customizations_by_type,
            "customizations_by_scope": customizations_by_scope,
            "communities": len(self.community_profiles),
            "recent_customizations": self._get_recent_customizations(10)
        }
    
    def _get_recent_customizations(self, count: int) -> List[Dict[str, Any]]:
        """Get recent customizations"""
        sorted_customizations = sorted(self.customizations.values(), key=lambda c: c.created_at, reverse=True)
        recent_customizations = sorted_customizations[:count]
        
        return [
            {
                "customization_id": c.customization_id,
                "title": c.title,
                "type": c.customization_type.value,
                "scope": c.scope.value,
                "community_id": c.community_id,
                "created_at": c.created_at.isoformat(),
                "mesh_compatibility": c.mesh_compatibility
            }
            for c in recent_customizations
        ]
    
    def export_customizations(self, community_id: Optional[str] = None) -> str:
        """Export customizations as JSON string"""
        try:
            export_data = {
                "metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "exported_by": self.node_id,
                    "version": "1.0"
                },
                "customizations": [],
                "community_profiles": []
            }
            
            # Export customizations
            customizations_to_export = self.customizations.values()
            if community_id:
                customizations_to_export = [c for c in customizations_to_export if c.community_id == community_id]
            
            for customization in customizations_to_export:
                export_data["customizations"].append(customization.to_dict())
            
            # Export community profiles
            profiles_to_export = self.community_profiles.values()
            if community_id:
                profiles_to_export = [p for p in profiles_to_export if p.community_id == community_id]
            
            for profile in profiles_to_export:
                export_data["community_profiles"].append(profile.to_dict())
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to export customizations: {e}")
            return ""
    
    def import_customizations(self, customizations_json: str) -> bool:
        """Import customizations from JSON string"""
        try:
            data = json.loads(customizations_json)
            
            # Import customizations
            for customization_data in data.get("customizations", []):
                customization = LocalCustomization.from_dict(customization_data)
                self.customizations[customization.customization_id] = customization
            
            # Import community profiles
            for profile_data in data.get("community_profiles", []):
                profile = CommunityProfile(**profile_data)
                self.community_profiles[profile.community_id] = profile
            
            logger.info(f"Imported {len(data.get('customizations', []))} customizations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import customizations: {e}")
            return False
