"""
Privacy Filter - Privacy Protection System

Implements comprehensive privacy protection for data synchronization
in The Mesh network, with configurable privacy levels, data anonymization,
and selective sharing policies.

Key Features:
- Multi-level privacy protection
- Configurable privacy policies  
- Data anonymization and pseudonymization
- Selective data sharing controls
- Privacy-preserving analytics
"""

import hashlib
import secrets
import time
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from enum import Enum
from collections import defaultdict
import logging

from .data_chunker import DataChunk, ChunkType, PrivacyLevel
from ..trust.trust_ledger import TrustLedger, TrustScore


class PrivacyPolicy(Enum):
    """Privacy policy types"""
    OPEN = "open"                    # No restrictions
    PSEUDONYMOUS = "pseudonymous"    # Identity hidden but trackable
    ANONYMOUS = "anonymous"          # Fully anonymous
    SELECTIVE = "selective"          # User-defined selective sharing
    ENCRYPTED = "encrypted"          # End-to-end encryption required
    PRIVATE = "private"              # No sharing allowed


class DataSensitivity(Enum):
    """Data sensitivity levels"""
    PUBLIC = 1       # Safe to share publicly
    INTERNAL = 2     # Safe within trusted network  
    CONFIDENTIAL = 3 # Restricted sharing
    SENSITIVE = 4    # High privacy requirements
    CRITICAL = 5     # Maximum protection required


class FilterAction(Enum):
    """Actions privacy filter can take"""
    ALLOW = "allow"                 # Allow data sharing
    DENY = "deny"                  # Block data sharing
    ANONYMIZE = "anonymize"        # Anonymize before sharing
    ENCRYPT = "encrypt"            # Encrypt before sharing  
    REDACT = "redact"              # Remove sensitive parts
    TRANSFORM = "transform"        # Transform data structure


@dataclass
class PrivacyRule:
    """Individual privacy rule"""
    rule_id: str
    name: str
    data_pattern: str              # Regex pattern for data matching
    sensitivity_level: DataSensitivity
    target_peers: Optional[Set[str]] = None  # Specific peers (None = all)
    allowed_privacy_levels: Set[PrivacyLevel] = field(default_factory=set)
    filter_action: FilterAction = FilterAction.ANONYMIZE
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    enabled: bool = True


@dataclass 
class PrivacyContext:
    """Context for privacy decisions"""
    requesting_peer: str
    trust_score: float
    data_type: str
    intended_use: Optional[str] = None
    privacy_level_requested: PrivacyLevel = PrivacyLevel.ANONYMOUS
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterDecision:
    """Result of privacy filtering"""
    original_data: Any
    filtered_data: Optional[Any]
    action_taken: FilterAction
    rules_applied: List[str]
    privacy_level_achieved: PrivacyLevel
    confidence: float              # 0.0 to 1.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAnonymizer:
    """Handles data anonymization and pseudonymization"""
    
    def __init__(self):
        self.anonymization_salt = secrets.token_bytes(32)
        self.pseudonym_cache: Dict[str, str] = {}
        
        # Common patterns for sensitive data
        self.sensitive_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b', 
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'user_id': r'\buser_id[:\s]*[a-zA-Z0-9_-]+\b',
            'session_id': r'\bsession[_\s]?id[:\s]*[a-zA-Z0-9_-]+\b'
        }
        
    def anonymize_data(self, data: Any, anonymization_level: PrivacyLevel) -> Tuple[Any, Dict[str, Any]]:
        """Anonymize data according to privacy level"""
        metadata = {}
        
        if anonymization_level == PrivacyLevel.PUBLIC:
            return data, metadata
            
        if isinstance(data, str):
            return self._anonymize_string(data, anonymization_level, metadata)
        elif isinstance(data, dict):
            return self._anonymize_dict(data, anonymization_level, metadata)
        elif isinstance(data, list):
            return self._anonymize_list(data, anonymization_level, metadata)
        else:
            return data, metadata
            
    def _anonymize_string(self, text: str, level: PrivacyLevel, metadata: Dict) -> Tuple[str, Dict]:
        """Anonymize sensitive patterns in text"""
        anonymized_text = text
        patterns_found = []
        
        for pattern_name, pattern in self.sensitive_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                patterns_found.append(pattern_name)
                
                for match in matches:
                    if level == PrivacyLevel.PSEUDONYMOUS:
                        # Consistent pseudonym for same value
                        replacement = self._get_pseudonym(match, pattern_name)
                    else:
                        # Full anonymization
                        replacement = self._get_anonymous_replacement(pattern_name)
                        
                    anonymized_text = anonymized_text.replace(match, replacement)
                    
        metadata['patterns_anonymized'] = patterns_found
        return anonymized_text, metadata
        
    def _anonymize_dict(self, data: Dict, level: PrivacyLevel, metadata: Dict) -> Tuple[Dict, Dict]:
        """Anonymize dictionary data"""
        anonymized_dict = {}
        
        for key, value in data.items():
            # Anonymize key if it contains sensitive patterns
            anonymized_key = key
            if self._is_sensitive_key(key):
                if level == PrivacyLevel.PSEUDONYMOUS:
                    anonymized_key = self._get_pseudonym(key, 'dict_key')
                else:
                    anonymized_key = f"field_{hashlib.sha256(key.encode()).hexdigest()[:8]}"
                    
            # Recursively anonymize value
            anonymized_value, value_metadata = self.anonymize_data(value, level)
            anonymized_dict[anonymized_key] = anonymized_value
            
            # Merge metadata
            if value_metadata:
                metadata[f"field_{anonymized_key}"] = value_metadata
                
        return anonymized_dict, metadata
        
    def _anonymize_list(self, data: List, level: PrivacyLevel, metadata: Dict) -> Tuple[List, Dict]:
        """Anonymize list data"""
        anonymized_list = []
        
        for i, item in enumerate(data):
            anonymized_item, item_metadata = self.anonymize_data(item, level)
            anonymized_list.append(anonymized_item)
            
            if item_metadata:
                metadata[f"item_{i}"] = item_metadata
                
        return anonymized_list, metadata
        
    def _get_pseudonym(self, original: str, pattern_type: str) -> str:
        """Get consistent pseudonym for a value"""
        cache_key = f"{pattern_type}:{original}"
        
        if cache_key in self.pseudonym_cache:
            return self.pseudonym_cache[cache_key]
            
        # Generate deterministic pseudonym
        hash_input = f"{original}:{self.anonymization_salt.hex()}:{pattern_type}".encode()
        pseudonym_hash = hashlib.sha256(hash_input).hexdigest()
        
        # Format based on pattern type
        if pattern_type == 'email':
            pseudonym = f"user{pseudonym_hash[:8]}@example.com"
        elif pattern_type == 'phone':
            pseudonym = f"555-{pseudonym_hash[:3]}-{pseudonym_hash[3:7]}"
        elif pattern_type == 'user_id':
            pseudonym = f"user_{pseudonym_hash[:12]}"
        else:
            pseudonym = pseudonym_hash[:16]
            
        self.pseudonym_cache[cache_key] = pseudonym
        return pseudonym
        
    def _get_anonymous_replacement(self, pattern_type: str) -> str:
        """Get anonymous replacement for sensitive data"""
        replacements = {
            'email': '[REDACTED_EMAIL]',
            'phone': '[REDACTED_PHONE]', 
            'ssn': '[REDACTED_SSN]',
            'ip_address': '[REDACTED_IP]',
            'credit_card': '[REDACTED_CC]',
            'user_id': '[REDACTED_USER]',
            'session_id': '[REDACTED_SESSION]'
        }
        return replacements.get(pattern_type, '[REDACTED]')
        
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if dictionary key is sensitive"""
        sensitive_keys = {
            'password', 'secret', 'token', 'key', 'auth', 'credential',
            'user_id', 'email', 'phone', 'address', 'ssn', 'credit_card'
        }
        return key.lower() in sensitive_keys or any(sk in key.lower() for sk in sensitive_keys)


class PrivacyFilterSystem:
    """Main privacy filtering system"""
    
    def __init__(self, local_peer_id: str, trust_ledger: TrustLedger):
        self.local_peer_id = local_peer_id
        self.trust_ledger = trust_ledger
        self.anonymizer = DataAnonymizer()
        
        # Privacy rules and policies
        self.privacy_rules: Dict[str, PrivacyRule] = {}
        self.default_policy = PrivacyPolicy.PSEUDONYMOUS
        self.peer_privacy_preferences: Dict[str, PrivacyPolicy] = {}
        
        # Filter statistics
        self.filter_stats = defaultdict(int)
        self.blocked_requests = []
        
        # Configuration
        self.min_trust_for_sensitive = 0.8
        self.max_redacted_percentage = 0.5  # Max 50% of data can be redacted
        
        self.logger = logging.getLogger(__name__)
        
        # Setup default rules
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default privacy rules"""
        
        # Rule for sensitive personal data
        self.add_privacy_rule(
            "sensitive_personal_data",
            data_pattern=r".*(email|phone|ssn|address).*",
            sensitivity_level=DataSensitivity.SENSITIVE,
            filter_action=FilterAction.ANONYMIZE,
            allowed_privacy_levels={PrivacyLevel.ANONYMOUS, PrivacyLevel.ENCRYPTED}
        )
        
        # Rule for authentication data  
        self.add_privacy_rule(
            "auth_data",
            data_pattern=r".*(password|secret|token|key|credential).*",
            sensitivity_level=DataSensitivity.CRITICAL,
            filter_action=FilterAction.DENY
        )
        
        # Rule for internal IDs
        self.add_privacy_rule(
            "internal_ids", 
            data_pattern=r".*(user_id|session_id|internal_id).*",
            sensitivity_level=DataSensitivity.CONFIDENTIAL,
            filter_action=FilterAction.ANONYMIZE,
            allowed_privacy_levels={PrivacyLevel.PSEUDONYMOUS, PrivacyLevel.ANONYMOUS}
        )
        
    def add_privacy_rule(self, 
                        name: str,
                        data_pattern: str,
                        sensitivity_level: DataSensitivity,
                        filter_action: FilterAction = FilterAction.ANONYMIZE,
                        target_peers: Optional[Set[str]] = None,
                        allowed_privacy_levels: Optional[Set[PrivacyLevel]] = None,
                        **kwargs) -> str:
        """Add a new privacy rule"""
        
        rule_id = secrets.token_hex(8)
        
        rule = PrivacyRule(
            rule_id=rule_id,
            name=name,
            data_pattern=data_pattern,
            sensitivity_level=sensitivity_level,
            target_peers=target_peers,
            allowed_privacy_levels=allowed_privacy_levels or {PrivacyLevel.ANONYMOUS},
            filter_action=filter_action,
            **kwargs
        )
        
        self.privacy_rules[rule_id] = rule
        self.logger.info(f"Added privacy rule: {name}")
        return rule_id
        
    def remove_privacy_rule(self, rule_id: str) -> bool:
        """Remove a privacy rule"""
        if rule_id in self.privacy_rules:
            del self.privacy_rules[rule_id]
            return True
        return False
        
    def set_peer_privacy_preference(self, peer_id: str, policy: PrivacyPolicy):
        """Set privacy policy preference for a specific peer"""
        self.peer_privacy_preferences[peer_id] = policy
        self.logger.info(f"Set privacy policy {policy.value} for peer {peer_id}")
        
    async def filter_data_for_sharing(self, 
                                    data: Any, 
                                    context: PrivacyContext) -> FilterDecision:
        """Filter data before sharing with another peer"""
        
        # Get trust score for requesting peer
        trust_score = context.trust_score
        if trust_score == 0:
            trust_record = self.trust_ledger.get_trust_score(context.requesting_peer)
            trust_score = trust_record.composite_score if trust_record else 0.0
            
        # Determine applicable privacy policy
        peer_policy = self.peer_privacy_preferences.get(
            context.requesting_peer, 
            self.default_policy
        )
        
        # Find matching privacy rules
        matching_rules = self._find_matching_rules(data, context)
        
        # Make filtering decision
        decision = await self._make_filter_decision(
            data, context, peer_policy, matching_rules, trust_score
        )
        
        # Update statistics
        self.filter_stats[decision.action_taken.value] += 1
        if decision.action_taken == FilterAction.DENY:
            self.blocked_requests.append({
                'peer': context.requesting_peer,
                'reason': 'privacy_policy',
                'timestamp': time.time()
            })
            
        return decision
        
    def _find_matching_rules(self, data: Any, context: PrivacyContext) -> List[PrivacyRule]:
        """Find privacy rules that match the given data"""
        matching_rules = []
        data_str = self._data_to_string(data)
        
        for rule in self.privacy_rules.values():
            if not rule.enabled:
                continue
                
            # Check if rule has expired
            if rule.expires_at and time.time() > rule.expires_at:
                continue
                
            # Check if rule applies to this peer
            if rule.target_peers and context.requesting_peer not in rule.target_peers:
                continue
                
            # Check if data pattern matches
            if re.search(rule.data_pattern, data_str, re.IGNORECASE | re.MULTILINE):
                matching_rules.append(rule)
                
        return matching_rules
        
    async def _make_filter_decision(self, 
                                  data: Any,
                                  context: PrivacyContext, 
                                  peer_policy: PrivacyPolicy,
                                  matching_rules: List[PrivacyRule],
                                  trust_score: float) -> FilterDecision:
        """Make the final filtering decision"""
        
        # Start with most restrictive action from matching rules
        most_restrictive_action = FilterAction.ALLOW
        rules_applied = []
        warnings = []
        
        # Check each matching rule
        for rule in matching_rules:
            # Check if requested privacy level is allowed
            if context.privacy_level_requested not in rule.allowed_privacy_levels:
                if rule.filter_action == FilterAction.DENY:
                    return FilterDecision(
                        original_data=data,
                        filtered_data=None,
                        action_taken=FilterAction.DENY,
                        rules_applied=[rule.rule_id],
                        privacy_level_achieved=PrivacyLevel.PRIVATE,
                        confidence=1.0,
                        warnings=[f"Data blocked by rule: {rule.name}"]
                    )
                    
            # Update most restrictive action
            if self._is_more_restrictive(rule.filter_action, most_restrictive_action):
                most_restrictive_action = rule.filter_action
                
            rules_applied.append(rule.rule_id)
            
        # Check trust-based restrictions
        if trust_score < self.min_trust_for_sensitive:
            # Check if data contains sensitive patterns
            if self._contains_sensitive_data(data):
                if most_restrictive_action == FilterAction.ALLOW:
                    most_restrictive_action = FilterAction.ANONYMIZE
                warnings.append("Low trust score - sensitive data anonymized")
                
        # Apply the filtering action
        if most_restrictive_action == FilterAction.ALLOW:
            filtered_data = data
            privacy_achieved = context.privacy_level_requested
            confidence = 1.0
            
        elif most_restrictive_action == FilterAction.DENY:
            filtered_data = None
            privacy_achieved = PrivacyLevel.PRIVATE
            confidence = 1.0
            
        elif most_restrictive_action == FilterAction.ANONYMIZE:
            filtered_data, anonymization_metadata = self.anonymizer.anonymize_data(
                data, context.privacy_level_requested
            )
            privacy_achieved = context.privacy_level_requested
            confidence = self._calculate_anonymization_confidence(anonymization_metadata)
            
        elif most_restrictive_action == FilterAction.REDACT:
            filtered_data = self._redact_sensitive_data(data)
            privacy_achieved = PrivacyLevel.ANONYMOUS
            confidence = 0.8  # Redaction is reasonably effective
            
        else:
            # Fallback to anonymization
            filtered_data, anonymization_metadata = self.anonymizer.anonymize_data(
                data, PrivacyLevel.ANONYMOUS
            )
            privacy_achieved = PrivacyLevel.ANONYMOUS
            confidence = 0.7
            
        return FilterDecision(
            original_data=data,
            filtered_data=filtered_data,
            action_taken=most_restrictive_action,
            rules_applied=rules_applied,
            privacy_level_achieved=privacy_achieved,
            confidence=confidence,
            warnings=warnings
        )
        
    def _data_to_string(self, data: Any) -> str:
        """Convert data to string for pattern matching"""
        if isinstance(data, str):
            return data
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str)
        else:
            return str(data)
            
    def _is_more_restrictive(self, action1: FilterAction, action2: FilterAction) -> bool:
        """Check if action1 is more restrictive than action2"""
        restrictiveness = {
            FilterAction.ALLOW: 0,
            FilterAction.TRANSFORM: 1,
            FilterAction.REDACT: 2,
            FilterAction.ANONYMIZE: 3,
            FilterAction.ENCRYPT: 4,
            FilterAction.DENY: 5
        }
        return restrictiveness[action1] > restrictiveness[action2]
        
    def _contains_sensitive_data(self, data: Any) -> bool:
        """Check if data contains sensitive patterns"""
        data_str = self._data_to_string(data)
        
        for pattern in self.anonymizer.sensitive_patterns.values():
            if re.search(pattern, data_str, re.IGNORECASE):
                return True
                
        return False
        
    def _redact_sensitive_data(self, data: Any) -> Any:
        """Redact sensitive data while preserving structure"""
        if isinstance(data, str):
            redacted = data
            for pattern_name, pattern in self.anonymizer.sensitive_patterns.items():
                replacement = self.anonymizer._get_anonymous_replacement(pattern_name)
                redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
            return redacted
            
        elif isinstance(data, dict):
            redacted_dict = {}
            for key, value in data.items():
                if self.anonymizer._is_sensitive_key(key):
                    redacted_dict[key] = '[REDACTED]'
                else:
                    redacted_dict[key] = self._redact_sensitive_data(value)
            return redacted_dict
            
        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]
            
        else:
            return data
            
    def _calculate_anonymization_confidence(self, metadata: Dict[str, Any]) -> float:
        """Calculate confidence in anonymization quality"""
        base_confidence = 0.9
        
        # Reduce confidence based on patterns found
        patterns_found = metadata.get('patterns_anonymized', [])
        confidence_penalty = len(patterns_found) * 0.05
        
        return max(base_confidence - confidence_penalty, 0.5)
        
    def validate_privacy_compliance(self, filtered_data: Any, original_data: Any) -> Dict[str, Any]:
        """Validate that filtered data meets privacy requirements"""
        validation_result = {
            'compliant': True,
            'issues': [],
            'privacy_score': 1.0
        }
        
        # Check for data leaks
        original_str = self._data_to_string(original_data)
        filtered_str = self._data_to_string(filtered_data)
        
        # Look for sensitive patterns in filtered data
        for pattern_name, pattern in self.anonymizer.sensitive_patterns.items():
            original_matches = re.findall(pattern, original_str, re.IGNORECASE)
            filtered_matches = re.findall(pattern, filtered_str, re.IGNORECASE)
            
            # Check for data leaks
            leaked_data = set(original_matches) & set(filtered_matches)
            if leaked_data:
                validation_result['compliant'] = False
                validation_result['issues'].append(f"Leaked {pattern_name}: {list(leaked_data)}")
                validation_result['privacy_score'] -= 0.2
                
        return validation_result
        
    def get_privacy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive privacy filter statistics"""
        total_requests = sum(self.filter_stats.values())
        recent_blocks = [b for b in self.blocked_requests 
                        if time.time() - b['timestamp'] < 86400]  # Last 24h
        
        return {
            'total_filter_requests': total_requests,
            'filter_actions': dict(self.filter_stats),
            'block_rate': self.filter_stats[FilterAction.DENY.value] / max(total_requests, 1),
            'anonymization_rate': self.filter_stats[FilterAction.ANONYMIZE.value] / max(total_requests, 1),
            'blocked_requests_24h': len(recent_blocks),
            'active_rules': len([r for r in self.privacy_rules.values() if r.enabled]),
            'peer_preferences': len(self.peer_privacy_preferences),
            'common_block_reasons': self._get_common_block_reasons(recent_blocks)
        }
        
    def _get_common_block_reasons(self, blocked_requests: List[Dict]) -> Dict[str, int]:
        """Get most common reasons for blocking requests"""
        reasons = defaultdict(int)
        for request in blocked_requests:
            reasons[request['reason']] += 1
        return dict(reasons)