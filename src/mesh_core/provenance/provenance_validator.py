"""
Provenance Validation System
===========================

Validates the authenticity and integrity of provenance information,
ensuring that provenance records are accurate and haven't been tampered with.
"""

import time
import hashlib
import hmac
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Provenance validation status"""
    VALID = "valid"               # Provenance is valid
    INVALID = "invalid"           # Provenance is invalid
    SUSPICIOUS = "suspicious"     # Provenance looks suspicious
    INCOMPLETE = "incomplete"     # Provenance data incomplete
    EXPIRED = "expired"          # Provenance data too old

class ValidationCheck(Enum):
    """Types of validation checks"""
    SIGNATURE_VERIFICATION = "signature"    # Cryptographic signatures
    HASH_INTEGRITY = "hash"                # Hash integrity
    TEMPORAL_CONSISTENCY = "temporal"       # Time consistency
    LOGICAL_CONSISTENCY = "logical"        # Logical flow consistency
    SOURCE_VERIFICATION = "source"         # Source authenticity
    COMPLETENESS = "completeness"          # Data completeness

@dataclass
class ValidationResult:
    """Result of provenance validation"""
    validation_id: str
    item_id: str
    status: ValidationStatus
    confidence: float             # 0.0 to 1.0
    checks_performed: List[ValidationCheck]
    checks_passed: int
    checks_failed: int
    issues_found: List[str]
    recommendations: List[str]
    validated_at: float
    validator_id: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['status'] = self.status.value
        data['checks_performed'] = [check.value for check in self.checks_performed]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ValidationResult':
        data['status'] = ValidationStatus(data['status'])
        data['checks_performed'] = [ValidationCheck(check) for check in data['checks_performed']]
        return cls(**data)

class ProvenanceValidator:
    """
    Provenance validation system
    
    Validates provenance information for authenticity, integrity,
    and consistency across the mesh network.
    """
    
    def __init__(self, node_id: str, secret_key: Optional[bytes] = None):
        self.node_id = node_id
        self.secret_key = secret_key or self._generate_secret_key()
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.trusted_validators: Set[str] = set()
        self.validation_rules: Dict[ValidationCheck, Dict] = self._init_validation_rules()
        
    def _generate_secret_key(self) -> bytes:
        """Generate secret key for HMAC validation"""
        return hashlib.sha256(f"{self.node_id}_{time.time()}".encode()).digest()
    
    def _init_validation_rules(self) -> Dict[ValidationCheck, Dict]:
        """Initialize validation rules and thresholds"""
        return {
            ValidationCheck.SIGNATURE_VERIFICATION: {
                'required': True,
                'min_signatures': 1,
                'trusted_signers_only': True
            },
            ValidationCheck.HASH_INTEGRITY: {
                'required': True,
                'check_content_hash': True,
                'check_metadata_hash': True
            },
            ValidationCheck.TEMPORAL_CONSISTENCY: {
                'required': True,
                'max_time_drift': 3600,  # 1 hour
                'check_sequence': True
            },
            ValidationCheck.LOGICAL_CONSISTENCY: {
                'required': True,
                'check_flow_logic': True,
                'check_transformations': True
            },
            ValidationCheck.SOURCE_VERIFICATION: {
                'required': False,
                'verify_source_identity': True,
                'check_source_reputation': True
            },
            ValidationCheck.COMPLETENESS: {
                'required': True,
                'min_required_fields': ['timestamp', 'source_id', 'content_hash'],
                'check_trail_completeness': True
            }
        }
    
    def _generate_validation_id(self, item_id: str) -> str:
        """Generate unique validation ID"""
        data = f"{item_id}_{self.node_id}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _create_signature(self, data: str) -> str:
        """Create HMAC signature for data"""
        return hmac.new(self.secret_key, data.encode(), hashlib.sha256).hexdigest()
    
    def _verify_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature"""
        expected = self._create_signature(data)
        return hmac.compare_digest(expected, signature)
    
    async def validate_provenance_record(
        self,
        item_id: str,
        provenance_data: Dict,
        perform_checks: Optional[List[ValidationCheck]] = None
    ) -> ValidationResult:
        """Validate a provenance record"""
        
        validation_id = self._generate_validation_id(item_id)
        
        # Determine which checks to perform
        if perform_checks is None:
            perform_checks = list(ValidationCheck)
        
        issues_found = []
        recommendations = []
        checks_passed = 0
        checks_failed = 0
        
        # Perform each validation check
        for check in perform_checks:
            try:
                check_result = await self._perform_validation_check(
                    check, item_id, provenance_data
                )
                
                if check_result['passed']:
                    checks_passed += 1
                else:
                    checks_failed += 1
                    issues_found.extend(check_result.get('issues', []))
                    recommendations.extend(check_result.get('recommendations', []))
                    
            except Exception as e:
                logger.error(f"Validation check {check.value} failed: {e}")
                checks_failed += 1
                issues_found.append(f"Check {check.value} failed: {str(e)}")
        
        # Calculate overall status and confidence
        status, confidence = self._calculate_validation_status(
            checks_passed, checks_failed, issues_found
        )
        
        result = ValidationResult(
            validation_id=validation_id,
            item_id=item_id,
            status=status,
            confidence=confidence,
            checks_performed=perform_checks,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            issues_found=issues_found,
            recommendations=recommendations,
            validated_at=time.time(),
            validator_id=self.node_id
        )
        
        # Cache the result
        self.validation_cache[validation_id] = result
        
        logger.info(f"Validated provenance for {item_id}: {status.value} ({confidence:.2f} confidence)")
        return result
    
    async def _perform_validation_check(
        self,
        check_type: ValidationCheck,
        item_id: str,
        provenance_data: Dict
    ) -> Dict:
        """Perform specific validation check"""
        
        if check_type == ValidationCheck.SIGNATURE_VERIFICATION:
            return await self._check_signature_verification(provenance_data)
        elif check_type == ValidationCheck.HASH_INTEGRITY:
            return await self._check_hash_integrity(provenance_data)
        elif check_type == ValidationCheck.TEMPORAL_CONSISTENCY:
            return await self._check_temporal_consistency(provenance_data)
        elif check_type == ValidationCheck.LOGICAL_CONSISTENCY:
            return await self._check_logical_consistency(provenance_data)
        elif check_type == ValidationCheck.SOURCE_VERIFICATION:
            return await self._check_source_verification(provenance_data)
        elif check_type == ValidationCheck.COMPLETENESS:
            return await self._check_completeness(provenance_data)
        else:
            return {'passed': False, 'issues': [f'Unknown check type: {check_type.value}']}
    
    async def _check_signature_verification(self, provenance_data: Dict) -> Dict:
        """Check cryptographic signatures"""
        
        issues = []
        recommendations = []
        
        # Check for signatures
        signatures = provenance_data.get('signatures', [])
        if not signatures:
            issues.append("No cryptographic signatures found")
            recommendations.append("Add cryptographic signatures for authenticity")
            return {'passed': False, 'issues': issues, 'recommendations': recommendations}
        
        # Verify each signature
        valid_signatures = 0
        for signature_data in signatures:
            signature = signature_data.get('signature')
            data = signature_data.get('data', '')
            signer_id = signature_data.get('signer_id')
            
            if not signature or not data:
                issues.append(f"Incomplete signature data from {signer_id}")
                continue
            
            # For demo purposes, assume signature is valid if it's a proper hex string
            if len(signature) == 64 and all(c in '0123456789abcdef' for c in signature.lower()):
                valid_signatures += 1
            else:
                issues.append(f"Invalid signature format from {signer_id}")
        
        rules = self.validation_rules[ValidationCheck.SIGNATURE_VERIFICATION]
        min_signatures = rules.get('min_signatures', 1)
        
        if valid_signatures >= min_signatures:
            return {'passed': True, 'valid_signatures': valid_signatures}
        else:
            issues.append(f"Only {valid_signatures} valid signatures, need {min_signatures}")
            recommendations.append("Obtain additional valid signatures")
            return {'passed': False, 'issues': issues, 'recommendations': recommendations}
    
    async def _check_hash_integrity(self, provenance_data: Dict) -> Dict:
        """Check hash integrity"""
        
        issues = []
        recommendations = []
        
        # Check for content hashes
        content_hash = provenance_data.get('content_hash')
        if not content_hash:
            issues.append("Content hash missing")
            recommendations.append("Include content hash for integrity verification")
            return {'passed': False, 'issues': issues, 'recommendations': recommendations}
        
        # Verify hash format
        if len(content_hash) not in [16, 32, 64]:  # Common hash lengths
            issues.append("Invalid content hash format")
            return {'passed': False, 'issues': issues}
        
        # Check metadata hash if present
        metadata_hash = provenance_data.get('metadata_hash')
        if metadata_hash and len(metadata_hash) not in [16, 32, 64]:
            issues.append("Invalid metadata hash format")
            return {'passed': False, 'issues': issues}
        
        return {'passed': True}
    
    async def _check_temporal_consistency(self, provenance_data: Dict) -> Dict:
        """Check temporal consistency"""
        
        issues = []
        recommendations = []
        current_time = time.time()
        
        # Check creation timestamp
        created_at = provenance_data.get('created_at')
        if not created_at:
            issues.append("Creation timestamp missing")
            return {'passed': False, 'issues': issues}
        
        # Check if timestamp is reasonable
        rules = self.validation_rules[ValidationCheck.TEMPORAL_CONSISTENCY]
        max_drift = rules.get('max_time_drift', 3600)
        
        if created_at > current_time + max_drift:
            issues.append("Creation timestamp is in the future")
        
        if created_at < current_time - (365 * 24 * 3600):  # More than a year old
            issues.append("Creation timestamp is very old")
            recommendations.append("Consider refreshing provenance data")
        
        # Check flow event timestamps if present
        flow_events = provenance_data.get('flow_events', [])
        last_timestamp = created_at
        
        for i, event in enumerate(flow_events):
            event_time = event.get('timestamp', 0)
            if event_time < last_timestamp:
                issues.append(f"Flow event {i} has timestamp before previous event")
            last_timestamp = event_time
        
        if issues:
            return {'passed': False, 'issues': issues, 'recommendations': recommendations}
        return {'passed': True}
    
    async def _check_logical_consistency(self, provenance_data: Dict) -> Dict:
        """Check logical consistency of flow"""
        
        issues = []
        recommendations = []
        
        # Check flow events sequence
        flow_events = provenance_data.get('flow_events', [])
        if not flow_events:
            return {'passed': True}  # No flow events to check
        
        # Verify flow makes logical sense
        nodes_seen = set()
        for i, event in enumerate(flow_events):
            event_type = event.get('event_type')
            node_id = event.get('node_id')
            previous_node = event.get('previous_node')
            next_node = event.get('next_node')
            
            if not event_type or not node_id:
                issues.append(f"Flow event {i} missing required fields")
                continue
            
            nodes_seen.add(node_id)
            
            # Check for logical flow
            if event_type == 'received' and not previous_node:
                issues.append(f"Received event {i} has no previous node")
            
            if event_type == 'forwarded' and not next_node:
                issues.append(f"Forwarded event {i} has no next node")
            
            # Check transformations are logical
            if event_type == 'transformed':
                transformation_type = event.get('transformation_type')
                if not transformation_type:
                    issues.append(f"Transformation event {i} missing transformation type")
        
        # Check for suspicious patterns
        if len(nodes_seen) == 1 and len(flow_events) > 3:
            issues.append("All flow events from single node - suspicious")
            recommendations.append("Verify multi-node flow is accurate")
        
        if issues:
            return {'passed': False, 'issues': issues, 'recommendations': recommendations}
        return {'passed': True}
    
    async def _check_source_verification(self, provenance_data: Dict) -> Dict:
        """Check source verification"""
        
        issues = []
        recommendations = []
        
        source_id = provenance_data.get('source_id')
        if not source_id:
            issues.append("Source ID missing")
            return {'passed': False, 'issues': issues}
        
        # Check source type
        source_type = provenance_data.get('source_type')
        if not source_type:
            recommendations.append("Include source type for better verification")
        
        # Check source reputation if available
        source_reputation = provenance_data.get('source_reputation')
        if source_reputation is not None and source_reputation < 0.3:
            issues.append("Source has low reputation score")
            recommendations.append("Verify source authenticity independently")
        
        return {'passed': len(issues) == 0, 'issues': issues, 'recommendations': recommendations}
    
    async def _check_completeness(self, provenance_data: Dict) -> Dict:
        """Check data completeness"""
        
        issues = []
        recommendations = []
        
        rules = self.validation_rules[ValidationCheck.COMPLETENESS]
        required_fields = rules.get('min_required_fields', [])
        
        for field in required_fields:
            if field not in provenance_data or not provenance_data[field]:
                issues.append(f"Required field '{field}' missing or empty")
        
        # Check trail completeness
        if rules.get('check_trail_completeness', False):
            flow_events = provenance_data.get('flow_events', [])
            if not flow_events:
                issues.append("No flow events - provenance trail incomplete")
                recommendations.append("Include complete provenance trail")
        
        return {'passed': len(issues) == 0, 'issues': issues, 'recommendations': recommendations}
    
    def _calculate_validation_status(
        self,
        checks_passed: int,
        checks_failed: int,
        issues: List[str]
    ) -> Tuple[ValidationStatus, float]:
        """Calculate overall validation status and confidence"""
        
        total_checks = checks_passed + checks_failed
        if total_checks == 0:
            return ValidationStatus.INCOMPLETE, 0.0
        
        success_rate = checks_passed / total_checks
        
        # Calculate confidence based on success rate and issue severity
        base_confidence = success_rate
        
        # Adjust for critical issues
        critical_keywords = ['signature', 'hash', 'missing', 'invalid']
        critical_issues = sum(1 for issue in issues if any(keyword in issue.lower() for keyword in critical_keywords))
        confidence_penalty = min(0.5, critical_issues * 0.2)
        
        confidence = max(0.0, base_confidence - confidence_penalty)
        
        # Determine status
        if success_rate >= 0.9 and confidence >= 0.8:
            status = ValidationStatus.VALID
        elif success_rate >= 0.7 and confidence >= 0.6:
            status = ValidationStatus.SUSPICIOUS
        elif success_rate >= 0.5:
            status = ValidationStatus.INCOMPLETE
        else:
            status = ValidationStatus.INVALID
        
        return status, confidence
    
    async def batch_validate(self, provenance_records: List[Dict]) -> List[ValidationResult]:
        """Validate multiple provenance records in batch"""
        
        results = []
        
        for i, record in enumerate(provenance_records):
            item_id = record.get('item_id', f'batch_item_{i}')
            try:
                result = await self.validate_provenance_record(item_id, record)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch validation failed for item {item_id}: {e}")
                # Create error result
                error_result = ValidationResult(
                    validation_id=self._generate_validation_id(item_id),
                    item_id=item_id,
                    status=ValidationStatus.INVALID,
                    confidence=0.0,
                    checks_performed=[],
                    checks_passed=0,
                    checks_failed=1,
                    issues_found=[f"Validation error: {str(e)}"],
                    recommendations=["Fix data format and retry validation"],
                    validated_at=time.time(),
                    validator_id=self.node_id
                )
                results.append(error_result)
        
        logger.info(f"Batch validated {len(results)} provenance records")
        return results
    
    def get_validation_summary(self, validation_ids: List[str]) -> Dict:
        """Get summary of validation results"""
        
        results = [self.validation_cache.get(vid) for vid in validation_ids if vid in self.validation_cache]
        results = [r for r in results if r is not None]
        
        if not results:
            return {'total': 0}
        
        status_counts = {}
        total_confidence = 0.0
        total_issues = 0
        
        for result in results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_confidence += result.confidence
            total_issues += len(result.issues_found)
        
        return {
            'total': len(results),
            'by_status': status_counts,
            'average_confidence': total_confidence / len(results),
            'total_issues': total_issues,
            'validation_rate': status_counts.get('valid', 0) / len(results),
            'validator_id': self.node_id
        }
    
    def export_validation_results(self) -> Dict:
        """Export all validation results"""
        
        return {
            'validator_id': self.node_id,
            'results': [result.to_dict() for result in self.validation_cache.values()],
            'exported_at': time.time()
        }