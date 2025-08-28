"""
Social Checksum System
=====================

Provides social verification mechanisms where communities can validate
information through collective verification, creating checksums based
on social consensus and community knowledge.
"""

import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import statistics

logger = logging.getLogger(__name__)

class VerificationMethod(Enum):
    """Methods of social verification"""
    CROWD_VERIFICATION = "crowd"         # Crowd-based verification
    EXPERT_PANEL = "expert"             # Expert panel verification
    COMMUNITY_VOTE = "vote"             # Community voting
    PEER_REVIEW = "peer_review"         # Peer review process
    SOCIAL_PROOF = "social_proof"       # Social proof mechanism
    DELEGATION = "delegation"           # Delegated verification

class ChecksumType(Enum):
    """Types of social checksums"""
    FACTUAL_ACCURACY = "factual"        # Factual accuracy checksum
    SOURCE_CREDIBILITY = "credibility"   # Source credibility checksum
    CONTENT_QUALITY = "quality"         # Content quality checksum
    BIAS_ASSESSMENT = "bias"            # Bias assessment checksum
    RELEVANCE = "relevance"             # Relevance checksum
    COMPLETENESS = "completeness"       # Completeness checksum

@dataclass
class VerificationContribution:
    """Individual contribution to social verification"""
    contributor_id: str
    item_id: str
    checksum_type: ChecksumType
    verification_method: VerificationMethod
    assessment_score: float  # 0.0 to 1.0
    confidence: float
    reasoning: str
    evidence: List[str]
    contributed_at: float
    weight: float  # Contributor's weight in calculation
    metadata: Dict
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['checksum_type'] = self.checksum_type.value
        data['verification_method'] = self.verification_method.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationContribution':
        data['checksum_type'] = ChecksumType(data['checksum_type'])
        data['verification_method'] = VerificationMethod(data['verification_method'])
        return cls(**data)

@dataclass
class ChecksumResult:
    """Result of social checksum calculation"""
    checksum_id: str
    item_id: str
    checksum_type: ChecksumType
    social_score: float
    confidence_level: float
    consensus_strength: float
    participant_count: int
    expert_weight: float
    community_agreement: float
    verification_methods_used: List[VerificationMethod]
    computed_at: float
    valid_until: float
    contributions: List[VerificationContribution]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['checksum_type'] = self.checksum_type.value
        data['verification_methods_used'] = [method.value for method in self.verification_methods_used]
        data['contributions'] = [contrib.to_dict() for contrib in self.contributions]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChecksumResult':
        data['checksum_type'] = ChecksumType(data['checksum_type'])
        data['verification_methods_used'] = [VerificationMethod(method) for method in data['verification_methods_used']]
        data['contributions'] = [VerificationContribution.from_dict(contrib) for contrib in data['contributions']]
        return cls(**data)

class SocialChecksum:
    """
    Social checksum verification system
    
    Enables communities to collectively verify information quality,
    accuracy, and other attributes through social consensus mechanisms.
    """
    
    def __init__(self, node_id: str, reputation_engine=None):
        self.node_id = node_id
        self.reputation_engine = reputation_engine
        self.active_verifications: Dict[str, Dict] = {}  # item_id -> verification state
        self.checksum_results: Dict[str, ChecksumResult] = {}
        self.contributor_weights: Dict[str, Dict] = {}  # contributor_id -> weights by checksum_type
        self.verification_parameters: Dict[str, float] = self._init_verification_parameters()
        
    def _init_verification_parameters(self) -> Dict[str, float]:
        """Initialize verification parameters"""
        return {
            'min_contributors': 3,           # Minimum contributors for checksum
            'expert_weight_multiplier': 2.0, # Weight multiplier for experts
            'confidence_threshold': 0.7,     # Minimum confidence for valid checksum
            'consensus_threshold': 0.6,      # Minimum consensus for agreement
            'checksum_validity_hours': 168,  # Checksum validity period (1 week)
            'reputation_weight_factor': 0.3  # Factor for reputation in weighting
        }
    
    def _generate_checksum_id(self, item_id: str, checksum_type: ChecksumType) -> str:
        """Generate unique checksum ID"""
        data = f"{item_id}_{checksum_type.value}_{time.time()}_{self.node_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def initiate_social_verification(
        self,
        item_id: str,
        content: str,
        checksum_types: List[ChecksumType],
        verification_methods: List[VerificationMethod],
        target_contributors: Optional[List[str]] = None,
        timeout_hours: int = 48
    ) -> Dict[ChecksumType, str]:
        """Initiate social verification for multiple checksum types"""
        
        checksum_ids = {}
        
        for checksum_type in checksum_types:
            checksum_id = self._generate_checksum_id(item_id, checksum_type)
            
            # Create verification state
            verification_state = {
                'checksum_id': checksum_id,
                'item_id': item_id,
                'content': content,
                'checksum_type': checksum_type,
                'verification_methods': verification_methods,
                'target_contributors': target_contributors or [],
                'contributions': [],
                'started_at': time.time(),
                'expires_at': time.time() + (timeout_hours * 3600),
                'status': 'active'
            }
            
            self.active_verifications[f"{item_id}_{checksum_type.value}"] = verification_state
            checksum_ids[checksum_type] = checksum_id
            
            # Request verifications
            await self._request_social_verifications(verification_state)
        
        logger.info(f"Initiated social verification for {item_id} with {len(checksum_types)} checksum types")
        return checksum_ids
    
    async def _request_social_verifications(self, verification_state: Dict):
        """Request social verifications from community"""
        
        # In real implementation, would broadcast verification requests
        # to community members, experts, or specific contributors
        
        checksum_type = verification_state['checksum_type']
        methods = verification_state['verification_methods']
        
        logger.info(f"Requesting social verification for checksum type {checksum_type.value} using methods: {[m.value for m in methods]}")
        
        # Would send network messages to request contributions
    
    async def contribute_verification(
        self,
        item_id: str,
        checksum_type: ChecksumType,
        contributor_id: str,
        assessment_score: float,
        confidence: float,
        verification_method: VerificationMethod,
        reasoning: str = "",
        evidence: Optional[List[str]] = None
    ) -> bool:
        """Contribute to social verification"""
        
        if evidence is None:
            evidence = []
            
        verification_key = f"{item_id}_{checksum_type.value}"
        verification_state = self.active_verifications.get(verification_key)
        
        if not verification_state:
            logger.error(f"No active verification for {item_id} / {checksum_type.value}")
            return False
        
        if time.time() > verification_state['expires_at']:
            logger.warning(f"Verification for {item_id} / {checksum_type.value} has expired")
            return False
        
        # Check if contributor already contributed
        existing_contributions = verification_state['contributions']
        for contrib in existing_contributions:
            if contrib.contributor_id == contributor_id:
                logger.warning(f"Contributor {contributor_id} already contributed to {verification_key}")
                return False
        
        # Calculate contributor weight
        weight = await self._calculate_contributor_weight(contributor_id, checksum_type)
        
        # Create contribution
        contribution = VerificationContribution(
            contributor_id=contributor_id,
            item_id=item_id,
            checksum_type=checksum_type,
            verification_method=verification_method,
            assessment_score=max(0.0, min(1.0, assessment_score)),
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning,
            evidence=evidence,
            contributed_at=time.time(),
            weight=weight,
            metadata={'verification_key': verification_key}
        )
        
        # Add contribution
        verification_state['contributions'].append(contribution)
        
        # Check if we can compute checksum
        await self._check_for_checksum_completion(verification_state)
        
        logger.info(f"Added contribution from {contributor_id} for {verification_key}")
        return True
    
    async def _calculate_contributor_weight(self, contributor_id: str, checksum_type: ChecksumType) -> float:
        """Calculate weight of contributor for checksum type"""
        
        base_weight = 1.0
        
        # Check if contributor is an expert (high reputation)
        if self.reputation_engine:
            try:
                reputation = await self.reputation_engine.get_reputation_score(contributor_id)
                
                # High reputation contributors get higher weight
                reputation_boost = reputation * self.verification_parameters['reputation_weight_factor']
                
                # Expert-level contributors get multiplier
                if reputation >= 0.8:
                    expert_multiplier = self.verification_parameters['expert_weight_multiplier']
                    base_weight *= expert_multiplier
                
                base_weight += reputation_boost
                
            except Exception as e:
                logger.warning(f"Failed to get reputation for contributor {contributor_id}: {e}")
        
        # Check stored weights for this checksum type
        contributor_weights = self.contributor_weights.get(contributor_id, {})
        type_weight = contributor_weights.get(checksum_type.value, 1.0)
        
        return base_weight * type_weight
    
    async def _check_for_checksum_completion(self, verification_state: Dict):
        """Check if social checksum can be computed"""
        
        contributions = verification_state['contributions']
        min_contributors = self.verification_parameters['min_contributors']
        
        # Check if we have enough contributors
        if len(contributions) < min_contributors:
            return
        
        # Check if we have enough confidence
        avg_confidence = statistics.mean(c.confidence for c in contributions)
        if avg_confidence < self.verification_parameters['confidence_threshold']:
            return
        
        # Compute social checksum
        checksum_result = await self._compute_social_checksum(verification_state)
        
        # Store result
        self.checksum_results[checksum_result.checksum_id] = checksum_result
        
        # Mark verification as completed
        verification_state['status'] = 'completed'
        verification_state['result'] = checksum_result
        
        # Clean up active verification
        verification_key = f"{verification_state['item_id']}_{verification_state['checksum_type'].value}"
        if verification_key in self.active_verifications:
            del self.active_verifications[verification_key]
        
        logger.info(f"Computed social checksum {checksum_result.checksum_id}")
    
    async def _compute_social_checksum(self, verification_state: Dict) -> ChecksumResult:
        """Compute social checksum from contributions"""
        
        contributions = verification_state['contributions']
        checksum_id = verification_state['checksum_id']
        
        if not contributions:
            # Return default result
            return ChecksumResult(
                checksum_id=checksum_id,
                item_id=verification_state['item_id'],
                checksum_type=verification_state['checksum_type'],
                social_score=0.5,
                confidence_level=0.0,
                consensus_strength=0.0,
                participant_count=0,
                expert_weight=0.0,
                community_agreement=0.0,
                verification_methods_used=[],
                computed_at=time.time(),
                valid_until=time.time() + (self.verification_parameters['checksum_validity_hours'] * 3600),
                contributions=[]
            )
        
        # Calculate weighted social score
        total_weighted_score = sum(c.assessment_score * c.weight * c.confidence for c in contributions)
        total_weight = sum(c.weight * c.confidence for c in contributions)
        social_score = total_weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Calculate confidence level
        confidence_scores = [c.confidence for c in contributions]
        confidence_level = statistics.mean(confidence_scores)
        
        # Calculate consensus strength (how much contributors agree)
        assessment_scores = [c.assessment_score for c in contributions]
        if len(assessment_scores) > 1:
            score_variance = statistics.variance(assessment_scores)
            consensus_strength = max(0.0, 1.0 - (score_variance * 2))  # Lower variance = higher consensus
        else:
            consensus_strength = 1.0
        
        # Calculate expert weight (proportion of expert contributions)
        expert_contributions = sum(1 for c in contributions if c.weight > 1.5)
        expert_weight = expert_contributions / len(contributions)
        
        # Calculate community agreement
        threshold = self.verification_parameters['consensus_threshold']
        agreeing_contributors = sum(1 for c in contributions if c.assessment_score >= threshold)
        community_agreement = agreeing_contributors / len(contributions)
        
        # Get verification methods used
        methods_used = list(set(c.verification_method for c in contributions))
        
        return ChecksumResult(
            checksum_id=checksum_id,
            item_id=verification_state['item_id'],
            checksum_type=verification_state['checksum_type'],
            social_score=social_score,
            confidence_level=confidence_level,
            consensus_strength=consensus_strength,
            participant_count=len(contributions),
            expert_weight=expert_weight,
            community_agreement=community_agreement,
            verification_methods_used=methods_used,
            computed_at=time.time(),
            valid_until=time.time() + (self.verification_parameters['checksum_validity_hours'] * 3600),
            contributions=contributions
        )
    
    async def get_checksum_result(self, checksum_id: str) -> Optional[ChecksumResult]:
        """Get social checksum result by ID"""
        
        result = self.checksum_results.get(checksum_id)
        
        # Check if checksum is still valid
        if result and time.time() > result.valid_until:
            logger.warning(f"Checksum {checksum_id} has expired")
            return None
        
        return result
    
    async def get_item_checksums(self, item_id: str) -> List[ChecksumResult]:
        """Get all social checksums for an item"""
        
        results = []
        current_time = time.time()
        
        for result in self.checksum_results.values():
            if result.item_id == item_id and current_time <= result.valid_until:
                results.append(result)
        
        return results
    
    async def get_checksum_summary(self, item_id: str) -> Dict:
        """Get summary of all checksums for an item"""
        
        checksums = await self.get_item_checksums(item_id)
        
        if not checksums:
            return {
                'item_id': item_id,
                'total_checksums': 0,
                'overall_score': 0.5
            }
        
        # Calculate overall metrics
        scores_by_type = {}
        total_participants = 0
        avg_confidence = 0.0
        
        for checksum in checksums:
            checksum_type = checksum.checksum_type.value
            scores_by_type[checksum_type] = {
                'score': checksum.social_score,
                'confidence': checksum.confidence_level,
                'consensus': checksum.consensus_strength,
                'participants': checksum.participant_count
            }
            total_participants += checksum.participant_count
            avg_confidence += checksum.confidence_level
        
        # Calculate overall score (weighted average)
        if checksums:
            overall_score = statistics.mean(c.social_score for c in checksums)
            avg_confidence /= len(checksums)
        else:
            overall_score = 0.5
        
        return {
            'item_id': item_id,
            'total_checksums': len(checksums),
            'overall_score': overall_score,
            'average_confidence': avg_confidence,
            'total_participants': total_participants,
            'scores_by_type': scores_by_type,
            'verification_methods': list(set(
                method.value for checksum in checksums 
                for method in checksum.verification_methods_used
            ))
        }
    
    async def update_contributor_weight(
        self,
        contributor_id: str,
        checksum_type: ChecksumType,
        new_weight: float
    ):
        """Update contributor weight for specific checksum type"""
        
        if contributor_id not in self.contributor_weights:
            self.contributor_weights[contributor_id] = {}
        
        self.contributor_weights[contributor_id][checksum_type.value] = max(0.1, min(5.0, new_weight))
        
        logger.info(f"Updated weight for {contributor_id} in {checksum_type.value}: {new_weight}")
    
    async def get_contributor_statistics(self, contributor_id: str) -> Dict:
        """Get statistics for a contributor"""
        
        # Find all contributions by this contributor
        all_contributions = []
        for result in self.checksum_results.values():
            for contrib in result.contributions:
                if contrib.contributor_id == contributor_id:
                    all_contributions.append(contrib)
        
        if not all_contributions:
            return {
                'contributor_id': contributor_id,
                'total_contributions': 0
            }
        
        # Calculate statistics
        total_contributions = len(all_contributions)
        avg_confidence = statistics.mean(c.confidence for c in all_contributions)
        avg_assessment = statistics.mean(c.assessment_score for c in all_contributions)
        
        # Count by checksum type
        type_counts = {}
        for contrib in all_contributions:
            checksum_type = contrib.checksum_type.value
            type_counts[checksum_type] = type_counts.get(checksum_type, 0) + 1
        
        # Count by verification method
        method_counts = {}
        for contrib in all_contributions:
            method = contrib.verification_method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'contributor_id': contributor_id,
            'total_contributions': total_contributions,
            'average_confidence': avg_confidence,
            'average_assessment_score': avg_assessment,
            'contributions_by_type': type_counts,
            'contributions_by_method': method_counts,
            'current_weights': self.contributor_weights.get(contributor_id, {}),
            'first_contribution': min(c.contributed_at for c in all_contributions),
            'latest_contribution': max(c.contributed_at for c in all_contributions)
        }
    
    def get_system_statistics(self) -> Dict:
        """Get overall system statistics"""
        
        if not self.checksum_results:
            return {
                'total_checksums': 0,
                'active_verifications': len(self.active_verifications),
                'node_id': self.node_id
            }
        
        results = list(self.checksum_results.values())
        
        # Count by checksum type
        type_counts = {}
        for result in results:
            checksum_type = result.checksum_type.value
            type_counts[checksum_type] = type_counts.get(checksum_type, 0) + 1
        
        # Calculate average scores
        avg_social_score = statistics.mean(r.social_score for r in results)
        avg_confidence = statistics.mean(r.confidence_level for r in results)
        avg_consensus = statistics.mean(r.consensus_strength for r in results)
        
        # Count participants
        total_participants = sum(r.participant_count for r in results)
        unique_contributors = len(set(
            contrib.contributor_id for result in results
            for contrib in result.contributions
        ))
        
        return {
            'total_checksums': len(results),
            'active_verifications': len(self.active_verifications),
            'by_checksum_type': type_counts,
            'average_social_score': avg_social_score,
            'average_confidence': avg_confidence,
            'average_consensus': avg_consensus,
            'total_participants': total_participants,
            'unique_contributors': unique_contributors,
            'registered_contributors': len(self.contributor_weights),
            'node_id': self.node_id
        }