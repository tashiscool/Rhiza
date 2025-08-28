"""
Consensus Validation System
==========================

Validates information through distributed consensus mechanisms,
ensuring that truth claims are verified by multiple independent nodes.
"""

import asyncio
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import statistics

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Results of consensus validation"""
    CONSENSUS_REACHED = "consensus"      # Strong consensus achieved
    MAJORITY_AGREE = "majority"         # Majority agreement
    MIXED_RESULTS = "mixed"             # Mixed validation results
    INSUFFICIENT_DATA = "insufficient"  # Not enough validators
    CONSENSUS_FAILED = "failed"         # Failed to reach consensus

class ConsensusThreshold(Enum):
    """Consensus threshold levels"""
    SIMPLE_MAJORITY = "simple"          # >50% agreement
    STRONG_MAJORITY = "strong"          # >66% agreement  
    SUPERMAJORITY = "super"             # >80% agreement
    UNANIMOUS = "unanimous"             # 100% agreement

class ValidatorQuality(Enum):
    """Quality levels of validators"""
    UNKNOWN = "unknown"                 # Unknown validator quality
    LOW = "low"                        # Low-quality validator
    MEDIUM = "medium"                  # Medium-quality validator
    HIGH = "high"                      # High-quality validator
    EXPERT = "expert"                  # Expert-level validator

@dataclass
class ValidationSubmission:
    """Individual validation submission"""
    submission_id: str
    validator_id: str
    item_id: str
    validation_result: bool
    confidence_score: float
    reasoning: str
    evidence: List[str]
    submitted_at: float
    validator_quality: ValidatorQuality
    metadata: Dict
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['validator_quality'] = self.validator_quality.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ValidationSubmission':
        data['validator_quality'] = ValidatorQuality(data['validator_quality'])
        return cls(**data)

@dataclass
class ConsensusResult:
    """Result of consensus validation process"""
    consensus_id: str
    item_id: str
    validation_result: ValidationResult
    consensus_confidence: float
    agreement_percentage: float
    participating_validators: int
    required_threshold: ConsensusThreshold
    submissions: List[ValidationSubmission]
    consensus_reached_at: float
    final_verdict: bool
    quality_weighted_score: float
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['validation_result'] = self.validation_result.value
        data['required_threshold'] = self.required_threshold.value
        data['submissions'] = [sub.to_dict() for sub in self.submissions]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConsensusResult':
        data['validation_result'] = ValidationResult(data['validation_result'])
        data['required_threshold'] = ConsensusThreshold(data['required_threshold'])
        data['submissions'] = [ValidationSubmission.from_dict(sub) for sub in data['submissions']]
        return cls(**data)

class ConsensusValidator:
    """
    Distributed consensus validation system
    
    Coordinates validation across multiple nodes to reach consensus
    on the validity of information claims.
    """
    
    def __init__(self, node_id: str, reputation_engine=None):
        self.node_id = node_id
        self.reputation_engine = reputation_engine
        self.active_validations: Dict[str, Dict] = {}  # item_id -> validation state
        self.consensus_history: Dict[str, ConsensusResult] = {}
        self.validator_pool: Set[str] = set()
        self.quality_thresholds: Dict[ValidatorQuality, float] = self._init_quality_thresholds()
        self.consensus_parameters: Dict[str, float] = self._init_consensus_parameters()
        
    def _init_quality_thresholds(self) -> Dict[ValidatorQuality, float]:
        """Initialize validator quality thresholds"""
        return {
            ValidatorQuality.UNKNOWN: 0.0,
            ValidatorQuality.LOW: 0.3,
            ValidatorQuality.MEDIUM: 0.6,
            ValidatorQuality.HIGH: 0.8,
            ValidatorQuality.EXPERT: 0.9
        }
    
    def _init_consensus_parameters(self) -> Dict[str, float]:
        """Initialize consensus algorithm parameters"""
        return {
            'min_validators': 3,          # Minimum validators for consensus
            'max_wait_time': 300,         # Maximum wait time in seconds
            'quality_weight_factor': 2.0, # How much to weight validator quality
            'confidence_threshold': 0.7,  # Minimum confidence for consensus
            'reputation_weight': 0.3      # Weight of reputation in quality assessment
        }
    
    def _generate_consensus_id(self, item_id: str) -> str:
        """Generate unique consensus ID"""
        data = f"{item_id}_{self.node_id}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_submission_id(self, validator_id: str, item_id: str) -> str:
        """Generate unique submission ID"""
        data = f"{validator_id}_{item_id}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    
    async def assess_validator_quality(self, validator_id: str) -> ValidatorQuality:
        """Assess the quality of a validator"""
        
        if not self.reputation_engine:
            return ValidatorQuality.MEDIUM  # Default for unknown validators
        
        try:
            # Get reputation metrics
            reputation = await self.reputation_engine.get_reputation_score(validator_id)
            
            # Map reputation to quality level
            if reputation >= 0.9:
                return ValidatorQuality.EXPERT
            elif reputation >= 0.8:
                return ValidatorQuality.HIGH
            elif reputation >= 0.6:
                return ValidatorQuality.MEDIUM
            elif reputation >= 0.3:
                return ValidatorQuality.LOW
            else:
                return ValidatorQuality.UNKNOWN
                
        except Exception as e:
            logger.warning(f"Failed to assess validator quality for {validator_id}: {e}")
            return ValidatorQuality.UNKNOWN
    
    async def initiate_consensus_validation(
        self,
        item_id: str,
        content: str,
        threshold: ConsensusThreshold = ConsensusThreshold.STRONG_MAJORITY,
        target_validators: Optional[List[str]] = None,
        timeout_seconds: int = 300
    ) -> str:
        """Initiate consensus validation for an item"""
        
        consensus_id = self._generate_consensus_id(item_id)
        
        # Initialize validation state
        validation_state = {
            'consensus_id': consensus_id,
            'item_id': item_id,
            'content': content,
            'threshold': threshold,
            'target_validators': target_validators or list(self.validator_pool),
            'submissions': [],
            'started_at': time.time(),
            'expires_at': time.time() + timeout_seconds,
            'status': 'pending'
        }
        
        self.active_validations[item_id] = validation_state
        
        # Request validations from target validators
        await self._request_validations(validation_state)
        
        logger.info(f"Initiated consensus validation {consensus_id} for item {item_id}")
        return consensus_id
    
    async def _request_validations(self, validation_state: Dict):
        """Request validations from target validators"""
        
        # In a real implementation, this would send validation requests
        # to other nodes in the network. For now, we'll simulate this.
        
        target_validators = validation_state['target_validators']
        logger.info(f"Requesting validations from {len(target_validators)} validators")
        
        # In practice, would send network messages like:
        # await self.network.send_validation_request(validator_id, {
        #     'consensus_id': validation_state['consensus_id'],
        #     'item_id': validation_state['item_id'],
        #     'content': validation_state['content'],
        #     'expires_at': validation_state['expires_at']
        # })
    
    async def submit_validation(
        self,
        consensus_id: str,
        validator_id: str,
        validation_result: bool,
        confidence_score: float,
        reasoning: str = "",
        evidence: Optional[List[str]] = None
    ) -> bool:
        """Submit a validation result for consensus"""
        
        if evidence is None:
            evidence = []
            
        # Find the validation state
        validation_state = None
        for item_id, state in self.active_validations.items():
            if state['consensus_id'] == consensus_id:
                validation_state = state
                break
        
        if not validation_state:
            logger.error(f"Consensus validation {consensus_id} not found")
            return False
        
        # Check if validation is still active
        if time.time() > validation_state['expires_at']:
            logger.warning(f"Validation {consensus_id} has expired")
            return False
        
        # Check if validator already submitted
        existing_submissions = validation_state['submissions']
        for submission in existing_submissions:
            if submission.validator_id == validator_id:
                logger.warning(f"Validator {validator_id} already submitted for {consensus_id}")
                return False
        
        # Assess validator quality
        validator_quality = await self.assess_validator_quality(validator_id)
        
        # Create submission
        submission = ValidationSubmission(
            submission_id=self._generate_submission_id(validator_id, validation_state['item_id']),
            validator_id=validator_id,
            item_id=validation_state['item_id'],
            validation_result=validation_result,
            confidence_score=max(0.0, min(1.0, confidence_score)),
            reasoning=reasoning,
            evidence=evidence,
            submitted_at=time.time(),
            validator_quality=validator_quality,
            metadata={'consensus_id': consensus_id}
        )
        
        # Add submission to validation state
        validation_state['submissions'].append(submission)
        
        # Check if we can reach consensus
        await self._check_for_consensus(validation_state)
        
        logger.info(f"Received validation from {validator_id} for consensus {consensus_id}")
        return True
    
    async def _check_for_consensus(self, validation_state: Dict):
        """Check if consensus has been reached"""
        
        submissions = validation_state['submissions']
        threshold = validation_state['threshold']
        
        # Check minimum validators
        min_validators = self.consensus_parameters['min_validators']
        if len(submissions) < min_validators:
            return  # Need more submissions
        
        # Calculate consensus metrics
        consensus_result = await self._calculate_consensus(submissions, threshold)
        
        # Check if consensus is reached
        if consensus_result.validation_result in [ValidationResult.CONSENSUS_REACHED, ValidationResult.MAJORITY_AGREE]:
            # Consensus reached - finalize
            await self._finalize_consensus(validation_state, consensus_result)
        elif time.time() > validation_state['expires_at']:
            # Timeout - finalize with current results
            await self._finalize_consensus(validation_state, consensus_result)
    
    async def _calculate_consensus(self, submissions: List[ValidationSubmission], threshold: ConsensusThreshold) -> ConsensusResult:
        """Calculate consensus from submissions"""
        
        if not submissions:
            return ConsensusResult(
                consensus_id="",
                item_id="",
                validation_result=ValidationResult.INSUFFICIENT_DATA,
                consensus_confidence=0.0,
                agreement_percentage=0.0,
                participating_validators=0,
                required_threshold=threshold,
                submissions=[],
                consensus_reached_at=time.time(),
                final_verdict=False,
                quality_weighted_score=0.0
            )
        
        # Calculate basic agreement
        positive_votes = sum(1 for s in submissions if s.validation_result)
        total_votes = len(submissions)
        agreement_percentage = positive_votes / total_votes if total_votes > 0 else 0.0
        
        # Calculate quality-weighted score
        quality_weights = {
            ValidatorQuality.UNKNOWN: 1.0,
            ValidatorQuality.LOW: 1.2,
            ValidatorQuality.MEDIUM: 1.5,
            ValidatorQuality.HIGH: 2.0,
            ValidatorQuality.EXPERT: 3.0
        }
        
        weighted_positive = sum(
            quality_weights[s.validator_quality] * s.confidence_score
            for s in submissions if s.validation_result
        )
        weighted_total = sum(
            quality_weights[s.validator_quality] * s.confidence_score
            for s in submissions
        )
        
        quality_weighted_score = weighted_positive / weighted_total if weighted_total > 0 else 0.0
        
        # Calculate overall confidence
        confidence_scores = [s.confidence_score for s in submissions]
        avg_confidence = statistics.mean(confidence_scores)
        confidence_std = statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0
        consensus_confidence = avg_confidence * (1.0 - min(0.5, confidence_std))
        
        # Determine validation result
        threshold_values = {
            ConsensusThreshold.SIMPLE_MAJORITY: 0.5,
            ConsensusThreshold.STRONG_MAJORITY: 0.66,
            ConsensusThreshold.SUPERMAJORITY: 0.8,
            ConsensusThreshold.UNANIMOUS: 1.0
        }
        
        required_threshold = threshold_values[threshold]
        
        if quality_weighted_score >= required_threshold and consensus_confidence >= self.consensus_parameters['confidence_threshold']:
            if agreement_percentage == 1.0:
                validation_result = ValidationResult.CONSENSUS_REACHED
            else:
                validation_result = ValidationResult.MAJORITY_AGREE
        elif quality_weighted_score >= 0.5:
            validation_result = ValidationResult.MIXED_RESULTS
        else:
            validation_result = ValidationResult.CONSENSUS_FAILED
        
        # Final verdict
        final_verdict = validation_result in [ValidationResult.CONSENSUS_REACHED, ValidationResult.MAJORITY_AGREE]
        
        return ConsensusResult(
            consensus_id=submissions[0].metadata.get('consensus_id', ''),
            item_id=submissions[0].item_id,
            validation_result=validation_result,
            consensus_confidence=consensus_confidence,
            agreement_percentage=agreement_percentage,
            participating_validators=total_votes,
            required_threshold=threshold,
            submissions=submissions,
            consensus_reached_at=time.time(),
            final_verdict=final_verdict,
            quality_weighted_score=quality_weighted_score
        )
    
    async def _finalize_consensus(self, validation_state: Dict, consensus_result: ConsensusResult):
        """Finalize consensus validation"""
        
        item_id = validation_state['item_id']
        
        # Store consensus result
        self.consensus_history[consensus_result.consensus_id] = consensus_result
        
        # Update validation state
        validation_state['status'] = 'completed'
        validation_state['result'] = consensus_result
        
        # Remove from active validations
        if item_id in self.active_validations:
            del self.active_validations[item_id]
        
        # Update reputation of validators based on consensus
        await self._update_validator_reputations(consensus_result)
        
        logger.info(f"Consensus finalized for {item_id}: {consensus_result.validation_result.value} ({consensus_result.consensus_confidence:.2f} confidence)")
    
    async def _update_validator_reputations(self, consensus_result: ConsensusResult):
        """Update validator reputations based on consensus results"""
        
        if not self.reputation_engine:
            return
        
        # Determine the "correct" answer based on consensus
        consensus_answer = consensus_result.final_verdict
        
        for submission in consensus_result.submissions:
            validator_id = submission.validator_id
            
            # Calculate reputation impact
            if submission.validation_result == consensus_answer:
                # Validator agreed with consensus - positive impact
                impact = 0.1 * submission.confidence_score
            else:
                # Validator disagreed with consensus - negative impact
                impact = -0.1 * submission.confidence_score
            
            # Weight impact by consensus confidence
            impact *= consensus_result.consensus_confidence
            
            try:
                await self.reputation_engine.record_interaction(
                    validator_id,
                    'validation',  # Assuming this maps to an InteractionType
                    impact,
                    context={
                        'consensus_id': consensus_result.consensus_id,
                        'item_id': consensus_result.item_id,
                        'consensus_result': consensus_result.validation_result.value
                    },
                    verified=True
                )
            except Exception as e:
                logger.error(f"Failed to update reputation for validator {validator_id}: {e}")
    
    async def get_consensus_result(self, consensus_id: str) -> Optional[ConsensusResult]:
        """Get consensus result by ID"""
        return self.consensus_history.get(consensus_id)
    
    async def get_consensus_for_item(self, item_id: str) -> Optional[ConsensusResult]:
        """Get consensus result for item"""
        for result in self.consensus_history.values():
            if result.item_id == item_id:
                return result
        return None
    
    async def add_validator(self, validator_id: str) -> bool:
        """Add validator to the pool"""
        self.validator_pool.add(validator_id)
        logger.info(f"Added validator {validator_id} to pool")
        return True
    
    async def remove_validator(self, validator_id: str) -> bool:
        """Remove validator from the pool"""
        if validator_id in self.validator_pool:
            self.validator_pool.remove(validator_id)
            logger.info(f"Removed validator {validator_id} from pool")
            return True
        return False
    
    def get_validation_statistics(self) -> Dict:
        """Get validation system statistics"""
        
        if not self.consensus_history:
            return {
                'total_validations': 0,
                'validator_pool_size': len(self.validator_pool)
            }
        
        results = list(self.consensus_history.values())
        
        # Count by result type
        result_counts = {}
        total_validators = 0
        total_confidence = 0.0
        
        for result in results:
            result_type = result.validation_result.value
            result_counts[result_type] = result_counts.get(result_type, 0) + 1
            total_validators += result.participating_validators
            total_confidence += result.consensus_confidence
        
        return {
            'total_validations': len(results),
            'by_result_type': result_counts,
            'average_validators_per_validation': total_validators / len(results),
            'average_consensus_confidence': total_confidence / len(results),
            'validator_pool_size': len(self.validator_pool),
            'active_validations': len(self.active_validations),
            'consensus_success_rate': result_counts.get('consensus', 0) / len(results),
            'node_id': self.node_id
        }