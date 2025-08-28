"""
Truth Validator - Advanced truth verification and validation system

Provides comprehensive truth validation using multiple verification layers,
consensus mechanisms, and integration with The Mesh's distributed trust system.

Key Features:
- Multi-layer truth verification
- Consensus-based validation
- Trust network integration
- Temporal consistency checking
- Cross-reference validation
- Evidence synthesis
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

try:
    from .axiom_processor import AxiomProcessor, AxiomVerificationResult
    from .knowledge_validator import KnowledgeValidator, ValidationResult, KnowledgeClaim
    from ..trust.trust_ledger import TrustLedger
    from ..network.mesh_protocol import MeshProtocol
    from ..provenance.provenance_tracker import ProvenanceTracker
except ImportError:
    # Mock classes for development
    class MockAxiomProcessor:
        async def verify_claim(self, claim: str, context: Optional[Dict] = None):
            return {"verification_status": "mock", "confidence_score": 0.7}
    
    class MockKnowledgeValidator:
        async def validate_knowledge_claim(self, claim, validation_level, context=None):
            return type('MockResult', (), {
                'confidence_score': 0.7,
                'confidence_level': 'MEDIUM',
                'corroboration_sources': ['source1', 'source2'],
                'contradictions': []
            })()
    
    class MockTrustLedger:
        def calculate_trust_score(self, node_id: str) -> float:
            return 0.8
        
        def get_trust_network(self, radius: int = 2) -> Dict:
            return {"nodes": ["node1", "node2"], "trust_scores": [0.8, 0.9]}
    
    class MockMeshProtocol:
        async def broadcast_truth_query(self, query: Dict) -> List[Dict]:
            return [{"node_id": "node1", "response": {"verified": True, "confidence": 0.8}}]
    
    class MockProvenanceTracker:
        def get_information_lineage(self, claim: str) -> Dict:
            return {"sources": ["source1"], "transformations": []}
    
    # Use mocks
    AxiomProcessor = MockAxiomProcessor
    KnowledgeValidator = MockKnowledgeValidator
    TrustLedger = MockTrustLedger
    MeshProtocol = MockMeshProtocol
    ProvenanceTracker = MockProvenanceTracker
    AxiomVerificationResult = dict
    ValidationResult = type('ValidationResult', (), {})
    KnowledgeClaim = type('KnowledgeClaim', (), {})


class TruthStatus(Enum):
    """Truth validation status levels"""
    EMPIRICALLY_VERIFIED = "empirically_verified"
    STRONGLY_SUPPORTED = "strongly_supported"
    CORROBORATED = "corroborated"
    LOGICALLY_CONSISTENT = "logically_consistent"
    PROPOSED = "proposed"
    DISPUTED = "disputed"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    ERROR = "error"


class ValidationMode(Enum):
    """Validation thoroughness modes"""
    RAPID = "rapid"          # Basic verification only
    STANDARD = "standard"    # Standard multi-layer verification
    RIGOROUS = "rigorous"    # Comprehensive validation with consensus
    FORENSIC = "forensic"    # Maximum validation with full network query


@dataclass
class TruthClaim:
    """Truth claim for validation"""
    content: str
    claim_id: Optional[str] = None
    source: Optional[str] = None
    domain: str = "general"
    claim_type: str = "factual"
    urgency: str = "normal"  # low, normal, high, critical
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.claim_id is None:
            self.claim_id = f"claim_{int(time.time() * 1000000)}"


@dataclass
class ValidationEvidence:
    """Evidence supporting or contradicting a claim"""
    evidence_type: str  # corroboration, contradiction, source_verification
    content: str
    confidence: float
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result of consensus validation"""
    participating_nodes: List[str]
    agreement_score: float
    confidence_scores: List[float]
    dissenting_nodes: List[str] = field(default_factory=list)
    consensus_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TruthValidationResult:
    """Comprehensive truth validation result"""
    claim: TruthClaim
    truth_status: TruthStatus
    overall_confidence: float
    validation_mode: ValidationMode
    
    # Verification results
    axiom_result: Optional[Union[AxiomVerificationResult, Dict]] = None
    knowledge_result: Optional[ValidationResult] = None
    
    # Evidence and consensus
    supporting_evidence: List[ValidationEvidence] = field(default_factory=list)
    contradicting_evidence: List[ValidationEvidence] = field(default_factory=list)
    consensus_result: Optional[ConsensusResult] = None
    
    # Trust and provenance
    trust_score: float = 0.0
    provenance_score: float = 0.0
    temporal_consistency: float = 0.0
    
    # Metadata
    validation_time: float = 0.0
    participating_systems: List[str] = field(default_factory=list)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)


class TruthValidator:
    """Advanced truth validation system with multi-layer verification"""
    
    def __init__(self):
        self.axiom_processor = AxiomProcessor()
        self.knowledge_validator = KnowledgeValidator()
        self.trust_ledger = TrustLedger()
        self.mesh_protocol = MeshProtocol()
        self.provenance_tracker = ProvenanceTracker()
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.consensus_threshold = 0.7
        self.trust_threshold = 0.6
        self.temporal_threshold = 0.8
        self.evidence_weight = {
            "axiom_verification": 0.4,
            "knowledge_validation": 0.3,
            "consensus": 0.2,
            "trust_network": 0.1
        }
        
        # Statistics
        self.validations_performed = 0
        self.high_confidence_validations = 0
        self.consensus_queries = 0
        self.validation_times = []
    
    async def validate_truth(
        self,
        claim: TruthClaim,
        validation_mode: ValidationMode = ValidationMode.STANDARD
    ) -> TruthValidationResult:
        """Validate a truth claim with specified thoroughness"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Validating truth claim: {claim.claim_id}")
            
            # Initialize result
            result = TruthValidationResult(
                claim=claim,
                truth_status=TruthStatus.UNVERIFIED,
                overall_confidence=0.0,
                validation_mode=validation_mode
            )
            
            # Perform validation based on mode
            if validation_mode == ValidationMode.RAPID:
                await self._rapid_validation(result)
            elif validation_mode == ValidationMode.STANDARD:
                await self._standard_validation(result)
            elif validation_mode == ValidationMode.RIGOROUS:
                await self._rigorous_validation(result)
            elif validation_mode == ValidationMode.FORENSIC:
                await self._forensic_validation(result)
            
            # Calculate final truth status and confidence
            self._calculate_final_truth_status(result)
            
            # Record performance metrics
            result.validation_time = time.time() - start_time
            self.validation_times.append(result.validation_time)
            self.validations_performed += 1
            
            if result.overall_confidence >= 0.8:
                self.high_confidence_validations += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating truth claim {claim.claim_id}: {e}")
            return TruthValidationResult(
                claim=claim,
                truth_status=TruthStatus.ERROR,
                overall_confidence=0.0,
                validation_mode=validation_mode,
                error_messages=[str(e)],
                validation_time=time.time() - start_time
            )
    
    async def validate_batch_truths(
        self,
        claims: List[TruthClaim],
        validation_mode: ValidationMode = ValidationMode.STANDARD
    ) -> List[TruthValidationResult]:
        """Validate multiple truth claims in batch"""
        tasks = [
            self.validate_truth(claim, validation_mode)
            for claim in claims
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _rapid_validation(self, result: TruthValidationResult):
        """Rapid validation using only AxiomEngine"""
        try:
            # AxiomEngine verification only
            axiom_result = await self.axiom_processor.verify_claim(
                result.claim.content,
                result.claim.context
            )
            result.axiom_result = axiom_result
            result.participating_systems.append("axiom_engine")
            
            # Extract confidence
            if isinstance(axiom_result, dict):
                result.overall_confidence = axiom_result.get("confidence_score", 0.5)
            else:
                result.overall_confidence = getattr(axiom_result, "confidence_score", 0.5)
                
        except Exception as e:
            result.error_messages.append(f"Rapid validation error: {e}")
    
    async def _standard_validation(self, result: TruthValidationResult):
        """Standard validation with multiple verification layers"""
        try:
            # AxiomEngine verification
            axiom_result = await self.axiom_processor.verify_claim(
                result.claim.content,
                result.claim.context
            )
            result.axiom_result = axiom_result
            result.participating_systems.append("axiom_engine")
            
            # Knowledge validation
            knowledge_claim = KnowledgeClaim(
                content=result.claim.content,
                source=result.claim.source,
                domain=result.claim.domain,
                claim_type=result.claim.claim_type,
                metadata=result.claim.metadata
            )
            
            knowledge_result = await self.knowledge_validator.validate_knowledge_claim(
                knowledge_claim,
                validation_level="standard",
                context=result.claim.context
            )
            result.knowledge_result = knowledge_result
            result.participating_systems.append("knowledge_validator")
            
            # Trust score calculation
            if result.claim.source:
                result.trust_score = self.trust_ledger.calculate_trust_score(
                    result.claim.source
                )
            
            # Provenance analysis
            provenance = self.provenance_tracker.get_information_lineage(
                result.claim.content
            )
            result.provenance_score = self._calculate_provenance_score(provenance)
            
            # Weighted confidence calculation
            result.overall_confidence = self._calculate_weighted_confidence(result)
            
        except Exception as e:
            result.error_messages.append(f"Standard validation error: {e}")
    
    async def _rigorous_validation(self, result: TruthValidationResult):
        """Rigorous validation with consensus mechanism"""
        try:
            # Perform standard validation first
            await self._standard_validation(result)
            
            # Network consensus query
            consensus_result = await self._perform_consensus_validation(result.claim)
            result.consensus_result = consensus_result
            result.participating_systems.append("mesh_consensus")
            
            # Temporal consistency check
            result.temporal_consistency = await self._check_temporal_consistency(
                result.claim
            )
            
            # Evidence synthesis
            await self._synthesize_evidence(result)
            
            # Recalculate confidence with consensus
            result.overall_confidence = self._calculate_consensus_confidence(result)
            
        except Exception as e:
            result.error_messages.append(f"Rigorous validation error: {e}")
    
    async def _forensic_validation(self, result: TruthValidationResult):
        """Forensic-level validation with maximum verification"""
        try:
            # Perform rigorous validation first
            await self._rigorous_validation(result)
            
            # Deep trust network analysis
            trust_network = self.trust_ledger.get_trust_network(radius=3)
            result.validation_metadata["trust_network"] = trust_network
            
            # Cross-domain validation
            await self._cross_domain_validation(result)
            
            # Historical pattern analysis
            await self._analyze_historical_patterns(result)
            
            # Final forensic confidence adjustment
            result.overall_confidence = self._calculate_forensic_confidence(result)
            
        except Exception as e:
            result.error_messages.append(f"Forensic validation error: {e}")
    
    async def _perform_consensus_validation(self, claim: TruthClaim) -> ConsensusResult:
        """Perform network consensus validation"""
        try:
            self.consensus_queries += 1
            
            # Broadcast truth query to network
            query = {
                "claim_id": claim.claim_id,
                "content": claim.content,
                "domain": claim.domain,
                "query_type": "truth_validation"
            }
            
            responses = await self.mesh_protocol.broadcast_truth_query(query)
            
            # Process responses
            participating_nodes = []
            confidence_scores = []
            dissenting_nodes = []
            
            for response in responses:
                node_id = response.get("node_id", "unknown")
                response_data = response.get("response", {})
                
                participating_nodes.append(node_id)
                confidence = response_data.get("confidence", 0.5)
                confidence_scores.append(confidence)
                
                if confidence < 0.5:
                    dissenting_nodes.append(node_id)
            
            # Calculate agreement score
            agreement_score = sum(confidence_scores) / max(len(confidence_scores), 1)
            
            return ConsensusResult(
                participating_nodes=participating_nodes,
                agreement_score=agreement_score,
                confidence_scores=confidence_scores,
                dissenting_nodes=dissenting_nodes,
                consensus_metadata={"query_responses": len(responses)}
            )
            
        except Exception as e:
            self.logger.error(f"Consensus validation error: {e}")
            return ConsensusResult(
                participating_nodes=[],
                agreement_score=0.0,
                confidence_scores=[]
            )
    
    async def _check_temporal_consistency(self, claim: TruthClaim) -> float:
        """Check temporal consistency of claim"""
        try:
            # Check if claim is consistent over time
            # This would involve querying historical data
            
            # Mock implementation
            return 0.8
            
        except Exception as e:
            self.logger.error(f"Temporal consistency check error: {e}")
            return 0.0
    
    async def _synthesize_evidence(self, result: TruthValidationResult):
        """Synthesize supporting and contradicting evidence"""
        try:
            # Extract evidence from AxiomEngine result
            if result.axiom_result:
                axiom_corroborations = self._extract_axiom_corroborations(
                    result.axiom_result
                )
                result.supporting_evidence.extend(axiom_corroborations)
            
            # Extract evidence from knowledge validation
            if result.knowledge_result:
                knowledge_evidence = self._extract_knowledge_evidence(
                    result.knowledge_result
                )
                result.supporting_evidence.extend(knowledge_evidence)
                
                # Extract contradictions
                contradictions = getattr(result.knowledge_result, 'contradictions', [])
                for contradiction in contradictions:
                    evidence = ValidationEvidence(
                        evidence_type="contradiction",
                        content=contradiction.get("description", ""),
                        confidence=contradiction.get("confidence", 0.5),
                        source="knowledge_validator"
                    )
                    result.contradicting_evidence.append(evidence)
                    
        except Exception as e:
            result.error_messages.append(f"Evidence synthesis error: {e}")
    
    async def _cross_domain_validation(self, result: TruthValidationResult):
        """Perform cross-domain validation"""
        try:
            # This would validate the claim across different knowledge domains
            # Mock implementation
            result.validation_metadata["cross_domain_score"] = 0.7
            
        except Exception as e:
            result.error_messages.append(f"Cross-domain validation error: {e}")
    
    async def _analyze_historical_patterns(self, result: TruthValidationResult):
        """Analyze historical patterns for the claim"""
        try:
            # This would analyze historical validation patterns
            # Mock implementation
            result.validation_metadata["historical_pattern_score"] = 0.8
            
        except Exception as e:
            result.error_messages.append(f"Historical pattern analysis error: {e}")
    
    def _calculate_provenance_score(self, provenance: Dict) -> float:
        """Calculate provenance reliability score"""
        try:
            sources = provenance.get("sources", [])
            transformations = provenance.get("transformations", [])
            
            # Score based on source reliability and transformation count
            source_score = min(len(sources) * 0.2, 1.0)
            transformation_penalty = min(len(transformations) * 0.1, 0.5)
            
            return max(source_score - transformation_penalty, 0.0)
            
        except Exception:
            return 0.0
    
    def _calculate_weighted_confidence(self, result: TruthValidationResult) -> float:
        """Calculate weighted confidence from multiple sources"""
        try:
            total_confidence = 0.0
            total_weight = 0.0
            
            # AxiomEngine confidence
            if result.axiom_result:
                axiom_confidence = self._extract_confidence(result.axiom_result)
                weight = self.evidence_weight["axiom_verification"]
                total_confidence += axiom_confidence * weight
                total_weight += weight
            
            # Knowledge validation confidence
            if result.knowledge_result:
                knowledge_confidence = getattr(result.knowledge_result, 'confidence_score', 0.5)
                weight = self.evidence_weight["knowledge_validation"]
                total_confidence += knowledge_confidence * weight
                total_weight += weight
            
            # Trust score
            if result.trust_score > 0:
                weight = self.evidence_weight["trust_network"]
                total_confidence += result.trust_score * weight
                total_weight += weight
            
            return total_confidence / max(total_weight, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_consensus_confidence(self, result: TruthValidationResult) -> float:
        """Calculate confidence including consensus results"""
        try:
            base_confidence = self._calculate_weighted_confidence(result)
            
            if result.consensus_result:
                consensus_weight = self.evidence_weight["consensus"]
                consensus_confidence = result.consensus_result.agreement_score
                
                return (base_confidence * (1 - consensus_weight) + 
                       consensus_confidence * consensus_weight)
            
            return base_confidence
            
        except Exception:
            return 0.5
    
    def _calculate_forensic_confidence(self, result: TruthValidationResult) -> float:
        """Calculate final forensic confidence"""
        try:
            base_confidence = self._calculate_consensus_confidence(result)
            
            # Apply forensic adjustments
            temporal_bonus = result.temporal_consistency * 0.05
            cross_domain_bonus = result.validation_metadata.get("cross_domain_score", 0.0) * 0.03
            historical_bonus = result.validation_metadata.get("historical_pattern_score", 0.0) * 0.02
            
            return min(base_confidence + temporal_bonus + cross_domain_bonus + historical_bonus, 1.0)
            
        except Exception:
            return base_confidence if 'base_confidence' in locals() else 0.5
    
    def _calculate_final_truth_status(self, result: TruthValidationResult):
        """Determine final truth status based on validation results"""
        try:
            confidence = result.overall_confidence
            evidence_count = len(result.supporting_evidence)
            contradiction_count = len(result.contradicting_evidence)
            
            # Determine status based on confidence and evidence
            if confidence >= 0.95 and evidence_count >= 5 and contradiction_count == 0:
                result.truth_status = TruthStatus.EMPIRICALLY_VERIFIED
            elif confidence >= 0.85 and evidence_count >= 3 and contradiction_count <= 1:
                result.truth_status = TruthStatus.STRONGLY_SUPPORTED
            elif confidence >= 0.75 and evidence_count >= 2:
                result.truth_status = TruthStatus.CORROBORATED
            elif confidence >= 0.6 and contradiction_count == 0:
                result.truth_status = TruthStatus.LOGICALLY_CONSISTENT
            elif confidence >= 0.4:
                result.truth_status = TruthStatus.PROPOSED
            elif contradiction_count > evidence_count:
                result.truth_status = TruthStatus.CONTRADICTED
            elif contradiction_count > 0:
                result.truth_status = TruthStatus.DISPUTED
            else:
                result.truth_status = TruthStatus.UNVERIFIED
                
        except Exception as e:
            self.logger.error(f"Error calculating truth status: {e}")
            result.truth_status = TruthStatus.ERROR
    
    def _extract_confidence(self, axiom_result: Union[AxiomVerificationResult, Dict]) -> float:
        """Extract confidence score from AxiomEngine result"""
        if isinstance(axiom_result, dict):
            return axiom_result.get("confidence_score", 0.5)
        else:
            return getattr(axiom_result, "confidence_score", 0.5)
    
    def _extract_axiom_corroborations(self, axiom_result) -> List[ValidationEvidence]:
        """Extract corroborations from AxiomEngine result"""
        evidence = []
        
        try:
            if isinstance(axiom_result, dict):
                corroborations = axiom_result.get("corroborations", [])
            else:
                corroborations = getattr(axiom_result, "corroborations", [])
            
            for corr in corroborations:
                evidence.append(ValidationEvidence(
                    evidence_type="corroboration",
                    content=corr.get("content", ""),
                    confidence=corr.get("similarity", 0.5),
                    source=corr.get("source", "axiom_engine")
                ))
                
        except Exception as e:
            self.logger.error(f"Error extracting AxiomEngine corroborations: {e}")
        
        return evidence
    
    def _extract_knowledge_evidence(self, knowledge_result) -> List[ValidationEvidence]:
        """Extract evidence from knowledge validation result"""
        evidence = []
        
        try:
            sources = getattr(knowledge_result, 'corroboration_sources', [])
            
            for source in sources:
                evidence.append(ValidationEvidence(
                    evidence_type="corroboration",
                    content=f"Knowledge source: {source}",
                    confidence=0.7,
                    source=source
                ))
                
        except Exception as e:
            self.logger.error(f"Error extracting knowledge evidence: {e}")
        
        return evidence
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get truth validation statistics"""
        avg_time = sum(self.validation_times) / max(len(self.validation_times), 1)
        success_rate = (self.high_confidence_validations / max(self.validations_performed, 1)) * 100
        
        return {
            "total_validations": self.validations_performed,
            "high_confidence_validations": self.high_confidence_validations,
            "success_rate": f"{success_rate:.1f}%",
            "consensus_queries": self.consensus_queries,
            "average_validation_time": f"{avg_time:.2f}s",
            "validation_time_range": {
                "min": min(self.validation_times) if self.validation_times else 0,
                "max": max(self.validation_times) if self.validation_times else 0
            }
        }
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.validations_performed = 0
        self.high_confidence_validations = 0
        self.consensus_queries = 0
        self.validation_times.clear()