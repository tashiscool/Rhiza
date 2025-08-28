"""
Knowledge Validator - Advanced claim analysis and validation

Provides comprehensive knowledge claim validation using AxiomEngine's
verification capabilities combined with The Mesh's distributed consensus.

Key Features:
- Multi-source knowledge validation
- Semantic claim analysis
- Evidence correlation
- Knowledge graph integration
- Consensus-based validation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from .axiom_processor import AxiomProcessor, AxiomVerificationResult
    from ..trust.trust_ledger import TrustLedger
    from ..provenance.provenance_tracker import ProvenanceTracker
except ImportError:
    # Mock classes for development
    class MockAxiomProcessor:
        async def verify_claim(self, claim: str, context: Optional[Dict] = None):
            return {"verification_status": "mock", "confidence_score": 0.5}
    
    class MockTrustLedger:
        def calculate_trust_score(self, node_id: str) -> float:
            return 0.8
    
    class MockProvenanceTracker:
        def track_claim_origin(self, claim: str, metadata: Dict) -> str:
            return "mock_trace_id"
    
    AxiomProcessor = MockAxiomProcessor
    TrustLedger = MockTrustLedger
    ProvenanceTracker = MockProvenanceTracker
    AxiomVerificationResult = dict


class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    FORENSIC = "forensic"


class ConfidenceLevel(Enum):
    """Knowledge confidence levels"""
    VERY_LOW = 0.0
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    ABSOLUTE = 1.0


@dataclass
class KnowledgeClaim:
    """Knowledge claim for validation"""
    content: str
    source: Optional[str] = None
    domain: str = "general"
    claim_type: str = "factual"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationResult:
    """Result of knowledge validation"""
    claim: KnowledgeClaim
    validation_level: ValidationLevel
    confidence_score: float
    confidence_level: ConfidenceLevel
    axiom_result: Optional[Union[AxiomVerificationResult, Dict]] = None
    consensus_score: float = 0.0
    evidence_count: int = 0
    corroboration_sources: List[str] = None
    contradictions: List[Dict] = None
    provenance_trace: Optional[str] = None
    validation_time: float = 0.0
    validation_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.corroboration_sources is None:
            self.corroboration_sources = []
        if self.contradictions is None:
            self.contradictions = []
        if self.validation_metadata is None:
            self.validation_metadata = {}
        
        # Determine confidence level from score
        if self.confidence_score >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.confidence_score >= 0.7:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.5:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 0.3:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW


class KnowledgeValidator:
    """Advanced knowledge claim validator"""
    
    def __init__(self):
        self.axiom_processor = AxiomProcessor()
        self.trust_ledger = TrustLedger()
        self.provenance_tracker = ProvenanceTracker()
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.consensus_threshold = 0.75
        self.min_corroboration_sources = 2
        self.contradiction_tolerance = 0.1
        
        # Statistics
        self.validations_performed = 0
        self.high_confidence_results = 0
        self.contradictions_found = 0
    
    async def validate_knowledge_claim(
        self,
        claim: KnowledgeClaim,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate a knowledge claim with specified thoroughness"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(f"Validating claim: {claim.content[:100]}...")
            
            # Track provenance
            provenance_trace = self.provenance_tracker.track_claim_origin(
                claim.content, claim.metadata or {}
            )
            
            # Perform AxiomEngine verification
            axiom_result = await self.axiom_processor.verify_claim(
                claim.content, context
            )
            
            # Perform validation based on level
            if validation_level == ValidationLevel.BASIC:
                result = await self._basic_validation(claim, axiom_result)
            elif validation_level == ValidationLevel.STANDARD:
                result = await self._standard_validation(claim, axiom_result)
            elif validation_level == ValidationLevel.COMPREHENSIVE:
                result = await self._comprehensive_validation(claim, axiom_result)
            elif validation_level == ValidationLevel.FORENSIC:
                result = await self._forensic_validation(claim, axiom_result)
            else:
                result = await self._standard_validation(claim, axiom_result)
            
            # Add common metadata
            result.provenance_trace = provenance_trace
            result.validation_time = asyncio.get_event_loop().time() - start_time
            result.validation_level = validation_level
            result.axiom_result = axiom_result
            
            # Update statistics
            self.validations_performed += 1
            if result.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
                self.high_confidence_results += 1
            if result.contradictions:
                self.contradictions_found += len(result.contradictions)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating claim: {e}")
            return ValidationResult(
                claim=claim,
                validation_level=validation_level,
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                validation_metadata={"error": str(e)}
            )
    
    async def validate_batch_claims(
        self,
        claims: List[KnowledgeClaim],
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> List[ValidationResult]:
        """Validate multiple claims in batch"""
        tasks = [
            self.validate_knowledge_claim(claim, validation_level)
            for claim in claims
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _basic_validation(
        self,
        claim: KnowledgeClaim,
        axiom_result: Union[AxiomVerificationResult, Dict]
    ) -> ValidationResult:
        """Basic validation using only AxiomEngine results"""
        confidence_score = self._extract_confidence_score(axiom_result)
        
        return ValidationResult(
            claim=claim,
            validation_level=ValidationLevel.BASIC,
            confidence_score=confidence_score,
            confidence_level=ConfidenceLevel.MEDIUM,
            evidence_count=1
        )
    
    async def _standard_validation(
        self,
        claim: KnowledgeClaim,
        axiom_result: Union[AxiomVerificationResult, Dict]
    ) -> ValidationResult:
        """Standard validation with corroboration analysis"""
        confidence_score = self._extract_confidence_score(axiom_result)
        corroborations = self._extract_corroborations(axiom_result)
        
        # Calculate consensus score
        consensus_score = await self._calculate_consensus_score(claim, corroborations)
        
        # Adjust confidence based on consensus
        adjusted_confidence = self._adjust_confidence_for_consensus(
            confidence_score, consensus_score
        )
        
        return ValidationResult(
            claim=claim,
            validation_level=ValidationLevel.STANDARD,
            confidence_score=adjusted_confidence,
            confidence_level=ConfidenceLevel.MEDIUM,
            consensus_score=consensus_score,
            evidence_count=len(corroborations) + 1,
            corroboration_sources=[c.get("source", "unknown") for c in corroborations]
        )
    
    async def _comprehensive_validation(
        self,
        claim: KnowledgeClaim,
        axiom_result: Union[AxiomVerificationResult, Dict]
    ) -> ValidationResult:
        """Comprehensive validation with contradiction detection"""
        # Start with standard validation
        result = await self._standard_validation(claim, axiom_result)
        
        # Find contradictions
        contradictions = await self._find_contradictions(claim)
        result.contradictions = contradictions
        
        # Adjust confidence for contradictions
        if contradictions:
            contradiction_penalty = min(len(contradictions) * 0.1, 0.4)
            result.confidence_score = max(
                result.confidence_score - contradiction_penalty,
                0.0
            )
        
        # Perform semantic analysis
        semantic_metadata = await self._perform_semantic_analysis(claim)
        result.validation_metadata.update(semantic_metadata)
        
        return result
    
    async def _forensic_validation(
        self,
        claim: KnowledgeClaim,
        axiom_result: Union[AxiomVerificationResult, Dict]
    ) -> ValidationResult:
        """Forensic-level validation with deep analysis"""
        # Start with comprehensive validation
        result = await self._comprehensive_validation(claim, axiom_result)
        
        # Deep source analysis
        source_analysis = await self._analyze_sources(claim, result.corroboration_sources)
        result.validation_metadata["source_analysis"] = source_analysis
        
        # Trust network analysis
        trust_analysis = await self._analyze_trust_network(claim)
        result.validation_metadata["trust_analysis"] = trust_analysis
        
        # Historical consistency check
        consistency_analysis = await self._check_historical_consistency(claim)
        result.validation_metadata["consistency_analysis"] = consistency_analysis
        
        # Final confidence adjustment based on forensic analysis
        forensic_boost = self._calculate_forensic_confidence_boost(
            source_analysis, trust_analysis, consistency_analysis
        )
        result.confidence_score = min(result.confidence_score + forensic_boost, 1.0)
        
        return result
    
    def _extract_confidence_score(self, axiom_result: Union[AxiomVerificationResult, Dict]) -> float:
        """Extract confidence score from AxiomEngine result"""
        if isinstance(axiom_result, dict):
            return axiom_result.get("confidence_score", 0.5)
        else:
            return getattr(axiom_result, "confidence_score", 0.5)
    
    def _extract_corroborations(self, axiom_result: Union[AxiomVerificationResult, Dict]) -> List[Dict]:
        """Extract corroborations from AxiomEngine result"""
        if isinstance(axiom_result, dict):
            return axiom_result.get("corroborations", [])
        else:
            return getattr(axiom_result, "corroborations", [])
    
    async def _calculate_consensus_score(
        self,
        claim: KnowledgeClaim,
        corroborations: List[Dict]
    ) -> float:
        """Calculate consensus score from corroborations"""
        if not corroborations:
            return 0.0
        
        # Simple consensus calculation
        similarity_scores = [c.get("similarity", 0.0) for c in corroborations]
        return sum(similarity_scores) / len(similarity_scores)
    
    def _adjust_confidence_for_consensus(self, base_confidence: float, consensus_score: float) -> float:
        """Adjust confidence based on consensus"""
        if consensus_score >= self.consensus_threshold:
            return min(base_confidence + 0.1, 1.0)
        elif consensus_score < 0.3:
            return max(base_confidence - 0.1, 0.0)
        return base_confidence
    
    async def _find_contradictions(self, claim: KnowledgeClaim) -> List[Dict]:
        """Find contradictory claims"""
        # Mock implementation - in practice would search for contradictory evidence
        contradictions = []
        
        # Check for common contradictory patterns
        content_lower = claim.content.lower()
        if "always" in content_lower or "never" in content_lower:
            contradictions.append({
                "type": "absolute_claim",
                "description": "Absolute claims are often contradicted by exceptions",
                "confidence": 0.3
            })
        
        return contradictions
    
    async def _perform_semantic_analysis(self, claim: KnowledgeClaim) -> Dict[str, Any]:
        """Perform semantic analysis on claim"""
        return {
            "word_count": len(claim.content.split()),
            "sentence_count": claim.content.count('.') + claim.content.count('!') + claim.content.count('?'),
            "has_numbers": any(char.isdigit() for char in claim.content),
            "has_dates": "202" in claim.content or "201" in claim.content,
            "complexity": "high" if len(claim.content.split()) > 50 else "medium"
        }
    
    async def _analyze_sources(self, claim: KnowledgeClaim, sources: List[str]) -> Dict[str, Any]:
        """Analyze source reliability"""
        return {
            "source_count": len(sources),
            "unique_domains": len(set(sources)),
            "diversity_score": len(set(sources)) / max(len(sources), 1)
        }
    
    async def _analyze_trust_network(self, claim: KnowledgeClaim) -> Dict[str, Any]:
        """Analyze trust network for claim"""
        return {
            "trust_score": 0.8,  # Mock score
            "network_reach": 10,  # Mock reach
            "trust_distribution": "normal"  # Mock distribution
        }
    
    async def _check_historical_consistency(self, claim: KnowledgeClaim) -> Dict[str, Any]:
        """Check historical consistency of claim"""
        return {
            "historical_support": 0.7,  # Mock support
            "consistency_score": 0.8,  # Mock consistency
            "temporal_stability": "stable"  # Mock stability
        }
    
    def _calculate_forensic_confidence_boost(
        self,
        source_analysis: Dict,
        trust_analysis: Dict,
        consistency_analysis: Dict
    ) -> float:
        """Calculate confidence boost from forensic analysis"""
        source_boost = source_analysis.get("diversity_score", 0) * 0.1
        trust_boost = trust_analysis.get("trust_score", 0) * 0.05
        consistency_boost = consistency_analysis.get("consistency_score", 0) * 0.05
        
        return source_boost + trust_boost + consistency_boost
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        success_rate = (
            self.high_confidence_results / max(self.validations_performed, 1)
        ) * 100
        
        return {
            "total_validations": self.validations_performed,
            "high_confidence_results": self.high_confidence_results,
            "success_rate": f"{success_rate:.1f}%",
            "contradictions_found": self.contradictions_found,
            "avg_contradictions_per_validation": (
                self.contradictions_found / max(self.validations_performed, 1)
            )
        }
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.validations_performed = 0
        self.high_confidence_results = 0
        self.contradictions_found = 0