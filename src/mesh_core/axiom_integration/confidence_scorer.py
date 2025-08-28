"""
Confidence Scorer - Advanced confidence scoring and reliability assessment

Provides sophisticated confidence scoring using multi-dimensional analysis,
temporal factors, source reliability, and consensus mechanisms for
The Mesh's truth verification system.

Key Features:
- Multi-dimensional confidence scoring
- Temporal confidence decay and boosting
- Source reliability assessment
- Network consensus integration
- Dynamic threshold adjustment
- Confidence calibration
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

try:
    from ..trust.trust_ledger import TrustLedger
    from ..network.mesh_protocol import MeshProtocol
    from ..provenance.provenance_tracker import ProvenanceTracker
except ImportError:
    # Mock classes for development
    class MockTrustLedger:
        def calculate_trust_score(self, node_id: str) -> float:
            return 0.8
        
        def get_node_reputation(self, node_id: str) -> Dict:
            return {"reputation": 0.8, "interactions": 100, "reliability": 0.9}
    
    class MockMeshProtocol:
        async def get_network_consensus(self, query: Dict) -> Dict:
            return {"consensus_score": 0.75, "participating_nodes": 5}
    
    class MockProvenanceTracker:
        def get_source_reliability(self, source: str) -> float:
            return 0.7
        
        def get_information_chain_quality(self, claim: str) -> Dict:
            return {"quality_score": 0.8, "chain_length": 3, "verification_points": 2}
    
    TrustLedger = MockTrustLedger
    MeshProtocol = MockMeshProtocol
    ProvenanceTracker = MockProvenanceTracker


class ConfidenceFactorType(Enum):
    """Types of confidence factors"""
    SOURCE_RELIABILITY = "source_reliability"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    CROSS_VALIDATION = "cross_validation"
    NETWORK_CONSENSUS = "network_consensus"
    EVIDENCE_STRENGTH = "evidence_strength"
    LOGICAL_CONSISTENCY = "logical_consistency"
    EXPERT_VALIDATION = "expert_validation"
    HISTORICAL_ACCURACY = "historical_accuracy"


class ConfidenceLevel(Enum):
    """Confidence level classifications"""
    ABSOLUTE = (0.95, 1.0, "absolute")
    VERY_HIGH = (0.85, 0.95, "very_high")
    HIGH = (0.75, 0.85, "high")
    MODERATE_HIGH = (0.65, 0.75, "moderate_high")
    MODERATE = (0.55, 0.65, "moderate")
    MODERATE_LOW = (0.45, 0.55, "moderate_low")
    LOW = (0.35, 0.45, "low")
    VERY_LOW = (0.25, 0.35, "very_low")
    MINIMAL = (0.0, 0.25, "minimal")
    
    def __init__(self, min_score, max_score, label):
        self.min_score = min_score
        self.max_score = max_score
        self.label = label
    
    @classmethod
    def from_score(cls, score: float):
        for level in cls:
            if level.min_score <= score < level.max_score:
                return level
        return cls.ABSOLUTE if score >= 0.95 else cls.MINIMAL


@dataclass
class ConfidenceFactor:
    """Individual confidence factor"""
    factor_type: ConfidenceFactorType
    value: float
    weight: float
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_value(self) -> float:
        return self.value * self.weight


@dataclass
class ScoreComponents:
    """Components that make up a confidence score"""
    base_score: float
    factors: List[ConfidenceFactor] = field(default_factory=list)
    adjustments: Dict[str, float] = field(default_factory=dict)
    temporal_decay: float = 0.0
    consensus_boost: float = 0.0
    calibration_factor: float = 1.0
    
    @property
    def factor_contribution(self) -> float:
        return sum(factor.weighted_value for factor in self.factors)
    
    @property
    def adjustment_total(self) -> float:
        return sum(self.adjustments.values())


@dataclass
class ConfidenceScore:
    """Comprehensive confidence score with detailed breakdown"""
    final_score: float
    confidence_level: ConfidenceLevel
    components: ScoreComponents
    calculation_method: str
    calculation_time: datetime = field(default_factory=datetime.utcnow)
    
    # Quality metrics
    reliability: float = 0.0
    stability: float = 0.0
    consensus_agreement: float = 0.0
    
    # Metadata
    participating_sources: List[str] = field(default_factory=list)
    validation_history: List[Dict] = field(default_factory=list)
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceScorer:
    """Advanced confidence scoring system"""
    
    def __init__(self):
        self.trust_ledger = TrustLedger()
        self.mesh_protocol = MeshProtocol()
        self.provenance_tracker = ProvenanceTracker()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.default_weights = {
            ConfidenceFactorType.SOURCE_RELIABILITY: 0.25,
            ConfidenceFactorType.TEMPORAL_CONSISTENCY: 0.15,
            ConfidenceFactorType.CROSS_VALIDATION: 0.20,
            ConfidenceFactorType.NETWORK_CONSENSUS: 0.15,
            ConfidenceFactorType.EVIDENCE_STRENGTH: 0.15,
            ConfidenceFactorType.LOGICAL_CONSISTENCY: 0.10
        }
        
        # Temporal settings
        self.temporal_decay_rate = 0.1  # per day
        self.temporal_boost_window = timedelta(hours=24)
        
        # Calibration settings
        self.calibration_history = deque(maxlen=1000)
        self.calibration_samples = defaultdict(list)
        
        # Statistics
        self.scores_calculated = 0
        self.calibration_adjustments = 0
        self.consensus_queries = 0
    
    async def calculate_confidence_score(
        self,
        claim: str,
        base_evidence: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        custom_weights: Optional[Dict[ConfidenceFactorType, float]] = None
    ) -> ConfidenceScore:
        """Calculate comprehensive confidence score"""
        try:
            self.logger.info(f"Calculating confidence score for claim: {claim[:100]}...")
            
            # Initialize components
            components = ScoreComponents(
                base_score=base_evidence.get("base_confidence", 0.5)
            )
            
            # Use custom weights or defaults
            weights = custom_weights or self.default_weights
            
            # Calculate individual factors
            await self._calculate_source_reliability(claim, components, weights, context)
            await self._calculate_temporal_consistency(claim, components, weights, context)
            await self._calculate_cross_validation(claim, components, weights, context)
            await self._calculate_network_consensus(claim, components, weights, context)
            await self._calculate_evidence_strength(base_evidence, components, weights)
            await self._calculate_logical_consistency(claim, components, weights, context)
            
            # Apply temporal decay
            components.temporal_decay = self._calculate_temporal_decay(base_evidence)
            
            # Apply consensus boost
            components.consensus_boost = await self._calculate_consensus_boost(claim, context)
            
            # Calculate raw score
            raw_score = self._calculate_raw_score(components)
            
            # Apply calibration
            components.calibration_factor = self._get_calibration_factor(raw_score, context)
            final_score = self._apply_calibration(raw_score, components.calibration_factor)
            
            # Create confidence score object
            confidence_score = ConfidenceScore(
                final_score=final_score,
                confidence_level=ConfidenceLevel.from_score(final_score),
                components=components,
                calculation_method="multi_dimensional_weighted"
            )
            
            # Calculate quality metrics
            await self._calculate_quality_metrics(confidence_score, claim, context)
            
            # Record for calibration
            self._record_score_for_calibration(confidence_score, context)
            
            self.scores_calculated += 1
            return confidence_score
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return self._create_error_score(str(e))
    
    async def calculate_batch_confidence_scores(
        self,
        claims_and_evidence: List[Tuple[str, Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ConfidenceScore]:
        """Calculate confidence scores for multiple claims"""
        tasks = [
            self.calculate_confidence_score(claim, evidence, context)
            for claim, evidence in claims_and_evidence
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def recalibrate_scoring_model(self, validation_data: List[Dict[str, Any]]):
        """Recalibrate the scoring model based on validation feedback"""
        try:
            self.logger.info("Recalibrating confidence scoring model...")
            
            # Analyze prediction accuracy
            accuracy_by_level = defaultdict(list)
            
            for validation in validation_data:
                predicted_score = validation.get("predicted_score", 0.5)
                actual_outcome = validation.get("actual_outcome", 0.5)
                predicted_level = ConfidenceLevel.from_score(predicted_score)
                
                accuracy = 1.0 - abs(predicted_score - actual_outcome)
                accuracy_by_level[predicted_level].append(accuracy)
            
            # Update calibration factors
            for level, accuracies in accuracy_by_level.items():
                avg_accuracy = sum(accuracies) / len(accuracies)
                self.calibration_samples[level] = accuracies[-100:]  # Keep recent samples
                
                self.logger.info(f"Calibration for {level.label}: {avg_accuracy:.3f}")
            
            self.calibration_adjustments += 1
            
        except Exception as e:
            self.logger.error(f"Error recalibrating scoring model: {e}")
    
    async def _calculate_source_reliability(
        self,
        claim: str,
        components: ScoreComponents,
        weights: Dict,
        context: Optional[Dict]
    ):
        """Calculate source reliability factor"""
        try:
            sources = context.get("sources", []) if context else []
            if not sources:
                return
            
            # Calculate average source reliability
            reliabilities = []
            for source in sources:
                reliability = self.provenance_tracker.get_source_reliability(source)
                trust_score = self.trust_ledger.calculate_trust_score(source)
                combined_reliability = (reliability + trust_score) / 2
                reliabilities.append(combined_reliability)
            
            avg_reliability = sum(reliabilities) / len(reliabilities)
            
            factor = ConfidenceFactor(
                factor_type=ConfidenceFactorType.SOURCE_RELIABILITY,
                value=avg_reliability,
                weight=weights[ConfidenceFactorType.SOURCE_RELIABILITY],
                source="source_analysis",
                metadata={"source_count": len(sources), "individual_reliabilities": reliabilities}
            )
            components.factors.append(factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating source reliability: {e}")
    
    async def _calculate_temporal_consistency(
        self,
        claim: str,
        components: ScoreComponents,
        weights: Dict,
        context: Optional[Dict]
    ):
        """Calculate temporal consistency factor"""
        try:
            # Check temporal consistency of the claim
            # This would analyze how the claim has been validated over time
            
            # Mock implementation
            temporal_score = 0.8  # Would be calculated from historical data
            
            factor = ConfidenceFactor(
                factor_type=ConfidenceFactorType.TEMPORAL_CONSISTENCY,
                value=temporal_score,
                weight=weights[ConfidenceFactorType.TEMPORAL_CONSISTENCY],
                source="temporal_analysis",
                metadata={"consistency_period": "30_days"}
            )
            components.factors.append(factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal consistency: {e}")
    
    async def _calculate_cross_validation(
        self,
        claim: str,
        components: ScoreComponents,
        weights: Dict,
        context: Optional[Dict]
    ):
        """Calculate cross-validation factor"""
        try:
            # Cross-validate with multiple independent sources
            corroborations = context.get("corroborations", []) if context else []
            
            if corroborations:
                # Calculate cross-validation strength
                unique_sources = set(corr.get("source", "") for corr in corroborations)
                avg_similarity = sum(corr.get("similarity", 0.5) for corr in corroborations) / len(corroborations)
                
                # Factor in diversity and similarity
                diversity_bonus = min(len(unique_sources) / 5.0, 1.0)  # Up to 5 sources
                cross_validation_score = avg_similarity * (0.7 + 0.3 * diversity_bonus)
                
                factor = ConfidenceFactor(
                    factor_type=ConfidenceFactorType.CROSS_VALIDATION,
                    value=cross_validation_score,
                    weight=weights[ConfidenceFactorType.CROSS_VALIDATION],
                    source="cross_validation",
                    metadata={
                        "corroboration_count": len(corroborations),
                        "unique_sources": len(unique_sources),
                        "avg_similarity": avg_similarity
                    }
                )
                components.factors.append(factor)
                
        except Exception as e:
            self.logger.error(f"Error calculating cross-validation: {e}")
    
    async def _calculate_network_consensus(
        self,
        claim: str,
        components: ScoreComponents,
        weights: Dict,
        context: Optional[Dict]
    ):
        """Calculate network consensus factor"""
        try:
            # Query network for consensus
            query = {
                "claim": claim,
                "query_type": "consensus_check"
            }
            
            consensus_result = await self.mesh_protocol.get_network_consensus(query)
            self.consensus_queries += 1
            
            consensus_score = consensus_result.get("consensus_score", 0.5)
            participating_nodes = consensus_result.get("participating_nodes", 0)
            
            # Adjust for participation level
            participation_factor = min(participating_nodes / 10.0, 1.0)  # Up to 10 nodes
            adjusted_consensus = consensus_score * (0.6 + 0.4 * participation_factor)
            
            factor = ConfidenceFactor(
                factor_type=ConfidenceFactorType.NETWORK_CONSENSUS,
                value=adjusted_consensus,
                weight=weights[ConfidenceFactorType.NETWORK_CONSENSUS],
                source="network_consensus",
                metadata={
                    "raw_consensus": consensus_score,
                    "participating_nodes": participating_nodes,
                    "participation_factor": participation_factor
                }
            )
            components.factors.append(factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating network consensus: {e}")
    
    async def _calculate_evidence_strength(
        self,
        base_evidence: Dict[str, Any],
        components: ScoreComponents,
        weights: Dict
    ):
        """Calculate evidence strength factor"""
        try:
            # Analyze strength of provided evidence
            evidence_count = base_evidence.get("evidence_count", 0)
            citation_count = len(base_evidence.get("citations", []))
            verified_citations = len([
                c for c in base_evidence.get("citations", [])
                if c.get("status") == "VALID_AND_LIVE"
            ])
            
            # Calculate evidence strength
            count_score = min(evidence_count / 5.0, 1.0)  # Up to 5 pieces of evidence
            citation_score = min(citation_count / 3.0, 1.0)  # Up to 3 citations
            verification_score = verified_citations / max(citation_count, 1)
            
            evidence_strength = (count_score + citation_score + verification_score) / 3
            
            factor = ConfidenceFactor(
                factor_type=ConfidenceFactorType.EVIDENCE_STRENGTH,
                value=evidence_strength,
                weight=weights[ConfidenceFactorType.EVIDENCE_STRENGTH],
                source="evidence_analysis",
                metadata={
                    "evidence_count": evidence_count,
                    "citation_count": citation_count,
                    "verified_citations": verified_citations
                }
            )
            components.factors.append(factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating evidence strength: {e}")
    
    async def _calculate_logical_consistency(
        self,
        claim: str,
        components: ScoreComponents,
        weights: Dict,
        context: Optional[Dict]
    ):
        """Calculate logical consistency factor"""
        try:
            # Analyze logical consistency of the claim
            contradictions = context.get("contradictions", []) if context else []
            
            # Simple logical consistency check
            if not contradictions:
                consistency_score = 0.9
            else:
                # Penalize for contradictions
                contradiction_penalty = min(len(contradictions) * 0.2, 0.8)
                consistency_score = max(0.9 - contradiction_penalty, 0.1)
            
            factor = ConfidenceFactor(
                factor_type=ConfidenceFactorType.LOGICAL_CONSISTENCY,
                value=consistency_score,
                weight=weights[ConfidenceFactorType.LOGICAL_CONSISTENCY],
                source="logical_analysis",
                metadata={"contradiction_count": len(contradictions)}
            )
            components.factors.append(factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating logical consistency: {e}")
    
    def _calculate_temporal_decay(self, base_evidence: Dict[str, Any]) -> float:
        """Calculate temporal decay factor"""
        try:
            timestamp_str = base_evidence.get("timestamp")
            if not timestamp_str:
                return 0.0
            
            # Parse timestamp (assuming ISO format)
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = timestamp_str
            
            # Calculate age in days
            age_days = (datetime.utcnow() - timestamp).days
            
            # Apply exponential decay
            decay = self.temporal_decay_rate * age_days
            return min(decay, 0.5)  # Cap at 50% decay
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal decay: {e}")
            return 0.0
    
    async def _calculate_consensus_boost(
        self,
        claim: str,
        context: Optional[Dict]
    ) -> float:
        """Calculate consensus boost factor"""
        try:
            # Recent validation boost
            recent_validations = context.get("recent_validations", []) if context else []
            
            if recent_validations:
                recent_count = len([
                    v for v in recent_validations
                    if datetime.utcnow() - v.get("timestamp", datetime.min) < self.temporal_boost_window
                ])
                return min(recent_count * 0.05, 0.2)  # Up to 20% boost
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_raw_score(self, components: ScoreComponents) -> float:
        """Calculate raw confidence score from components"""
        try:
            # Start with base score
            raw_score = components.base_score
            
            # Add weighted factor contributions
            raw_score += components.factor_contribution
            
            # Add adjustments
            raw_score += components.adjustment_total
            
            # Apply temporal decay
            raw_score -= components.temporal_decay
            
            # Add consensus boost
            raw_score += components.consensus_boost
            
            # Clamp to [0, 1] range
            return max(0.0, min(raw_score, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating raw score: {e}")
            return 0.5
    
    def _get_calibration_factor(self, raw_score: float, context: Optional[Dict]) -> float:
        """Get calibration factor for the raw score"""
        try:
            confidence_level = ConfidenceLevel.from_score(raw_score)
            
            # Get historical accuracy for this confidence level
            if confidence_level in self.calibration_samples:
                samples = self.calibration_samples[confidence_level]
                if samples:
                    avg_accuracy = sum(samples) / len(samples)
                    # Adjust calibration based on accuracy
                    # If accuracy is low, reduce confidence
                    return 0.5 + (avg_accuracy * 0.5)
            
            # Default calibration
            return 1.0
            
        except Exception:
            return 1.0
    
    def _apply_calibration(self, raw_score: float, calibration_factor: float) -> float:
        """Apply calibration to raw score"""
        try:
            calibrated_score = raw_score * calibration_factor
            return max(0.0, min(calibrated_score, 1.0))
        except Exception:
            return raw_score
    
    async def _calculate_quality_metrics(
        self,
        confidence_score: ConfidenceScore,
        claim: str,
        context: Optional[Dict]
    ):
        """Calculate quality metrics for the confidence score"""
        try:
            # Reliability: consistency of factors
            factor_values = [f.value for f in confidence_score.components.factors]
            if factor_values:
                mean_factor = sum(factor_values) / len(factor_values)
                variance = sum((v - mean_factor) ** 2 for v in factor_values) / len(factor_values)
                confidence_score.reliability = max(0.0, 1.0 - math.sqrt(variance))
            
            # Stability: based on temporal consistency
            temporal_factors = [
                f for f in confidence_score.components.factors
                if f.factor_type == ConfidenceFactorType.TEMPORAL_CONSISTENCY
            ]
            if temporal_factors:
                confidence_score.stability = temporal_factors[0].value
            
            # Consensus agreement: from network consensus
            consensus_factors = [
                f for f in confidence_score.components.factors
                if f.factor_type == ConfidenceFactorType.NETWORK_CONSENSUS
            ]
            if consensus_factors:
                confidence_score.consensus_agreement = consensus_factors[0].value
                
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
    
    def _record_score_for_calibration(
        self,
        confidence_score: ConfidenceScore,
        context: Optional[Dict]
    ):
        """Record score for future calibration"""
        try:
            calibration_record = {
                "score": confidence_score.final_score,
                "level": confidence_score.confidence_level,
                "timestamp": confidence_score.calculation_time,
                "factors": len(confidence_score.components.factors)
            }
            
            self.calibration_history.append(calibration_record)
            
        except Exception as e:
            self.logger.error(f"Error recording calibration data: {e}")
    
    def _create_error_score(self, error_message: str) -> ConfidenceScore:
        """Create an error confidence score"""
        components = ScoreComponents(base_score=0.0)
        return ConfidenceScore(
            final_score=0.0,
            confidence_level=ConfidenceLevel.MINIMAL,
            components=components,
            calculation_method="error",
            calculation_metadata={"error": error_message}
        )
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get confidence scoring statistics"""
        recent_scores = list(self.calibration_history)[-100:]  # Last 100 scores
        
        if recent_scores:
            avg_score = sum(record["score"] for record in recent_scores) / len(recent_scores)
            score_distribution = defaultdict(int)
            for record in recent_scores:
                score_distribution[record["level"].label] += 1
        else:
            avg_score = 0.0
            score_distribution = {}
        
        return {
            "total_scores_calculated": self.scores_calculated,
            "calibration_adjustments": self.calibration_adjustments,
            "consensus_queries": self.consensus_queries,
            "average_recent_score": avg_score,
            "score_distribution": dict(score_distribution),
            "calibration_samples_count": sum(len(samples) for samples in self.calibration_samples.values())
        }
    
    def reset_statistics(self):
        """Reset scoring statistics"""
        self.scores_calculated = 0
        self.calibration_adjustments = 0
        self.consensus_queries = 0
        self.calibration_history.clear()
        self.calibration_samples.clear()