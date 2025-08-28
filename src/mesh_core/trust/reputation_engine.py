"""
Reputation Engine System
=======================

Advanced reputation calculation engine that combines multiple trust signals
and historical data to compute comprehensive reputation scores for network nodes.
"""

import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class ReputationMetric(Enum):
    """Types of reputation metrics"""
    OVERALL_REPUTATION = "overall"        # Combined reputation score
    INFORMATION_QUALITY = "info_quality"  # Quality of information provided
    NETWORK_CONTRIBUTION = "contribution" # Contribution to network
    RELIABILITY = "reliability"          # Node reliability
    TRUSTWORTHINESS = "trustworthiness"  # General trustworthiness
    EXPERTISE = "expertise"              # Domain expertise
    SOCIAL_STANDING = "social"           # Social reputation

class InteractionType(Enum):
    """Types of interactions affecting reputation"""
    INFORMATION_SHARING = "info_share"    # Shared information
    VALIDATION = "validation"            # Validated content
    COOPERATION = "cooperation"          # Cooperative behavior
    CHALLENGE = "challenge"              # Challenged claims
    SUPPORT = "support"                 # Supported others
    VIOLATION = "violation"             # Policy violation
    EXCELLENCE = "excellence"           # Exceptional performance

@dataclass
class ReputationMetrics:
    """Comprehensive reputation metrics for a node"""
    node_id: str
    overall_reputation: float
    information_quality: float
    network_contribution: float
    reliability: float
    trustworthiness: float
    expertise: float
    social_standing: float
    last_updated: float
    interaction_count: int
    confidence_level: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReputationMetrics':
        return cls(**data)

@dataclass
class InteractionRecord:
    """Record of an interaction affecting reputation"""
    interaction_id: str
    node_id: str
    observer_id: str
    interaction_type: InteractionType
    impact_score: float  # -1.0 to 1.0
    timestamp: float
    context: Dict
    verified: bool
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['interaction_type'] = self.interaction_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'InteractionRecord':
        data['interaction_type'] = InteractionType(data['interaction_type'])
        return cls(**data)

class ReputationEngine:
    """
    Advanced reputation calculation system
    
    Computes and maintains reputation scores based on observed interactions,
    trust signals, and historical behavior patterns.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.reputation_scores: Dict[str, ReputationMetrics] = {}
        self.interaction_history: Dict[str, List[InteractionRecord]] = {}
        self.reputation_weights: Dict[ReputationMetric, float] = self._init_reputation_weights()
        self.decay_parameters: Dict[str, float] = self._init_decay_parameters()
        
    def _init_reputation_weights(self) -> Dict[ReputationMetric, float]:
        """Initialize weights for different reputation components"""
        return {
            ReputationMetric.INFORMATION_QUALITY: 0.25,
            ReputationMetric.NETWORK_CONTRIBUTION: 0.20,
            ReputationMetric.RELIABILITY: 0.20,
            ReputationMetric.TRUSTWORTHINESS: 0.15,
            ReputationMetric.EXPERTISE: 0.10,
            ReputationMetric.SOCIAL_STANDING: 0.10
        }
    
    def _init_decay_parameters(self) -> Dict[str, float]:
        """Initialize reputation decay parameters"""
        return {
            'half_life_days': 90,      # Reputation half-life in days
            'min_retention': 0.1,      # Minimum reputation retention
            'decay_rate': 0.008,       # Daily decay rate
            'boost_threshold': 0.8,    # Score above which decay is slower
            'penalty_multiplier': 1.5  # Multiplier for negative interactions
        }
    
    def _generate_interaction_id(self, node_id: str, interaction_type: InteractionType) -> str:
        """Generate unique interaction ID"""
        import hashlib
        data = f"{node_id}:{interaction_type.value}:{time.time()}:{self.node_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def record_interaction(
        self,
        node_id: str,
        interaction_type: InteractionType,
        impact_score: float,
        context: Optional[Dict] = None,
        verified: bool = False
    ) -> InteractionRecord:
        """Record an interaction that affects reputation"""
        
        if context is None:
            context = {}
            
        # Clamp impact score
        impact_score = max(-1.0, min(1.0, impact_score))
        
        # Create interaction record
        interaction = InteractionRecord(
            interaction_id=self._generate_interaction_id(node_id, interaction_type),
            node_id=node_id,
            observer_id=self.node_id,
            interaction_type=interaction_type,
            impact_score=impact_score,
            timestamp=time.time(),
            context=context,
            verified=verified
        )
        
        # Store interaction
        if node_id not in self.interaction_history:
            self.interaction_history[node_id] = []
        self.interaction_history[node_id].append(interaction)
        
        # Update reputation scores
        await self._update_reputation_scores(node_id, interaction)
        
        logger.info(f"Recorded interaction: {node_id} -> {interaction_type.value} (impact: {impact_score:.3f})")
        return interaction
    
    async def _update_reputation_scores(self, node_id: str, new_interaction: InteractionRecord):
        """Update reputation scores based on new interaction"""
        
        # Get or create reputation metrics
        if node_id not in self.reputation_scores:
            self.reputation_scores[node_id] = ReputationMetrics(
                node_id=node_id,
                overall_reputation=0.5,
                information_quality=0.5,
                network_contribution=0.5,
                reliability=0.5,
                trustworthiness=0.5,
                expertise=0.5,
                social_standing=0.5,
                last_updated=time.time(),
                interaction_count=0,
                confidence_level=0.1
            )
        
        metrics = self.reputation_scores[node_id]
        
        # Apply temporal decay first
        await self._apply_temporal_decay(metrics)
        
        # Update specific metrics based on interaction type
        await self._update_specific_metrics(metrics, new_interaction)
        
        # Recalculate overall reputation
        await self._calculate_overall_reputation(metrics)
        
        # Update metadata
        metrics.last_updated = time.time()
        metrics.interaction_count += 1
        
        # Update confidence level based on interaction count and time
        await self._update_confidence_level(metrics)
    
    async def _apply_temporal_decay(self, metrics: ReputationMetrics):
        """Apply temporal decay to reputation scores"""
        
        if metrics.last_updated == 0:
            return
            
        # Calculate time elapsed
        time_elapsed = time.time() - metrics.last_updated
        days_elapsed = time_elapsed / 86400  # Convert to days
        
        if days_elapsed < 1:  # Less than a day, no decay
            return
        
        # Calculate decay factor
        decay_rate = self.decay_parameters['decay_rate']
        decay_factor = math.exp(-decay_rate * days_elapsed)
        
        # Apply different decay rates based on current score
        boost_threshold = self.decay_parameters['boost_threshold']
        min_retention = self.decay_parameters['min_retention']
        
        # Decay each metric
        for metric_name in ['information_quality', 'network_contribution', 'reliability', 
                           'trustworthiness', 'expertise', 'social_standing']:
            current_value = getattr(metrics, metric_name)
            
            # High-performing nodes decay slower
            if current_value > boost_threshold:
                effective_decay = decay_factor + (1 - decay_factor) * 0.5
            else:
                effective_decay = decay_factor
            
            # Apply decay with minimum retention
            new_value = max(min_retention, current_value * effective_decay)
            setattr(metrics, metric_name, new_value)
    
    async def _update_specific_metrics(self, metrics: ReputationMetrics, interaction: InteractionRecord):
        """Update specific metrics based on interaction type"""
        
        impact = interaction.impact_score
        learning_rate = 0.1  # How quickly reputation changes
        
        # Apply penalty multiplier for negative interactions
        if impact < 0:
            impact *= self.decay_parameters['penalty_multiplier']
        
        # Update metrics based on interaction type
        if interaction.interaction_type == InteractionType.INFORMATION_SHARING:
            await self._update_metric(metrics, 'information_quality', impact, learning_rate)
            await self._update_metric(metrics, 'network_contribution', impact * 0.5, learning_rate)
            
        elif interaction.interaction_type == InteractionType.VALIDATION:
            await self._update_metric(metrics, 'reliability', impact, learning_rate)
            await self._update_metric(metrics, 'trustworthiness', impact * 0.7, learning_rate)
            
        elif interaction.interaction_type == InteractionType.COOPERATION:
            await self._update_metric(metrics, 'social_standing', impact, learning_rate)
            await self._update_metric(metrics, 'network_contribution', impact * 0.8, learning_rate)
            
        elif interaction.interaction_type == InteractionType.CHALLENGE:
            # Challenging false information is positive, but reduces social standing
            await self._update_metric(metrics, 'trustworthiness', abs(impact) * 0.5, learning_rate)
            await self._update_metric(metrics, 'social_standing', impact * 0.3, learning_rate)
            
        elif interaction.interaction_type == InteractionType.SUPPORT:
            await self._update_metric(metrics, 'social_standing', impact, learning_rate)
            await self._update_metric(metrics, 'cooperation', impact * 0.6, learning_rate)
            
        elif interaction.interaction_type == InteractionType.VIOLATION:
            # Policy violations hurt multiple metrics
            await self._update_metric(metrics, 'trustworthiness', impact, learning_rate * 1.5)
            await self._update_metric(metrics, 'reliability', impact * 0.8, learning_rate)
            await self._update_metric(metrics, 'social_standing', impact * 0.6, learning_rate)
            
        elif interaction.interaction_type == InteractionType.EXCELLENCE:
            # Excellence boosts all metrics slightly
            for metric in ['information_quality', 'reliability', 'trustworthiness', 'expertise']:
                await self._update_metric(metrics, metric, impact * 0.4, learning_rate * 0.5)
    
    async def _update_metric(self, metrics: ReputationMetrics, metric_name: str, impact: float, learning_rate: float):
        """Update individual metric with impact"""
        
        current_value = getattr(metrics, metric_name)
        
        # Apply learning rate and clamp to [0, 1]
        adjustment = impact * learning_rate
        new_value = max(0.0, min(1.0, current_value + adjustment))
        
        setattr(metrics, metric_name, new_value)
    
    async def _calculate_overall_reputation(self, metrics: ReputationMetrics):
        """Calculate overall reputation from component metrics"""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in self.reputation_weights.items():
            if metric == ReputationMetric.OVERALL_REPUTATION:
                continue
                
            metric_name = metric.value.replace('_', '_')
            if metric_name in ['info_quality']:
                metric_name = 'information_quality'
            elif metric_name in ['contribution']:
                metric_name = 'network_contribution'
            elif metric_name in ['social']:
                metric_name = 'social_standing'
            
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                weighted_sum += value * weight
                total_weight += weight
        
        if total_weight > 0:
            metrics.overall_reputation = weighted_sum / total_weight
    
    async def _update_confidence_level(self, metrics: ReputationMetrics):
        """Update confidence level based on interaction history"""
        
        # Confidence increases with more interactions and recent activity
        interaction_factor = min(1.0, metrics.interaction_count / 50)  # Max confidence at 50 interactions
        
        # Recent activity factor
        days_since_update = (time.time() - metrics.last_updated) / 86400
        recency_factor = max(0.1, math.exp(-days_since_update / 30))  # Decay over 30 days
        
        # Combined confidence
        metrics.confidence_level = (interaction_factor * 0.7 + recency_factor * 0.3)
    
    async def get_reputation_metrics(self, node_id: str) -> Optional[ReputationMetrics]:
        """Get reputation metrics for node"""
        
        if node_id not in self.reputation_scores:
            return None
            
        metrics = self.reputation_scores[node_id]
        
        # Apply decay before returning
        await self._apply_temporal_decay(metrics)
        await self._calculate_overall_reputation(metrics)
        
        return metrics
    
    async def get_reputation_score(self, node_id: str, metric: ReputationMetric = ReputationMetric.OVERALL_REPUTATION) -> float:
        """Get specific reputation score for node"""
        
        metrics = await self.get_reputation_metrics(node_id)
        if not metrics:
            return 0.5  # Neutral score for unknown nodes
        
        if metric == ReputationMetric.OVERALL_REPUTATION:
            return metrics.overall_reputation
        elif metric == ReputationMetric.INFORMATION_QUALITY:
            return metrics.information_quality
        elif metric == ReputationMetric.NETWORK_CONTRIBUTION:
            return metrics.network_contribution
        elif metric == ReputationMetric.RELIABILITY:
            return metrics.reliability
        elif metric == ReputationMetric.TRUSTWORTHINESS:
            return metrics.trustworthiness
        elif metric == ReputationMetric.EXPERTISE:
            return metrics.expertise
        elif metric == ReputationMetric.SOCIAL_STANDING:
            return metrics.social_standing
        
        return 0.5
    
    async def get_interaction_history(self, node_id: str, limit: Optional[int] = None) -> List[InteractionRecord]:
        """Get interaction history for node"""
        
        history = self.interaction_history.get(node_id, [])
        
        # Sort by timestamp (most recent first)
        sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            return sorted_history[:limit]
        return sorted_history
    
    async def get_reputation_trend(self, node_id: str, days: int = 30) -> Dict:
        """Get reputation trend over time period"""
        
        cutoff_time = time.time() - (days * 86400)
        history = self.interaction_history.get(node_id, [])
        
        # Filter to time period
        recent_interactions = [
            interaction for interaction in history
            if interaction.timestamp >= cutoff_time
        ]
        
        if not recent_interactions:
            return {
                'trend': 'stable',
                'change': 0.0,
                'interaction_count': 0,
                'average_impact': 0.0
            }
        
        # Calculate trend
        impacts = [interaction.impact_score for interaction in recent_interactions]
        average_impact = statistics.mean(impacts)
        
        # Determine trend direction
        if average_impact > 0.1:
            trend = 'improving'
        elif average_impact < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': average_impact,
            'interaction_count': len(recent_interactions),
            'average_impact': average_impact,
            'period_days': days
        }
    
    async def get_top_nodes(self, metric: ReputationMetric = ReputationMetric.OVERALL_REPUTATION, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top nodes by reputation metric"""
        
        node_scores = []
        
        for node_id, metrics in self.reputation_scores.items():
            score = await self.get_reputation_score(node_id, metric)
            node_scores.append((node_id, score))
        
        # Sort by score (highest first)
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        return node_scores[:limit]
    
    async def calculate_node_similarity(self, node_id1: str, node_id2: str) -> float:
        """Calculate similarity between two nodes' reputation profiles"""
        
        metrics1 = await self.get_reputation_metrics(node_id1)
        metrics2 = await self.get_reputation_metrics(node_id2)
        
        if not metrics1 or not metrics2:
            return 0.0
        
        # Compare all metrics
        metric_values1 = [
            metrics1.information_quality,
            metrics1.network_contribution,
            metrics1.reliability,
            metrics1.trustworthiness,
            metrics1.expertise,
            metrics1.social_standing
        ]
        
        metric_values2 = [
            metrics2.information_quality,
            metrics2.network_contribution,
            metrics2.reliability,
            metrics2.trustworthiness,
            metrics2.expertise,
            metrics2.social_standing
        ]
        
        # Calculate cosine similarity
        dot_product = sum(v1 * v2 for v1, v2 in zip(metric_values1, metric_values2))
        magnitude1 = math.sqrt(sum(v1 * v1 for v1 in metric_values1))
        magnitude2 = math.sqrt(sum(v2 * v2 for v2 in metric_values2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_reputation_statistics(self) -> Dict:
        """Get overall reputation system statistics"""
        
        if not self.reputation_scores:
            return {'total_nodes': 0}
        
        # Calculate statistics
        all_scores = [metrics.overall_reputation for metrics in self.reputation_scores.values()]
        all_interactions = sum(len(history) for history in self.interaction_history.values())
        
        return {
            'total_nodes': len(self.reputation_scores),
            'total_interactions': all_interactions,
            'average_reputation': statistics.mean(all_scores) if all_scores else 0.5,
            'reputation_std_dev': statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
            'min_reputation': min(all_scores) if all_scores else 0.5,
            'max_reputation': max(all_scores) if all_scores else 0.5,
            'high_reputation_nodes': sum(1 for score in all_scores if score > 0.7),
            'low_reputation_nodes': sum(1 for score in all_scores if score < 0.3),
            'observer_id': self.node_id
        }
    
    def export_reputation_data(self) -> Dict:
        """Export reputation data for backup or transfer"""
        
        return {
            'node_id': self.node_id,
            'reputation_scores': {
                node_id: metrics.to_dict()
                for node_id, metrics in self.reputation_scores.items()
            },
            'interaction_history': {
                node_id: [interaction.to_dict() for interaction in interactions]
                for node_id, interactions in self.interaction_history.items()
            },
            'exported_at': time.time()
        }